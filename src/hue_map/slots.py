"""Slot detection and cropping.

Finds the Anvil stockpile grid inside a screenshot by looking at the slot
interior colour (a warm olive/tan hue shared by every slot background),
clustering candidate blobs into rows and columns, and then producing a
bounding box for every grid cell - including slots that are partially
cut off at the edge of the screenshot.

Why colour-based detection
--------------------------
Earlier edge-contour approaches missed slots whose frame was clipped by
the screenshot border, merged with the inventory title bar, or had low
edge contrast after scaling. The interior tint, on the other hand, is
always present: empty slots are filled with it, and filled slots still
show it as a ring around the item. That means the same signal works for
both, and we never rely on seeing the full slot frame to find a slot.

Pipeline
--------
1. HSV-mask the screenshot to find pixels matching the slot interior.
2. Morphologically close small gaps and take connected components;
   filter by size, aspect, and shape.
3. Cluster candidate centres into rows and columns (1D mean-shift style).
4. Infer the row/column pitch and extend each axis outward so cells that
   sit beyond the last detected slot but are still visible get produced
   as well (handy for partial bottom/right rows).
5. For each grid cell, produce a bbox sized to the median slot, clipped
   to the image bounds. Cells whose visible area drops below
   ``GRID_MIN_VISIBLE_FRACTION`` are dropped entirely (they're cut off).
6. Only keep cells that overlap an actual detected candidate OR that are
   interior to the detected grid, so we don't emit spurious cells above
   the title bar or beside the real grid.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


LOGGER = logging.getLogger("hue_map.slots")


# -- slot-interior colour signature -----------------------------------------
# Ranges intentionally loose so slight tint shifts across screenshots and
# display settings still match. The slot-interior tan sits in a narrow wedge
# of HSV space well away from frame darks and UI whites.
SLOT_HUE_MIN = 5       # HSV hue is 0-179 in OpenCV; warm yellow-orange wedge
SLOT_HUE_MAX = 32
SLOT_SAT_MIN = 22      # avoid grey / white (low sat)
SLOT_SAT_MAX = 120     # avoid pure saturated colours (items)
SLOT_VAL_MIN = 55      # avoid near-black frame
SLOT_VAL_MAX = 170     # avoid near-white UI panels

# -- connected-component filters --------------------------------------------
SLOT_CLOSE_KERNEL = 3          # tiny close to sew 1-2 pixel gaps in the ring
SLOT_MIN_SIDE_FRACTION = 0.035 # slot side must be >= this fraction of min(h,w)
SLOT_MIN_SIDE_PIXELS = 28      # and also at least this many absolute pixels
# Max side: capped by an absolute ceiling AND by a fraction of the larger
# image dimension. Using max(h,w) rather than min prevents narrow
# screenshots (short inventory panels) from rejecting their own slots.
SLOT_MAX_SIDE_FRACTION = 0.35
SLOT_MAX_SIDE_PIXELS = 280
SLOT_MAX_ASPECT_RATIO = 1.6    # slots are square-ish; reject wildly oblong blobs
# Filled slots show tan as a ring around the item, so density drops into
# the 0.2-0.4 range. Empty slots approach 1.0. Reject truly spidery blobs.
SLOT_MIN_DENSITY = 0.15

# -- grid fitting -----------------------------------------------------------
GRID_AXIS_TOLERANCE_FRACTION = 0.35  # merge points within 35% of median pitch
GRID_MIN_VISIBLE_FRACTION = 0.55     # drop slots clipped to less than 55% area
# Row-extension probe: when detected rows don't cover the full grid
# (items covered every slot's tan ring too aggressively for CC), we probe
# the strip at the next expected row position. A real row has a tan
# fraction comparable to the detected rows; a phantom probe above the
# grid (into title bar / behind-UI area) has noticeably less. We gate on
# both an absolute floor and a *relative* threshold against the detected
# rows' median, which reliably rejects phantom extensions.
GRID_ROW_PROBE_ABSOLUTE_MIN = 0.30      # must exceed this floor
GRID_ROW_PROBE_RELATIVE_MIN = 0.70      # and this fraction of detected-row mean
GRID_ROW_PROBE_MIN_STRIP_COVERAGE = 0.70  # drop probes that mostly clip off-image

# -- border trim preserved for the classifier-compatible contract ----------
DEFAULT_BORDER_FRACTION = 0.06


@dataclass(frozen=True)
class SlotCrop:
    """A cropped slot image paired with its grid position."""

    slot_index: int   # flat reading-order index (0 = top-left)
    row: int          # 0-based row position in the grid
    col: int          # 0-based column position in the grid
    image: np.ndarray # BGR image for this slot (uint8, HxWx3)


def load_screenshot(path: str | Path) -> np.ndarray:
    """Load a screenshot from disk as a BGR NumPy array."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read screenshot: {path}")
    return image


def detect_and_crop_slots(
    image: np.ndarray,
    border_fraction: float = DEFAULT_BORDER_FRACTION,
) -> list[SlotCrop]:
    """Detect slots in a screenshot and return their cropped images.

    Args:
        image: Full stockpile screenshot as a BGR ``ndarray``.
        border_fraction: Fraction of each slot trimmed from the left, right
            and top to remove the slot frame. The bottom is preserved so
            the stack-count row stays in the histogram.

    Returns:
        A list of :class:`SlotCrop` ordered by reading order (row, column).
    """
    image_height, image_width = image.shape[:2]

    candidates = _find_slot_candidates(image)
    if len(candidates) < 2:
        LOGGER.warning("Too few slot candidates detected: %d", len(candidates))
        return []

    grid = _fit_grid(
        candidates,
        image=image,
        image_height=image_height,
        image_width=image_width,
    )
    if not grid:
        return []

    crops: list[SlotCrop] = []
    for slot_index, row, col, bbox in grid:
        x, y, w, h = bbox
        slot_image = image[y : y + h, x : x + w]
        if slot_image.size == 0:
            continue
        crops.append(
            SlotCrop(
                slot_index=slot_index,
                row=row,
                col=col,
                image=_apply_border_crop(slot_image, border_fraction),
            )
        )
    return crops


# -- candidate detection ----------------------------------------------------


def _find_slot_candidates(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Locate slot-sized blobs of slot-interior colour.

    Returns a list of (x, y, w, h) bboxes, one per candidate, without any
    assumptions about their grid relationship yet - that happens later.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[..., 0] >= SLOT_HUE_MIN) & (hsv[..., 0] <= SLOT_HUE_MAX)
        & (hsv[..., 1] >= SLOT_SAT_MIN) & (hsv[..., 1] <= SLOT_SAT_MAX)
        & (hsv[..., 2] >= SLOT_VAL_MIN) & (hsv[..., 2] <= SLOT_VAL_MAX)
    ).astype(np.uint8)

    if SLOT_CLOSE_KERNEL > 1:
        kernel = np.ones((SLOT_CLOSE_KERNEL, SLOT_CLOSE_KERNEL), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    height, width = image.shape[:2]
    min_side = max(SLOT_MIN_SIDE_PIXELS, int(min(height, width) * SLOT_MIN_SIDE_FRACTION))
    max_side = min(SLOT_MAX_SIDE_PIXELS, int(max(height, width) * SLOT_MAX_SIDE_FRACTION))

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    candidates: list[tuple[int, int, int, int]] = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if w < min_side or h < min_side:
            continue
        if w > max_side or h > max_side:
            continue
        if max(w, h) / max(1, min(w, h)) > SLOT_MAX_ASPECT_RATIO:
            continue
        if area / max(1, w * h) < SLOT_MIN_DENSITY:
            continue
        candidates.append((x, y, w, h))

    LOGGER.debug("Slot candidates after filtering: %d", len(candidates))
    return candidates


# -- grid fitting -----------------------------------------------------------


def _fit_grid(
    candidates: list[tuple[int, int, int, int]],
    *,
    image: np.ndarray,
    image_height: int,
    image_width: int,
) -> list[tuple[int, int, int, tuple[int, int, int, int]]]:
    """Infer row/column axes from candidates and emit one bbox per grid cell."""
    widths = np.array([c[2] for c in candidates], dtype=np.int32)
    heights = np.array([c[3] for c in candidates], dtype=np.int32)
    median_w = int(np.median(widths))
    median_h = int(np.median(heights))

    centre_x = np.array([c[0] + c[2] / 2.0 for c in candidates])
    centre_y = np.array([c[1] + c[3] / 2.0 for c in candidates])

    # Cluster detected centres into distinct columns / rows. The tolerance
    # is proportional to the median slot size so a wide grid and a tight
    # grid both work.
    col_positions = _cluster_1d(centre_x, tolerance=median_w * GRID_AXIS_TOLERANCE_FRACTION)
    row_positions = _cluster_1d(centre_y, tolerance=median_h * GRID_AXIS_TOLERANCE_FRACTION)
    if not col_positions or not row_positions:
        return []

    # Recover rows whose individual slots fragmented under heavy item
    # coverage (see ``_extend_rows_by_tan_probe``). This is a targeted
    # extension that only fires when a strip at the expected row has the
    # colour signature of a real row.
    if len(row_positions) >= 1:
        row_positions = _extend_rows_by_tan_probe(
            row_positions=row_positions,
            median_h=median_h,
            image=image,
            image_height=image_height,
        )

    # The cluster positions ARE the grid. We do not extrapolate beyond them:
    # a position that wasn't supported by any detected candidate would land
    # on empty image area (title bar, background) and produce bogus crops.
    # Every (row, col) in the cluster span is emitted as a grid cell.
    grid_cells: list[tuple[int, int, int, tuple[int, int, int, int]]] = []
    slot_index = 0
    for row_idx, cy in enumerate(row_positions):
        for col_idx, cx in enumerate(col_positions):
            full_x = int(round(cx - median_w / 2))
            full_y = int(round(cy - median_h / 2))
            full_w = median_w
            full_h = median_h

            clipped = _clip_bbox(full_x, full_y, full_w, full_h, image_width, image_height)
            if clipped is None:
                continue

            cx_cl, cy_cl, cw_cl, ch_cl = clipped
            visibility = (cw_cl * ch_cl) / max(1, full_w * full_h)
            if visibility < GRID_MIN_VISIBLE_FRACTION:
                continue

            grid_cells.append((slot_index, row_idx, col_idx, (cx_cl, cy_cl, cw_cl, ch_cl)))
            slot_index += 1

    LOGGER.info(
        "Grid fit: %d rows x %d cols -> %d slot cells",
        len(row_positions), len(col_positions), len(grid_cells),
    )
    return grid_cells


def _cluster_1d(values: np.ndarray, *, tolerance: float) -> list[float]:
    """Cluster 1D values: consecutive points within ``tolerance`` merge.

    Returns cluster representatives (medians) sorted ascending. This is a
    dead-simple single-pass clustering that only needs the sorted input -
    fine for <200 candidates and stable for our grid-fitting purposes.
    """
    if values.size == 0:
        return []
    sorted_values = np.sort(values)
    clusters: list[list[float]] = [[float(sorted_values[0])]]
    for value in sorted_values[1:]:
        if float(value) - clusters[-1][-1] <= tolerance:
            clusters[-1].append(float(value))
        else:
            clusters.append([float(value)])
    return [float(np.median(c)) for c in clusters]


def _extend_rows_by_tan_probe(
    *,
    row_positions: list[float],
    median_h: int,
    image: np.ndarray,
    image_height: int,
) -> list[float]:
    """Try to recover rows that connected-components missed.

    When every slot in a row is heavily occluded by an item, the slot-bg
    tan ring breaks into fragments too small to survive the CC filter. But
    if the row is still physically there, the horizontal strip at that row
    position will have plenty of tan pixels in aggregate.

    For each candidate extension (above the first detected row or below
    the last), we sample the full-width strip of height ~= median slot
    height centred at the expected row position. If at least
    ``GRID_ROW_PROBE_TAN_FRACTION`` of pixels in the strip match the
    slot-interior colour, the row is added; otherwise we stop probing in
    that direction. This guards against extending into the title bar,
    background, or off-screen area.
    """
    if not row_positions:
        return row_positions
    if len(row_positions) < 2:
        # Without a second row we can't estimate pitch reliably; fall back
        # to the median slot height as pitch.
        pitch = float(median_h)
    else:
        diffs = np.diff(sorted(row_positions))
        pitch = float(np.median(diffs))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tan_mask = (
        (hsv[..., 0] >= SLOT_HUE_MIN) & (hsv[..., 0] <= SLOT_HUE_MAX)
        & (hsv[..., 1] >= SLOT_SAT_MIN) & (hsv[..., 1] <= SLOT_SAT_MAX)
        & (hsv[..., 2] >= SLOT_VAL_MIN) & (hsv[..., 2] <= SLOT_VAL_MAX)
    )

    extended = sorted(row_positions)
    half_strip = max(4, int(median_h * 0.40))
    full_strip_height = 2 * half_strip

    def strip_tan_fraction(centre_y: float) -> tuple[float, float]:
        """Return (tan_fraction, strip_coverage) for the probe strip.

        ``strip_coverage`` is the fraction of the expected strip that
        actually landed inside the image - lets us reject probes that are
        mostly off-screen and therefore statistically unreliable.
        """
        requested_top = int(round(centre_y - half_strip))
        requested_bottom = int(round(centre_y + half_strip))
        y0 = max(0, requested_top)
        y1 = min(image_height, requested_bottom)
        if y1 <= y0:
            return 0.0, 0.0
        strip = tan_mask[y0:y1, :]
        coverage = (y1 - y0) / max(1, full_strip_height)
        return float(strip.mean()), coverage

    # Establish a baseline from the detected rows. A probe row must look
    # similar (in tan fraction) to the rows we already trust.
    detected_fractions = []
    for centre in row_positions:
        fraction, coverage = strip_tan_fraction(centre)
        if coverage >= GRID_ROW_PROBE_MIN_STRIP_COVERAGE:
            detected_fractions.append(fraction)

    if detected_fractions:
        baseline = float(np.median(detected_fractions))
        threshold = max(GRID_ROW_PROBE_ABSOLUTE_MIN, baseline * GRID_ROW_PROBE_RELATIVE_MIN)
    else:
        threshold = GRID_ROW_PROBE_ABSOLUTE_MIN

    def probe_passes(centre_y: float) -> bool:
        fraction, coverage = strip_tan_fraction(centre_y)
        if coverage < GRID_ROW_PROBE_MIN_STRIP_COVERAGE:
            return False
        return fraction >= threshold

    # Probe upwards (one row at a time, stop at first failure).
    while True:
        next_centre = extended[0] - pitch
        if next_centre - median_h / 2 < -median_h * 0.3:
            break
        if not probe_passes(next_centre):
            break
        extended.insert(0, next_centre)

    # Probe downwards.
    while True:
        next_centre = extended[-1] + pitch
        if next_centre + median_h / 2 > image_height + median_h * 0.3:
            break
        if not probe_passes(next_centre):
            break
        extended.append(next_centre)

    return extended


def _clip_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Clip a bbox to the image rect and return ``None`` if it has no area."""
    x_cl = max(0, x)
    y_cl = max(0, y)
    x2 = min(image_width, x + w)
    y2 = min(image_height, y + h)
    if x2 <= x_cl or y2 <= y_cl:
        return None
    return (x_cl, y_cl, x2 - x_cl, y2 - y_cl)


# -- border trim (unchanged contract) --------------------------------------


def _apply_border_crop(slot_image: np.ndarray, border_fraction: float) -> np.ndarray:
    """Trim a small border from the top/sides while keeping the full bottom."""
    height, width = slot_image.shape[:2]
    border_x = max(0, int(round(width * border_fraction)))
    border_y = max(0, int(round(height * border_fraction)))
    top = min(border_y, max(0, height - 1))
    left = min(border_x, max(0, width - 1))
    right = max(left + 1, width - border_x)
    return slot_image[top:height, left:right]
