"""Color-histogram-based item classifier.

This module builds an in-memory index of reference item histograms and
classifies query crops by comparing against it.

Similarity metric
-----------------
We use the Bhattacharyya coefficient ``sum(sqrt(q * r))`` rather than plain
histogram intersection ``sum(min(q, r))``. Intersection is biased against
sparse references: items like Javelin cover only ~7% of their PNG canvas, so
their histograms are spiky (mass concentrated in a few bins). Intersection
caps the score at the query's mass inside those bins, while broad references
accumulate many small partial overlaps. Bhattacharyya uses a geometric mean
and rewards *shape correlation*, so sparse templates aligned with the right
part of the query score fairly.

Masking
-------
Reference icons: mask out near-transparent (low alpha) and near-black
pixels. The wiki PNGs store transparency as alpha, so this isolates the
actual icon.

Query crops: masked by dropping near-black pixels AND pixels within a
per-channel tolerance of the slot background. The slot background color is
*detected per query* by sampling the four corners of the crop (the icon is
always centered, so corners are always background). This means the matcher
adapts automatically when different screenshots have subtly different slot
tints.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .paths import IMAGE_EXTENSIONS


LOGGER = logging.getLogger("hue_map.color_matcher")


# -- histogram shape --------------------------------------------------------
# OpenCV HSV ranges: H in [0, 180), S and V in [0, 256). We intentionally give
# hue more resolution than saturation/value because most items are identified
# primarily by hue; the 24 * 6 * 6 = 864 bin histogram is a good tradeoff
# between sensitivity and noise.
HUE_BINS = 24
SAT_BINS = 6
VAL_BINS = 6
HISTOGRAM_DIMS = (HUE_BINS, SAT_BINS, VAL_BINS)
HISTOGRAM_RANGES = (0, 180, 0, 256, 0, 256)
HISTOGRAM_CHANNELS = (0, 1, 2)
HISTOGRAM_SIZE = HUE_BINS * SAT_BINS * VAL_BINS

# -- masking thresholds ----------------------------------------------------
MIN_ALPHA = 24                  # pixels more transparent than this are dropped
MIN_VALUE = 18                  # pixels darker than this HSV value are dropped
BACKGROUND_RING_FRACTION = 1 / 12   # perimeter ring width as a fraction of min(h,w)
BACKGROUND_RING_MIN = 2             # minimum ring width in pixels
BACKGROUND_QUANTIZATION_LEVELS = 16  # per-channel buckets used to find the mode
BACKGROUND_CANDIDATE_BUCKETS = 6  # how many top perimeter buckets to consider
BACKGROUND_MIN_BRIGHTNESS = 70    # skip "too dark" buckets (likely slot frame)
BACKGROUND_MAX_BRIGHTNESS = 205   # skip "too bright" buckets (likely white bar / text)

# HSV tolerances for matching a pixel against the detected slot background.
# A pixel is considered background only if it is close on ALL THREE axes;
# this is strictly more discriminative than a BGR L-inf ball because item
# pixels that share slot-bg's warm hue (wooden handles, leather) are
# typically far more saturated and darker, so they escape the envelope.
BACKGROUND_HUE_TOLERANCE = 12      # half-width in OpenCV hue units (0-179)
BACKGROUND_SAT_TOLERANCE = 45      # half-width in saturation (0-255)
BACKGROUND_VAL_TOLERANCE = 45      # half-width in value/brightness (0-255)

# -- interior-hole filling ----------------------------------------------------
# Amorphous items (Animal_Shite, meat, leather) have interior colours that
# share the slot-bg hue, so HSV masking strips them, leaving just the outline.
# After initial masking we close the outline and fill any "holes" that are
# not connected to the image border - those are the item's interior pixels.
# The kernel must be large enough to bridge the gaps in an item's outline
# (Animal_Shite is essentially a ring of darker pixels around a tan interior,
# with breaks of several pixels between ring fragments). Too large would
# merge neighbouring feather strands or knife blades, so we cap at 7.
ITEM_CLOSE_KERNEL_FRACTION = 1 / 12   # close-kernel size as fraction of min(h,w)
ITEM_CLOSE_KERNEL_MIN = 3             # minimum odd kernel size
ITEM_CLOSE_KERNEL_MAX = 7             # maximum odd kernel size
# Only fill holes that are substantial. Amorphous items like Animal_Shite
# have one big interior hole (~15-20% of the slot) that should be filled;
# lattice items like Rope have many small gaps between strands (<5% each)
# that should NOT be filled - doing so would add slot-bg pixels to the
# histogram and ruin the match. The threshold is applied PER HOLE so a
# multi-region item is still filled correctly.
ITEM_HOLE_MIN_FRACTION = 0.08

# -- empty-slot detection ---------------------------------------------------
# An empty slot carries nothing but slot background plus a little UI chrome,
# so almost every pixel is masked out. Items that still appear non-empty in
# the sparsest case (Javelin, thin knives) typically keep ~7-10% of the
# crop; healthy filled slots keep 20%+. We use two signals:
#
#   1. Total kept-pixel fraction - catches obviously-empty slots.
#   2. Shape of the largest connected component - catches slots whose only
#      "content" is a thin UI strip (leftover quality bar, stack-count row
#      edge) rather than a real item.
#
# A real item's largest blob is always at least a few percent of the crop
# AND has a roughly item-like aspect ratio; a 1-pixel-tall strip failing
# either test is treated as empty.
EMPTY_SLOT_FRACTION = 0.04
EMPTY_SLOT_MIN_PIXELS = 60
EMPTY_SLOT_MAIN_BLOB_FRACTION = 0.03    # largest blob must cover >= this fraction
EMPTY_SLOT_MAIN_BLOB_MIN_PIXELS = 50    # and at least this many absolute pixels
EMPTY_SLOT_MAX_ASPECT_RATIO = 9.0       # longer / shorter side of blob's bbox
# Frame-leak detector: a largest blob whose bounding box spans almost the
# whole slot AND whose density inside that bbox is very low is a scattered
# outline of residual frame pixels, not a real item. Real items either have
# a bbox smaller than the slot (centred) or fill their bbox densely. A low
# *total* kept-pixel fraction is the third guardrail - real-but-thin items
# like Javelin can touch slot corners, but they still keep 15%+ of pixels
# whereas frame leaks max out around 10%.
EMPTY_SLOT_FRAME_BBOX_FRACTION = 0.90   # bbox must cover at least 90% of one axis
EMPTY_SLOT_FRAME_MIN_DENSITY = 0.10     # and the blob must fill < 10% of its bbox
EMPTY_SLOT_FRAME_MAX_TOTAL_FRACTION = 0.12  # and total kept pixels must be < 12%

# -- UI chrome detection ----------------------------------------------------
# Anvil slots carry three overlays we want to exclude from the item mask:
#
#   * a coloured quality badge in the top-left corner (green/orange/red cross)
#   * a white quality bar along the left edge
#   * the stack-count text in the bottom-right
#
# Each overlay has a reliable spatial location plus a colour signature
# (brightness / saturation). Region sizes are expressed as fractions of the
# crop dimensions so they adapt automatically to different slot sizes.

# Region fractions (height, width). Pixels outside these regions are never
# flagged as that particular kind of chrome.
CHROME_TOPLEFT_H_FRAC = 0.34      # top 34% rows covered by the top-left region
CHROME_TOPLEFT_W_FRAC = 0.30      # left 30% columns of the top-left region
CHROME_LEFT_EDGE_W_FRAC = 0.10    # leftmost 10% columns for the quality bar
CHROME_BOTTOMRIGHT_H_FRAC = 0.35  # bottom 35% rows for the stack-count area
CHROME_BOTTOMRIGHT_W_FRAC = 0.55  # right 55% columns for the stack-count area

# Signature thresholds. HSV with OpenCV conventions (H: 0-179, S/V: 0-255).
CHROME_SATURATED_MIN = 120        # "clearly saturated" - quality badge green/orange/red
CHROME_BRIGHT_MIN = 190           # "bright" - white bar + number text
CHROME_WHITE_SAT_MAX = 55         # "white-ish" - paired with CHROME_BRIGHT_MIN

# Only these reference sub-folders are consumed. A ``rejected/`` folder may
# sit alongside them and is ignored.
USABLE_REFERENCE_FOLDERS = ("wiki", "approved")

DEFAULT_TOP_K = 5
DEFAULT_MAX_PATHS_PER_ITEM = 3


@dataclass(frozen=True)
class ItemReference:
    """A single reference image with its pre-computed histogram."""

    item_name: str
    image_path: Path
    source_folder: str
    histogram: np.ndarray        # shape (HISTOGRAM_SIZE,), L1-normalized
    mask_pixel_count: int        # number of pixels that survived masking


@dataclass
class ReferenceIndex:
    """In-memory catalog of all reference histograms.

    ``histogram_matrix`` is a stacked ``(N, HISTOGRAM_SIZE)`` array so we can
    compute similarity against every reference with a single matrix op.
    """

    references_root: Path
    references: list[ItemReference]
    histogram_matrix: np.ndarray
    item_to_indices: dict[str, list[int]]

    @property
    def item_count(self) -> int:
        return len(self.item_to_indices)

    @property
    def reference_count(self) -> int:
        return len(self.references)


# -- public API -------------------------------------------------------------


def build_reference_index(references_root: str | Path) -> ReferenceIndex:
    """Scan ``references_root`` and build an in-memory histogram index.

    The expected folder layout is::

        references_root/
            <Item_Name>/
                wiki/*.png          (scraped reference art)
                approved/*.png      (human-verified crops, optional)
                rejected/*.png      (ignored)

    Raises:
        RuntimeError: If the root doesn't exist or has no usable references.
    """
    references_by_item = _load_reference_paths(references_root)
    if not references_by_item:
        raise RuntimeError(
            f"No usable item references found under {references_root}. "
            "Expected folders with wiki/ or approved/ images."
        )

    references: list[ItemReference] = []
    histograms: list[np.ndarray] = []
    item_to_indices: dict[str, list[int]] = {}

    for item_name, image_paths in references_by_item.items():
        for image_path in image_paths:
            fingerprint = compute_image_histogram(image_path, assume_reference=True)
            if fingerprint is None:
                LOGGER.warning("Skipping unreadable reference %s", image_path)
                continue

            histogram, mask_pixel_count = fingerprint
            if mask_pixel_count == 0:
                LOGGER.warning("Skipping reference with empty mask %s", image_path)
                continue

            reference = ItemReference(
                item_name=item_name,
                image_path=image_path,
                source_folder=_source_folder_name(image_path),
                histogram=histogram,
                mask_pixel_count=mask_pixel_count,
            )
            item_to_indices.setdefault(item_name, []).append(len(references))
            references.append(reference)
            histograms.append(histogram)

    if not references:
        raise RuntimeError(
            f"No item color references could be indexed from {references_root}"
        )

    histogram_matrix = np.stack(histograms).astype(np.float32)
    LOGGER.info(
        "Built color reference index: %d items, %d reference images",
        len(item_to_indices), len(references),
    )

    return ReferenceIndex(
        references_root=Path(references_root),
        references=references,
        histogram_matrix=histogram_matrix,
        item_to_indices=item_to_indices,
    )


def classify_item_crop(
    image: str | Path | np.ndarray,
    reference_index: ReferenceIndex,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_reference_paths_per_item: int = DEFAULT_MAX_PATHS_PER_ITEM,
) -> dict[str, Any]:
    """Classify a single item crop against the reference index.

    Args:
        image: Either a path to an image file or an in-memory BGR/BGRA
            ``ndarray``. BGR is assumed for 3-channel arrays.
        reference_index: Built via :func:`build_reference_index`.
        top_k: Number of candidate items to return.
        max_reference_paths_per_item: How many reference image paths to
            include per candidate (for human inspection).

    Returns:
        A dictionary with ``predicted_item``, ``confidence``, ``top_gap``
        (difference between the top two candidate scores), ``top_matches``,
        ``matched_reference_paths``, and ``error``.
    """
    image_path_str = ""
    loaded_bgra: np.ndarray | None = None

    if isinstance(image, (str, Path)):
        query_path = Path(image)
        image_path_str = str(query_path)
        if not query_path.exists():
            return _empty_result(image_path_str, error=f"Image crop does not exist: {query_path}")
        loaded_bgra = cv2.imread(str(query_path), cv2.IMREAD_UNCHANGED)
        if loaded_bgra is None:
            return _empty_result(image_path_str, error="Could not read or decode query image")
    else:
        if image.size == 0:
            return _empty_result(image_path_str, error="Empty image array")
        loaded_bgra = image

    if reference_index.histogram_matrix.size == 0 or not reference_index.references:
        return _empty_result(image_path_str, error="Color reference index is empty")

    # Build the query mask ahead of classification. We need it to decide
    # emptiness (shape + pixel count) and it feeds straight into the
    # histogram, so we compute it once and reuse.
    bgra_full = _ensure_bgra(loaded_bgra)
    bgr = bgra_full[..., :3]
    alpha = bgra_full[..., 3]
    total_pixels = int(bgr.shape[0] * bgr.shape[1])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    keep_mask = np.ones(alpha.shape, dtype=bool)
    keep_mask &= alpha >= MIN_ALPHA
    keep_mask &= hsv[..., 2] >= MIN_VALUE
    slot_background = _sample_slot_background(bgr)
    keep_mask &= ~_border_connected_background(hsv, slot_background)
    keep_mask &= ~_detect_ui_chrome_mask(bgr, hsv)
    keep_mask = _fill_item_interior(keep_mask)

    # Empty-slot short-circuit: either few pixels overall OR the largest
    # connected blob is too small / too thin to be a real item.
    if _is_empty_slot(keep_mask, total_pixels):
        return _empty_slot_result(image_path_str, int(keep_mask.sum()), total_pixels)

    fingerprint = _histogram_from_mask(hsv, keep_mask)
    query_histogram, query_pixels = fingerprint

    if query_pixels == 0:
        return _empty_result(image_path_str, error="Query mask is empty after background removal")

    # Bhattacharyya coefficient: geometric mean of the two distributions.
    # Range is [0, 1]; identical distributions give 1.
    similarities = np.sqrt(
        reference_index.histogram_matrix * query_histogram[None, :]
    ).sum(axis=1)

    # Group scores by item name (an item may have multiple reference images).
    scored_by_item: dict[str, list[tuple[float, Path]]] = {}
    for index, similarity in enumerate(similarities):
        reference = reference_index.references[index]
        scored_by_item.setdefault(reference.item_name, []).append(
            (float(similarity), reference.image_path)
        )

    # For each item keep the best reference score, plus a small number of
    # representative reference paths for human inspection.
    aggregated_matches: list[dict[str, Any]] = []
    for item_name, scored_references in scored_by_item.items():
        ranked = sorted(scored_references, key=lambda pair: pair[0], reverse=True)
        best_score = ranked[0][0]
        top_scores = [score for score, _ in ranked[:max_reference_paths_per_item]]
        mean_top_score = float(np.mean(top_scores)) if top_scores else best_score
        matched_paths = [str(path) for _, path in ranked[:max_reference_paths_per_item]]

        aggregated_matches.append(
            {
                "item": item_name,
                "confidence": round(best_score, 4),
                "best_score": round(best_score, 4),
                "mean_top_score": round(mean_top_score, 4),
                "reference_count": len(scored_references),
                "matched_reference_paths": matched_paths,
            }
        )

    top_matches = sorted(
        aggregated_matches, key=lambda item: item["confidence"], reverse=True
    )[:top_k]
    if not top_matches:
        return _empty_result(image_path_str, error="No color matches produced")

    predicted = top_matches[0]
    second_confidence = float(top_matches[1]["confidence"]) if len(top_matches) > 1 else 0.0
    confidence = float(predicted["confidence"])

    return {
        "image_path": image_path_str,
        "predicted_item": predicted["item"],
        "confidence": confidence,
        "top_gap": round(confidence - second_confidence, 4) if len(top_matches) > 1 else None,
        "top_matches": top_matches,
        "matched_reference_paths": predicted["matched_reference_paths"],
        "error": None,
    }


def batch_classify_item_crops(
    images: Iterable[str | Path | np.ndarray],
    reference_index: ReferenceIndex,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Convenience wrapper: classify every item in ``images`` in order."""
    return [classify_item_crop(image, reference_index, top_k=top_k) for image in images]


def compute_image_histogram(
    image_path: str | Path,
    *,
    assume_reference: bool,
) -> tuple[np.ndarray, int] | None:
    """Load ``image_path`` and compute its masked, normalized HSV histogram.

    Returns ``(histogram, kept_pixel_count)`` or ``None`` if the file can't
    be decoded.
    """
    bgra = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if bgra is None:
        return None
    return compute_image_histogram_from_array(bgra, assume_reference=assume_reference)


def compute_image_histogram_from_array(
    image: np.ndarray,
    *,
    assume_reference: bool,
) -> tuple[np.ndarray, int]:
    """Compute the masked, L1-normalized HSV histogram of an in-memory image.

    ``image`` may be grayscale, BGR, or BGRA (OpenCV's native ordering). For
    query crops we additionally detect and mask the slot background and UI
    chrome overlays; references rely on their alpha channel instead.
    """
    bgra = _ensure_bgra(image)
    bgr = bgra[..., :3]
    alpha = bgra[..., 3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Start permissive, then subtract pixels we don't want to count.
    mask = np.ones(alpha.shape, dtype=bool)
    mask &= alpha >= MIN_ALPHA                  # drop transparent pixels
    mask &= hsv[..., 2] >= MIN_VALUE            # drop near-black pixels

    if not assume_reference:
        # Slot background detection (see ``_sample_slot_background``) - done
        # per-query so we adapt to screenshots that have slightly different
        # slot tints without any hardcoded colour.
        slot_background = _sample_slot_background(bgr)
        mask &= ~_border_connected_background(hsv, slot_background)

        # UI chrome detection (quality badge, left bar, stack count).
        mask &= ~_detect_ui_chrome_mask(bgr, hsv)

        # Fallback for amorphous items whose interior is large enough to
        # be classified as slot-bg BUT doesn't connect to the border
        # because the outline is continuous enough to enclose it. These
        # were already kept by the border-connected test above, but the
        # interior fill still catches the corner case where the outline
        # closes up well enough for the interior region to be labelled as
        # bg and we need it back.
        mask = _fill_item_interior(mask)

    return _histogram_from_mask(hsv, mask)


def build_query_keep_mask(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the "item pixels" boolean mask for a slot crop.

    The same mask logic used by :func:`classify_item_crop` is exposed here so
    other modules (the pipeline, reporting) can produce cleaned previews
    consistent with what the classifier sees.

    Returns:
        A ``(keep_mask, slot_background)`` tuple. ``keep_mask`` is True where
        a pixel belongs to the item (not slot bg, not UI chrome, not black).
    """
    bgr_array = _ensure_bgra(bgr)[..., :3]
    slot_background = _sample_slot_background(bgr_array)
    hsv = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)

    keep = np.ones(bgr_array.shape[:2], dtype=bool)
    keep &= hsv[..., 2] >= MIN_VALUE
    keep &= ~_border_connected_background(hsv, slot_background)
    keep &= ~_detect_ui_chrome_mask(bgr_array, hsv)
    keep = _fill_item_interior(keep)
    return keep, slot_background


def cleaned_rgba(bgr: np.ndarray) -> np.ndarray:
    """Return a 4-channel BGRA version of ``bgr`` with non-item pixels transparent.

    Use this to render a "clean" thumbnail that shows the item icon on a
    transparent background, with the slot tint, quality badge, quality bar
    and stack count all removed.
    """
    bgr_array = _ensure_bgra(bgr)[..., :3]
    keep, _ = build_query_keep_mask(bgr_array)
    alpha = (keep.astype(np.uint8) * 255)
    return np.dstack([bgr_array, alpha])


# -- helpers ----------------------------------------------------------------


def _ensure_bgra(image: np.ndarray) -> np.ndarray:
    """Return a BGRA view of ``image`` regardless of its original channel count."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image


def _sample_slot_background(bgr: np.ndarray) -> np.ndarray:
    """Estimate the slot background colour from a perimeter ring.

    The icon is always roughly centred, so a thin ring around the edges of
    the crop is almost entirely slot background. Several UI elements muddy
    this:

    * Dark, rounded slot frames can dominate the perimeter on some UIs.
    * A white "quality bar" runs along the left edge of every slot.
    * Numeric stack counts spill into the bottom-right of the perimeter.

    A naive single-bucket mode can lock onto any of those and miss the
    actual flat slot tint. To be robust we:

    1. Gather perimeter-ring pixels.
    2. Quantise into coarse colour buckets and count occupancy.
    3. Consider the top-N most populous buckets.
    4. Reject candidates whose representative brightness is outside a
       plausible slot-tint range (too dark = frame shadow, too bright =
       white bar / number text).
    5. Return the median colour of the highest-count surviving bucket.

    If no candidate survives the brightness filter we fall back to the
    overall perimeter median - usually better than a guaranteed-wrong pick.
    """
    height, width = bgr.shape[:2]
    ring = max(BACKGROUND_RING_MIN, int(min(height, width) * BACKGROUND_RING_FRACTION))
    ring = min(ring, max(1, height // 2), max(1, width // 2))

    # Gather the four edges of the ring. Top/bottom include the full width;
    # left/right only fill the inner rows so corners aren't double-counted.
    top = bgr[:ring, :].reshape(-1, 3)
    bottom = bgr[-ring:, :].reshape(-1, 3)
    left = bgr[ring : height - ring, :ring].reshape(-1, 3)
    right = bgr[ring : height - ring, -ring:].reshape(-1, 3)
    perimeter = np.concatenate([top, bottom, left, right])

    if perimeter.size == 0:
        return np.array([0, 0, 0], dtype=np.int16)

    # Quantise each channel and count occupancy per 3D colour bucket.
    levels = BACKGROUND_QUANTIZATION_LEVELS
    bucket_size = 256 // levels
    quantised = (perimeter // bucket_size).astype(np.int32)
    keys = (quantised[:, 0] * levels + quantised[:, 1]) * levels + quantised[:, 2]
    counts = np.bincount(keys, minlength=levels ** 3)

    # Evaluate the top-N candidate buckets in order of popularity.
    top_candidates = np.argsort(counts)[::-1][:BACKGROUND_CANDIDATE_BUCKETS]
    for bucket in top_candidates:
        if counts[bucket] == 0:
            break
        bucket_pixels = perimeter[keys == bucket]
        representative = np.median(bucket_pixels, axis=0)
        # Approximate HSV "V" (brightness) as the max channel, matching the
        # definition used by cv2.cvtColor(BGR2HSV). Reject buckets that
        # represent slot frame shadow or white-bar / number-text pixels.
        brightness = float(representative.max())
        if BACKGROUND_MIN_BRIGHTNESS <= brightness <= BACKGROUND_MAX_BRIGHTNESS:
            return representative.astype(np.int16)

    # Fallback: overall perimeter median if nothing looks plausible.
    return np.median(perimeter, axis=0).astype(np.int16)


def _histogram_from_mask(hsv: np.ndarray, keep_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Compute the L1-normalised HSV histogram over the given mask.

    Shared between the classifier's query path (where we want to reuse a
    pre-built mask) and the reference-index builder (via
    ``compute_image_histogram_from_array``). Returns an all-zero histogram
    if no pixels survive the mask.
    """
    mask_u8 = keep_mask.astype(np.uint8)
    pixel_count = int(mask_u8.sum())
    if pixel_count == 0:
        return np.zeros(HISTOGRAM_SIZE, dtype=np.float32), 0

    histogram = cv2.calcHist(
        [hsv],
        list(HISTOGRAM_CHANNELS),
        mask_u8,
        list(HISTOGRAM_DIMS),
        list(HISTOGRAM_RANGES),
    )
    flattened = histogram.flatten().astype(np.float32)
    total = flattened.sum()
    if total > 0:
        flattened /= total
    return flattened, pixel_count


def _fill_item_interior(keep_mask: np.ndarray) -> np.ndarray:
    """Restore interior pixels of items that were falsely masked as slot bg.

    The HSV-distance bg check works well for pixels whose *hue* differs from
    the slot tint (weapons, metals, bones, fish). Amorphous organic items
    (Animal_Shite, meat, leather) have large interior regions that share
    the slot-bg hue, so the mask hollows them out, leaving only an outline.

    To restore those interiors we:

    1. Apply a morphological close so small gaps in the outline seal shut.
    2. Inspect the "bg-masked" region's connected components.
    3. Any component NOT touching the image border is an interior hole
       (genuine slot background is always connected to the border) - that
       hole is added back to the item mask.

    Items that legitimately have openings connected to the edge (e.g. a
    shield with a hole reaching to the boundary) are left alone because
    their gap remains reachable from the border.
    """
    height, width = keep_mask.shape
    if not keep_mask.any():
        return keep_mask

    # Morphological close: dilate then erode. Odd kernel size so it has a centre.
    raw_kernel = int(min(height, width) * ITEM_CLOSE_KERNEL_FRACTION)
    kernel_size = min(ITEM_CLOSE_KERNEL_MAX, max(ITEM_CLOSE_KERNEL_MIN, raw_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(keep_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    # Label the connected components of the *bg* (the complement of ``closed``).
    bg_region = (~closed).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(bg_region)
    if num_labels <= 1:
        return keep_mask | (closed & ~keep_mask)

    # Components touching the outer edge are genuine slot background.
    border_labels = set(labels[0, :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, -1].tolist())

    # Anything not in that set is an enclosed hole; restore it ONLY if it's
    # big enough to plausibly be the item's solid interior. Small holes
    # are lattice gaps (rope strands, chain links, feather fans) and
    # filling them adds slot-bg pixels the histogram shouldn't see.
    min_hole_pixels = int(round(height * width * ITEM_HOLE_MIN_FRACTION))
    interior_hole_mask = np.zeros_like(keep_mask, dtype=bool)
    for label in range(1, num_labels):
        if label in border_labels:
            continue
        hole = (labels == label)
        if int(hole.sum()) < min_hole_pixels:
            continue
        interior_hole_mask |= hole

    # Keep original outline pixels plus any substantial interior holes.
    return keep_mask | interior_hole_mask


def _border_connected_background(
    hsv: np.ndarray,
    slot_background_bgr: np.ndarray,
) -> np.ndarray:
    """Identify slot-bg pixels that are actually "outside" the item.

    Naive HSV matching has a nasty edge case: light-brown or tan item
    *highlights* (rope fibres, wooden handles) often sit inside the
    slot-bg HSV envelope and therefore get masked away, leaving the
    histogram biased toward only the item's dark/shadowed pixels. The
    reference icon, which has no such masking, ends up much brighter and
    the Bhattacharyya score collapses.

    Real slot background is always a large area connected to the image
    border. Item-coloured pixels that happen to match slot-bg's HSV are
    interior and do NOT connect to the border. So we first compute the
    naive HSV match and then keep only its border-connected connected
    components; enclosed bg-coloured regions are returned unmasked.

    (The ``_fill_item_interior`` post-pass still handles the corner case
    where an item's interior forms a genuine enclosed hole that we do
    want to keep as part of the item.)
    """
    candidate = _is_slot_background_pixel(hsv, slot_background_bgr)
    if not candidate.any():
        return candidate

    num_labels, labels = cv2.connectedComponents(candidate.astype(np.uint8))
    if num_labels <= 1:
        return candidate

    # Any component touching the outer edge is real slot background; flag
    # every pixel carrying one of those labels.
    border_labels = set(labels[0, :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, -1].tolist())
    border_labels.discard(0)  # 0 is the "background" (non-candidate) label

    if not border_labels:
        return np.zeros_like(candidate)

    label_array = np.arange(num_labels, dtype=np.int32)
    is_border = np.isin(label_array, list(border_labels))
    return is_border[labels]


def _is_slot_background_pixel(hsv: np.ndarray, slot_background_bgr: np.ndarray) -> np.ndarray:
    """Per-pixel bool mask: True where the pixel matches the slot background.

    Matching is performed in HSV space rather than BGR because item pixels
    that share the slot-bg's warm hue (wooden handles, leather, tan stone)
    are typically much more saturated and darker than the flat slot tint.
    Comparing hue / saturation / value independently lets us keep those
    item pixels while still masking genuine slot-bg pixels, even when the
    detected background colour is slightly off.
    """
    slot_bg_uint8 = np.asarray(slot_background_bgr, dtype=np.uint8).reshape(1, 1, 3)
    slot_bg_hsv = cv2.cvtColor(slot_bg_uint8, cv2.COLOR_BGR2HSV)[0, 0].astype(np.int16)

    # Hue is circular (0-179 in OpenCV), so the shortest distance around
    # the wheel is ``min(dh, 180 - dh)``.
    hue_delta = np.abs(hsv[..., 0].astype(np.int16) - int(slot_bg_hsv[0]))
    hue_delta = np.minimum(hue_delta, 180 - hue_delta)
    sat_delta = np.abs(hsv[..., 1].astype(np.int16) - int(slot_bg_hsv[1]))
    val_delta = np.abs(hsv[..., 2].astype(np.int16) - int(slot_bg_hsv[2]))

    return (
        (hue_delta <= BACKGROUND_HUE_TOLERANCE)
        & (sat_delta <= BACKGROUND_SAT_TOLERANCE)
        & (val_delta <= BACKGROUND_VAL_TOLERANCE)
    )


def _detect_ui_chrome_mask(bgr: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    """Flag pixels belonging to Anvil's UI chrome inside a slot crop.

    Three overlays are detected using spatial regions combined with colour
    signatures in HSV. Because the regions are defined as fractions of the
    crop dimensions, this works across different screenshot resolutions and
    slot sizes without any hardcoded pixel counts.

    Overlays handled:

    * ``quality badge`` - top-left corner, highly saturated.
    * ``quality bar``   - left edge, bright and low saturation.
    * ``stack count``   - bottom-right, bright and low saturation.
    """
    height, width = bgr.shape[:2]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    chrome = np.zeros((height, width), dtype=bool)

    # -- Top-left quality badge: clearly-coloured pixels in the TL sector. --
    tl_h = max(1, int(round(height * CHROME_TOPLEFT_H_FRAC)))
    tl_w = max(1, int(round(width * CHROME_TOPLEFT_W_FRAC)))
    top_left = (slice(0, tl_h), slice(0, tl_w))
    chrome[top_left] |= saturation[top_left] >= CHROME_SATURATED_MIN

    # -- Left quality bar: bright, low-saturation pixels along the left edge. --
    le_w = max(1, int(round(width * CHROME_LEFT_EDGE_W_FRAC)))
    left_edge = (slice(None), slice(0, le_w))
    chrome[left_edge] |= (
        (value[left_edge] >= CHROME_BRIGHT_MIN)
        & (saturation[left_edge] <= CHROME_WHITE_SAT_MAX)
    )

    # -- Bottom-right stack count: bright, low-sat pixels in the BR sector. --
    br_h_start = min(height - 1, int(round(height * (1 - CHROME_BOTTOMRIGHT_H_FRAC))))
    br_w_start = min(width - 1, int(round(width * (1 - CHROME_BOTTOMRIGHT_W_FRAC))))
    bottom_right = (slice(br_h_start, None), slice(br_w_start, None))
    chrome[bottom_right] |= (
        (value[bottom_right] >= CHROME_BRIGHT_MIN)
        & (saturation[bottom_right] <= CHROME_WHITE_SAT_MAX)
    )

    return chrome


def _load_reference_paths(references_root: str | Path) -> dict[str, list[Path]]:
    """Walk ``references_root`` and collect image paths grouped by item name."""
    root = Path(references_root)
    if not root.exists() or not root.is_dir():
        LOGGER.warning("References root missing or not a directory: %s", root)
        return {}

    references: dict[str, list[Path]] = {}
    for item_dir in sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    ):
        # Skip underscore-prefixed helper folders like _global_rejected.
        if item_dir.name.startswith("_"):
            continue
        item_paths: list[Path] = []
        for folder_name in USABLE_REFERENCE_FOLDERS:
            source_dir = item_dir / folder_name
            if source_dir.is_dir():
                item_paths.extend(_discover_images(source_dir))
        if item_paths:
            references[item_dir.name] = sorted(item_paths, key=lambda path: str(path).casefold())

    LOGGER.info(
        "Loaded %d item folder(s) with %d usable reference image(s) from %s",
        len(references),
        sum(len(paths) for paths in references.values()),
        root,
    )
    return references


def _discover_images(folder: Path) -> list[Path]:
    """Recursively list images inside ``folder`` while ignoring rejected/."""
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "rejected" not in {part.casefold() for part in path.parts}
    )


def _source_folder_name(image_path: Path) -> str:
    """Return whichever of ``wiki``/``approved`` this image lives inside."""
    for part in image_path.parts:
        if part in USABLE_REFERENCE_FOLDERS:
            return part
    return ""


def _empty_result(image_path: str, error: str | None = None) -> dict[str, Any]:
    """Return a uniformly-shaped empty / error classification result.

    Used when classification could not run (missing file, undecodable image,
    empty reference index). ``is_empty`` is False because this represents a
    *failure*, not a deliberately-empty slot.
    """
    return {
        "image_path": image_path,
        "predicted_item": "",
        "confidence": 0.0,
        "top_gap": None,
        "top_matches": [],
        "matched_reference_paths": [],
        "is_empty": False,
        "kept_pixel_count": 0,
        "kept_pixel_fraction": 0.0,
        "error": error,
    }


def _empty_slot_result(
    image_path: str,
    kept_pixels: int,
    total_pixels: int,
) -> dict[str, Any]:
    """Return a result indicating the slot is intentionally empty.

    The HTML report and downstream tooling can inspect ``is_empty`` to
    render an "empty" state rather than a low-confidence prediction.
    """
    fraction = kept_pixels / total_pixels if total_pixels > 0 else 0.0
    return {
        "image_path": image_path,
        "predicted_item": "",
        "confidence": 0.0,
        "top_gap": None,
        "top_matches": [],
        "matched_reference_paths": [],
        "is_empty": True,
        "kept_pixel_count": int(kept_pixels),
        "kept_pixel_fraction": round(fraction, 4),
        "error": None,
    }


def _is_empty_slot(keep_mask: np.ndarray, total_pixels: int) -> bool:
    """Return True if the keep mask indicates an empty slot.

    Two-stage check so that a thin strip of residual UI chrome doesn't get
    classified as an item just because it has "enough" pixels overall:

    1. **Gross pixel count** - if there are barely any kept pixels at all,
       the slot is empty regardless of shape.
    2. **Largest connected component** - a real item is contiguous and
       roughly proportioned; if the biggest blob is too small or has an
       extreme aspect ratio (e.g. a 3-pixel-tall horizontal strip spanning
       the bottom of the slot), the "content" is chrome leakage and the
       slot should be flagged empty.
    """
    if total_pixels <= 0:
        return True

    kept_pixels = int(keep_mask.sum())
    if kept_pixels < EMPTY_SLOT_MIN_PIXELS:
        return True
    if kept_pixels / total_pixels < EMPTY_SLOT_FRACTION:
        return True

    # Largest connected component analysis.
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        keep_mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return True

    # stats rows for labels 1..N-1 are the foreground components (index 0 is
    # the whole-image background label and is ignored).
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return True
    largest_component_row = 1 + int(areas.argmax())
    largest_area = int(stats[largest_component_row, cv2.CC_STAT_AREA])

    if largest_area < EMPTY_SLOT_MAIN_BLOB_MIN_PIXELS:
        return True
    if largest_area / total_pixels < EMPTY_SLOT_MAIN_BLOB_FRACTION:
        return True

    # Aspect-ratio sanity check on the largest blob's bounding box.
    width = int(stats[largest_component_row, cv2.CC_STAT_WIDTH])
    height = int(stats[largest_component_row, cv2.CC_STAT_HEIGHT])
    if width <= 0 or height <= 0:
        return True
    aspect = max(width, height) / max(1, min(width, height))
    if aspect > EMPTY_SLOT_MAX_ASPECT_RATIO:
        return True

    # Frame-leak detector: a bbox that spans almost the whole slot but is
    # sparsely filled AND has few total kept pixels is residual frame /
    # outline noise. Real but thin items (Javelin on certain screenshots)
    # may hit the span+density conditions, so the total-fraction gate lets
    # them through - their mask keeps 15%+ of pixels while frame leaks
    # stay below 12%.
    image_height = int(keep_mask.shape[0])
    image_width = int(keep_mask.shape[1])
    bbox_span = max(width / max(1, image_width), height / max(1, image_height))
    bbox_density = largest_area / max(1, width * height)
    total_fraction = kept_pixels / total_pixels
    if (
        bbox_span >= EMPTY_SLOT_FRAME_BBOX_FRACTION
        and bbox_density < EMPTY_SLOT_FRAME_MIN_DENSITY
        and total_fraction < EMPTY_SLOT_FRAME_MAX_TOTAL_FRACTION
    ):
        return True

    return False
