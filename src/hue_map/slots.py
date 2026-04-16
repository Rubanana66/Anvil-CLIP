"""Slot detection and cropping.

We reuse the slot-grid detection from the repo-root ``parser.py`` (it does
the heavy lifting of finding the stockpile region and splitting it into
individual slot bounding boxes), but we do our own cropping. The upstream
parser crops the bottom 28% off each slot to isolate the icon; we keep
the whole bottom so the stack-count row is part of what we histogram.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import REPO_ROOT


# Make the repo root importable so we can reuse parser.py without copying it.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import parser as anvil_parser  # noqa: E402  (import after sys.path tweak)


# Fraction of slot width/height trimmed off the left, right, and top edges to
# discard the slot frame. The bottom is never trimmed; that's the whole point.
DEFAULT_BORDER_FRACTION = 0.06

# Pixels added to each side of the parser-detected bbox before cropping.
# Zero by default - the parser's bboxes already contain full items in practice,
# and enlarging them re-introduces the slot frame which biases bg detection.
DEFAULT_BBOX_PAD = 0


@dataclass(frozen=True)
class SlotCrop:
    """A cropped slot image paired with its grid position in the stockpile."""

    slot_index: int   # flat reading-order index (0 = top-left)
    row: int          # 0-based row position in the grid
    col: int          # 0-based column position in the grid
    image: np.ndarray # BGR image for this slot (uint8, HxWx3)


def load_screenshot(path: str | Path) -> np.ndarray:
    """Load a screenshot from disk as a BGR NumPy array."""
    return anvil_parser.load_image(path)


def detect_and_crop_slots(
    image: np.ndarray,
    border_fraction: float = DEFAULT_BORDER_FRACTION,
    bbox_pad: int = DEFAULT_BBOX_PAD,
) -> list[SlotCrop]:
    """Detect slots in a screenshot and return their cropped images.

    Args:
        image: Full stockpile screenshot as a BGR ``ndarray``.
        border_fraction: Fraction of each slot trimmed from the left, right
            and top to remove the slot frame. The bottom is preserved.
        bbox_pad: Pixels added to each side of the parser's slot bbox before
            cropping. Gives a small safety margin so items are never clipped.

    Returns:
        A list of :class:`SlotCrop` ordered by (row, column).
    """
    # Find and isolate the stockpile grid region within the screenshot.
    stockpile_bbox = anvil_parser.detect_stockpile_region(image)
    if stockpile_bbox is None:
        # Fall back to the full image so we still produce *something*.
        stockpile_image = image
    else:
        stockpile_image = _crop_bbox(image, stockpile_bbox)

    # Find each slot and assign row/column positions.
    slot_bboxes = anvil_parser.detect_slot_grid(stockpile_image)
    positioned_slots = anvil_parser._slots_with_grid_positions(slot_bboxes)

    crops: list[SlotCrop] = []
    for slot_index, row, col, bbox in positioned_slots:
        # Pad the parser's bbox to avoid clipping items when the bbox is
        # slightly too tight, then clip back into the image bounds.
        padded = bbox.padded(bbox_pad) if bbox_pad > 0 else bbox
        clipped = padded.clipped(stockpile_image.shape)
        if clipped.is_empty:
            continue
        slot_image = _crop_bbox(stockpile_image, clipped)
        crops.append(
            SlotCrop(
                slot_index=slot_index,
                row=row,
                col=col,
                image=_apply_border_crop(slot_image, border_fraction),
            )
        )
    return crops


def _crop_bbox(image: np.ndarray, bbox) -> np.ndarray:
    """Slice ``image`` to the rectangle described by an anvil BoundingBox."""
    return image[bbox.y : bbox.y2, bbox.x : bbox.x2]


def _apply_border_crop(slot_image: np.ndarray, border_fraction: float) -> np.ndarray:
    """Trim a small border from the top/sides while keeping the full bottom.

    Defensive guards ensure we never return an empty crop: if the requested
    border would eat the whole image we fall back to the original bounds.
    """
    height, width = slot_image.shape[:2]
    border_x = max(0, int(round(width * border_fraction)))
    border_y = max(0, int(round(height * border_fraction)))
    top = min(border_y, max(0, height - 1))
    left = min(border_x, max(0, width - 1))
    right = max(left + 1, width - border_x)
    return slot_image[top:height, left:right]
