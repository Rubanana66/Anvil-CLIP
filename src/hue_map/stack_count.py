"""Stack-count reading for Anvil inventory slots.

Pipeline:

1. Binarise bright/white pixels across the whole slot.
2. Find connected components, filter to digit-like shapes in the
   bottom-right corner.
3. Cluster survivors by shared baseline Y (digits of the same number
   end at the same row).
4. Score each cluster by (count of components, how low it sits, how
   densely packed its members are) and pick the winner. The scoring
   explicitly prefers clusters that sit at the very bottom of the slot
   - stack counts are always at the bottom, so lower beats larger.
5. Tight-crop around the winning cluster and hand it to EasyOCR.

Why EasyOCR
-----------
Published benchmarks put EasyOCR at 95%+ on small digit crops versus
Tesseract's ~90%. It ships its own pretrained model so there's no
external binary to install - just ``pip install easyocr``.

Semantic rule (applied in ``pipeline.py``):

    Anvil never prints "1" next to a single item, so a non-empty slot
    that yields no OCR result is interpreted as a stack of one. Empty
    slots stay ``value=None``.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


LOGGER = logging.getLogger("hue_map.stack_count")


# --- OCR backend -----------------------------------------------------------
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None  # type: ignore[assignment]
    EASYOCR_AVAILABLE = False


# Lazily constructed EasyOCR reader. Loading the model takes a couple of
# seconds so we only do it once per process.
_EASYOCR_READER = None


def _get_easyocr_reader():
    """Return a shared EasyOCR Reader instance, constructing it lazily."""
    global _EASYOCR_READER
    if not EASYOCR_AVAILABLE:
        return None
    if _EASYOCR_READER is None:
        # CPU inference is plenty fast for the tight digit crops and
        # avoids CUDA setup pain.
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _EASYOCR_READER


def is_ocr_available() -> bool:
    return EASYOCR_AVAILABLE


def describe_ocr_status() -> str:
    """Human-readable one-line summary for pipeline startup logging."""
    if EASYOCR_AVAILABLE:
        return "Stack-count OCR ready (EasyOCR backend, CPU inference)."
    return (
        "Stack-count OCR disabled: EasyOCR not installed. "
        "Run `py -m pip install easyocr` to enable."
    )


# --- digit-component detection -----------------------------------------
# Every threshold is expressed as a fraction of slot height / width so
# it stays valid at any screenshot resolution.

# Binarisation: keep bright / low-saturation pixels. Loose enough to
# preserve anti-aliased glyph edges.
WHITE_MIN_VALUE = 170
WHITE_MAX_SATURATION = 85

# Position filter: digits always sit in the BOTTOM-RIGHT of the slot.
# These are intentionally tight; item pixels leaking high or to the
# left fail the filter before they can compete with real digits.
DIGIT_MIN_X_FRACTION = 0.25
DIGIT_MIN_Y_FRACTION = 0.52

# Shape filter: single digits are small and relatively narrow.
DIGIT_MIN_AREA_PIXELS = 3
DIGIT_MAX_AREA_FRACTION = 0.045
DIGIT_MIN_HEIGHT_FRACTION = 0.06
DIGIT_MAX_HEIGHT_FRACTION = 0.35
DIGIT_MAX_WIDTH_FRACTION = 0.22

# Digits of the same number share a bottom-Y (text baseline).
DIGIT_BASELINE_TOLERANCE_FRACTION = 0.09

# Cluster scoring weights. The WINNING cluster is whichever maximises
#   score = (number of members) + BOTTOM_WEIGHT * (baseline_y / slot_h)
# so a cluster sitting at the very bottom can beat a larger cluster
# higher up. This matters when item pixels (e.g. cabbage leaves at
# y=0.60) form a 2-3 component cluster that would otherwise outscore
# a single-digit number at y=0.85.
CLUSTER_BOTTOM_WEIGHT = 10.0

# Upscale + padding for OCR.
OCR_UPSCALE_FACTOR = 4
OCR_BORDER_PADDING = 20

# Minimum pixel count before attempting OCR.
MIN_WHITE_PIXELS = 10


@dataclass(frozen=True)
class StackCount:
    """Result of stack-count recognition for one slot."""

    value: Optional[int]        # parsed integer; None when unreadable / absent
    raw_text: str               # raw OCR output (for debugging)
    confidence: float           # 0-1 model confidence (EasyOCR's own score)
    pixel_count: int            # white pixels in the tight digit crop
    binary_image: Optional[np.ndarray] = None   # preprocessed mask passed to OCR

    def to_dict(self) -> dict:
        """JSON-safe representation (drops the numpy array)."""
        return {
            "value": self.value,
            "raw_text": self.raw_text,
            "confidence": round(float(self.confidence), 3),
            "pixel_count": int(self.pixel_count),
        }


# --- public API -----------------------------------------------------------


def read_stack_count(slot_bgr: np.ndarray) -> StackCount:
    """Read the stack count from a slot crop."""
    if slot_bgr is None or slot_bgr.size == 0:
        return _empty_result()

    binary = _detect_digit_binary(slot_bgr)
    if binary.size == 0:
        return _empty_result()

    pixel_count = int(np.count_nonzero(binary))
    if pixel_count < MIN_WHITE_PIXELS:
        return StackCount(
            value=None, raw_text="", confidence=0.0,
            pixel_count=pixel_count, binary_image=binary,
        )

    reader = _get_easyocr_reader()
    if reader is None:
        return StackCount(
            value=None, raw_text="", confidence=0.0,
            pixel_count=pixel_count, binary_image=binary,
        )

    upscaled = _upscale_and_pad(binary)
    raw_text, confidence = _run_easyocr(reader, upscaled)
    value, _ = _parse_digits(raw_text)
    return StackCount(
        value=value, raw_text=raw_text, confidence=confidence,
        pixel_count=pixel_count, binary_image=binary,
    )


def batch_read_stack_counts(slot_bgrs: list[np.ndarray]) -> list[StackCount]:
    return [read_stack_count(image) for image in slot_bgrs]


# --- digit-component detection -------------------------------------------


def _detect_digit_binary(slot_bgr: np.ndarray) -> np.ndarray:
    """Return a tight binary image containing just the stack-count glyphs."""
    slot_height, slot_width = slot_bgr.shape[:2]

    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    white_mask = (
        (hsv[..., 2] >= WHITE_MIN_VALUE)
        & (hsv[..., 1] <= WHITE_MAX_SATURATION)
    ).astype(np.uint8) * 255
    closed = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:
        return np.zeros((slot_height, slot_width), dtype=np.uint8)

    min_x = slot_width * DIGIT_MIN_X_FRACTION
    min_y = slot_height * DIGIT_MIN_Y_FRACTION
    min_height = max(4, int(slot_height * DIGIT_MIN_HEIGHT_FRACTION))
    max_height = max(min_height + 1, int(slot_height * DIGIT_MAX_HEIGHT_FRACTION))
    max_width = max(4, int(slot_height * DIGIT_MAX_WIDTH_FRACTION))
    max_area = max(DIGIT_MIN_AREA_PIXELS + 1, int(slot_height * slot_width * DIGIT_MAX_AREA_FRACTION))

    candidates: list[dict] = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < DIGIT_MIN_AREA_PIXELS or area > max_area:
            continue
        if h < min_height or h > max_height:
            continue
        if w > max_width:
            continue
        centre_x = x + w / 2.0
        centre_y = y + h / 2.0
        if centre_x < min_x or centre_y < min_y:
            continue

        candidates.append(
            {
                "label": label,
                "x": x, "y": y, "w": w, "h": h,
                "bottom": y + h,
            }
        )

    if not candidates:
        return np.zeros((slot_height, slot_width), dtype=np.uint8)

    # Baseline clustering: digits of the same number end at the same bottom Y.
    baseline_tolerance = max(3, int(slot_height * DIGIT_BASELINE_TOLERANCE_FRACTION))
    best_cluster: list[dict] = []
    best_score = -1.0
    for seed in candidates:
        cluster = [
            cand for cand in candidates
            if abs(cand["bottom"] - seed["bottom"]) <= baseline_tolerance
        ]
        # Score: cluster size plus a bonus proportional to how low the
        # cluster's baseline sits in the slot. A lower baseline means
        # "closer to the bottom" = more likely to be a real stack count.
        cluster_bottom = max(cand["bottom"] for cand in cluster)
        bottom_fraction = cluster_bottom / max(1, slot_height)
        score = len(cluster) + CLUSTER_BOTTOM_WEIGHT * bottom_fraction
        if score > best_score:
            best_score = score
            best_cluster = cluster

    if not best_cluster:
        return np.zeros((slot_height, slot_width), dtype=np.uint8)

    full_output = np.zeros_like(closed)
    for cand in best_cluster:
        full_output[labels == cand["label"]] = 255

    pad = max(2, int(slot_height * 0.03))
    x_start = max(0, min(cand["x"] for cand in best_cluster) - pad)
    x_end = min(slot_width, max(cand["x"] + cand["w"] for cand in best_cluster) + pad)
    y_start = max(0, min(cand["y"] for cand in best_cluster) - pad)
    y_end = min(slot_height, max(cand["y"] + cand["h"] for cand in best_cluster) + pad)
    return full_output[y_start:y_end, x_start:x_end]


def _upscale_and_pad(binary: np.ndarray) -> np.ndarray:
    """Enlarge with bicubic interpolation and pad with black before OCR.

    EasyOCR is a neural model trained on anti-aliased rendered text, so
    it works best on inputs whose edges look natural. Bicubic upscale
    produces smooth edges; nearest-neighbour (which we used with the old
    Tesseract backend) produces chunky pixel squares that the neural
    model doesn't recognise as consistent strokes. The same "6" that
    reads at confidence 0 with nearest upscales to 0.9999 with cubic.
    """
    height, width = binary.shape[:2]
    new_size = (width * OCR_UPSCALE_FACTOR, height * OCR_UPSCALE_FACTOR)
    upscaled = cv2.resize(binary, new_size, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(
        upscaled,
        OCR_BORDER_PADDING, OCR_BORDER_PADDING,
        OCR_BORDER_PADDING, OCR_BORDER_PADDING,
        cv2.BORDER_CONSTANT,
        value=0,
    )


# --- OCR + parsing --------------------------------------------------------


_OCR_FAILURE_LOGGED = False


def _run_easyocr(reader, binary: np.ndarray) -> tuple[str, float]:
    """Run EasyOCR and return (text, confidence) for the best result."""
    global _OCR_FAILURE_LOGGED
    try:
        results = reader.readtext(binary, allowlist="0123456789", detail=1, paragraph=False)
    except Exception as exc:
        if not _OCR_FAILURE_LOGGED:
            LOGGER.warning("EasyOCR invocation failed: %s", exc)
            _OCR_FAILURE_LOGGED = True
        return "", 0.0
    if not results:
        return "", 0.0
    best = max(results, key=lambda item: float(item[2]))
    text = str(best[1]).strip()
    confidence = float(best[2])
    return text, max(0.0, min(1.0, confidence))


def _parse_digits(raw_text: str) -> tuple[Optional[int], float]:
    """Extract the first contiguous digit run."""
    if not raw_text:
        return None, 0.0
    match = re.search(r"\d+", raw_text)
    if not match:
        return None, 0.0
    value = int(match.group())
    total_chars = len(raw_text.strip())
    digit_chars = sum(1 for c in raw_text if c.isdigit())
    confidence = digit_chars / max(1, total_chars)
    return value, min(1.0, max(0.0, confidence))


def _empty_result() -> StackCount:
    return StackCount(value=None, raw_text="", confidence=0.0, pixel_count=0)
