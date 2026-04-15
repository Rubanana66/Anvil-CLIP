from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LOGGER = logging.getLogger("anvil_parser")
ImageArray = np.ndarray
SCHEMA_VERSION = "2.0"


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return self.x + (self.width / 2)

    @property
    def center_y(self) -> float:
        return self.y + (self.height / 2)

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def shifted(self, dx: int, dy: int) -> "BoundingBox":
        return BoundingBox(self.x + dx, self.y + dy, self.width, self.height)

    def padded(self, padding: int) -> "BoundingBox":
        return BoundingBox(
            self.x - padding,
            self.y - padding,
            self.width + (padding * 2),
            self.height + (padding * 2),
        )

    def clipped(self, image_shape: tuple[int, ...]) -> "BoundingBox":
        image_height, image_width = image_shape[:2]
        x1 = max(0, min(self.x, image_width))
        y1 = max(0, min(self.y, image_height))
        x2 = max(0, min(self.x2, image_width))
        y2 = max(0, min(self.y2, image_height))
        return BoundingBox(x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    def to_dict(self) -> dict[str, int]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
        }


@dataclass(frozen=True)
class SlotCrop:
    slot_index: int
    row: int
    col: int
    slot_bbox: BoundingBox
    item_bbox: BoundingBox
    quality_bbox: BoundingBox
    number_bbox: BoundingBox
    slot_path: Path
    item_path: Path
    quality_path: Path
    number_path: Path

    @property
    def icon_bbox(self) -> BoundingBox:
        """Compatibility alias for older icon-matcher code."""
        return self.item_bbox

    @property
    def icon_path(self) -> Path:
        """Compatibility alias for older icon-matcher code."""
        return self.item_path

    def to_dict(self) -> dict[str, Any]:
        data = {
            "slot_index": self.slot_index,
            "row": self.row,
            "col": self.col,
            "slot_bbox": self.slot_bbox.to_dict(),
            "item_bbox": self.item_bbox.to_dict(),
            "quality_bbox": self.quality_bbox.to_dict(),
            "number_bbox": self.number_bbox.to_dict(),
            "slot_path": str(self.slot_path),
            "item_path": str(self.item_path),
            "quality_path": str(self.quality_path),
            "number_path": str(self.number_path),
        }

        # Keep the current pipeline usable while newer modules migrate to
        # item_path/item_bbox naming.
        data["icon_bbox"] = data["item_bbox"]
        data["icon_path"] = data["item_path"]
        return data


@dataclass(frozen=True)
class ParseResult:
    image_path: Path
    output_dir: Path
    stockpile_bbox: BoundingBox | None
    slot_count: int
    slots: list[SlotCrop]
    debug_paths: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "image_path": str(self.image_path),
            "output_dir": str(self.output_dir),
            "stockpile_found": self.stockpile_bbox is not None,
            "stockpile_bbox": self.stockpile_bbox.to_dict() if self.stockpile_bbox else None,
            "slot_count": self.slot_count,
            "coordinate_space": {
                "slot_bbox": "full_image_pixels",
                "item_bbox": "slot_pixels",
                "quality_bbox": "slot_pixels",
                "number_bbox": "slot_pixels",
            },
            "slots": [slot.to_dict() for slot in self.slots],
            "debug_paths": {name: str(path) for name, path in self.debug_paths.items()},
        }


def load_image(path: str | Path) -> ImageArray:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Screenshot not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read screenshot as an image: {image_path}")

    LOGGER.debug("Loaded image %s with shape %s", image_path, image.shape)
    return image


def preprocess_image(image: ImageArray) -> ImageArray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # CLAHE keeps the slot borders visible across darker screenshots and
    # slightly compressed captures without baking in a fixed color threshold.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    edges = cv2.Canny(gray, 45, 135)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return edges


def detect_stockpile_region(image: ImageArray) -> BoundingBox | None:
    candidates = _raw_slot_candidates(image)
    slots = _select_best_grid(candidates)
    if len(slots) < 2:
        LOGGER.warning("Could not find enough repeated slot boxes to isolate the stockpile.")
        return None

    stockpile_bbox = _union_boxes(slots).padded(10).clipped(image.shape)
    LOGGER.info("Detected stockpile/grid region: %s", stockpile_bbox)
    return stockpile_bbox


def detect_slot_grid(stockpile_image: ImageArray) -> list[BoundingBox]:
    candidates = _raw_slot_candidates(stockpile_image)
    slots = _select_best_grid(candidates)
    slots = _drop_header_like_slot_rows(slots)
    slots = _sort_reading_order(slots)
    LOGGER.info("Detected %d visible slot box(es).", len(slots))
    return slots


def make_item_crop(slot_image: ImageArray) -> BoundingBox:
    # The item crop intentionally avoids the lower strip where stack counts are
    # drawn. Crops may still overlap the quality crop; consumers use them for
    # different recognition tasks.
    return _relative_crop_bbox(
        slot_image,
        left=0.07,
        top=0.07,
        right=0.93,
        bottom=0.72,
    )


def make_quality_crop(slot_image: ImageArray) -> BoundingBox:
    # MVP heuristic for the main UI style: quality markers live in the upper
    # left area when present. Empty quality crops are still useful review data.
    return _relative_crop_bbox(
        slot_image,
        left=0.04,
        top=0.04,
        right=0.38,
        bottom=0.34,
    )


def make_number_crop(slot_image: ImageArray) -> BoundingBox:
    return _relative_crop_bbox(
        slot_image,
        left=0.42,
        top=0.52,
        right=0.98,
        bottom=0.97,
    )


def make_icon_crop(slot_image: ImageArray) -> BoundingBox:
    """Compatibility wrapper. New code should call make_item_crop()."""
    return make_item_crop(slot_image)


def extract_slot_crops(
    image: ImageArray,
    slot_bboxes: list[BoundingBox],
    output_dir: str | Path,
) -> list[SlotCrop]:
    output_path = Path(output_dir)
    slot_dir = output_path / "slots"
    item_dir = output_path / "items"
    quality_dir = output_path / "quality"
    number_dir = output_path / "numbers"
    for directory in (slot_dir, item_dir, quality_dir, number_dir):
        directory.mkdir(parents=True, exist_ok=True)

    slots: list[SlotCrop] = []
    ordered = _slots_with_grid_positions(slot_bboxes)

    for slot_index, row, col, slot_bbox in ordered:
        clipped_slot_bbox = slot_bbox.clipped(image.shape)
        if clipped_slot_bbox.is_empty:
            LOGGER.warning("Skipping empty slot bbox %s", slot_bbox)
            continue

        slot_image = _crop(image, clipped_slot_bbox)
        if slot_image.size == 0:
            LOGGER.warning("Skipping empty slot crop for slot %d at %s", slot_index, clipped_slot_bbox)
            continue

        item_bbox = make_item_crop(slot_image)
        quality_bbox = make_quality_crop(slot_image)
        number_bbox = make_number_crop(slot_image)

        item_image = _crop(slot_image, item_bbox)
        quality_image = _crop(slot_image, quality_bbox)
        number_image = _crop(slot_image, number_bbox)

        suffix = f"{slot_index:03d}_r{row:02d}_c{col:02d}.png"
        slot_path = slot_dir / f"slot_{suffix}"
        item_path = item_dir / f"item_{suffix}"
        quality_path = quality_dir / f"quality_{suffix}"
        number_path = number_dir / f"number_{suffix}"

        _write_image(slot_path, slot_image)
        _write_image(item_path, item_image)
        _write_image(quality_path, quality_image)
        _write_image(number_path, number_image)

        slots.append(
            SlotCrop(
                slot_index=slot_index,
                row=row,
                col=col,
                slot_bbox=clipped_slot_bbox,
                item_bbox=item_bbox,
                quality_bbox=quality_bbox,
                number_bbox=number_bbox,
                slot_path=slot_path,
                item_path=item_path,
                quality_path=quality_path,
                number_path=number_path,
            )
        )

    LOGGER.info("Saved %d slot crop set(s) into %s", len(slots), output_path)
    return slots


def parse_screenshot(
    image_path: str | Path,
    output_dir: str | Path = "output_debug",
) -> ParseResult:
    source_path = Path(image_path)
    image = load_image(source_path)

    run_dir = _make_run_output_dir(Path(output_dir), source_path)
    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    debug_paths: dict[str, Path] = {}

    original_path = overlays_dir / "original.png"
    _write_image(original_path, image)
    debug_paths["original"] = original_path

    preprocessed_path = overlays_dir / "preprocessed_edges.png"
    _write_image(preprocessed_path, preprocess_image(image))
    debug_paths["preprocessed_edges"] = preprocessed_path

    stockpile_bbox = detect_stockpile_region(image)
    region_bbox = stockpile_bbox or BoundingBox(0, 0, image.shape[1], image.shape[0])
    stockpile_image = _crop(image, region_bbox)

    stockpile_path = overlays_dir / "stockpile_region.png"
    _write_image(stockpile_path, stockpile_image)
    debug_paths["stockpile_region"] = stockpile_path

    stockpile_overlay = image.copy()
    if stockpile_bbox:
        _draw_box(stockpile_overlay, stockpile_bbox, (80, 220, 120), 3, "stockpile")
    else:
        _draw_box(stockpile_overlay, region_bbox, (80, 180, 255), 2, "fallback_full_image")
    stockpile_overlay_path = overlays_dir / "stockpile_region_overlay.png"
    _write_image(stockpile_overlay_path, stockpile_overlay)
    debug_paths["stockpile_region_overlay"] = stockpile_overlay_path

    relative_slots = detect_slot_grid(stockpile_image)
    absolute_slots = [slot.shifted(region_bbox.x, region_bbox.y) for slot in relative_slots]
    absolute_slots = _sort_reading_order(absolute_slots)

    slots_overlay = image.copy()
    for index, slot_bbox in enumerate(absolute_slots):
        _draw_box(slots_overlay, slot_bbox, (60, 220, 255), 2, str(index))
    slots_overlay_path = overlays_dir / "slot_boxes_overlay.png"
    _write_image(slots_overlay_path, slots_overlay)
    debug_paths["slot_boxes_overlay"] = slots_overlay_path

    slots = extract_slot_crops(image, absolute_slots, run_dir)

    crop_overlay = image.copy()
    for slot in slots:
        _draw_box(crop_overlay, slot.slot_bbox, (60, 220, 255), 1, str(slot.slot_index))
        _draw_box(crop_overlay, slot.item_bbox.shifted(slot.slot_bbox.x, slot.slot_bbox.y), (80, 220, 120), 1, "item")
        _draw_box(crop_overlay, slot.quality_bbox.shifted(slot.slot_bbox.x, slot.slot_bbox.y), (255, 180, 60), 1, "quality")
        _draw_box(crop_overlay, slot.number_bbox.shifted(slot.slot_bbox.x, slot.slot_bbox.y), (80, 120, 255), 1, "number")
    crop_overlay_path = overlays_dir / "crop_boxes_overlay.png"
    _write_image(crop_overlay_path, crop_overlay)
    debug_paths["crop_boxes_overlay"] = crop_overlay_path

    debug_paths.update(_write_crop_contact_sheets(slots, overlays_dir))

    json_path = run_dir / "parse_result.json"
    debug_paths["json_summary"] = json_path
    debug_paths["parse_result_json"] = json_path

    result = ParseResult(
        image_path=source_path,
        output_dir=run_dir,
        stockpile_bbox=stockpile_bbox,
        slot_count=len(slots),
        slots=slots,
        debug_paths=debug_paths,
    )
    save_parse_result_json(result, json_path)
    LOGGER.info("Wrote parse JSON summary to %s", json_path)

    return result


def save_parse_result_json(parse_result: ParseResult, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(parse_result.to_dict(), indent=2), encoding="utf-8")


def _raw_slot_candidates(image: ImageArray) -> list[BoundingBox]:
    edge_maps = _slot_edge_maps(image)
    image_height, image_width = image.shape[:2]
    min_dimension = min(image_height, image_width)
    min_side = max(34, int(min_dimension * 0.045))
    max_side = min(240, int(min_dimension * 0.45))

    candidates: list[BoundingBox] = []
    for edge_map in edge_maps:
        contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width < min_side or height < min_side:
                continue
            if width > max_side or height > max_side:
                continue

            aspect = width / max(height, 1)
            if not 0.62 <= aspect <= 1.55:
                continue

            area = width * height
            if area < int((min_side * min_side) * 0.65):
                continue

            contour_area = cv2.contourArea(contour)
            extent = contour_area / max(area, 1)
            if extent < 0.03:
                continue

            candidates.append(BoundingBox(x, y, width, height))

    LOGGER.debug("Raw slot candidates: %d", len(candidates))
    return candidates


def _slot_edge_maps(image: ImageArray) -> list[ImageArray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    simple_edges = cv2.Canny(gray, 50, 120)
    enhanced_edges = preprocess_image(image)

    # A thresholded gradient map catches some UI borders that are low-contrast
    # after game scaling but still form repeated rectangles.
    gradient_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    gradient_x_abs = cv2.convertScaleAbs(gradient_x)
    gradient_y_abs = cv2.convertScaleAbs(gradient_y)
    gradient = cv2.addWeighted(gradient_x_abs, 0.5, gradient_y_abs, 0.5, 0)
    _, gradient_edges = cv2.threshold(gradient, 35, 255, cv2.THRESH_BINARY)
    gradient_edges = cv2.morphologyEx(gradient_edges, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)

    return [simple_edges, enhanced_edges, gradient_edges]


def _select_best_grid(candidates: list[BoundingBox]) -> list[BoundingBox]:
    if not candidates:
        return []

    best_group: list[BoundingBox] = []
    best_score = float("-inf")

    for seed in candidates:
        width_tolerance = max(8, int(seed.width * 0.18))
        height_tolerance = max(10, int(seed.height * 0.25))

        group = [
            candidate
            for candidate in candidates
            if abs(candidate.width - seed.width) <= width_tolerance
            and abs(candidate.height - seed.height) <= height_tolerance
        ]
        group = _dedupe_rectangles(group)
        if len(group) < 2:
            continue

        row_clusters = _cluster_values(
            [box.center_y for box in group],
            tolerance=max(10, int(seed.height * 0.42)),
        )
        col_clusters = _cluster_values(
            [box.center_x for box in group],
            tolerance=max(10, int(seed.width * 0.42)),
        )

        median_width = float(np.median([box.width for box in group]))
        median_height = float(np.median([box.height for box in group]))
        aspect_penalty = abs((median_width / max(median_height, 1.0)) - 1.0)
        regularity_score = len(row_clusters) * len(col_clusters)
        size_score = (median_width * median_height) / 1000.0
        score = (len(group) * 8.0) + regularity_score + size_score - (aspect_penalty * 8.0)

        if len(col_clusters) < 2:
            score -= 15.0

        if score > best_score:
            best_score = score
            best_group = group

    if not best_group:
        return []

    median_width = float(np.median([box.width for box in best_group]))
    median_height = float(np.median([box.height for box in best_group]))
    filtered = [
        box
        for box in best_group
        if box.width >= median_width * 0.72
        and box.height >= median_height * 0.70
        and box.width <= median_width * 1.28
        and box.height <= median_height * 1.32
    ]
    return _complete_regular_grid(_dedupe_rectangles(filtered))


def _dedupe_rectangles(rectangles: list[BoundingBox]) -> list[BoundingBox]:
    sorted_rectangles = sorted(rectangles, key=_slot_quality_score, reverse=True)
    deduped: list[BoundingBox] = []

    for box in sorted_rectangles:
        if any(_box_iou(box, existing) > 0.42 or _box_inside(box, existing) for existing in deduped):
            continue
        deduped.append(box)

    return deduped


def _complete_regular_grid(slots: list[BoundingBox]) -> list[BoundingBox]:
    if len(slots) < 4:
        return slots

    median_width = float(np.median([slot.width for slot in slots]))
    median_height = float(np.median([slot.height for slot in slots]))
    row_tolerance = max(10, int(median_height * 0.50))
    col_tolerance = max(10, int(median_width * 0.50))

    row_clusters = _cluster_boxes_by_center(slots, axis="y", tolerance=row_tolerance)
    col_clusters = _cluster_boxes_by_center(slots, axis="x", tolerance=col_tolerance)
    if len(row_clusters) < 1 or len(col_clusters) < 2:
        return slots

    expected_count = len(row_clusters) * len(col_clusters)
    missing_count = expected_count - len(slots)
    if missing_count <= 0 or missing_count > max(3, len(col_clusters)):
        return slots

    completed = list(slots)
    for row in row_clusters:
        row_center = float(np.mean([slot.center_y for slot in row]))
        row_height = int(round(float(np.median([slot.height for slot in row]))))
        for col in col_clusters:
            col_center = float(np.mean([slot.center_x for slot in col]))
            exists = any(
                abs(slot.center_y - row_center) <= row_tolerance
                and abs(slot.center_x - col_center) <= col_tolerance
                for slot in completed
            )
            if exists:
                continue

            synthesized = BoundingBox(
                int(round(col_center - (median_width / 2))),
                int(round(row_center - (row_height / 2))),
                int(round(median_width)),
                max(1, row_height),
            )
            LOGGER.debug("Synthesized missing grid slot at %s", synthesized)
            completed.append(synthesized)

    return _dedupe_rectangles(completed)


def _cluster_boxes_by_center(
    boxes: list[BoundingBox],
    axis: str,
    tolerance: int | float,
) -> list[list[BoundingBox]]:
    key = (lambda box: box.center_y) if axis == "y" else (lambda box: box.center_x)
    clusters: list[list[BoundingBox]] = []
    for box in sorted(boxes, key=key):
        if not clusters:
            clusters.append([box])
            continue

        cluster_center = float(np.mean([key(existing) for existing in clusters[-1]]))
        if abs(key(box) - cluster_center) > tolerance:
            clusters.append([box])
        else:
            clusters[-1].append(box)
    return clusters


def _slot_quality_score(box: BoundingBox) -> float:
    aspect = box.width / max(box.height, 1)
    aspect_score = 1.0 - min(abs(aspect - 1.0), 0.8) / 0.8
    size_score = box.area / 1000.0
    small_penalty = 1.5 if box.width < 72 or box.height < 72 else 0.0
    return size_score + (aspect_score * 4.0) - small_penalty


def _box_iou(left: BoundingBox, right: BoundingBox) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection = intersection_width * intersection_height
    if intersection == 0:
        return 0.0
    union = left.area + right.area - intersection
    return intersection / max(union, 1)


def _box_inside(inner: BoundingBox, outer: BoundingBox) -> bool:
    if inner.area >= outer.area:
        return False
    return outer.x <= inner.center_x <= outer.x2 and outer.y <= inner.center_y <= outer.y2


def _cluster_values(values: list[float], tolerance: int | float) -> list[list[float]]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or abs(value - float(np.mean(clusters[-1]))) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return clusters


def _sort_reading_order(slots: list[BoundingBox]) -> list[BoundingBox]:
    if not slots:
        return []

    median_height = float(np.median([slot.height for slot in slots]))
    row_tolerance = max(10, int(median_height * 0.45))

    rows: list[list[BoundingBox]] = []
    for slot in sorted(slots, key=lambda box: box.center_y):
        if not rows or abs(slot.center_y - float(np.mean([box.center_y for box in rows[-1]]))) > row_tolerance:
            rows.append([slot])
        else:
            rows[-1].append(slot)

    ordered: list[BoundingBox] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda box: box.center_x))
    return ordered


def _drop_header_like_slot_rows(slots: list[BoundingBox]) -> list[BoundingBox]:
    if len(slots) < 6:
        return slots

    median_height = float(np.median([slot.height for slot in slots]))
    rows = _cluster_boxes_by_center(
        slots,
        axis="y",
        tolerance=max(10, int(median_height * 0.45)),
    )
    if len(rows) < 2:
        return slots

    row_centers = [float(np.mean([slot.center_y for slot in row])) for row in rows]
    first_gap = row_centers[1] - row_centers[0]

    # Stockpile category tabs can create a fake row of square-ish contours just
    # above the real slots. Real slot rows are spaced roughly one slot height
    # apart; tab contours sit much closer to the first real row.
    if first_gap >= median_height * 0.85:
        return slots

    first_row = rows[0]
    second_row = rows[1]
    if len(first_row) < max(2, int(len(second_row) * 0.6)):
        return slots

    filtered = [slot for row in rows[1:] for slot in row]
    LOGGER.info(
        "Dropped %d header-like slot candidate(s) above the stockpile grid.",
        len(first_row),
    )
    return filtered


def _slots_with_grid_positions(slots: list[BoundingBox]) -> list[tuple[int, int, int, BoundingBox]]:
    if not slots:
        return []

    median_height = float(np.median([slot.height for slot in slots]))
    row_tolerance = max(10, int(median_height * 0.45))

    rows: list[list[BoundingBox]] = []
    for slot in sorted(slots, key=lambda box: box.center_y):
        if not rows or abs(slot.center_y - float(np.mean([box.center_y for box in rows[-1]]))) > row_tolerance:
            rows.append([slot])
        else:
            rows[-1].append(slot)

    positioned: list[tuple[int, int, int, BoundingBox]] = []
    slot_index = 0
    for row_index, row in enumerate(rows):
        for col_index, slot in enumerate(sorted(row, key=lambda box: box.center_x)):
            positioned.append((slot_index, row_index, col_index, slot))
            slot_index += 1
    return positioned


def _union_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    x1 = min(box.x for box in boxes)
    y1 = min(box.y for box in boxes)
    x2 = max(box.x2 for box in boxes)
    y2 = max(box.y2 for box in boxes)
    return BoundingBox(x1, y1, x2 - x1, y2 - y1)


def _relative_crop_bbox(
    image: ImageArray,
    *,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> BoundingBox:
    height, width = image.shape[:2]
    x1 = int(round(width * left))
    y1 = int(round(height * top))
    x2 = int(round(width * right))
    y2 = int(round(height * bottom))

    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1

    return BoundingBox(x1, y1, x2 - x1, y2 - y1).clipped(image.shape)


def _crop(image: ImageArray, bbox: BoundingBox) -> ImageArray:
    clipped = bbox.clipped(image.shape)
    return image[clipped.y : clipped.y2, clipped.x : clipped.x2]


def _write_image(path: Path, image: ImageArray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.size == 0:
        raise ValueError(f"Cannot write an empty image crop: {path}")
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write image: {path}")


def _draw_box(
    image: ImageArray,
    bbox: BoundingBox,
    color: tuple[int, int, int],
    thickness: int,
    label: str | None = None,
) -> None:
    if bbox.is_empty:
        return

    cv2.rectangle(image, (bbox.x, bbox.y), (bbox.x2, bbox.y2), color, thickness)
    if label:
        text_y = bbox.y - 5 if bbox.y > 18 else bbox.y + 15
        cv2.putText(
            image,
            label,
            (bbox.x + 3, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def _write_crop_contact_sheets(slots: list[SlotCrop], overlays_dir: Path) -> dict[str, Path]:
    if not slots:
        return {}

    debug_paths: dict[str, Path] = {}
    sheet_specs = {
        "slots_contact_sheet": [(slot.slot_index, slot.slot_path) for slot in slots],
        "items_contact_sheet": [(slot.slot_index, slot.item_path) for slot in slots],
        "quality_contact_sheet": [(slot.slot_index, slot.quality_path) for slot in slots],
        "numbers_contact_sheet": [(slot.slot_index, slot.number_path) for slot in slots],
    }

    for name, entries in sheet_specs.items():
        output_path = overlays_dir / f"{name}.png"
        _write_contact_sheet(entries, output_path)
        debug_paths[name] = output_path

    return debug_paths


def _write_contact_sheet(entries: list[tuple[int, Path]], output_path: Path) -> None:
    if not entries:
        return

    thumb_size = 96
    label_height = 18
    padding = 8
    columns = min(5, max(1, len(entries)))
    rows = int(np.ceil(len(entries) / columns))

    sheet_width = (columns * thumb_size) + ((columns + 1) * padding)
    sheet_height = (rows * (thumb_size + label_height)) + ((rows + 1) * padding)
    sheet = np.full((sheet_height, sheet_width, 3), 34, dtype=np.uint8)

    for entry_index, (slot_index, image_path) in enumerate(entries):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            LOGGER.warning("Could not read crop for contact sheet: %s", image_path)
            continue

        row = entry_index // columns
        col = entry_index % columns
        x = padding + col * (thumb_size + padding)
        y = padding + row * (thumb_size + label_height + padding)

        resized = _resize_to_fit(image, thumb_size, thumb_size)
        y_offset = y + (thumb_size - resized.shape[0]) // 2
        x_offset = x + (thumb_size - resized.shape[1]) // 2
        sheet[y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]] = resized

        cv2.putText(
            sheet,
            f"{slot_index:03d}",
            (x, y + thumb_size + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    _write_image(output_path, sheet)


def _resize_to_fit(image: ImageArray, max_width: int, max_height: int) -> ImageArray:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image

    scale = min(max_width / width, max_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _make_run_output_dir(base_dir: Path, image_path: Path) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem).strip("_") or "screenshot"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{safe_stem}_{timestamp}"

    suffix = 1
    unique_dir = run_dir
    while unique_dir.exists():
        suffix += 1
        unique_dir = base_dir / f"{safe_stem}_{timestamp}_{suffix:02d}"

    unique_dir.mkdir(parents=True, exist_ok=False)
    return unique_dir


def main() -> None:
    cli = argparse.ArgumentParser(description="Parse Anvil Empires stockpile screenshots into crop sets.")
    cli.add_argument("image_path", type=Path, help="Path to a full stockpile screenshot.")
    cli.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output_debug"),
        help="Base directory where parser outputs are saved.",
    )
    cli.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level. Default: INFO.",
    )
    args = cli.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    try:
        result = parse_screenshot(args.image_path, args.output_dir)
    except Exception as exc:
        LOGGER.error("Parse failed: %s", exc)
        raise SystemExit(1) from exc

    print(f"Stockpile found: {'yes' if result.stockpile_bbox else 'no'}")
    print(f"Slot count: {result.slot_count}")
    print(f"Output folder: {result.output_dir}")
    json_path = result.debug_paths.get("json_summary")
    if json_path:
        print(f"Parse result JSON: {json_path}")


if __name__ == "__main__":
    main()
