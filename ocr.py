from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


LOGGER = logging.getLogger("anvil_ocr")

DEFAULT_OUTPUT_DIR = Path("data") / "ocr"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DIGIT_ACCEPT_THRESHOLD = 0.22
DIGIT_TEMPLATES: dict[str, list[np.ndarray]] | None = None


def read_amount_from_crop(image_path: str | Path) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        return _empty_ocr_result(path, error=f"Number crop does not exist: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return _empty_ocr_result(path, error=f"Could not read number crop: {path}")

    mask = _number_foreground_mask(image)
    image_height, image_width = mask.shape[:2]
    lower_mask = mask.copy()
    lower_mask[: int(image_height * 0.35), :] = 0

    contours, _ = cv2.findContours(lower_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < 3 or height < 8:
            continue
        if height < image_height * 0.12:
            continue
        if y > image_height * 0.94:
            continue
        if height > image_height * 0.55 and width <= max(6, int(image_width * 0.05)):
            continue
        if height > width * 4 and width <= max(6, int(image_width * 0.06)):
            continue
        boxes.append((x, y, width, height))

    boxes = _merge_digit_boxes(boxes)
    if not boxes:
        return _empty_ocr_result(path, raw_ocr_text="", error=None)

    max_bottom = max(y + height for _, y, _, height in boxes)
    boxes = [box for box in boxes if box[1] + box[3] >= max_bottom - 12]

    raw_digits: list[str] = []
    scores: list[float] = []
    for box in boxes:
        digit_image = _crop_digit(mask, box)
        digit, score = _match_digit(digit_image)
        raw_digits.append(digit)
        scores.append(score)

    raw_text = "".join(raw_digits)
    confidence = round(float(np.mean(scores)), 4) if scores else 0.0
    amount = raw_text if raw_text and min(scores) >= DIGIT_ACCEPT_THRESHOLD else ""

    return {
        "image_path": str(path),
        "amount": amount,
        "ocr_confidence": confidence,
        "raw_ocr_text": raw_text,
        "backend": "template_digit_matcher",
        "digit_scores": [round(float(score), 4) for score in scores],
        "error": None,
    }


def batch_read_amounts(image_paths: list[str | Path]) -> list[dict[str, Any]]:
    return [read_amount_from_crop(image_path) for image_path in image_paths]


def discover_number_crops(path: str | Path) -> list[Path]:
    input_path = Path(path)
    if input_path.is_file():
        return [input_path]

    if (input_path / "numbers").is_dir():
        input_path = input_path / "numbers"

    if not input_path.is_dir():
        return []

    return sorted(
        candidate
        for candidate in input_path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_json(results: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def _number_foreground_mask(gray: np.ndarray) -> np.ndarray:
    enlarged = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(enlarged, (3, 3), 0)

    _, mask = cv2.threshold(blurred, 135, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _merge_digit_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda item: item[0])
    merged: list[tuple[int, int, int, int]] = []
    for box in boxes:
        x, y, width, height = box
        if not merged:
            merged.append(box)
            continue

        previous_x, previous_y, previous_width, previous_height = merged[-1]
        gap = x - (previous_x + previous_width)
        vertical_overlap = min(y + height, previous_y + previous_height) - max(y, previous_y)
        if gap <= 2 and vertical_overlap > min(height, previous_height) * 0.35:
            x1 = min(previous_x, x)
            y1 = min(previous_y, y)
            x2 = max(previous_x + previous_width, x + width)
            y2 = max(previous_y + previous_height, y + height)
            merged[-1] = (x1, y1, x2 - x1, y2 - y1)
        else:
            merged.append(box)

    median_height = float(np.median([height for _, _, _, height in merged]))
    merged = [box for box in merged if box[3] >= median_height * 0.55]
    return sorted(merged, key=lambda item: item[0])


def _crop_digit(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = box
    pad = 3
    y1 = max(0, y - pad)
    x1 = max(0, x - pad)
    y2 = min(mask.shape[0], y + height + pad)
    x2 = min(mask.shape[1], x + width + pad)
    digit = mask[y1:y2, x1:x2]
    return cv2.resize(digit, (28, 36), interpolation=cv2.INTER_AREA)


def _digit_templates() -> dict[str, list[np.ndarray]]:
    global DIGIT_TEMPLATES
    if DIGIT_TEMPLATES is not None:
        return DIGIT_TEMPLATES

    templates: dict[str, list[np.ndarray]] = {digit: [] for digit in "0123456789"}
    font_paths = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf"),
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
        Path(r"C:\Windows\Fonts\verdana.ttf"),
    ]

    for font_path in font_paths:
        if not font_path.exists():
            continue
        for font_size in range(24, 42, 2):
            try:
                font = ImageFont.truetype(str(font_path), font_size)
            except OSError:
                continue
            for stroke_width in (0, 1):
                for digit in "0123456789":
                    canvas = Image.new("L", (60, 70), 0)
                    draw = ImageDraw.Draw(canvas)
                    draw.text(
                        (8, 4),
                        digit,
                        font=font,
                        fill=255,
                        stroke_width=stroke_width,
                        stroke_fill=255,
                    )
                    template = _normalise_template(np.asarray(canvas))
                    if template is not None:
                        templates[digit].append(template)

    for digit in "0123456789":
        canvas = np.zeros((48, 36), dtype=np.uint8)
        cv2.putText(
            canvas,
            digit,
            (3, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            255,
            2,
            cv2.LINE_AA,
        )
        template = _normalise_template(canvas)
        if template is not None:
            templates[digit].append(template)

    DIGIT_TEMPLATES = templates
    return templates


def _normalise_template(template: np.ndarray) -> np.ndarray | None:
    _, binary = cv2.threshold(template, 30, 255, cv2.THRESH_BINARY)
    y, x = np.where(binary > 0)
    if len(x) == 0 or len(y) == 0:
        return None
    cropped = binary[y.min() : y.max() + 1, x.min() : x.max() + 1]
    return cv2.resize(cropped, (28, 36), interpolation=cv2.INTER_AREA)


def _match_digit(digit_image: np.ndarray) -> tuple[str, float]:
    best_digit = ""
    best_score = -1.0
    query = digit_image.astype(np.float32) / 255.0

    for digit, templates in _digit_templates().items():
        for template in templates:
            target = template.astype(np.float32) / 255.0
            score = float(cv2.matchTemplate(query, target, cv2.TM_CCOEFF_NORMED)[0][0])
            if score > best_score:
                best_digit = digit
                best_score = score

    return best_digit, best_score


def _empty_ocr_result(path: Path, raw_ocr_text: str = "", error: str | None = None) -> dict[str, Any]:
    return {
        "image_path": str(path),
        "amount": "",
        "ocr_confidence": 0.0,
        "raw_ocr_text": raw_ocr_text,
        "backend": "template_digit_matcher",
        "digit_scores": [],
        "error": error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read stack amounts from number crops.")
    parser.add_argument("input_path", type=Path, help="A number crop, numbers/ folder, or parser output folder.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "ocr_results.json",
        help="Where JSON OCR results are written.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level. Default: INFO.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    image_paths = discover_number_crops(args.input_path)
    if not image_paths:
        raise SystemExit(f"No number crops found in {args.input_path}")

    results = batch_read_amounts(image_paths)
    write_json(results, args.output_path)

    print("Anvil OCR summary")
    print(f"Number crops: {len(results)}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
