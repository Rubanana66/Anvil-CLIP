"""End-to-end pipeline: screenshot(s) -> HTML report(s).

CLI entry point::

    py -m hue_map.pipeline                # process every image in src/input/
    py -m hue_map.pipeline my_shot.png    # process a single image
    py -m hue_map.pipeline --input ...    # explicit override

For each screenshot we:

1. Load the image.
2. Detect slots and crop them (keeping the full bottom of each slot).
3. Classify each slot crop against the color-histogram reference index.
4. Write ``<output_dir>/<screenshot_stem>/report.html``, ``results.json``
   and ``slots/*.png`` containing the individual slot crops.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2

from . import color_matcher, reporting, slots
from .paths import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REFERENCES_DIR,
    IMAGE_EXTENSIONS,
)


LOGGER = logging.getLogger("hue_map.pipeline")


DEFAULT_TOP_K = color_matcher.DEFAULT_TOP_K
DEFAULT_BORDER_FRACTION = slots.DEFAULT_BORDER_FRACTION


def process_screenshot(
    screenshot_path: Path,
    reference_index: color_matcher.ReferenceIndex,
    output_dir: Path,
    *,
    top_k: int = DEFAULT_TOP_K,
    border_fraction: float = DEFAULT_BORDER_FRACTION,
) -> dict[str, Any]:
    """Process one screenshot end-to-end and write a report.

    Returns a summary dict with the resulting file paths and slot count.
    """
    screenshot_path = Path(screenshot_path)
    image = slots.load_screenshot(screenshot_path)

    slot_crops = slots.detect_and_crop_slots(image, border_fraction=border_fraction)
    if not slot_crops:
        raise SystemExit(f"No slots detected in {screenshot_path}")

    run_dir = output_dir / screenshot_path.stem
    slots_dir = run_dir / "slots"
    slots_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for crop in slot_crops:
        # Persist a cleaned BGRA version of the slot crop so the HTML report
        # shows the item on a transparent background, without the slot tint,
        # quality badge, quality bar, or stack count.
        crop_filename = f"slot_{crop.slot_index:03d}_r{crop.row:02d}_c{crop.col:02d}.png"
        crop_path = slots_dir / crop_filename
        cleaned = color_matcher.cleaned_rgba(crop.image)
        cv2.imwrite(str(crop_path), cleaned)

        # Classify from the raw BGR array - the classifier does its own
        # masking so it sees exactly the same pixels as the cleaned preview.
        match = color_matcher.classify_item_crop(crop.image, reference_index, top_k=top_k)
        results.append(
            {
                "slot_index": crop.slot_index,
                "row": crop.row,
                "col": crop.col,
                "path": crop_path,
                "match": match,
            }
        )

    report_path = run_dir / "report.html"
    report_path.write_text(
        reporting.render_html(screenshot_path, results),
        encoding="utf-8",
    )

    results_path = run_dir / "results.json"
    results_path.write_text(
        json.dumps([_serialisable(result) for result in results], indent=2),
        encoding="utf-8",
    )

    LOGGER.info("Report: %s", report_path)
    return {
        "screenshot_path": str(screenshot_path),
        "output_dir": str(run_dir),
        "report_path": str(report_path),
        "results_path": str(results_path),
        "slot_count": len(results),
    }


def iter_input_images(input_path: Path) -> list[Path]:
    """Return all image files in ``input_path`` (or ``[input_path]`` if a file)."""
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )


def _serialisable(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a result dict into something ``json.dumps`` can handle."""
    serialised = dict(result)
    serialised["path"] = str(result["path"])
    return serialised


# -- CLI --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect slots, colour-match each item crop, and emit a "
            "self-contained HTML report per screenshot."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help="Screenshot file or directory of screenshots. Default: src/input/",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=DEFAULT_REFERENCES_DIR,
        help="Reference library root. Default: src/references/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where reports are written. Default: src/output/",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--border-fraction",
        type=float,
        default=DEFAULT_BORDER_FRACTION,
        help=(
            "Fraction trimmed from the left, right and top of each slot. "
            "The bottom is never trimmed."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    image_paths = iter_input_images(args.input)
    if not image_paths:
        raise SystemExit(
            f"No input images found in {args.input}. Drop PNG/JPG screenshots there first."
        )

    LOGGER.info("Found %d screenshot(s) to process", len(image_paths))
    reference_index = color_matcher.build_reference_index(args.references)

    args.output.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for image_path in image_paths:
        LOGGER.info("Processing %s", image_path)
        summary = process_screenshot(
            image_path,
            reference_index,
            args.output,
            top_k=args.top_k,
            border_fraction=args.border_fraction,
        )
        summaries.append(summary)

    print("Hue-Map pipeline summary")
    print(f"References: {reference_index.item_count} items / {reference_index.reference_count} images")
    for summary in summaries:
        print(
            f"  {Path(summary['screenshot_path']).name}: "
            f"{summary['slot_count']} slots -> {summary['report_path']}"
        )


if __name__ == "__main__":
    main()
