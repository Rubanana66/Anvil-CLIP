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
import base64
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from . import color_matcher, reporting, slots, stack_count
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
        # Skip this screenshot but keep processing the rest of the batch.
        LOGGER.warning("No slots detected in %s; skipping.", screenshot_path)
        return {
            "screenshot_path": str(screenshot_path),
            "output_dir": "",
            "report_path": "",
            "results_path": "",
            "slot_count": 0,
            "skipped": True,
        }

    run_dir = output_dir / screenshot_path.stem
    slots_dir = run_dir / "slots"
    numbers_dir = run_dir / "numbers"
    slots_dir.mkdir(parents=True, exist_ok=True)
    numbers_dir.mkdir(parents=True, exist_ok=True)

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

        # Read the stack-count number printed in the slot's bottom-right.
        # Uses its own preprocessing pipeline tuned for white UI text,
        # completely independent from the item classifier.
        count = stack_count.read_stack_count(crop.image)

        # Persist the preprocessed binary image so the HTML report can
        # show exactly what Tesseract saw. Skip if there's no binary
        # (which only happens on image-load failure).
        count_image_path = ""
        if count.binary_image is not None and count.binary_image.size > 0:
            count_image_file = numbers_dir / f"number_{crop.slot_index:03d}_r{crop.row:02d}_c{crop.col:02d}.png"
            cv2.imwrite(str(count_image_file), count.binary_image)
            count_image_path = str(count_image_file)

        count_dict = count.to_dict()
        count_dict["image_path"] = count_image_path

        # Semantic rule: Anvil doesn't print a "1" next to a single item,
        # so a slot that's classified as a real item but yielded no OCR
        # reading is interpreted as a stack of one. Empty slots stay
        # None; only OCR-failed-but-item-present slots get the default.
        if not match.get("is_empty") and count_dict["value"] is None:
            count_dict["value"] = 1
            count_dict["assumed_single"] = True
        else:
            count_dict["assumed_single"] = False

        results.append(
            {
                "slot_index": crop.slot_index,
                "row": crop.row,
                "col": crop.col,
                "path": crop_path,
                "match": match,
                "stack_count": count_dict,
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


# -- alternative input sources ---------------------------------------------
# A caller that doesn't have a file on disk can still feed the pipeline a
# screenshot through ``--base64`` (inline string), ``--base64-file`` (path
# to a file containing a base64 payload, useful when the string is too
# long for a Windows command line), or ``--stdin`` (raw PNG/JPG bytes via
# a pipe). We write the decoded bytes to a timestamped ``_transient/``
# folder under the output dir so the rest of the pipeline - which expects
# a path - just works. The path is returned for processing.

_TRANSIENT_FOLDER_NAME = "_transient"


def _transient_dir(output_root: Path) -> Path:
    """Directory where decoded base64 / stdin images are saved temporarily."""
    transient = output_root / _TRANSIENT_FOLDER_NAME
    transient.mkdir(parents=True, exist_ok=True)
    return transient


def _save_bytes_as_screenshot(
    data: bytes,
    *,
    output_root: Path,
    label: str,
) -> Path:
    """Write raw image bytes to the transient folder and return the new path."""
    if not data:
        raise SystemExit("Received an empty image payload.")
    stem = _safe_label_stem(label)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = _transient_dir(output_root) / f"{stem}_{stamp}.png"
    destination.write_bytes(data)
    LOGGER.info("Saved transient screenshot to %s (%d bytes)", destination, len(data))
    return destination


def _decode_base64_payload(payload: str) -> bytes:
    """Decode a base64 string, tolerating ``data:image/...;base64,`` prefixes."""
    text = payload.strip()
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    # Whitespace and newlines inside the base64 string are legal but some
    # decoders complain; strip them so the decode is always clean.
    text = "".join(text.split())
    try:
        return base64.b64decode(text, validate=False)
    except (ValueError, base64.binascii.Error) as exc:
        raise SystemExit(f"Could not decode base64 payload: {exc}") from exc


def _safe_label_stem(value: str) -> str:
    """Sanitise a short hint so it's safe to use as a filename stem."""
    allowed = []
    for char in value.strip() or "input":
        if char.isalnum() or char in "-_":
            allowed.append(char)
        else:
            allowed.append("_")
    cleaned = "".join(allowed).strip("_")
    return cleaned[:40] or "input"


def resolve_input_images(args: argparse.Namespace) -> list[Path]:
    """Resolve CLI arguments into a concrete list of screenshot paths.

    The flags are mutually exclusive in argparse, so exactly one of
    (base64, base64-file, stdin, positional path) is active per call.
    """
    if args.base64:
        data = _decode_base64_payload(args.base64)
        return [_save_bytes_as_screenshot(data, output_root=args.output, label=args.label or "b64_input")]
    if args.base64_file:
        text = Path(args.base64_file).read_text(encoding="utf-8")
        data = _decode_base64_payload(text)
        label = args.label or Path(args.base64_file).stem
        return [_save_bytes_as_screenshot(data, output_root=args.output, label=label)]
    if args.stdin:
        data = sys.stdin.buffer.read()
        return [_save_bytes_as_screenshot(data, output_root=args.output, label=args.label or "stdin_input")]
    return iter_input_images(args.input)


def _serialisable(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a result dict into something ``json.dumps`` can handle."""
    serialised = dict(result)
    serialised["path"] = str(result["path"])
    return serialised


# -- CLI --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect slots, colour-match each item crop, read stack counts, "
            "and emit a self-contained HTML report per screenshot.\n\n"
            "Input can be provided four ways:\n"
            "  1. Positional path: run on a file or directory of screenshots.\n"
            "  2. --base64 <string>: decode a base64-encoded image inline.\n"
            "  3. --base64-file <path>: same as above but the base64 text\n"
            "     lives in a file (useful when the payload is too long for\n"
            "     a Windows command-line argument).\n"
            "  4. --stdin: read raw PNG / JPG bytes piped from stdin."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    # Alternative input sources. Mutually exclusive - you either point
    # at a file/directory or feed a payload through one of these flags.
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--base64",
        metavar="STRING",
        help=(
            "Decode the given base64 string as an image and run the "
            "pipeline on it. Accepts 'data:image/...;base64,...' URIs."
        ),
    )
    source_group.add_argument(
        "--base64-file",
        type=Path,
        metavar="FILE",
        help="Read a base64 string from FILE and decode it as an image.",
    )
    source_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read raw image bytes (PNG/JPG/...) from stdin.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help=(
            "Optional short label used as the output folder name when "
            "processing base64 / stdin input. Default: a timestamped name."
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

    args.output.mkdir(parents=True, exist_ok=True)

    image_paths = resolve_input_images(args)
    if not image_paths:
        # Only the positional-path case produces an empty list here; the
        # base64/stdin paths raise SystemExit on empty payloads directly.
        raise SystemExit(
            f"No input images found in {args.input}. "
            "Drop PNG/JPG screenshots there or pass --base64 / --stdin."
        )

    LOGGER.info("Found %d screenshot(s) to process", len(image_paths))
    LOGGER.info(stack_count.describe_ocr_status())
    reference_index = color_matcher.build_reference_index(args.references)

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
