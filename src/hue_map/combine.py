"""Merge every per-screenshot report into one HTML page.

Walks ``src/output/<screenshot>/results.json`` and ``slots/``, then writes a
single ``src/output/combined_report.html`` with one section per screenshot.
Each section includes the source screenshot, a summary ribbon, and the same
per-slot prediction table rendered by :mod:`hue_map.reporting`. An index at
the top links to each section so you can jump around long runs.

CLI::

    py -m hue_map.combine                          # default paths
    py -m hue_map.combine --output combined.html   # custom output
    py -m hue_map.combine --sort name              # sort sections by name

``results.json`` carries the slot-crop paths; the crops themselves still
live on disk and are embedded as base64 data URIs in the merged page, so
the resulting file is self-contained.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from . import reporting
from .paths import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, IMAGE_EXTENSIONS


LOGGER = logging.getLogger("hue_map.combine")


def collect_reports(output_root: Path, input_root: Path) -> list[tuple[Path, list[dict[str, Any]]]]:
    """Return ``(screenshot_path, results)`` tuples for every report found.

    ``screenshot_path`` is resolved against ``input_root`` when possible so
    the original image can be embedded into the combined report. If the
    source screenshot is missing we still include the section, just without
    the big preview at the top.
    """
    items: list[tuple[Path, list[dict[str, Any]]]] = []

    # Map screenshot stems to their source files so we can reattach the
    # source image even if the output folder was renamed since the run.
    input_by_stem: dict[str, Path] = {}
    if input_root.exists() and input_root.is_dir():
        for candidate in input_root.iterdir():
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                input_by_stem.setdefault(candidate.stem, candidate)

    for entry in sorted(output_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        results_file = entry / "results.json"
        if not results_file.exists():
            continue

        try:
            results = json.loads(results_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            LOGGER.warning("Skipping unreadable %s: %s", results_file, exc)
            continue

        screenshot_path = input_by_stem.get(entry.name, input_root / f"{entry.name}.png")
        items.append((screenshot_path, results))

    return items


def sort_items(
    items: list[tuple[Path, list[dict[str, Any]]]],
    *,
    strategy: str,
) -> list[tuple[Path, list[dict[str, Any]]]]:
    """Sort the list of ``(screenshot_path, results)`` pairs."""
    if strategy == "name":
        return sorted(items, key=lambda item: item[0].name.casefold())
    if strategy == "slots":
        # Most slots first - useful to triage the biggest inventories first.
        return sorted(items, key=lambda item: len(item[1]), reverse=True)
    if strategy == "weakest":
        # Sections with the most "weak / uncertain" predictions first.
        return sorted(items, key=lambda item: _uncertain_count(item[1]), reverse=True)
    return items


def _uncertain_count(results: list[dict[str, Any]]) -> int:
    """Number of results whose prediction is not in the "confident" bucket.

    Empty slots are not counted as uncertain - they're a deliberate state,
    not a failed prediction.
    """
    uncertain = 0
    for result in results:
        match = result.get("match", {})
        if match.get("is_empty"):
            continue
        confidence = float(match.get("confidence", 0.0) or 0.0)
        gap = match.get("top_gap")
        if reporting._confidence_class(confidence, gap) != "pred-green":
            uncertain += 1
    return uncertain


# -- CLI --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing per-screenshot report folders. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory holding the source screenshots. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination HTML file. Default: <output-root>/combined_report.html",
    )
    parser.add_argument(
        "--sort",
        choices=("name", "slots", "weakest", "none"),
        default="name",
        help="Section order. Default: name",
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

    if not args.output_root.exists() or not args.output_root.is_dir():
        raise SystemExit(f"Output root does not exist: {args.output_root}")

    items = collect_reports(args.output_root, args.input_root)
    if not items:
        raise SystemExit(
            f"No report sections found under {args.output_root}. "
            "Run `py -m hue_map.pipeline` first."
        )

    items = sort_items(items, strategy=args.sort)
    html = reporting.render_combined_html(items)

    output_path = args.output or (args.output_root / "combined_report.html")
    output_path.write_text(html, encoding="utf-8")

    total_slots = sum(len(results) for _, results in items)
    print(
        f"Combined {len(items)} screenshot(s) / {total_slots} slot(s) -> {output_path} "
        f"({output_path.stat().st_size / 1024:.0f} KB)"
    )


if __name__ == "__main__":
    main()
