from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import item_matcher


LOGGER = logging.getLogger("anvil_quality_matcher")

DEFAULT_QUALITY_REFERENCES_ROOT = Path("quality_references")
DEFAULT_OUTPUT_DIR = Path("data") / "quality_matches"
IMAGE_EXTENSIONS = item_matcher.IMAGE_EXTENSIONS
USABLE_REFERENCE_FOLDERS = item_matcher.USABLE_REFERENCE_FOLDERS


@dataclass
class QualityReferenceIndex:
    references_root: Path
    item_index: item_matcher.ItemReferenceIndex

    @property
    def quality_count(self) -> int:
        return self.item_index.item_count

    @property
    def reference_count(self) -> int:
        return self.item_index.reference_count


_QUALITY_INDEX_CACHE: dict[tuple[str, str, str, str], QualityReferenceIndex] = {}


def load_quality_references(references_root: str | Path) -> dict[str, list[Path]]:
    root = Path(references_root)
    if not root.exists():
        LOGGER.warning("Quality references root does not exist: %s", root)
        return {}
    if not root.is_dir():
        LOGGER.warning("Quality references root is not a directory: %s", root)
        return {}

    references: dict[str, list[Path]] = {}
    for quality_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.casefold()):
        image_paths: list[Path] = []
        for folder_name in USABLE_REFERENCE_FOLDERS:
            source_dir = quality_dir / folder_name
            if source_dir.is_dir():
                image_paths.extend(_discover_images(source_dir))
        if image_paths:
            references[quality_dir.name] = sorted(image_paths, key=lambda path: str(path).casefold())

    LOGGER.info(
        "Loaded %d quality folder(s) with %d usable reference image(s) from %s",
        len(references),
        sum(len(paths) for paths in references.values()),
        root,
    )
    return references


def classify_quality_crop(image_path: str | Path, references_root: str | Path) -> dict[str, Any]:
    try:
        reference_index = _get_quality_reference_index(references_root)
    except Exception as exc:
        LOGGER.warning("Quality matching unavailable for %s: %s", image_path, exc)
        return _empty_quality_result(Path(image_path), error=str(exc))

    return _classify_quality_crop_with_index(image_path, reference_index)


def batch_classify_quality_crops(image_paths: list[str | Path], references_root: str | Path) -> list[dict[str, Any]]:
    try:
        reference_index = _get_quality_reference_index(references_root)
    except Exception as exc:
        LOGGER.warning("Quality matching unavailable: %s", exc)
        return [_empty_quality_result(Path(image_path), error=str(exc)) for image_path in image_paths]

    return [_classify_quality_crop_with_index(image_path, reference_index) for image_path in image_paths]


def discover_quality_crops(path: str | Path) -> list[Path]:
    input_path = Path(path)
    if input_path.is_file():
        return [input_path]

    if (input_path / "quality").is_dir():
        input_path = input_path / "quality"

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


def _get_quality_reference_index(
    references_root: str | Path,
    model_name: str = item_matcher.MODEL_NAME,
    pretrained: str = item_matcher.PRETRAINED,
    device: str | None = None,
) -> QualityReferenceIndex:
    root = Path(references_root)
    resolved_device = device or "auto"
    cache_key = (str(root.resolve() if root.exists() else root), model_name, pretrained, resolved_device)
    cached = _QUALITY_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not load_quality_references(root):
        raise RuntimeError(
            f"No usable quality references found under {root}. "
            "Expected quality folders with wiki/ or approved/ images."
        )

    item_index = item_matcher.build_item_reference_index(
        root,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    quality_index = QualityReferenceIndex(references_root=root, item_index=item_index)
    _QUALITY_INDEX_CACHE[cache_key] = quality_index
    return quality_index


def _classify_quality_crop_with_index(
    image_path: str | Path,
    reference_index: QualityReferenceIndex,
) -> dict[str, Any]:
    item_result = item_matcher.classify_item_crop(image_path, reference_index.item_index)
    top_matches = [
        {
            "quality": match["item"],
            "confidence": match["confidence"],
            "best_score": match["best_score"],
            "mean_top_score": match["mean_top_score"],
            "reference_count": match["reference_count"],
            "matched_reference_paths": match["matched_reference_paths"],
        }
        for match in item_result.get("top_matches", [])
    ]

    return {
        "image_path": item_result.get("image_path", str(image_path)),
        "detected_quality": item_result.get("predicted_item", ""),
        "quality_confidence": float(item_result.get("confidence", 0.0) or 0.0),
        "top_gap": item_result.get("top_gap"),
        "top_matches": top_matches,
        "matched_reference_paths": item_result.get("matched_reference_paths", []),
        "method": "openclip_reference_matching",
        "error": item_result.get("error"),
    }


def _empty_quality_result(image_path: Path, error: str | None = None) -> dict[str, Any]:
    return {
        "image_path": str(image_path),
        "detected_quality": "",
        "quality_confidence": 0.0,
        "top_gap": None,
        "top_matches": [],
        "matched_reference_paths": [],
        "method": "openclip_reference_matching",
        "error": error,
    }


def _discover_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "rejected" not in {part.casefold() for part in path.parts}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify quality crops against quality_references/*/{wiki,approved}.")
    parser.add_argument("input_path", type=Path, help="A quality crop, quality/ folder, or parser output folder.")
    parser.add_argument(
        "--references-root",
        type=Path,
        default=DEFAULT_QUALITY_REFERENCES_ROOT,
        help="Root containing one folder per quality. Default: quality_references",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "quality_matches.json",
        help="Where JSON results are written.",
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

    image_paths = discover_quality_crops(args.input_path)
    if not image_paths:
        raise SystemExit(f"No quality crops found in {args.input_path}")

    results = batch_classify_quality_crops(image_paths, args.references_root)
    write_json(results, args.output_path)

    print("Anvil quality match summary")
    print(f"Quality crops: {len(results)}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
