from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:
    import open_clip
    import torch
except ImportError as exc:
    open_clip = None
    torch = None
    OPENCLIP_IMPORT_ERROR: ImportError | None = exc
else:
    OPENCLIP_IMPORT_ERROR = None


LOGGER = logging.getLogger("anvil_item_matcher")

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"
DEFAULT_REFERENCES_ROOT = Path("references")
DEFAULT_OUTPUT_DIR = Path("data") / "item_matches"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
USABLE_REFERENCE_FOLDERS = ("wiki", "approved")


@dataclass(frozen=True)
class ItemReference:
    item_name: str
    image_path: Path
    source_folder: str
    embedding: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_name": self.item_name,
            "image_path": str(self.image_path),
            "source_folder": self.source_folder,
        }


@dataclass
class ItemReferenceIndex:
    references_root: Path
    model_name: str
    pretrained: str
    device: str
    model: Any
    preprocess: Any
    references: list[ItemReference]
    embedding_matrix: np.ndarray
    item_to_indices: dict[str, list[int]]

    @property
    def item_count(self) -> int:
        return len(self.item_to_indices)

    @property
    def reference_count(self) -> int:
        return len(self.references)


def load_item_references(references_root: str | Path) -> dict[str, list[Path]]:
    root = Path(references_root)
    if not root.exists():
        LOGGER.warning("Item references root does not exist: %s", root)
        return {}
    if not root.is_dir():
        LOGGER.warning("Item references root is not a directory: %s", root)
        return {}

    references: dict[str, list[Path]] = {}
    for item_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.casefold()):
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


def build_item_reference_index(
    references_root: str | Path,
    model_name: str = MODEL_NAME,
    pretrained: str = PRETRAINED,
    device: str | None = None,
) -> ItemReferenceIndex:
    references_by_item = load_item_references(references_root)
    if not references_by_item:
        raise RuntimeError(
            f"No usable item references found under {references_root}. "
            "Expected item folders with wiki/ or approved/ images."
        )

    _raise_if_openclip_missing()
    resolved_device = _resolve_device(device)
    LOGGER.info("Loading OpenCLIP %s (%s) on %s", model_name, pretrained, resolved_device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=resolved_device,
    )
    model.eval()

    references: list[ItemReference] = []
    embeddings: list[np.ndarray] = []
    item_to_indices: dict[str, list[int]] = {}

    for item_name, image_paths in references_by_item.items():
        for image_path in image_paths:
            try:
                embedding = _encode_image(image_path, model, preprocess, resolved_device)
            except Exception as exc:
                LOGGER.warning("Skipping unreadable item reference %s: %s", image_path, exc)
                continue

            reference = ItemReference(
                item_name=item_name,
                image_path=image_path,
                source_folder=_source_folder_name(image_path),
                embedding=embedding,
            )
            item_to_indices.setdefault(item_name, []).append(len(references))
            references.append(reference)
            embeddings.append(embedding)

    if not references:
        raise RuntimeError(f"No item reference images could be encoded from {references_root}")

    embedding_matrix = np.stack(embeddings).astype(np.float32)
    LOGGER.info("Built item reference index: %d items, %d references", len(item_to_indices), len(references))

    return ItemReferenceIndex(
        references_root=Path(references_root),
        model_name=model_name,
        pretrained=pretrained,
        device=resolved_device,
        model=model,
        preprocess=preprocess,
        references=references,
        embedding_matrix=embedding_matrix,
        item_to_indices=item_to_indices,
    )


def classify_item_crop(
    image_path: str | Path,
    reference_index: ItemReferenceIndex,
    top_k: int = 5,
    max_reference_paths_per_item: int = 3,
) -> dict[str, Any]:
    query_path = Path(image_path)
    if not query_path.exists():
        return _empty_item_result(query_path, error=f"Image crop does not exist: {query_path}")

    if reference_index.embedding_matrix.size == 0 or not reference_index.references:
        return _empty_item_result(query_path, error="Item reference index is empty")

    try:
        query_embedding = _encode_image(query_path, reference_index.model, reference_index.preprocess, reference_index.device)
    except Exception as exc:
        LOGGER.warning("Could not encode item crop %s: %s", query_path, exc)
        return _empty_item_result(query_path, error=str(exc))

    similarities = reference_index.embedding_matrix @ query_embedding
    scored_by_item: dict[str, list[tuple[float, Path]]] = {}
    for index, similarity in enumerate(similarities):
        reference = reference_index.references[index]
        scored_by_item.setdefault(reference.item_name, []).append((float(similarity), reference.image_path))

    aggregated_matches: list[dict[str, Any]] = []
    for item_name, scored_references in scored_by_item.items():
        ranked_references = sorted(scored_references, key=lambda item: item[0], reverse=True)
        best_score = ranked_references[0][0]
        top_reference_scores = [score for score, _ in ranked_references[:max_reference_paths_per_item]]
        mean_top_score = float(np.mean(top_reference_scores)) if top_reference_scores else best_score
        matched_paths = [str(path) for _, path in ranked_references[:max_reference_paths_per_item]]

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

    top_matches = sorted(aggregated_matches, key=lambda item: item["confidence"], reverse=True)[:top_k]
    if not top_matches:
        return _empty_item_result(query_path, error="No item matches produced")

    predicted = top_matches[0]
    second_confidence = float(top_matches[1]["confidence"]) if len(top_matches) > 1 else 0.0
    confidence = float(predicted["confidence"])

    return {
        "image_path": str(query_path),
        "predicted_item": predicted["item"],
        "confidence": confidence,
        "top_gap": round(confidence - second_confidence, 4) if len(top_matches) > 1 else None,
        "top_matches": top_matches,
        "matched_reference_paths": predicted["matched_reference_paths"],
        "error": None,
    }


def batch_classify_item_crops(
    image_paths: list[str | Path],
    reference_index: ItemReferenceIndex,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    return [classify_item_crop(image_path, reference_index, top_k=top_k) for image_path in image_paths]


def discover_item_crops(path: str | Path) -> list[Path]:
    input_path = Path(path)
    if input_path.is_file():
        return [input_path]

    if (input_path / "items").is_dir():
        input_path = input_path / "items"

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


def _discover_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "rejected" not in {part.casefold() for part in path.parts}
    )


def _encode_image(image_path: Path, model: Any, preprocess: Any, device: str) -> np.ndarray:
    _raise_if_openclip_missing()
    image = _load_pil_image(image_path)
    with torch.inference_mode():
        tensor = preprocess(image).unsqueeze(0).to(device)
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _load_pil_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, (82, 80, 70, 255))
    background.alpha_composite(image)
    return ImageOps.exif_transpose(background.convert("RGB"))


def _source_folder_name(image_path: Path) -> str:
    for part in image_path.parts:
        if part in USABLE_REFERENCE_FOLDERS:
            return part
    return ""


def _resolve_device(device: str | None) -> str:
    _raise_if_openclip_missing()
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _raise_if_openclip_missing() -> None:
    if OPENCLIP_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing dependency: open_clip_torch. Install dependencies with "
            "`py -m pip install -r requirements.txt`."
        ) from OPENCLIP_IMPORT_ERROR


def _empty_item_result(image_path: Path, error: str | None = None) -> dict[str, Any]:
    return {
        "image_path": str(image_path),
        "predicted_item": "",
        "confidence": 0.0,
        "top_gap": None,
        "top_matches": [],
        "matched_reference_paths": [],
        "error": error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify item crops against references/*/{wiki,approved}.")
    parser.add_argument("input_path", type=Path, help="An item crop, items/ folder, or parser output folder.")
    parser.add_argument(
        "--references-root",
        type=Path,
        default=DEFAULT_REFERENCES_ROOT,
        help="Root containing one folder per item. Default: references",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "item_matches.json",
        help="Where JSON results are written.",
    )
    parser.add_argument("--model", default=MODEL_NAME, help=f"OpenCLIP model name. Default: {MODEL_NAME}")
    parser.add_argument("--pretrained", default=PRETRAINED, help=f"OpenCLIP weights. Default: {PRETRAINED}")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Torch device. Default: auto")
    parser.add_argument("--top-k", type=int, default=5, help="Number of item candidates to keep. Default: 5")
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

    image_paths = discover_item_crops(args.input_path)
    if not image_paths:
        raise SystemExit(f"No item crops found in {args.input_path}")

    reference_index = build_item_reference_index(
        args.references_root,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )
    results = batch_classify_item_crops(image_paths, reference_index, top_k=args.top_k)
    write_json(results, args.output_path)

    print("Anvil item match summary")
    print(f"Reference items: {reference_index.item_count}")
    print(f"Reference images: {reference_index.reference_count}")
    print(f"Item crops: {len(results)}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
