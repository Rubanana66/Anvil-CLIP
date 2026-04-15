from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

try:
    import open_clip
except ImportError as exc:
    print(
        "Missing dependency: open_clip_torch\n\n"
        "Install dependencies with:\n"
        "  py -m pip install -r requirements.txt\n",
    )
    raise SystemExit(1) from exc


LOGGER = logging.getLogger("anvil_icon_matcher")

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"
DEFAULT_ICON_DIR = Path("itemlist")
DEFAULT_CACHE_DIR = Path(".cache") / "icon_matcher"
DEFAULT_OUTPUT_DIR = Path("output_debug") / "icon_matches"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class IconReference:
    item_name: str
    icon_path: Path
    embedding: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_name": self.item_name,
            "icon_path": str(self.icon_path),
        }


@dataclass(frozen=True)
class IconCandidate:
    item_name: str
    icon_path: Path
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_name": self.item_name,
            "icon_path": str(self.icon_path),
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class IconMatch:
    query_path: Path
    best_item: str
    best_icon_path: Path
    similarity: float
    review_needed: bool
    candidates: list[IconCandidate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_path": str(self.query_path),
            "best_item": self.best_item,
            "best_icon_path": str(self.best_icon_path),
            "similarity": self.similarity,
            "review_needed": self.review_needed,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class IconMatcher:
    def __init__(
        self,
        icon_dir: Path = DEFAULT_ICON_DIR,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        model_name: str = MODEL_NAME,
        pretrained: str = PRETRAINED,
        device: str | None = None,
        review_threshold: float = 0.24,
        margin_threshold: float = 0.025,
    ) -> None:
        self.icon_dir = icon_dir
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.review_threshold = review_threshold
        self.margin_threshold = margin_threshold

        LOGGER.info("Loading OpenCLIP %s (%s) on %s", model_name, pretrained, self.device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.model.eval()

        self.references = self._load_or_build_reference_cache()
        if not self.references:
            raise RuntimeError(f"No reference icons found in {icon_dir}")

        self.reference_matrix = np.stack([reference.embedding for reference in self.references])
        LOGGER.info("Loaded %d reference icon embeddings.", len(self.references))

    def match_icon(self, icon_path: Path, top_k: int = 5) -> IconMatch:
        embedding = self._encode_image(icon_path)
        similarities = self.reference_matrix @ embedding
        top_indices = np.argsort(-similarities)[:top_k]

        candidates = [
            IconCandidate(
                item_name=self.references[index].item_name,
                icon_path=self.references[index].icon_path,
                similarity=round(float(similarities[index]), 4),
            )
            for index in top_indices
        ]
        best = candidates[0]
        second_similarity = candidates[1].similarity if len(candidates) > 1 else -1.0
        margin = best.similarity - second_similarity
        review_needed = best.similarity < self.review_threshold or margin < self.margin_threshold

        return IconMatch(
            query_path=icon_path,
            best_item=best.item_name,
            best_icon_path=best.icon_path,
            similarity=best.similarity,
            review_needed=review_needed,
            candidates=candidates,
        )

    def match_many(self, icon_paths: list[Path], top_k: int = 5) -> list[IconMatch]:
        return [self.match_icon(icon_path, top_k=top_k) for icon_path in icon_paths]

    def _load_or_build_reference_cache(self) -> list[IconReference]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path()
        icon_paths = sorted(
            path for path in self.icon_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        current_signature = _icon_dir_signature(icon_paths)

        if cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=True)
                if str(cached["signature"]) == current_signature:
                    LOGGER.info("Using cached reference embeddings: %s", cache_path)
                    names = cached["names"].tolist()
                    paths = [Path(value) for value in cached["paths"].tolist()]
                    embeddings = cached["embeddings"]
                    return [
                        IconReference(name, path, embedding.astype(np.float32))
                        for name, path, embedding in zip(names, paths, embeddings)
                    ]
                LOGGER.info("Reference icon cache is stale; rebuilding.")
            except Exception as exc:
                LOGGER.warning("Could not read icon cache %s: %s", cache_path, exc)

        references = []
        for index, icon_path in enumerate(icon_paths, start=1):
            LOGGER.debug("[%d/%d] Encoding %s", index, len(icon_paths), icon_path.name)
            references.append(
                IconReference(
                    item_name=item_name_from_icon_path(icon_path),
                    icon_path=icon_path,
                    embedding=self._encode_image(icon_path),
                )
            )

        np.savez_compressed(
            cache_path,
            signature=current_signature,
            names=np.array([reference.item_name for reference in references], dtype=object),
            paths=np.array([str(reference.icon_path) for reference in references], dtype=object),
            embeddings=np.stack([reference.embedding for reference in references]),
        )
        LOGGER.info("Saved reference embedding cache: %s", cache_path)
        return references

    def _cache_path(self) -> Path:
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{self.model_name}_{self.pretrained}")
        return self.cache_dir / f"{safe_model}_references.npz"

    @torch.inference_mode()
    def _encode_image(self, image_path: Path) -> np.ndarray:
        image = load_pil_image(image_path)
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).detach().cpu().numpy().astype(np.float32)


def load_pil_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")

    # Reference icons often have transparency. Compositing them onto a neutral
    # stockpile-like background makes them closer to parser icon crops.
    background = Image.new("RGBA", image.size, (82, 80, 70, 255))
    background.alpha_composite(image)
    image = background.convert("RGB")
    return ImageOps.exif_transpose(image)


def item_name_from_icon_path(icon_path: Path) -> str:
    return icon_path.stem.replace("_", " ")


def discover_icon_crops(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    if (path / "icons").is_dir():
        path = path / "icons"

    return sorted(
        candidate for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_json(matches: list[IconMatch], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([match.to_dict() for match in matches], indent=2),
        encoding="utf-8",
    )


def write_csv(matches: list[IconMatch], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_path",
                "best_item",
                "similarity",
                "review_needed",
                "top_candidates",
            ],
        )
        writer.writeheader()
        for match in matches:
            writer.writerow(
                {
                    "query_path": str(match.query_path),
                    "best_item": match.best_item,
                    "similarity": match.similarity,
                    "review_needed": match.review_needed,
                    "top_candidates": " | ".join(
                        f"{candidate.item_name}:{candidate.similarity:.4f}"
                        for candidate in match.candidates
                    ),
                }
            )


def _icon_dir_signature(icon_paths: list[Path]) -> str:
    parts = [
        f"{path.name}:{path.stat().st_size}:{int(path.stat().st_mtime)}"
        for path in icon_paths
    ]
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match Anvil item icon crops against itemlist icons using OpenCLIP.")
    parser.add_argument(
        "input_path",
        type=Path,
        help="One icon crop, an icons folder, or a parser output folder containing icons/.",
    )
    parser.add_argument(
        "--icon-dir",
        type=Path,
        default=DEFAULT_ICON_DIR,
        help="Reference icon folder. Default: itemlist",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where icon_matches.json/csv are written.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"OpenCLIP model name. Default: {MODEL_NAME}",
    )
    parser.add_argument(
        "--pretrained",
        default=PRETRAINED,
        help=f"OpenCLIP pretrained weights. Default: {PRETRAINED}",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Torch device. Default: cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidates to keep per icon. Default: 5",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.24,
        help="Flag review if best similarity is below this. Default: 0.24",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.025,
        help="Flag review if top match barely beats second match. Default: 0.025",
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

    icon_paths = discover_icon_crops(args.input_path)
    if not icon_paths:
        raise SystemExit(f"No icon crops found in {args.input_path}")

    matcher = IconMatcher(
        icon_dir=args.icon_dir,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        review_threshold=args.review_threshold,
        margin_threshold=args.margin_threshold,
    )
    matches = matcher.match_many(icon_paths, top_k=args.top_k)

    output_json = args.output_dir / "icon_matches.json"
    output_csv = args.output_dir / "icon_matches.csv"
    write_json(matches, output_json)
    write_csv(matches, output_csv)

    print()
    print("Anvil icon match summary")
    print("------------------------")
    print(f"Reference icons: {len(matcher.references)}")
    print(f"Query icons: {len(matches)}")
    print(f"JSON: {output_json}")
    print(f"CSV:  {output_csv}")
    for match in matches:
        review = " REVIEW" if match.review_needed else ""
        print(f"{match.query_path.name}: {match.best_item} ({match.similarity:.4f}){review}")


if __name__ == "__main__":
    main()
