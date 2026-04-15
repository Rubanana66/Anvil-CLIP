from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import item_matcher
import ocr
import quality_matcher


LOGGER = logging.getLogger("anvil_pipeline")

ITEM_AUTO_APPROVE_THRESHOLD = 0.30
QUALITY_AUTO_APPROVE_THRESHOLD = 0.30
OCR_AUTO_APPROVE_THRESHOLD = 0.45
TOP_GAP_MINIMUM = 0.025
DEFAULT_OUTPUT_DIR = Path("data") / "incoming_results"
DEFAULT_CONFUSION_GROUPS_PATH = Path("confusion_groups.json")


@dataclass(frozen=True)
class ReviewThresholds:
    item_auto_approve_threshold: float = ITEM_AUTO_APPROVE_THRESHOLD
    quality_auto_approve_threshold: float = QUALITY_AUTO_APPROVE_THRESHOLD
    ocr_auto_approve_threshold: float = OCR_AUTO_APPROVE_THRESHOLD
    top_gap_minimum: float = TOP_GAP_MINIMUM


def run_pipeline(
    parse_result_path: str | Path,
    references_root: str | Path = item_matcher.DEFAULT_REFERENCES_ROOT,
    quality_references_root: str | Path = quality_matcher.DEFAULT_QUALITY_REFERENCES_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    confusion_groups_path: str | Path | None = DEFAULT_CONFUSION_GROUPS_PATH,
    scan_id: str | None = None,
    model_name: str = item_matcher.MODEL_NAME,
    pretrained: str = item_matcher.PRETRAINED,
    device: str | None = None,
    thresholds: ReviewThresholds = ReviewThresholds(),
) -> dict[str, Any]:
    parse_path = Path(parse_result_path)
    parse_result = load_parse_result(parse_path)
    parse_dir = parse_path.resolve().parent
    resolved_scan_id = scan_id or _default_scan_id(parse_result, parse_path)
    confusion_groups = load_confusion_groups(confusion_groups_path)

    item_index: item_matcher.ItemReferenceIndex | None = None
    item_index_error: str | None = None
    try:
        item_index = item_matcher.build_item_reference_index(
            references_root,
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )
    except Exception as exc:
        item_index_error = str(exc)
        LOGGER.warning("Item matcher unavailable; all item predictions will require review: %s", exc)

    slots = list(parse_result.get("slots", []))
    quality_query_paths = [
        _resolve_crop_path(slot.get("quality_path"), parse_dir) or Path("__missing_quality_crop__")
        for slot in slots
    ]
    number_query_paths = [
        _resolve_crop_path(slot.get("number_path"), parse_dir) or Path("__missing_number_crop__")
        for slot in slots
    ]
    quality_results = quality_matcher.batch_classify_quality_crops(quality_query_paths, quality_references_root)
    ocr_results = ocr.batch_read_amounts(number_query_paths)

    records: list[dict[str, Any]] = []
    for index, slot in enumerate(slots):
        record = process_slot(
            slot,
            parse_dir=parse_dir,
            scan_id=resolved_scan_id,
            references_root=Path(references_root),
            quality_references_root=Path(quality_references_root),
            item_index=item_index,
            item_index_error=item_index_error,
            confusion_groups=confusion_groups,
            thresholds=thresholds,
            quality_result=quality_results[index],
            ocr_result=ocr_results[index],
        )
        records.append(record)

    pending_reviews = [record for record in records if record["review_needed"]]
    trusted_results = [record for record in records if record["auto_approved"]]
    alerts = [record for record in records if record["alert_needed"]]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pending_path = output_path / "pending_reviews.json"
    trusted_path = output_path / "trusted_results.json"
    alerts_path = output_path / "alerts.json"

    _write_json(pending_path, pending_reviews)
    _write_json(trusted_path, trusted_results)
    _write_json(alerts_path, alerts)

    summary = {
        "scan_id": resolved_scan_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parse_result_path": str(parse_path),
        "slot_count": len(records),
        "pending_review_count": len(pending_reviews),
        "trusted_count": len(trusted_results),
        "alert_count": len(alerts),
        "thresholds": asdict(thresholds),
        "output_paths": {
            "pending_reviews": str(pending_path),
            "trusted_results": str(trusted_path),
            "alerts": str(alerts_path),
        },
    }
    _write_json(output_path / "pipeline_summary.json", summary)
    return summary


def process_slot(
    slot: dict[str, Any],
    *,
    parse_dir: Path,
    scan_id: str,
    references_root: Path,
    quality_references_root: Path,
    item_index: item_matcher.ItemReferenceIndex | None,
    item_index_error: str | None,
    confusion_groups: dict[str, list[str]],
    thresholds: ReviewThresholds,
    quality_result: dict[str, Any],
    ocr_result: dict[str, Any],
) -> dict[str, Any]:
    slot_index = int(slot.get("slot_index", 0))
    slot_path = _resolve_crop_path(slot.get("slot_path"), parse_dir)
    item_path = _resolve_crop_path(slot.get("item_path") or slot.get("icon_path"), parse_dir)
    quality_path = _resolve_crop_path(slot.get("quality_path"), parse_dir)
    number_path = _resolve_crop_path(slot.get("number_path"), parse_dir)

    item_result = _classify_item_safe(item_path, item_index, item_index_error)

    record = {
        "scan_id": scan_id,
        "slot_index": slot_index,
        "row": int(slot.get("row", 0) or 0),
        "col": int(slot.get("col", 0) or 0),
        "slot_path": str(slot_path) if slot_path else "",
        "item_path": str(item_path) if item_path else "",
        "quality_path": str(quality_path) if quality_path else "",
        "number_path": str(number_path) if number_path else "",
        "predicted_item": item_result.get("predicted_item", ""),
        "item_confidence": float(item_result.get("confidence", 0.0) or 0.0),
        "top_item_matches": item_result.get("top_matches", []),
        "matched_item_reference_paths": item_result.get("matched_reference_paths", []),
        "predicted_quality": quality_result.get("detected_quality", ""),
        "quality_confidence": float(quality_result.get("quality_confidence", 0.0) or 0.0),
        "top_quality_matches": quality_result.get("top_matches", []),
        "matched_quality_reference_paths": quality_result.get("matched_reference_paths", []),
        "amount": str(ocr_result.get("amount", "") or ""),
        "ocr_confidence": float(ocr_result.get("ocr_confidence", 0.0) or 0.0),
        "raw_ocr_text": str(ocr_result.get("raw_ocr_text", "") or ""),
        "recognition_errors": {
            "item": item_result.get("error"),
            "quality": quality_result.get("error"),
            "ocr": ocr_result.get("error"),
        },
        "reference_update_targets": _reference_update_targets(
            predicted_item=item_result.get("predicted_item", ""),
            predicted_quality=quality_result.get("detected_quality", ""),
            references_root=references_root,
            quality_references_root=quality_references_root,
        ),
    }

    review_needed, review_reasons, auto_approved, alert_needed = evaluate_review_policy(
        record,
        thresholds=thresholds,
        confusion_groups=confusion_groups,
    )
    record["review_needed"] = review_needed
    record["review_reason"] = review_reasons
    record["auto_approved"] = auto_approved
    record["alert_needed"] = alert_needed
    return record


def load_parse_result(path: str | Path) -> dict[str, Any]:
    parse_path = Path(path)
    if not parse_path.exists():
        raise FileNotFoundError(f"parse_result.json not found: {parse_path}")
    with parse_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Parse result must be a JSON object: {parse_path}")
    if "slots" not in data or not isinstance(data["slots"], list):
        raise ValueError(f"Parse result does not contain a slots list: {parse_path}")
    return data


def load_confusion_groups(path: str | Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}

    confusion_path = Path(path)
    if not confusion_path.exists():
        LOGGER.info("No confusion_groups.json found at %s; confusion policy disabled.", confusion_path)
        return {}

    try:
        data = json.loads(confusion_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("Could not parse confusion groups %s: %s", confusion_path, exc)
        return {}

    groups: dict[str, list[str]] = {}
    for group_name, item_names in data.items():
        if not isinstance(item_names, list):
            LOGGER.warning("Ignoring confusion group %s because it is not a list.", group_name)
            continue
        groups[str(group_name)] = [str(item_name) for item_name in item_names]
    return groups


def evaluate_review_policy(
    record: dict[str, Any],
    *,
    thresholds: ReviewThresholds,
    confusion_groups: dict[str, list[str]],
) -> tuple[bool, list[str], bool, bool]:
    reasons: list[str] = []

    item_confidence = float(record.get("item_confidence", 0.0) or 0.0)
    quality_confidence = float(record.get("quality_confidence", 0.0) or 0.0)
    ocr_confidence = float(record.get("ocr_confidence", 0.0) or 0.0)
    amount = str(record.get("amount", "") or "")

    if item_confidence < thresholds.item_auto_approve_threshold:
        reasons.append("item_confidence_below_threshold")
    if quality_confidence < thresholds.quality_auto_approve_threshold:
        reasons.append("quality_confidence_below_threshold")
    if ocr_confidence < thresholds.ocr_auto_approve_threshold:
        reasons.append("ocr_confidence_below_threshold")
    if _amount_is_suspicious(amount):
        reasons.append("amount_empty_or_suspicious")

    top_item_matches = record.get("top_item_matches", [])
    top_gap = _top_match_gap(top_item_matches)
    if top_gap is not None and top_gap < thresholds.top_gap_minimum:
        reasons.append("top_item_matches_too_close")

    confusion_group_name = _confusion_group_for_item(record.get("predicted_item", ""), confusion_groups)
    if confusion_group_name:
        reasons.append(f"confusion_prone_item:{confusion_group_name}")

    close_group = _close_confusion_group_match(top_item_matches, confusion_groups, thresholds.top_gap_minimum)
    if close_group:
        reasons.append(f"confusion_group_top_matches_close:{close_group}")

    recognition_errors = record.get("recognition_errors", {})
    for stage_name, error in recognition_errors.items():
        if error:
            reasons.append(f"{stage_name}_recognition_error")

    review_needed = bool(reasons)
    auto_approved = not review_needed
    alert_needed = (
        review_needed
        or item_confidence < thresholds.item_auto_approve_threshold
        or quality_confidence < thresholds.quality_auto_approve_threshold
        or ocr_confidence < thresholds.ocr_auto_approve_threshold
        or _amount_is_suspicious(amount)
    )
    return review_needed, sorted(set(reasons)), auto_approved, alert_needed


def _classify_item_safe(
    image_path: Path | None,
    item_index: item_matcher.ItemReferenceIndex | None,
    item_index_error: str | None,
) -> dict[str, Any]:
    if image_path is None:
        return {
            "image_path": "",
            "predicted_item": "",
            "confidence": 0.0,
            "top_gap": None,
            "top_matches": [],
            "matched_reference_paths": [],
            "error": "Missing item crop path",
        }
    if item_index is None:
        return {
            "image_path": str(image_path),
            "predicted_item": "",
            "confidence": 0.0,
            "top_gap": None,
            "top_matches": [],
            "matched_reference_paths": [],
            "error": item_index_error or "Item matcher unavailable",
        }
    return item_matcher.classify_item_crop(image_path, item_index)


def _resolve_crop_path(value: Any, parse_dir: Path) -> Path | None:
    if not value:
        return None

    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.exists():
        return path

    candidate = parse_dir / path
    if candidate.exists():
        return candidate

    return path


def _default_scan_id(parse_result: dict[str, Any], parse_path: Path) -> str:
    output_dir = parse_result.get("output_dir")
    if output_dir:
        return Path(str(output_dir)).name
    return parse_path.parent.name or parse_path.stem


def _top_match_gap(top_matches: list[dict[str, Any]]) -> float | None:
    if len(top_matches) < 2:
        return None
    first = float(top_matches[0].get("confidence", 0.0) or 0.0)
    second = float(top_matches[1].get("confidence", 0.0) or 0.0)
    return first - second


def _confusion_group_for_item(item_name: Any, confusion_groups: dict[str, list[str]]) -> str | None:
    item = str(item_name or "")
    if not item:
        return None
    for group_name, item_names in confusion_groups.items():
        if item in item_names:
            return group_name
    return None


def _close_confusion_group_match(
    top_matches: list[dict[str, Any]],
    confusion_groups: dict[str, list[str]],
    minimum_gap: float,
) -> str | None:
    if len(top_matches) < 2:
        return None

    for group_name, item_names in confusion_groups.items():
        group_matches = [
            match
            for match in top_matches
            if str(match.get("item", "")) in item_names
        ]
        if len(group_matches) < 2:
            continue

        first = float(group_matches[0].get("confidence", 0.0) or 0.0)
        second = float(group_matches[1].get("confidence", 0.0) or 0.0)
        if first - second < minimum_gap:
            return group_name

    return None


def _amount_is_suspicious(amount: str) -> bool:
    if not amount:
        return True
    if not amount.isdigit():
        return True
    value = int(amount)
    if value <= 0:
        return True
    return len(amount) > 6


def _reference_update_targets(
    *,
    predicted_item: str,
    predicted_quality: str,
    references_root: Path,
    quality_references_root: Path,
) -> dict[str, Any]:
    item_targets: dict[str, str] = {}
    if predicted_item:
        item_root = references_root / predicted_item
        item_targets = {
            "predicted_item_approved_dir": str(item_root / "approved"),
            "predicted_item_rejected_dir": str(item_root / "rejected"),
        }

    quality_targets: dict[str, str] = {}
    if predicted_quality:
        quality_root = quality_references_root / predicted_quality
        quality_targets = {
            "predicted_quality_approved_dir": str(quality_root / "approved"),
            "predicted_quality_rejected_dir": str(quality_root / "rejected"),
        }

    return {
        "item": item_targets,
        "quality": quality_targets,
        "note": (
            "Review app should copy confirmed good crops to the corrected label's approved/ folder, "
            "and unusable crops to rejected/. This pipeline never writes those folders automatically."
        ),
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Anvil stockpile recognition on a parser parse_result.json.")
    parser.add_argument("parse_result", type=Path, help="Path to parser output parse_result.json.")
    parser.add_argument(
        "--references-root",
        type=Path,
        default=item_matcher.DEFAULT_REFERENCES_ROOT,
        help="Item references root. Default: references",
    )
    parser.add_argument(
        "--quality-references-root",
        type=Path,
        default=quality_matcher.DEFAULT_QUALITY_REFERENCES_ROOT,
        help="Quality references root. Default: quality_references",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for pending_reviews.json, trusted_results.json, and alerts.json.",
    )
    parser.add_argument(
        "--confusion-groups",
        type=Path,
        default=DEFAULT_CONFUSION_GROUPS_PATH,
        help="Optional confusion_groups.json path.",
    )
    parser.add_argument("--scan-id", help="Stable scan id. Default: parser output folder name.")
    parser.add_argument("--model", default=item_matcher.MODEL_NAME, help=f"OpenCLIP model. Default: {item_matcher.MODEL_NAME}")
    parser.add_argument("--pretrained", default=item_matcher.PRETRAINED, help=f"OpenCLIP weights. Default: {item_matcher.PRETRAINED}")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Torch device. Default: auto")
    parser.add_argument("--item-threshold", type=float, default=ITEM_AUTO_APPROVE_THRESHOLD)
    parser.add_argument("--quality-threshold", type=float, default=QUALITY_AUTO_APPROVE_THRESHOLD)
    parser.add_argument("--ocr-threshold", type=float, default=OCR_AUTO_APPROVE_THRESHOLD)
    parser.add_argument("--top-gap-minimum", type=float, default=TOP_GAP_MINIMUM)
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

    thresholds = ReviewThresholds(
        item_auto_approve_threshold=args.item_threshold,
        quality_auto_approve_threshold=args.quality_threshold,
        ocr_auto_approve_threshold=args.ocr_threshold,
        top_gap_minimum=args.top_gap_minimum,
    )
    summary = run_pipeline(
        args.parse_result,
        references_root=args.references_root,
        quality_references_root=args.quality_references_root,
        output_dir=args.output_dir,
        confusion_groups_path=args.confusion_groups,
        scan_id=args.scan_id,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        thresholds=thresholds,
    )

    print("Anvil recognition pipeline summary")
    print(f"Scan ID: {summary['scan_id']}")
    print(f"Slots: {summary['slot_count']}")
    print(f"Pending reviews: {summary['pending_review_count']}")
    print(f"Trusted results: {summary['trusted_count']}")
    print(f"Alerts: {summary['alert_count']}")
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()
