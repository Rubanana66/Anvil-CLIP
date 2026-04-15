from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st


LOGGER = logging.getLogger("anvil_review_app")

DEFAULT_REFERENCES_ROOT = Path("references")
DEFAULT_QUALITY_REFERENCES_ROOT = Path("quality_references")
DEFAULT_INCOMING_ROOT = Path("data") / "incoming_results"
DEFAULT_PENDING_REVIEWS_PATH = DEFAULT_INCOMING_ROOT / "pending_reviews.json"
DEFAULT_TRUSTED_RESULTS_PATH = DEFAULT_INCOMING_ROOT / "trusted_results.json"
DEFAULT_ALERTS_PATH = DEFAULT_INCOMING_ROOT / "alerts.json"
DEFAULT_LOGS_ROOT = Path("logs")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
REFERENCE_SUBDIRS = ("wiki", "approved")


st.set_page_config(page_title="Reference Review Tool", layout="wide")


def load_pending_reviews(path: str | Path) -> list[dict[str, Any]]:
    return _load_json_list(path)


def load_trusted_results(path: str | Path) -> list[dict[str, Any]]:
    return _load_json_list(path)


def load_alerts(path: str | Path) -> list[dict[str, Any]]:
    return _load_json_list(path)


def load_item_dirs(references_root: str | Path) -> list[str]:
    root = Path(references_root)
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )


def load_quality_dirs(quality_references_root: str | Path) -> list[str]:
    root = Path(quality_references_root)
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )


def list_reference_images(item_dir: str | Path) -> list[Path]:
    root = Path(item_dir)
    return _list_usable_reference_images(root)


def list_quality_reference_images(quality_dir: str | Path) -> list[Path]:
    root = Path(quality_dir)
    return _list_usable_reference_images(root)


def make_unique_filename(prefix: str, scan_id: str, slot_index: int) -> str:
    safe_prefix = _safe_name(prefix) or "crop"
    safe_scan = _safe_name(scan_id) or "scan"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{safe_prefix}_{safe_scan}_slot{slot_index:03d}_{timestamp}.png"


def approve_review_item(
    review_item: dict[str, Any],
    references_root: str | Path,
    quality_references_root: str | Path,
    logs_root: str | Path,
) -> dict[str, Any]:
    scan_id = str(review_item.get("scan_id", "scan"))
    slot_index = int(review_item.get("slot_index", 0))
    predicted_item = str(review_item.get("predicted_item", "") or "").strip()
    predicted_quality = str(review_item.get("predicted_quality", "") or "").strip()

    result = _base_action_result("approve_all", review_item)
    if predicted_item:
        item_dir = _ensure_label_dirs(Path(references_root), predicted_item, include_meta=True)
        _copy_crop_into_result(
            result,
            source=review_item.get("item_path"),
            destination_dir=item_dir / "approved",
            prefix=f"item_{predicted_item}",
            scan_id=scan_id,
            slot_index=slot_index,
        )
    else:
        result["errors"].append("Cannot approve item crop because predicted_item is empty.")

    if predicted_quality:
        quality_dir = _ensure_label_dirs(Path(quality_references_root), predicted_quality, include_meta=False)
        _copy_crop_into_result(
            result,
            source=review_item.get("quality_path"),
            destination_dir=quality_dir / "approved",
            prefix=f"quality_{predicted_quality}",
            scan_id=scan_id,
            slot_index=slot_index,
        )
    else:
        result["errors"].append("Cannot approve quality crop because predicted_quality is empty.")

    result["status"] = "ok" if result["copied_files"] and not result["errors"] else "partial"
    append_jsonl(Path(logs_root) / "actions.jsonl", result)
    return result


def correct_review_item(
    review_item: dict[str, Any],
    selected_item: str,
    selected_quality: str,
    references_root: str | Path,
    quality_references_root: str | Path,
    logs_root: str | Path,
    selected_amount: str | None = None,
) -> dict[str, Any]:
    selected_item = selected_item.strip()
    selected_quality = selected_quality.strip()
    scan_id = str(review_item.get("scan_id", "scan"))
    slot_index = int(review_item.get("slot_index", 0))

    result = _base_action_result("correct_and_save", review_item)
    result["corrected_item"] = selected_item
    result["corrected_quality"] = selected_quality
    result["corrected_amount"] = selected_amount if selected_amount is not None else review_item.get("amount", "")

    if selected_item:
        item_dir = _ensure_label_dirs(Path(references_root), selected_item, include_meta=True)
        _copy_crop_into_result(
            result,
            source=review_item.get("item_path"),
            destination_dir=item_dir / "approved",
            prefix=f"item_{selected_item}",
            scan_id=scan_id,
            slot_index=slot_index,
        )
    else:
        result["errors"].append("Cannot save corrected item crop because selected_item is empty.")

    if selected_quality:
        quality_dir = _ensure_label_dirs(Path(quality_references_root), selected_quality, include_meta=False)
        _copy_crop_into_result(
            result,
            source=review_item.get("quality_path"),
            destination_dir=quality_dir / "approved",
            prefix=f"quality_{selected_quality}",
            scan_id=scan_id,
            slot_index=slot_index,
        )
    else:
        result["errors"].append("Cannot save corrected quality crop because selected_quality is empty.")

    result["status"] = "ok" if result["copied_files"] and not result["errors"] else "partial"
    corrections_row = {
        **result,
        "previous_item": review_item.get("predicted_item", ""),
        "previous_quality": review_item.get("predicted_quality", ""),
    }
    append_jsonl(Path(logs_root) / "corrections.jsonl", corrections_row)
    append_jsonl(Path(logs_root) / "actions.jsonl", result)
    return result


def reject_review_item(
    review_item: dict[str, Any],
    references_root: str | Path,
    quality_references_root: str | Path,
    logs_root: str | Path,
) -> dict[str, Any]:
    scan_id = str(review_item.get("scan_id", "scan"))
    slot_index = int(review_item.get("slot_index", 0))
    predicted_item = str(review_item.get("predicted_item", "") or "").strip()
    predicted_quality = str(review_item.get("predicted_quality", "") or "").strip()

    result = _base_action_result("reject_crop", review_item)
    if predicted_item:
        item_rejected_dir = _ensure_label_dirs(Path(references_root), predicted_item, include_meta=True) / "rejected"
    else:
        item_rejected_dir = Path(references_root) / "_global_rejected" / "items"

    if predicted_quality:
        quality_rejected_dir = _ensure_label_dirs(Path(quality_references_root), predicted_quality, include_meta=False) / "rejected"
    else:
        quality_rejected_dir = Path(quality_references_root) / "_global_rejected" / "quality"

    _copy_crop_into_result(
        result,
        source=review_item.get("item_path"),
        destination_dir=item_rejected_dir,
        prefix=f"rejected_item_{predicted_item or 'unknown'}",
        scan_id=scan_id,
        slot_index=slot_index,
    )
    _copy_crop_into_result(
        result,
        source=review_item.get("quality_path"),
        destination_dir=quality_rejected_dir,
        prefix=f"rejected_quality_{predicted_quality or 'unknown'}",
        scan_id=scan_id,
        slot_index=slot_index,
    )

    result["status"] = "ok" if result["copied_files"] and not result["errors"] else "partial"
    append_jsonl(Path(logs_root) / "actions.jsonl", result)
    return result


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _load_json_list(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        st.sidebar.error(f"Could not parse JSON: {input_path}")
        LOGGER.warning("Could not parse JSON %s: %s", input_path, exc)
        return []
    if not isinstance(data, list):
        LOGGER.warning("Expected a JSON list in %s", input_path)
        return []
    return [item for item in data if isinstance(item, dict)]


def _list_usable_reference_images(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []

    images: list[Path] = []
    for subdir in REFERENCE_SUBDIRS:
        source_dir = root / subdir
        if source_dir.is_dir():
            images.extend(
                path
                for path in source_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and "rejected" not in {part.casefold() for part in path.parts}
            )
    return sorted(images, key=lambda path: str(path).casefold())


def _ensure_label_dirs(root: Path, label: str, *, include_meta: bool) -> Path:
    safe_label = _safe_label(label)
    label_dir = root / safe_label
    for subdir in ("wiki", "approved", "rejected"):
        (label_dir / subdir).mkdir(parents=True, exist_ok=True)

    if include_meta:
        meta_path = label_dir / "meta.json"
        if not meta_path.exists():
            meta_path.write_text(
                json.dumps(
                    {
                        "label": safe_label,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "created_by": "Reference Review Tool",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    return label_dir


def _safe_label(label: str) -> str:
    cleaned = _safe_name(label)
    return cleaned or "Unknown"


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned


def _copy_crop_into_result(
    result: dict[str, Any],
    *,
    source: Any,
    destination_dir: Path,
    prefix: str,
    scan_id: str,
    slot_index: int,
) -> None:
    source_path = Path(str(source or ""))
    if not source or not source_path.exists() or not source_path.is_file():
        result["errors"].append(f"Missing source crop: {source}")
        return

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / make_unique_filename(prefix, scan_id, slot_index)
    try:
        shutil.copy2(source_path, destination_path)
    except OSError as exc:
        result["errors"].append(f"Could not copy {source_path} to {destination_path}: {exc}")
        return

    result["copied_files"].append(
        {
            "source": str(source_path),
            "destination": str(destination_path),
        }
    )


def _base_action_result(action: str, review_item: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "status": "started",
        "scan_id": review_item.get("scan_id", ""),
        "slot_index": review_item.get("slot_index", ""),
        "predicted_item": review_item.get("predicted_item", ""),
        "predicted_quality": review_item.get("predicted_quality", ""),
        "amount": review_item.get("amount", ""),
        "item_confidence": review_item.get("item_confidence", 0.0),
        "quality_confidence": review_item.get("quality_confidence", 0.0),
        "ocr_confidence": review_item.get("ocr_confidence", 0.0),
        "review_reason": review_item.get("review_reason", []),
        "alert_needed": review_item.get("alert_needed", False),
        "slot_path": review_item.get("slot_path", ""),
        "item_path": review_item.get("item_path", ""),
        "quality_path": review_item.get("quality_path", ""),
        "number_path": review_item.get("number_path", ""),
        "copied_files": [],
        "errors": [],
    }


def item_key(review_item: dict[str, Any]) -> str:
    return f"{review_item.get('scan_id', '')}::{review_item.get('slot_index', '')}"


def apply_filters(
    reviews: list[dict[str, Any]],
    *,
    resolved: dict[str, str],
    only_unresolved: bool,
    only_review_needed: bool,
    only_alerts: bool,
    item_search: str,
    sort_low_confidence_first: bool,
) -> list[dict[str, Any]]:
    search = item_search.strip().casefold()
    filtered: list[dict[str, Any]] = []

    for review in reviews:
        key = item_key(review)
        if only_unresolved and key in resolved:
            continue
        if only_review_needed and not review.get("review_needed", False):
            continue
        if only_alerts and not review.get("alert_needed", False):
            continue
        if search and search not in _search_blob(review):
            continue
        filtered.append(review)

    if sort_low_confidence_first:
        filtered.sort(
            key=lambda item: (
                not bool(item.get("alert_needed", False)),
                _safe_float(item.get("item_confidence", 0.0)),
                _safe_float(item.get("quality_confidence", 0.0)),
                _safe_float(item.get("ocr_confidence", 0.0)),
            )
        )

    return filtered


def mark_resolved(review_item: dict[str, Any], action: str) -> None:
    st.session_state.resolved[item_key(review_item)] = action


def increment_counter(name: str) -> None:
    st.session_state[name] = int(st.session_state.get(name, 0)) + 1


def _search_blob(review: dict[str, Any]) -> str:
    pieces = [
        review.get("predicted_item", ""),
        review.get("predicted_quality", ""),
        review.get("amount", ""),
        " ".join(str(reason) for reason in review.get("review_reason", [])),
    ]
    for match in review.get("top_item_matches", []):
        pieces.append(str(match.get("item", "")))
    for match in review.get("top_quality_matches", []):
        pieces.append(str(match.get("quality", "")))
    return " ".join(pieces).casefold()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_percent(value: Any) -> str:
    return f"{_safe_float(value):.4f}"


def _options_with_value(options: list[str], value: str) -> list[str]:
    clean = [option for option in options if option]
    if value and value not in clean:
        clean.insert(0, value)
    return clean


def _option_index(options: list[str], value: str) -> int | None:
    if not options:
        return None
    if value in options:
        return options.index(value)
    return 0


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def show_crop(label: str, path_value: Any) -> None:
    st.markdown(f"**{label}**")
    path = Path(str(path_value or ""))
    if not path.exists() or not path.is_file():
        st.warning("Missing crop")
        st.caption(str(path_value or ""))
        return
    st.image(str(path), use_container_width=True)
    st.caption(path.name)


def show_top_matches(title: str, matches: list[dict[str, Any]], label_key: str) -> None:
    st.markdown(f"**{title}**")
    if not matches:
        st.caption("No matches available")
        return

    rows = []
    for match in matches[:8]:
        rows.append(
            {
                "label": match.get(label_key, match.get("item", match.get("quality", ""))),
                "confidence": match.get("confidence", ""),
                "refs": match.get("reference_count", ""),
            }
        )
    st.table(rows)

    with st.expander(f"{title} reference paths"):
        for match in matches[:8]:
            label = match.get(label_key, match.get("item", match.get("quality", "")))
            st.markdown(f"**{label}**")
            for reference_path in match.get("matched_reference_paths", []):
                st.caption(reference_path)


def show_reference_previews(title: str, image_paths: list[Path], max_images: int = 8) -> None:
    st.markdown(f"**{title}**")
    if not image_paths:
        st.caption("No wiki/approved reference images yet")
        return

    columns = st.columns(min(4, max(1, min(len(image_paths), max_images))))
    for index, image_path in enumerate(image_paths[:max_images]):
        with columns[index % len(columns)]:
            st.image(str(image_path), use_container_width=True)
            st.caption(f"{image_path.parent.name}/{image_path.name}")


def show_reason_badges(review_item: dict[str, Any]) -> None:
    reasons = [str(reason) for reason in review_item.get("review_reason", [])]
    if review_item.get("alert_needed", False):
        st.error("Alert needed")
    if review_item.get("review_needed", False):
        st.warning("Needs manual review")
    if not reasons:
        st.success("No review reasons")
        return

    friendly_messages = {
        "item_confidence_below_threshold": "Low item confidence",
        "quality_confidence_below_threshold": "Low quality confidence",
        "ocr_confidence_below_threshold": "OCR uncertain",
        "top_item_matches_too_close": "Top matches too close",
        "amount_empty_or_suspicious": "Amount empty or suspicious",
    }
    for reason in reasons:
        text = friendly_messages.get(reason, reason)
        if "confusion" in reason:
            text = f"Confusion-prone match: {reason}"
        st.caption(f"- {text}")


def sidebar_inputs() -> dict[str, Any]:
    st.sidebar.header("Paths")
    references_root = Path(st.sidebar.text_input("References path", str(DEFAULT_REFERENCES_ROOT)))
    quality_references_root = Path(st.sidebar.text_input("Quality references path", str(DEFAULT_QUALITY_REFERENCES_ROOT)))
    pending_path = Path(st.sidebar.text_input("Pending reviews path", str(DEFAULT_PENDING_REVIEWS_PATH)))
    trusted_path = Path(st.sidebar.text_input("Trusted results path", str(DEFAULT_TRUSTED_RESULTS_PATH)))
    alerts_path = Path(st.sidebar.text_input("Alerts path", str(DEFAULT_ALERTS_PATH)))
    logs_root = Path(st.sidebar.text_input("Logs path", str(DEFAULT_LOGS_ROOT)))

    st.sidebar.header("Filters")
    only_unresolved = st.sidebar.checkbox("Only unresolved", value=True)
    only_review_needed = st.sidebar.checkbox("Only review_needed", value=False)
    only_alerts = st.sidebar.checkbox("Only alerts", value=False)
    sort_low_confidence_first = st.sidebar.checkbox("Low confidence first", value=True)
    item_search = st.sidebar.text_input("Item search", "")

    return {
        "references_root": references_root,
        "quality_references_root": quality_references_root,
        "pending_path": pending_path,
        "trusted_path": trusted_path,
        "alerts_path": alerts_path,
        "logs_root": logs_root,
        "only_unresolved": only_unresolved,
        "only_review_needed": only_review_needed,
        "only_alerts": only_alerts,
        "sort_low_confidence_first": sort_low_confidence_first,
        "item_search": item_search,
    }


def initialize_session_state() -> None:
    defaults = {
        "current_index": 0,
        "resolved": {},
        "approved_count": 0,
        "corrected_count": 0,
        "rejected_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_summary(
    *,
    pending_reviews: list[dict[str, Any]],
    trusted_results: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
    filtered_reviews: list[dict[str, Any]],
) -> None:
    unresolved_count = sum(1 for item in pending_reviews if item_key(item) not in st.session_state.resolved)
    alert_count = len(alerts) if alerts else sum(1 for item in pending_reviews if item.get("alert_needed", False))
    auto_approved_count = len(trusted_results)

    st.sidebar.header("Progress")
    if filtered_reviews:
        st.sidebar.write(f"{st.session_state.current_index + 1} / {len(filtered_reviews)}")
        st.sidebar.progress((st.session_state.current_index + 1) / len(filtered_reviews))
    else:
        st.sidebar.write("0 / 0")
        st.sidebar.progress(0)

    st.sidebar.header("Summary")
    st.sidebar.metric("Unresolved", unresolved_count)
    st.sidebar.metric("Alerts", alert_count)
    st.sidebar.metric("Auto-approved", auto_approved_count)
    st.sidebar.metric("Approved this session", int(st.session_state.get("approved_count", 0)))
    st.sidebar.metric("Corrected this session", int(st.session_state.get("corrected_count", 0)))
    st.sidebar.metric("Rejected this session", int(st.session_state.get("rejected_count", 0)))


def show_item_details(review_item: dict[str, Any]) -> None:
    st.subheader("Details")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Predicted item", str(review_item.get("predicted_item", "") or ""))
    metric_cols[1].metric("Item confidence", _format_percent(review_item.get("item_confidence", 0.0)))
    metric_cols[2].metric("Predicted quality", str(review_item.get("predicted_quality", "") or ""))
    metric_cols[3].metric("Quality confidence", _format_percent(review_item.get("quality_confidence", 0.0)))

    metric_cols = st.columns(3)
    metric_cols[0].metric("Amount", str(review_item.get("amount", "") or ""))
    metric_cols[1].metric("OCR confidence", _format_percent(review_item.get("ocr_confidence", 0.0)))
    metric_cols[2].metric("Alert flag", "yes" if review_item.get("alert_needed", False) else "no")

    show_reason_badges(review_item)

    match_cols = st.columns(2)
    with match_cols[0]:
        show_top_matches("Top item matches", list(review_item.get("top_item_matches", [])), "item")
    with match_cols[1]:
        show_top_matches("Top quality matches", list(review_item.get("top_quality_matches", [])), "quality")


def show_navigation(filtered_reviews: list[dict[str, Any]]) -> None:
    previous_col, next_col = st.columns(2)
    with previous_col:
        if st.button("Previous", use_container_width=True, disabled=st.session_state.current_index <= 0):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
            _rerun()
    with next_col:
        if st.button(
            "Next",
            use_container_width=True,
            disabled=st.session_state.current_index >= len(filtered_reviews) - 1,
        ):
            st.session_state.current_index = min(len(filtered_reviews) - 1, st.session_state.current_index + 1)
            _rerun()


def inject_table_css() -> None:
    st.markdown(
        """
        <style>
        .review-table-header {
            background: #1f2937;
            color: white;
            font-weight: 700;
            border: 1px solid #cbd5e1;
            padding: 0.25rem 0.35rem;
            min-height: 2rem;
            font-size: 0.84rem;
        }
        .review-table-cell {
            border-left: 1px solid #d8dee9;
            border-bottom: 1px solid #d8dee9;
            padding: 0.25rem 0.35rem;
            min-height: 5.5rem;
            font-size: 0.84rem;
        }
        .review-row-alert {
            border-left: 4px solid #ef4444;
            padding-left: 0.35rem;
        }
        .review-row-ok {
            border-left: 4px solid #22c55e;
            padding-left: 0.35rem;
        }
        div[data-testid="stImage"] img {
            max-height: 72px;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    headers = [
        "Screenshot",
        "Slot",
        "Row",
        "Col",
        "Slot Crop",
        "Item Crop",
        "Detected Amount",
        "Detected Item",
        "Item Confidence",
        "Review Needed",
        "Top Candidates",
        "Correct Item",
        "Correct Amount",
        "Actions",
    ]
    widths = [2.2, 0.6, 0.6, 0.6, 1.1, 1.1, 1.1, 1.8, 1.1, 1.2, 4.0, 2.0, 1.2, 2.6]
    cols = st.columns(widths)
    for col, header in zip(cols, headers):
        col.markdown(f'<div class="review-table-header">{header}</div>', unsafe_allow_html=True)


def render_review_table(
    reviews: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
    item_options: list[str],
    quality_options: list[str],
) -> None:
    inject_table_css()
    st.caption(
        "Accept saves the detected item crop to approved/. Reject prediction means the crop is good but the label is wrong: "
        "choose the correct item/quality and it will save to that approved folder. Use Bad crop only for unusable crops."
    )
    render_header()
    for visible_index, review_item in enumerate(reviews):
        render_review_row(
            review_item,
            visible_index=visible_index,
            settings=settings,
            item_options=item_options,
            quality_options=quality_options,
        )


def render_review_row(
    review_item: dict[str, Any],
    *,
    visible_index: int,
    settings: dict[str, Any],
    item_options: list[str],
    quality_options: list[str],
) -> None:
    row_key = item_key(review_item)
    predicted_item = str(review_item.get("predicted_item", "") or "")
    predicted_quality = str(review_item.get("predicted_quality", "") or "")
    amount = str(review_item.get("amount", "") or "")
    row_class = "review-row-alert" if review_item.get("alert_needed", False) else "review-row-ok"
    widths = [2.2, 0.6, 0.6, 0.6, 1.1, 1.1, 1.1, 1.8, 1.1, 1.2, 4.0, 2.0, 1.2, 2.6]

    cols = st.columns(widths)
    cols[0].markdown(f'<div class="review-table-cell {row_class}">{_screenshot_label(review_item)}</div>', unsafe_allow_html=True)
    cols[1].markdown(f'<div class="review-table-cell">{review_item.get("slot_index", "")}</div>', unsafe_allow_html=True)
    cols[2].markdown(f'<div class="review-table-cell">{review_item.get("row", "")}</div>', unsafe_allow_html=True)
    cols[3].markdown(f'<div class="review-table-cell">{review_item.get("col", "")}</div>', unsafe_allow_html=True)

    with cols[4]:
        _show_thumb(review_item.get("slot_path"))
    with cols[5]:
        _show_thumb(review_item.get("item_path"))

    cols[6].markdown(f'<div class="review-table-cell">{amount or "?"}</div>', unsafe_allow_html=True)
    cols[7].markdown(f'<div class="review-table-cell">{predicted_item or "?"}</div>', unsafe_allow_html=True)
    cols[8].markdown(f'<div class="review-table-cell">{_format_percent(review_item.get("item_confidence", 0.0))}</div>', unsafe_allow_html=True)
    cols[9].markdown(
        f'<div class="review-table-cell">{"TRUE" if review_item.get("review_needed", False) else "FALSE"}'
        f'<br>{_reason_summary(review_item)}</div>',
        unsafe_allow_html=True,
    )
    cols[10].markdown(f'<div class="review-table-cell">{format_top_candidates(review_item)}</div>', unsafe_allow_html=True)

    item_choices = _options_with_value(item_options, predicted_item)
    quality_choices = _options_with_value(quality_options, predicted_quality)
    with cols[11]:
        selected_item = st.selectbox(
            "Correct item",
            item_choices,
            index=_option_index(item_choices, predicted_item),
            key=f"item_{row_key}",
            label_visibility="collapsed",
            placeholder="Select item",
        )
        item_override = st.text_input(
            "New item",
            value="",
            key=f"item_override_{row_key}",
            label_visibility="collapsed",
            placeholder="or type new item",
        )
        selected_quality = st.selectbox(
            "Quality",
            quality_choices,
            index=_option_index(quality_choices, predicted_quality) if predicted_quality else None,
            key=f"quality_{row_key}",
            label_visibility="collapsed",
            placeholder="Quality",
        )
    with cols[12]:
        selected_amount = st.text_input(
            "Correct amount",
            value=amount,
            key=f"amount_{row_key}",
            label_visibility="collapsed",
        )

    final_item = item_override.strip() or str(selected_item or "")
    final_quality = str(selected_quality or predicted_quality or "")
    with cols[13]:
        approve_disabled = not predicted_item
        if st.button("Accept", key=f"accept_{row_key}", disabled=approve_disabled, use_container_width=True):
            result = approve_review_item(
                review_item,
                settings["references_root"],
                settings["quality_references_root"],
                settings["logs_root"],
            )
            mark_resolved(review_item, "approved")
            increment_counter("approved_count")
            _flash_action_result("Accepted", result)
            _rerun()

        reject_disabled = not final_item or not final_quality
        if st.button("Reject prediction", key=f"reject_prediction_{row_key}", disabled=reject_disabled, use_container_width=True):
            result = correct_review_item(
                review_item,
                final_item,
                final_quality,
                settings["references_root"],
                settings["quality_references_root"],
                settings["logs_root"],
                selected_amount=selected_amount,
            )
            mark_resolved(review_item, "corrected")
            increment_counter("corrected_count")
            _flash_action_result("Saved correction", result)
            _rerun()

        bad_crop = st.checkbox("Bad crop", key=f"bad_crop_{row_key}")
        if st.button("Reject crop", key=f"reject_crop_{row_key}", disabled=not bad_crop, use_container_width=True):
            result = reject_review_item(
                review_item,
                settings["references_root"],
                settings["quality_references_root"],
                settings["logs_root"],
            )
            mark_resolved(review_item, "rejected")
            increment_counter("rejected_count")
            _flash_action_result("Rejected crop", result)
            _rerun()

    with st.expander(f"Details for slot {review_item.get('slot_index', '')}: reference previews / raw JSON"):
        detail_cols = st.columns([1, 1, 1])
        with detail_cols[0]:
            show_crop("Quality crop", review_item.get("quality_path"))
            show_crop("Number crop", review_item.get("number_path"))
        with detail_cols[1]:
            show_reference_previews(
                f"Item refs: {final_item or predicted_item or 'none'}",
                list_reference_images(settings["references_root"] / _safe_label(final_item or predicted_item)),
                max_images=6,
            )
        with detail_cols[2]:
            show_reference_previews(
                f"Quality refs: {final_quality or predicted_quality or 'none'}",
                list_quality_reference_images(settings["quality_references_root"] / _safe_label(final_quality or predicted_quality)),
                max_images=6,
            )
        st.json(review_item)


def _show_thumb(path_value: Any) -> None:
    path = Path(str(path_value or ""))
    if path.exists() and path.is_file():
        st.image(str(path), width=72)
    else:
        st.caption("missing")


def _screenshot_label(review_item: dict[str, Any]) -> str:
    screenshot = review_item.get("screenshot_name") or review_item.get("scan_id") or ""
    return str(screenshot)


def _reason_summary(review_item: dict[str, Any]) -> str:
    reasons = [str(reason) for reason in review_item.get("review_reason", [])]
    if review_item.get("alert_needed", False) and "alert" not in reasons:
        reasons.insert(0, "alert")
    return "<br>".join(reasons[:3])


def format_top_candidates(review_item: dict[str, Any]) -> str:
    matches = review_item.get("top_item_matches", [])
    if not matches:
        return ""
    parts = []
    for match in matches[:5]:
        label = str(match.get("item", ""))
        confidence = _format_percent(match.get("confidence", 0.0))
        parts.append(f"{label}:{confidence}")
    return " | ".join(parts)


def _flash_action_result(message: str, result: dict[str, Any]) -> None:
    copied = len(result.get("copied_files", []))
    errors = result.get("errors", [])
    if errors:
        st.warning(f"{message}: copied {copied} file(s), but: {'; '.join(errors)}")
    else:
        st.success(f"{message}: copied {copied} file(s).")


def main() -> None:
    initialize_session_state()
    st.title("Reference Review Tool")

    settings = sidebar_inputs()
    pending_reviews = load_pending_reviews(settings["pending_path"])
    trusted_results = load_trusted_results(settings["trusted_path"])
    alerts = load_alerts(settings["alerts_path"])

    filtered_reviews = apply_filters(
        pending_reviews,
        resolved=st.session_state.resolved,
        only_unresolved=settings["only_unresolved"],
        only_review_needed=settings["only_review_needed"],
        only_alerts=settings["only_alerts"],
        item_search=settings["item_search"],
        sort_low_confidence_first=settings["sort_low_confidence_first"],
    )

    if st.session_state.current_index >= len(filtered_reviews):
        st.session_state.current_index = max(0, len(filtered_reviews) - 1)

    show_summary(
        pending_reviews=pending_reviews,
        trusted_results=trusted_results,
        alerts=alerts,
        filtered_reviews=filtered_reviews,
    )

    if not pending_reviews:
        st.info("No pending review items were loaded. Check the pending reviews path in the sidebar.")
        return
    if not filtered_reviews:
        st.success("No items match the current filters.")
        return

    item_dirs = load_item_dirs(settings["references_root"])
    quality_dirs = load_quality_dirs(settings["quality_references_root"])
    render_review_table(
        filtered_reviews,
        settings=settings,
        item_options=item_dirs,
        quality_options=quality_dirs,
    )


if __name__ == "__main__":
    main()
