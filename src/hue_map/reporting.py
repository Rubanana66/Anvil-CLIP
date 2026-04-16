"""Self-contained HTML report generation.

We base64-embed every image so the report is a single file that can be
opened directly from disk, e-mailed, or committed to source control without
needing the accompanying slot crops or references on hand.

Colour coding
-------------
Each prediction row is annotated green / orange / red based on confidence
and the gap to the runner-up:

* **green**  confident and clear winner
* **orange** predicted, but close to the runner-up
* **red**    low confidence overall

Thresholds live in constants below and can be tweaked in one place.
"""
from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path
from typing import Any


# Thresholds used to colour-code a prediction row.
CONF_GREEN_MIN = 0.50         # minimum confidence to be considered "green"
CONF_GAP_GREEN_MIN = 0.10     # minimum gap-to-runner-up for "green"
CONF_ORANGE_MIN = 0.30        # below this we mark the row red


def render_html(
    screenshot_path: Path,
    results: list[dict[str, Any]],
) -> str:
    """Render a full-page HTML report as a string.

    Args:
        screenshot_path: Path to the original screenshot (used for the title
            and the embedded preview at the top of the report).
        results: A list of dicts, each with ``row``, ``col``, ``slot_index``,
            ``path`` (path to the saved slot crop), and ``match`` (the dict
            returned by ``classify_item_crop``).
    """
    screenshot_uri = _encode_image_data_uri(screenshot_path)
    stats = _summary_stats(results)
    rows_html = "\n".join(_render_row(result) for result in results)

    return _TEMPLATE.format(
        title=html.escape(screenshot_path.name),
        screenshot_uri=screenshot_uri,
        slot_count=stats["slot_count"],
        green=stats["green"],
        orange=stats["orange"],
        red=stats["red"],
        empty=stats["empty"],
        rows=rows_html,
    )


def render_combined_html(items: list[tuple[Path, list[dict[str, Any]]]]) -> str:
    """Render a single HTML page containing one section per screenshot.

    ``items`` is a list of ``(screenshot_path, results)`` tuples in the order
    the screenshots should appear. The page starts with an index listing
    every screenshot (with per-section confidence counts) that links to the
    matching section below; each section embeds the source screenshot and
    the per-slot table just like an individual report.
    """
    if not items:
        return _COMBINED_EMPTY_PAGE

    # Build the navigation index and the sections in one pass so they share
    # the same anchor ids and per-screenshot stats.
    index_items: list[str] = []
    sections: list[str] = []
    totals = {"slot_count": 0, "green": 0, "orange": 0, "red": 0, "empty": 0}

    for index, (screenshot_path, results) in enumerate(items):
        anchor = f"s{index:03d}"
        stats = _summary_stats(results)
        for key in totals:
            totals[key] += stats[key]

        index_items.append(
            f'<li><a href="#{anchor}">'
            f'{html.escape(screenshot_path.name)}</a> '
            f'<span class="index-stats">'
            f'{stats["slot_count"]} slots &middot; '
            f'<span style="color: var(--green)">{stats["green"]}</span> / '
            f'<span style="color: var(--orange)">{stats["orange"]}</span> / '
            f'<span style="color: var(--red)">{stats["red"]}</span> / '
            f'<span style="color: var(--muted)">{stats["empty"]} empty</span>'
            f'</span></li>'
        )

        rows_html = "\n".join(_render_row(result) for result in results)
        screenshot_uri = _encode_image_data_uri(screenshot_path)
        sections.append(
            _COMBINED_SECTION.format(
                anchor=anchor,
                title=html.escape(screenshot_path.name),
                slot_count=stats["slot_count"],
                green=stats["green"],
                orange=stats["orange"],
                red=stats["red"],
                empty=stats["empty"],
                screenshot_uri=screenshot_uri,
                rows=rows_html,
            )
        )

    return _COMBINED_TEMPLATE.format(
        screenshot_count=len(items),
        total_slots=totals["slot_count"],
        total_green=totals["green"],
        total_orange=totals["orange"],
        total_red=totals["red"],
        total_empty=totals["empty"],
        index="\n".join(index_items),
        sections="\n".join(sections),
    )


# -- row rendering ----------------------------------------------------------


def _render_row(result: dict[str, Any]) -> str:
    """Render one table row for a single slot's classification result."""
    match = result["match"]
    top_matches = list(match.get("top_matches", []))
    primary = top_matches[0] if top_matches else {}
    runners = top_matches[1:5]
    is_empty = bool(match.get("is_empty"))

    confidence = float(match.get("confidence", 0.0) or 0.0)
    gap = match.get("top_gap")
    confidence_class = "pred-empty" if is_empty else _confidence_class(confidence, gap)
    gap_text = f"{gap:.3f}" if isinstance(gap, (int, float)) else "–"

    slot_uri = _encode_image_data_uri(result["path"])
    primary_refs = primary.get("matched_reference_paths", []) if primary else []
    primary_uri = _encode_image_data_uri(primary_refs[0]) if primary_refs else ""
    predicted_name = html.escape(str(primary.get("item", "?") or "?"))
    count_badge_html = _render_count_badge(result.get("stack_count"))
    count_preview_html = _render_count_preview(result.get("stack_count"))

    # Empty slots don't really have a prediction, so collapse the prediction
    # and runners-up columns to a simple placeholder for readability.
    if is_empty:
        kept_fraction = float(match.get("kept_pixel_fraction", 0.0) or 0.0)
        return f"""
          <tr class="pred-empty">
            <td class="pos">r{result['row']}&thinsp;c{result['col']}<br>#{result['slot_index']:03d}</td>
            <td class="slot"><img class="slot-img" src="{slot_uri}" alt="empty slot {result['slot_index']}"></td>
            <td class="pred" colspan="3">
              <div class="pred-name">empty slot</div>
              <div class="pred-score">kept {kept_fraction * 100:.1f}% of pixels</div>
            </td>
          </tr>
        """

    # Build the runners-up block (thumbnail + name + score for each).
    runner_cells: list[str] = []
    for runner in runners:
        runner_refs = runner.get("matched_reference_paths", []) or []
        runner_uri = _encode_image_data_uri(runner_refs[0]) if runner_refs else ""
        runner_name = html.escape(str(runner.get("item", "") or ""))
        runner_score = float(runner.get("confidence", 0.0) or 0.0)
        runner_cells.append(
            f"""
            <div class="runner">
              <img src="{runner_uri}" alt="{runner_name}">
              <div class="runner-label">{runner_name}</div>
              <div class="runner-score">{runner_score:.3f}</div>
            </div>
            """
        )
    runners_html = "".join(runner_cells) if runner_cells else '<span class="runner-score">—</span>'

    return f"""
      <tr class="{confidence_class}">
        <td class="pos">r{result['row']}&thinsp;c{result['col']}<br>#{result['slot_index']:03d}</td>
        <td class="slot"><img class="slot-img" src="{slot_uri}" alt="slot {result['slot_index']}"></td>
        <td class="pred">
          <div class="pred-name">{predicted_name}{count_badge_html}</div>
          <div class="pred-score">conf {confidence:.3f}</div>
          <div class="pred-gap">gap {gap_text}</div>
          {count_preview_html}
        </td>
        <td class="ref"><img class="ref-img" src="{primary_uri}" alt="{predicted_name} reference"></td>
        <td class="runners-cell"><div class="runners">{runners_html}</div></td>
      </tr>
    """


def _render_count_badge(stack_count: dict[str, Any] | None) -> str:
    """Return a small inline HTML badge showing the detected stack count.

    - Value read by OCR: solid badge (`x164`).
    - No number visible but item present: dashed "assumed single" badge
      (`x1`) so the reader can distinguish default-to-1 from confident OCR.
    """
    if not stack_count:
        return ""
    value = stack_count.get("value")
    if value is None:
        return ""
    assumed = bool(stack_count.get("assumed_single"))
    confidence = float(stack_count.get("confidence", 0.0) or 0.0)
    if assumed:
        cls = "count-badge count-badge-assumed"
        title = "No number detected; assumed single item"
    else:
        cls = "count-badge"
        title = f"OCR confidence: {confidence:.0%}"
    return f' <span class="{cls}" title="{title}">&times;{int(value)}</span>'


def _render_count_preview(stack_count: dict[str, Any] | None) -> str:
    """Render the preprocessed number crop so a human can eyeball what OCR saw.

    The preview is an inline base64 PNG of the binarised number region -
    white digit pixels on a black background, the exact image that was
    handed to Tesseract. Shown regardless of whether OCR succeeded so we
    can debug both classes of failure (OCR misread vs no white pixels).
    """
    if not stack_count:
        return ""
    image_path = stack_count.get("image_path")
    if not image_path:
        return ""
    uri = _encode_image_data_uri(image_path)
    if not uri:
        return ""
    raw = html.escape(str(stack_count.get("raw_text", "") or ""))
    pixels = int(stack_count.get("pixel_count", 0) or 0)
    tooltip = f"OCR input (white px: {pixels}, raw: {raw or '-'})"
    return (
        f'<img class="count-preview" src="{uri}" alt="number crop" title="{tooltip}">'
    )


def _confidence_class(confidence: float, gap: float | None) -> str:
    """Pick a CSS class name based on confidence and runner-up gap."""
    effective_gap = gap if gap is not None else 0.0
    if confidence >= CONF_GREEN_MIN and effective_gap >= CONF_GAP_GREEN_MIN:
        return "pred-green"
    if confidence >= CONF_ORANGE_MIN:
        return "pred-orange"
    return "pred-red"


def _summary_stats(results: list[dict[str, Any]]) -> dict[str, int]:
    """Count slots by confidence bucket for the ribbon at the top of the report."""
    counts = {"slot_count": len(results), "green": 0, "orange": 0, "red": 0, "empty": 0}
    for result in results:
        match = result.get("match", {})
        if match.get("is_empty"):
            counts["empty"] += 1
            continue
        confidence = float(match.get("confidence", 0.0) or 0.0)
        gap = match.get("top_gap")
        klass = _confidence_class(confidence, gap)
        if klass == "pred-green":
            counts["green"] += 1
        elif klass == "pred-orange":
            counts["orange"] += 1
        else:
            counts["red"] += 1
    return counts


def _encode_image_data_uri(path: str | Path | None) -> str:
    """Encode an image file as a ``data:`` URI so it embeds directly in HTML."""
    if not path:
        return ""
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        return ""
    mime, _ = mimetypes.guess_type(str(resolved))
    if not mime:
        mime = "image/png"
    payload = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


# -- HTML shell -------------------------------------------------------------
# Outer template with CSS. Kept as a raw string (``{{`` and ``}}`` escape
# literal braces so ``str.format`` only replaces our named fields).
_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hue-Map: {title}</title>
<style>
  :root {{
    color-scheme: dark;
    --bg: #111418;
    --panel: #1b2128;
    --border: #2d343c;
    --text: #e6ebf1;
    --muted: #8b95a1;
    --green: #3fb950;
    --orange: #d29922;
    --red: #f85149;
  }}
  html, body {{
    background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    margin: 0; padding: 24px;
  }}
  h1 {{ margin: 0 0 8px 0; font-size: 20px; }}
  .meta {{ color: var(--muted); margin-bottom: 20px; font-size: 13px; }}
  .source {{
    display: block; max-width: 960px; max-height: 480px; width: auto; height: auto;
    border: 1px solid var(--border); border-radius: 8px; margin-bottom: 24px;
    image-rendering: pixelated;
  }}
  table {{
    border-collapse: collapse; width: 100%;
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    overflow: hidden;
  }}
  th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
  th {{
    background: #242b33; font-weight: 600; font-size: 12px; text-align: left;
    color: var(--muted); letter-spacing: 0.05em; text-transform: uppercase;
  }}
  tr:last-child td {{ border-bottom: none; }}
  td.pos {{ width: 70px; font-family: ui-monospace, Consolas, monospace; color: var(--muted); }}
  td.slot {{ width: 120px; }}
  td.ref {{ width: 110px; }}
  td.pred {{ width: 200px; }}
  img.slot-img, img.ref-img {{
    max-width: 100px; max-height: 100px; border: 1px solid var(--border);
    image-rendering: pixelated; border-radius: 4px;
    /* Subtle checkerboard so transparent pixels in masked previews are visible. */
    background-color: #0b0d10;
    background-image:
      linear-gradient(45deg, #151a20 25%, transparent 25%),
      linear-gradient(-45deg, #151a20 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, #151a20 75%),
      linear-gradient(-45deg, transparent 75%, #151a20 75%);
    background-size: 10px 10px;
    background-position: 0 0, 0 5px, 5px -5px, -5px 0px;
  }}
  .pred-name {{ font-weight: 600; font-size: 15px; margin-bottom: 4px; }}
  .pred-green .pred-name {{ color: var(--green); }}
  .pred-orange .pred-name {{ color: var(--orange); }}
  .pred-red .pred-name {{ color: var(--red); }}
  .pred-empty .pred-name {{ color: var(--muted); font-style: italic; }}
  .pred-empty td.slot img {{ opacity: 0.5; }}
  .count-badge {{
    display: inline-block; margin-left: 6px; padding: 1px 7px;
    border-radius: 10px; background: #2d343c; color: var(--text);
    font-family: ui-monospace, Consolas, monospace; font-size: 12px;
    font-weight: 600; vertical-align: middle;
  }}
  .count-badge-assumed {{
    background: transparent;
    border: 1px dashed var(--muted);
    color: var(--muted); font-weight: 500;
  }}
  .count-preview {{
    display: inline-block; margin-top: 6px;
    max-width: 120px; max-height: 40px;
    border: 1px solid var(--border); border-radius: 3px;
    background: #000; image-rendering: pixelated;
  }}
  .pred-score, .pred-gap {{ color: var(--muted); font-size: 12px; font-family: ui-monospace, monospace; }}
  .runners {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .runner {{ width: 80px; text-align: center; }}
  .runner img {{
    width: 64px; height: 64px; object-fit: contain;
    border: 1px solid var(--border); border-radius: 4px;
    background: #0b0d10; image-rendering: pixelated;
  }}
  .runner-label {{ font-size: 11px; margin-top: 2px; word-break: break-word; }}
  .runner-score {{ font-size: 11px; color: var(--muted); font-family: ui-monospace, monospace; }}
</style>
</head>
<body>
  <h1>Hue-Map: {title}</h1>
  <div class="meta">
    {slot_count} slots &middot;
    <span style="color: var(--green)">{green} confident</span> &middot;
    <span style="color: var(--orange)">{orange} uncertain</span> &middot;
    <span style="color: var(--red)">{red} weak</span> &middot;
    <span style="color: var(--muted)">{empty} empty</span>
  </div>
  <img class="source" src="{screenshot_uri}" alt="source screenshot">
  <table>
    <thead>
      <tr>
        <th>Pos</th>
        <th>Slot crop</th>
        <th>Prediction</th>
        <th>Top ref</th>
        <th>Runners-up</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""


# -- combined report --------------------------------------------------------
# Separate templates for the multi-screenshot view so the styling stays in
# one place. The combined page reuses the same CSS variables and row HTML
# as the single report; the only additions are the index and the per-section
# anchors / headers.

_COMBINED_SECTION = """
<section id="{anchor}">
  <h2>{title}</h2>
  <div class="meta">
    {slot_count} slots &middot;
    <span style="color: var(--green)">{green} confident</span> &middot;
    <span style="color: var(--orange)">{orange} uncertain</span> &middot;
    <span style="color: var(--red)">{red} weak</span> &middot;
    <span style="color: var(--muted)">{empty} empty</span>
    &middot; <a href="#top" class="back-to-top">back to top</a>
  </div>
  <img class="source" src="{screenshot_uri}" alt="{title}">
  <table>
    <thead>
      <tr>
        <th>Pos</th>
        <th>Slot crop</th>
        <th>Prediction</th>
        <th>Top ref</th>
        <th>Runners-up</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</section>
"""


_COMBINED_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hue-Map: combined report ({screenshot_count} screenshots)</title>
<style>
  :root {{
    color-scheme: dark;
    --bg: #111418;
    --panel: #1b2128;
    --border: #2d343c;
    --text: #e6ebf1;
    --muted: #8b95a1;
    --green: #3fb950;
    --orange: #d29922;
    --red: #f85149;
  }}
  html, body {{
    background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    margin: 0; padding: 24px;
  }}
  h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
  h2 {{ margin: 36px 0 8px 0; font-size: 18px; }}
  .meta {{ color: var(--muted); margin-bottom: 20px; font-size: 13px; }}
  .source {{
    display: block; max-width: 960px; max-height: 480px; width: auto; height: auto;
    border: 1px solid var(--border); border-radius: 8px; margin-bottom: 24px;
    image-rendering: pixelated;
  }}
  table {{
    border-collapse: collapse; width: 100%;
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    overflow: hidden; margin-bottom: 40px;
  }}
  th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
  th {{
    background: #242b33; font-weight: 600; font-size: 12px; text-align: left;
    color: var(--muted); letter-spacing: 0.05em; text-transform: uppercase;
  }}
  tr:last-child td {{ border-bottom: none; }}
  td.pos {{ width: 70px; font-family: ui-monospace, Consolas, monospace; color: var(--muted); }}
  td.slot {{ width: 120px; }}
  td.ref {{ width: 110px; }}
  td.pred {{ width: 200px; }}
  img.slot-img, img.ref-img {{
    max-width: 100px; max-height: 100px; border: 1px solid var(--border);
    image-rendering: pixelated; border-radius: 4px;
    background-color: #0b0d10;
    background-image:
      linear-gradient(45deg, #151a20 25%, transparent 25%),
      linear-gradient(-45deg, #151a20 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, #151a20 75%),
      linear-gradient(-45deg, transparent 75%, #151a20 75%);
    background-size: 10px 10px;
    background-position: 0 0, 0 5px, 5px -5px, -5px 0px;
  }}
  .pred-name {{ font-weight: 600; font-size: 15px; margin-bottom: 4px; }}
  .pred-green .pred-name {{ color: var(--green); }}
  .pred-orange .pred-name {{ color: var(--orange); }}
  .pred-red .pred-name {{ color: var(--red); }}
  .pred-empty .pred-name {{ color: var(--muted); font-style: italic; }}
  .pred-empty td.slot img {{ opacity: 0.5; }}
  .count-badge {{
    display: inline-block; margin-left: 6px; padding: 1px 7px;
    border-radius: 10px; background: #2d343c; color: var(--text);
    font-family: ui-monospace, Consolas, monospace; font-size: 12px;
    font-weight: 600; vertical-align: middle;
  }}
  .count-badge-assumed {{
    background: transparent;
    border: 1px dashed var(--muted);
    color: var(--muted); font-weight: 500;
  }}
  .count-preview {{
    display: inline-block; margin-top: 6px;
    max-width: 120px; max-height: 40px;
    border: 1px solid var(--border); border-radius: 3px;
    background: #000; image-rendering: pixelated;
  }}
  .pred-score, .pred-gap {{ color: var(--muted); font-size: 12px; font-family: ui-monospace, monospace; }}
  .runners {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .runner {{ width: 80px; text-align: center; }}
  .runner img {{
    width: 64px; height: 64px; object-fit: contain;
    border: 1px solid var(--border); border-radius: 4px;
    background: #0b0d10; image-rendering: pixelated;
  }}
  .runner-label {{ font-size: 11px; margin-top: 2px; word-break: break-word; }}
  .runner-score {{ font-size: 11px; color: var(--muted); font-family: ui-monospace, monospace; }}

  /* Combined-report specific */
  nav.index {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 14px 18px; margin-bottom: 32px;
  }}
  nav.index ul {{ margin: 0; padding: 0; list-style: none; }}
  nav.index li {{ padding: 4px 0; font-size: 13px; }}
  nav.index a {{ color: var(--text); text-decoration: none; }}
  nav.index a:hover {{ color: var(--green); }}
  .index-stats {{ color: var(--muted); font-family: ui-monospace, Consolas, monospace; font-size: 11px; margin-left: 8px; }}
  a.back-to-top {{ color: var(--muted); font-size: 12px; }}
  a.back-to-top:hover {{ color: var(--green); }}
  section {{ scroll-margin-top: 24px; }}
</style>
</head>
<body id="top">
  <h1>Hue-Map combined report</h1>
  <div class="meta">
    {screenshot_count} screenshots &middot; {total_slots} total slots &middot;
    <span style="color: var(--green)">{total_green} confident</span> &middot;
    <span style="color: var(--orange)">{total_orange} uncertain</span> &middot;
    <span style="color: var(--red)">{total_red} weak</span> &middot;
    <span style="color: var(--muted)">{total_empty} empty</span>
  </div>
  <nav class="index">
    <ul>
      {index}
    </ul>
  </nav>
  {sections}
</body>
</html>
"""


_COMBINED_EMPTY_PAGE = (
    "<!doctype html><html><body><h1>Hue-Map combined report</h1>"
    "<p>No reports found. Run <code>py -m hue_map.pipeline</code> first.</p>"
    "</body></html>"
)
