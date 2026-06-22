#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: E501
"""Post-process a scope_stats.jsonl into a self-contained HTML report.

The report uses inline CSS, inline SVG, and a small inline script for chart
expansion, with no external JavaScript or CDN dependency. It keeps the raw
JSONL schema unchanged and focuses the presentation on the diagnostic
questions scope_stats answers: which resource came closest to capacity, which
scope site produced the peak, and whether the pressure came from per-scope
allocation or accumulated backlog.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, NamedTuple

logger = logging.getLogger(__name__)


class Resource(NamedTuple):
    label: str
    unit: str
    divisor: int
    risk_metric: str


class Metric(NamedTuple):
    key: str
    label: str
    formula: str
    description: str
    fn: Callable[[dict, dict], int]


RESOURCES: dict[str, Resource] = {
    "task_window": Resource("Task window", "slots", 1, "scope_high_water"),
    "heap": Resource("Heap", "MiB", 1024 * 1024, "scope_high_water"),
    "dep_pool": Resource("Dep pool", "entries", 1, "scope_high_water"),
    "tensormap": Resource("TensorMap", "entries", 1, "real_occupancy"),
}


def _metrics(resource: str) -> list[Metric]:
    if resource == "tensormap":
        return [
            Metric(
                "real_occupancy",
                "Live entries",
                "end.tensormap",
                "TensorMap entries still in use at scope exit.",
                lambda _b, en: en.get("tensormap", 0),
            )
        ]

    s, e = f"{resource}_start", f"{resource}_end"
    return [
        Metric(
            "scope_high_water",
            "High water",
            "end.head - begin.tail",
            "Occupancy upper bound for this scope: entry backlog plus everything it allocated, "
            "not a realized peak and not bounded by capacity.",
            lambda b, en: en.get(e, 0) - b.get(s, 0),
        ),
        Metric(
            "real_occupancy",
            "Live at exit",
            "end.head - end.tail",
            "Resource still live at the scope exit instant.",
            lambda _b, en: en.get(e, 0) - en.get(s, 0),
        ),
        Metric(
            "scope_alloc",
            "Scope alloc",
            "end.head - begin.head",
            "How much this scope advanced the allocation frontier.",
            lambda b, en: en.get(e, 0) - b.get(e, 0),
        ),
    ]


def _load(jsonl_path: Path) -> tuple[dict, list[dict]]:
    lines = jsonl_path.read_text().splitlines()
    meta = json.loads(lines[0]) if lines else {}
    records = [json.loads(line) for line in lines[1:] if line.strip()]
    return meta, records


def _resource_size(meta: dict, resource: str, ring: int) -> int | None:
    cap = meta.get(f"{resource}_max")
    if isinstance(cap, list):
        return cap[ring] if 0 <= ring < len(cap) else None
    return cap


def _resource_has_data(pairs_by_ring: dict[int, list[tuple[dict, dict]]], resource: str) -> bool:
    if resource == "tensormap":
        return any("tensormap" in end for pairs in pairs_by_ring.values() for _begin, end in pairs)
    s, e = f"{resource}_start", f"{resource}_end"
    return any(
        s in begin and e in begin and s in end and e in end for pairs in pairs_by_ring.values() for begin, end in pairs
    )


def _pair_by_ring(records: list[dict]) -> dict[int, list[tuple[dict, dict]]]:
    pending: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    pairs: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for idx, rec in enumerate(records):
        rec["_record_index"] = idx
        ring, site, phase = rec["ring"], rec["site"], rec["phase"]
        if phase == "begin":
            pending[ring][site].append(rec)
        elif phase == "end":
            stack = pending[ring].get(site)
            if stack:
                pairs[ring].append((stack.pop(), rec))
            else:
                logger.warning("ring %s site %s: end without matching begin", ring, site)
    return pairs


def _esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _fmt_value(value: float, unit: str) -> str:
    if unit == "MiB":
        if abs(value - round(value)) < 0.05:
            return f"{value:.0f} {unit}"
        if value >= 100:
            return f"{value:.0f} {unit}"
        if value >= 10:
            return f"{value:.1f} {unit}"
        return f"{value:.2f} {unit}"
    if abs(value - round(value)) < 0.05:
        return f"{value:.0f} {unit}"
    return f"{value:.1f} {unit}"


def _fmt_pct(ratio: float | None) -> str:
    if ratio is None:
        return "n/a"
    return f"{ratio * 100:.2f}%"


def _scope_count_label(count: int) -> str:
    return f"{count} scope" if count == 1 else f"{count} scopes"


def _risk_class(ratio: float | None) -> str:
    if ratio is None:
        return "muted"
    if ratio >= 0.8:
        return "danger"
    if ratio >= 0.5:
        return "warn"
    return "ok"


def _use_bar_html(ratio: float | None) -> str:
    risk = _risk_class(ratio)
    pct = _fmt_pct(ratio)
    width = 0.0 if ratio is None else max(0.0, min(ratio, 1.0)) * 100
    return (
        f'<div class="use-bar {risk}" title="capacity use {pct}" aria-label="capacity use {pct}">'
        f'<span class="use-fill" style="width:{width:.2f}%"></span>'
        f'<span class="use-label">{pct}</span>'
        "</div>"
    )


def _series_for_resource(pairs: list[tuple[dict, dict]], resource: str) -> list[dict]:
    info = RESOURCES[resource]
    sites = [end.get("site", "?") for _begin, end in pairs]
    rings = [end.get("ring", begin.get("ring")) for begin, end in pairs]
    series = []
    for metric in _metrics(resource):
        # head/tail are monotonic non-wrapping counters, so each delta is exact
        # and non-negative — no wrap correction needed (see docs/dfx/scope-stats.md).
        raw_values = [metric.fn(begin, end) for begin, end in pairs]
        values = [v / info.divisor for v in raw_values]
        if values:
            peak_idx = max(range(len(values)), key=values.__getitem__)
            peak = values[peak_idx]
            site = sites[peak_idx]
            peak_context_ring = rings[peak_idx] if resource == "tensormap" else None
        else:
            peak_idx = -1
            peak = 0.0
            site = "?"
            peak_context_ring = None
        series.append(
            {
                "metric": metric,
                "class": "tensormap_live" if resource == "tensormap" else metric.key,
                "values": values,
                "sites": sites,
                "peak": peak,
                "peak_idx": peak_idx,
                "site": site,
                "peak_context_ring": peak_context_ring,
            }
        )
    return series


def _pairs_in_record_order(pairs_by_ring: dict[int, list[tuple[dict, dict]]]) -> list[tuple[dict, dict]]:
    pairs = [pair for ring_pairs in pairs_by_ring.values() for pair in ring_pairs]
    return sorted(pairs, key=lambda pair: pair[1].get("_record_index", 0))


def _risk_entry(meta: dict, resource: str, ring: int | None, pairs: list[tuple[dict, dict]]) -> dict:
    info = RESOURCES[resource]
    size = _resource_size(meta, resource, 0 if ring is None else ring)
    cap = (size / info.divisor) if size is not None else None
    series = _series_for_resource(pairs, resource)
    risk = next((s for s in series if s["metric"].key == info.risk_metric), series[0])
    ratio = risk["peak"] / cap if cap else None
    return {
        "resource": resource,
        "resource_label": info.label,
        "unit": info.unit,
        "ring": ring,
        "metric": risk["metric"],
        "peak": risk["peak"],
        "cap": cap,
        "ratio": ratio,
        "site": risk["site"],
        "scope_idx": risk["peak_idx"],
        "peak_context_ring": risk.get("peak_context_ring"),
        "pairs": len(pairs),
    }


def _risk_entries(meta: dict, pairs_by_ring: dict[int, list[tuple[dict, dict]]]) -> list[dict]:
    entries = []
    for resource in RESOURCES:
        if not _resource_has_data(pairs_by_ring, resource):
            continue
        if resource == "tensormap":
            pairs = _pairs_in_record_order(pairs_by_ring)
            if pairs:
                entries.append(_risk_entry(meta, resource, None, pairs))
            continue
        for ring in sorted(pairs_by_ring):
            pairs = pairs_by_ring[ring]
            if not pairs:
                continue
            entries.append(_risk_entry(meta, resource, ring, pairs))
    return entries


def _dominant_site(pairs_by_ring: dict[int, list[tuple[dict, dict]]]) -> tuple[str, int]:
    counts: Counter[str] = Counter()
    for pairs in pairs_by_ring.values():
        for _begin, end in pairs:
            counts[end.get("site", "?")] += 1
    if not counts:
        return "n/a", 0
    return counts.most_common(1)[0]


def _available_resources(pairs_by_ring: dict[int, list[tuple[dict, dict]]]) -> list[str]:
    return [resource for resource in RESOURCES if _resource_has_data(pairs_by_ring, resource)]


def _resource_list_text(resources: list[str]) -> str:
    labels = [RESOURCES[resource].label for resource in resources]
    if not labels:
        return "resources"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _formula_card(metric: Metric, extra: str = "") -> str:
    return (
        '<div class="formula-card">'
        f'<div class="formula-name"><span class="swatch m-{metric.key}"></span>{_esc(metric.label)}</div>'
        f"<code>{_esc(metric.formula)}</code>"
        f"<p>{_esc(metric.description)}{extra}</p>"
        "</div>"
    )


def _guide_card(title: str, marker_class: str, description: str) -> str:
    return (
        '<div class="formula-card">'
        f'<div class="formula-name"><span class="resource-dot {marker_class}"></span>{_esc(title)}</div>'
        f"<p>{_esc(description)}</p>"
        "</div>"
    )


def _resource_guide_cards(resources: list[str]) -> list[str]:
    resource_guides = {
        "task_window": (
            "Task window",
            "r-task_window",
            "Task slot pressure for each ring_depth.",
        ),
        "heap": (
            "Heap",
            "r-heap",
            "Output-buffer byte pressure for each ring_depth.",
        ),
        "dep_pool": (
            "Dep pool",
            "r-dep_pool",
            "Dependency-list entries used by scheduler fanout wiring for each ring_depth.",
        ),
        "tensormap": (
            "TensorMap",
            "r-tensormap",
            "Global live TensorMap entries used to resolve tensor producers.",
        ),
    }
    return [_guide_card(*resource_guides[resource]) for resource in resources if resource in resource_guides]


def _formula_guide_html(resources: list[str]) -> str:
    metrics = _metrics("heap")
    metric_cards = []
    for metric in metrics:
        extra = ""
        if metric.key == "scope_high_water":
            extra = " Use this line first when checking capacity risk."
        elif metric.key == "real_occupancy":
            extra = " Use it to see what remains live when the scope closes."
        elif metric.key == "scope_alloc":
            extra = " It is split into its own chart so small per-scope changes stay visible."
        metric_cards.append(_formula_card(metric, extra))
    resource_cards = _resource_guide_cards(resources)
    return (
        '<section class="panel formula-panel">'
        '<div class="section-head"><div><h2>Formula Guide</h2>'
        "<p>Resources name the measured pressure types. Chart metrics define the plotted lines and table peaks.</p>"
        "</div></div>"
        '<div class="formula-group"><div class="formula-group-title">Resources</div>'
        f'<div class="formula-grid">{"".join(resource_cards)}</div></div>'
        '<div class="formula-group"><div class="formula-group-title">Chart metrics</div>'
        f'<div class="formula-grid metric-grid">{"".join(metric_cards)}</div></div>'
        "</section>"
    )


def _summary_html(meta: dict, pairs_by_ring: dict[int, list[tuple[dict, dict]]], jsonl_name: str) -> str:
    entries = _risk_entries(meta, pairs_by_ring)
    resources = _available_resources(pairs_by_ring)
    pair_count = sum(len(pairs) for pairs in pairs_by_ring.values())
    site, site_count = _dominant_site(pairs_by_ring)
    fatal = bool(meta.get("fatal", False))
    dropped = int(meta.get("dropped", 0) or 0)
    status_label = "Fatal" if fatal else ("Dropped records" if dropped else "Healthy: no fatal, no dropped")
    status_class = "danger" if fatal else ("warn" if dropped else "ok")

    return f"""
<header class="report-head">
  <div>
    <p class="eyebrow">scope_stats report</p>
    <h1>{_esc(jsonl_name)}</h1>
    <div class="dominant-site">
      <span>Dominant scope site</span>
      <strong class="mono">{_esc(site)}</strong>
      <em>{site_count} paired scopes</em>
    </div>
    <div class="run-meta" aria-label="run metadata">
      <span class="status {status_class}">{_esc(status_label)}</span>
      <span>{int(meta.get("total", 0) or 0)} records</span>
      <span>{pair_count} total paired scopes</span>
      <span>{len(pairs_by_ring)} ring_depths</span>
    </div>
    <p class="subtle">Per-scope resource pressure across {_esc(_resource_list_text(resources))}. Use the
    formulas as definitions, then start diagnosis with Top Peaks.</p>
  </div>
</header>
{_formula_guide_html(resources)}
{_top_peaks_html(entries)}
"""


def _top_peaks_html(entries: list[dict]) -> str:
    rows = []
    resource_order = {resource: idx for idx, resource in enumerate(RESOURCES)}
    for entry in sorted(entries, key=lambda e: (e["ring"] is None, e["ring"] or 0, resource_order[e["resource"]])):
        cap = _fmt_value(entry["cap"], entry["unit"]) if entry["cap"] is not None else "n/a"
        ring_label = "global" if entry["ring"] is None else f"ring_depth {entry['ring']}"
        rows.append(
            "<tr>"
            f"<td>{_esc(entry['resource_label'])}</td>"
            f"<td>{_esc(ring_label)}</td>"
            f"<td>{_esc(entry['metric'].label)}</td>"
            f"<td>{_fmt_value(entry['peak'], entry['unit'])}</td>"
            f"<td>{cap}</td>"
            f"<td>{_use_bar_html(entry['ratio'])}</td>"
            f'<td class="mono">{_esc(entry["site"])}</td>'
            f"<td>#{entry['scope_idx']}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="8" class="subtle">No paired scopes found.</td></tr>')
    return (
        '<section class="panel">'
        '<div class="section-head"><div><h2>Top Peaks</h2>'
        "<p>Highlights the highest observed pressure, with capacity use and source site for quick diagnosis.</p>"
        "</div></div>"
        '<table class="peaks"><thead><tr><th>Resource</th><th>ring_depth</th><th>Metric</th>'
        "<th>Peak</th><th>Capacity</th><th>Use</th><th>Site</th><th>Scope</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


_SVG_W = 900
_SVG_H = 300
_PAD_L = 74
_PAD_R = 24
_PAD_T = 28
_PAD_B = 48


def _nice_tick_step(raw_step: float, *, integer: bool) -> float:
    if raw_step <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    base = 10**exponent
    candidates = [1, 2, 2.5, 5, 7.5, 10]
    steps = [candidate * base for candidate in candidates]
    if integer:
        steps = [step for step in steps if step >= 1 and abs(step - round(step)) < 1e-9]
    step = min(steps, key=lambda v: abs(v - raw_step), default=1.0)
    if integer:
        return max(1.0, step)
    return step


def _axis_max(peak: float) -> float:
    if peak <= 0:
        return 1.0
    return peak * 1.1


def _tick_values(y_max: float, unit: str, count: int = 6) -> list[float]:
    if y_max <= 0:
        return [0.0]
    step = _nice_tick_step(y_max / max(1, count - 1), integer=(unit != "MiB"))
    tick_count = max(1, math.floor(y_max / step))
    return [step * i for i in range(tick_count + 1)]


def _x_tick_indices(n: int, count: int = 6) -> list[int]:
    if n <= 0:
        return []
    max_idx = n - 1
    if max_idx == 0:
        return [0]
    step = int(_nice_tick_step(max_idx / max(1, count - 1), integer=True))
    step = max(1, step)
    return list(range(0, max_idx + 1, step))


def _svg_series_chart(series: list[dict], unit: str, show_axis_title: bool = False) -> str:
    values = [v for entry in series for v in entry["values"]]
    peak_value = max(values) if values else 0.0
    y_max = _axis_max(peak_value)
    n_points = max((len(entry["values"]) for entry in series), default=0)
    plot_w = _SVG_W - _PAD_L - _PAD_R
    plot_h = _SVG_H - _PAD_T - _PAD_B

    def sx(i: int, n: int) -> float:
        return _PAD_L + (plot_w * i / (n - 1) if n > 1 else plot_w / 2)

    def sy(v: float, axis_max: float) -> float:
        return _PAD_T + plot_h * (1 - v / axis_max)

    parts = [
        f'<svg viewBox="0 0 {_SVG_W} {_SVG_H}" class="chart" preserveAspectRatio="xMidYMid meet" '
        'role="button" tabindex="0" aria-label="Open enlarged chart">'
    ]
    for value in _tick_values(y_max, unit):
        y = sy(value, y_max)
        label = _fmt_value(value, unit)
        parts.append(f'<line x1="{_PAD_L}" y1="{y:.1f}" x2="{_SVG_W - _PAD_R}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{_PAD_L - 8}" y="{y + 3:.1f}" class="ylab" text-anchor="end">{_esc(label)}</text>')
    for idx in _x_tick_indices(n_points):
        x = sx(idx, n_points)
        parts.append(f'<line x1="{x:.1f}" y1="{_PAD_T}" x2="{x:.1f}" y2="{_PAD_T + plot_h}" class="xgrid"/>')
        parts.append(
            f'<line x1="{x:.1f}" y1="{_PAD_T + plot_h}" x2="{x:.1f}" y2="{_PAD_T + plot_h + 5}" class="axis"/>'
        )
        parts.append(f'<text x="{x:.1f}" y="{_SVG_H - 17}" class="xlab" text-anchor="middle">{idx}</text>')
    parts.append(f'<line x1="{_PAD_L}" y1="{_PAD_T}" x2="{_PAD_L}" y2="{_PAD_T + plot_h}" class="axis"/>')
    parts.append(
        f'<line x1="{_PAD_L}" y1="{_PAD_T + plot_h}" x2="{_SVG_W - _PAD_R}" y2="{_PAD_T + plot_h}" class="axis"/>'
    )
    if show_axis_title:
        if len(series) == 1 and series[0]["metric"].key == "real_occupancy":
            axis_title = series[0]["metric"].label.lower()
        elif len(series) == 1 and series[0]["metric"].key == "scope_alloc":
            axis_title = "scope alloc"
        else:
            axis_title = "high water / live at exit"
        parts.append(f'<text x="{_PAD_L}" y="16" class="axis-title">axis: {axis_title}</text>')
    parts.append(f'<text x="{_PAD_L}" y="{_SVG_H - 7}" class="alab">scope index</text>')

    for entry in series:
        metric = entry["metric"]
        css_class = entry["class"]
        values = entry["values"]
        n = len(values)
        if n == 0:
            continue
        pts = " ".join(f"{sx(i, n):.1f},{sy(v, y_max):.1f}" for i, v in enumerate(values))
        parts.append(f'<polyline points="{pts}" class="line m-{css_class}"/>')
        sites = entry["sites"]
        for i, value in enumerate(values):
            title = f"{metric.label}: scope #{i}, y={_fmt_value(value, unit)}, site={sites[i]}"
            parts.append(
                f'<circle cx="{sx(i, n):.1f}" cy="{sy(value, y_max):.1f}" r="5" class="hit m-{css_class}">'
                f"<title>{_esc(title)}</title></circle>"
            )
        idx = entry["peak_idx"]
        if idx >= 0:
            cx, cy = sx(idx, n), sy(values[idx], y_max)
            title = f"{metric.label} peak: {_fmt_value(values[idx], unit)} at scope #{idx} ({entry['site']})"
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" class="peak m-{css_class}">'
                f"<title>{_esc(title)}</title></circle>"
            )
    parts.append("</svg>")
    return "".join(parts)


def _legend_html(series: list[dict], unit: str, cap: float | None) -> str:
    items = []
    for entry in series:
        metric = entry["metric"]
        ratio = entry["peak"] / cap if cap else None
        items.append(
            "<li>"
            f'<span class="swatch m-{entry["class"]}"></span>'
            f'<span class="legend-main">{_esc(metric.label)}</span>'
            f'<span class="legend-note">peak {_fmt_value(entry["peak"], unit)} · use {_fmt_pct(ratio)}</span>'
            "</li>"
        )
    return (
        '<div class="aside-section">'
        '<div class="aside-title">Line colors</div>'
        f'<ul class="legend">{"".join(items)}</ul>'
        "</div>"
    )


def _max_use_html(risk: dict, unit: str, cap: float | None, ratio: float | None) -> str:
    cap_text = _fmt_value(cap, unit) if cap is not None else "n/a"
    scope_text = f"#{risk['peak_idx']}" if risk["peak_idx"] >= 0 else "n/a"
    use_class = _risk_class(ratio)
    use_text = f'<span class="pill {use_class}">{_fmt_pct(ratio)}</span>'
    context_ring = risk.get("peak_context_ring")
    context_row = ""
    if context_ring is not None:
        context_row = (
            f'<div class="aside-row"><span>Peak context ring_depth</span><strong>{_esc(context_ring)}</strong></div>'
        )
    return (
        '<div class="aside-section">'
        '<div class="aside-title">Max use</div>'
        f'<div class="aside-row"><span>Risk metric</span><strong>{_esc(risk["metric"].label)}</strong></div>'
        f'<div class="aside-row"><span>Peak</span><strong>{_fmt_value(risk["peak"], unit)}</strong></div>'
        f'<div class="aside-row"><span>Capacity</span><strong>{cap_text}</strong></div>'
        f'<div class="aside-row"><span>Use</span><strong>{use_text}</strong></div>'
        f'<div class="aside-row"><span>Peak scope</span><strong>{scope_text}</strong></div>'
        f"{context_row}"
        f'<div class="aside-row"><span>Peak site</span><strong class="mono">{_esc(risk["site"])}</strong></div>'
        "</div>"
    )


def _chart_stack_html(series: list[dict], unit: str) -> str:
    alloc_series = [entry for entry in series if entry["metric"].key == "scope_alloc"]
    pressure_series = [entry for entry in series if entry["metric"].key != "scope_alloc"]
    if not alloc_series or not pressure_series:
        title = series[0]["metric"].label if series else "Chart"
        return (
            '<div class="chart-stack">'
            f'<div class="chart-block"><div class="chart-heading">{_esc(title)}</div>'
            f"{_svg_series_chart(series, unit)}</div></div>"
        )
    return (
        '<div class="chart-stack">'
        '<div class="chart-block"><div class="chart-heading">Resource pressure</div>'
        f"{_svg_series_chart(pressure_series, unit, show_axis_title=False)}</div>"
        '<div class="chart-block"><div class="chart-heading">Scope alloc</div>'
        f"{_svg_series_chart(alloc_series, unit)}</div>"
        "</div>"
    )


def _scope_group_section(resource: str, title: str, pairs: list[tuple[dict, dict]], size: int | None) -> str:
    info = RESOURCES[resource]
    series = _series_for_resource(pairs, resource)
    cap = size / info.divisor if size is not None else None
    risk = next((s for s in series if s["metric"].key == info.risk_metric), series[0])
    ratio = risk["peak"] / cap if cap else None
    open_attr = " open" if len(pairs) > 1 else ""
    return f"""
<details class="ring-panel"{open_attr}>
  <summary>
    <span class="ring-title">{_esc(title)}</span>
    <span class="ring-meta">{_scope_count_label(len(pairs))}</span>
  </summary>
  <div class="ring-body">
    <div class="chart-wrap">{_chart_stack_html(series, info.unit)}</div>
    <div class="ring-aside">
      {_legend_html(series, info.unit, cap)}
      {_max_use_html(risk, info.unit, cap, ratio)}
    </div>
  </div>
</details>
"""


def _ring_section(resource: str, ring: int, pairs: list[tuple[dict, dict]], size: int | None) -> str:
    return _scope_group_section(resource, f"ring_depth {ring}", pairs, size)


def _resource_section(resource: str, meta: dict, pairs_by_ring: dict[int, list[tuple[dict, dict]]]) -> str:
    info = RESOURCES[resource]
    if not _resource_has_data(pairs_by_ring, resource):
        return ""
    if resource == "tensormap":
        pairs = _pairs_in_record_order(pairs_by_ring)
        if not pairs:
            return ""
        ring_sections = [_scope_group_section(resource, "global", pairs, _resource_size(meta, resource, 0))]
        description = (
            "Global TensorMap pressure. Charts merge live-entry samples from all scope ring_depths in record order."
        )
        return (
            '<section class="panel resource">'
            f'<div class="section-head"><div><h2>{_esc(info.label)}</h2>'
            f"<p>{_esc(description)}</p>"
            "</div></div>"
            f"{''.join(ring_sections)}</section>"
        )
    ring_sections = []
    for ring in sorted(pairs_by_ring):
        pairs = pairs_by_ring[ring]
        if pairs:
            ring_sections.append(_ring_section(resource, ring, pairs, _resource_size(meta, resource, ring)))
    if not ring_sections:
        return ""
    if resource == "dep_pool":
        description = (
            "Charts show scheduler-published dependency-list entry pressure with scope-index and observed-usage ticks."
        )
    else:
        description = "Charts split resource pressure from Scope alloc, each with scope-index and observed-usage ticks."
    return (
        '<section class="panel resource">'
        f'<div class="section-head"><div><h2>{_esc(info.label)}</h2>'
        f"<p>{_esc(info.label)} pressure by ring_depth. {_esc(description)}</p>"
        "</div></div>"
        f"{''.join(ring_sections)}</section>"
    )


_STYLE = """
:root{
  --bg:#f5f7fb; --panel:#ffffff; --text:#172033; --muted:#667085;
  --border:#d9dee7; --grid:#d7dee8; --axis:#7b8494;
  --blue:#2563eb; --green:#059669; --amber:#d97706; --orange:#ea580c; --red:#dc2626; --purple:#7c3aed;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.page{max-width:1220px;margin:0 auto;padding:24px}
.report-head{margin-bottom:14px}
.eyebrow{margin:0 0 4px;color:var(--muted);font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em}
h1{margin:0;font-size:26px;line-height:1.2}
h2{margin:0;font-size:18px;line-height:1.3}
p{margin:0}
.subtle{color:var(--muted);font-size:13px;line-height:1.45}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.dominant-site{display:flex;align-items:center;gap:8px;margin-top:9px;min-width:0}
.dominant-site span{color:var(--muted);font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.04em}
.dominant-site strong{font-size:17px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dominant-site em{color:var(--muted);font-size:12px;font-style:normal;white-space:nowrap}
.run-meta{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 8px;color:var(--muted);font-size:12px}
.run-meta span{display:inline-flex;align-items:center;border:1px solid var(--border);border-radius:999px;background:#fff;padding:3px 8px}
.status{border:1px solid var(--border);border-radius:999px;background:var(--panel);padding:3px 8px;font-size:12px;font-weight:700}
.status.ok,.pill.ok{color:#047857;background:#ecfdf3;border-color:#abefc6}
.status.warn,.pill.warn{color:#b54708;background:#fffaeb;border-color:#fedf89}
.status.danger,.pill.danger{color:#b42318;background:#fef3f2;border-color:#fecdca}
.pill.muted{color:var(--muted);background:#f2f4f7;border-color:var(--border)}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:8px}
.panel{padding:16px;margin:14px 0}
.section-head{display:flex;justify-content:space-between;gap:12px;margin-bottom:12px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em;border-bottom:1px solid var(--border);padding:8px}
td{border-bottom:1px solid #edf0f5;padding:8px;vertical-align:top}
tr:last-child td{border-bottom:0}
.pill{display:inline-flex;align-items:center;border:1px solid;border-radius:999px;padding:2px 8px;font-size:12px;font-weight:700;white-space:nowrap}
.use-bar{position:relative;width:132px;max-width:100%;height:22px;border:1px solid var(--border);border-radius:999px;background:#f2f4f7;overflow:hidden}
.use-fill{position:absolute;inset:0 auto 0 0;border-radius:999px}
.use-label{position:relative;z-index:1;display:flex;align-items:center;justify-content:flex-end;height:100%;padding:0 8px;font-size:12px;font-weight:750;color:var(--text)}
.use-bar.ok .use-fill{background:#abefc6}
.use-bar.warn .use-fill{background:#fedf89}
.use-bar.danger .use-fill{background:#fecdca}
.use-bar.muted .use-label{justify-content:center;color:var(--muted)}
.formula-group + .formula-group{margin-top:13px}
.formula-group-title{margin:0 0 7px;color:var(--muted);font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.05em}
.formula-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
.metric-grid{grid-template-columns:repeat(3,minmax(0,1fr))}
.formula-card{border:1px solid var(--border);border-radius:8px;background:#fff;padding:11px;min-width:0}
.formula-name{font-size:13px;font-weight:750;margin-bottom:7px}
.formula-card code{display:block;background:#f2f4f7;border-radius:5px;padding:5px 6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.formula-card p{color:var(--muted);font-size:12px;line-height:1.4;margin-top:7px}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:7px;vertical-align:-1px}
.resource-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;vertical-align:-1px}
.r-task_window{background:var(--blue)}
.r-heap{background:var(--green)}
.r-dep_pool{background:var(--amber)}
.r-tensormap{background:var(--purple)}
.m-scope_high_water{--series-color:var(--blue);stroke:var(--blue);fill:var(--blue);background:var(--blue)}
.m-real_occupancy{--series-color:var(--orange);stroke:var(--orange);fill:var(--orange);background:var(--orange)}
.m-scope_alloc{--series-color:var(--green);stroke:var(--green);fill:var(--green);background:var(--green)}
.m-tensormap_live{--series-color:var(--purple);stroke:var(--purple);fill:var(--purple);background:var(--purple)}
.ring-panel{border:1px solid var(--border);border-radius:8px;margin-top:10px;background:#fff}
.ring-panel summary{display:flex;align-items:center;gap:10px;cursor:pointer;padding:12px 14px}
.ring-title{font-size:15px;font-weight:750}
.ring-meta{color:var(--muted);font-size:12px;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ring-body{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:14px;padding:0 14px 14px}
.chart-wrap{min-width:0}
.chart-stack{display:grid;gap:10px}
.chart-block{min-width:0}
.chart-heading{margin:0 0 5px;color:var(--muted);font-size:12px;font-weight:750;text-transform:uppercase;letter-spacing:.04em}
.chart{width:100%;height:auto;display:block;background:#fff;border:1px solid #edf0f5;border-radius:6px;cursor:zoom-in}
.chart:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
.grid{stroke:var(--grid);stroke-width:1}
.xgrid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.line{fill:none;stroke-width:2}
.hit{fill:transparent;stroke:transparent;stroke-width:0;pointer-events:all;cursor:crosshair}
.hit:hover{fill:#fff;stroke:var(--series-color);stroke-width:3}
.peak{stroke-width:2;stroke:#fff}
.ylab,.xlab,.alab,.axis-title{font-size:11px;fill:var(--muted)}
.ring-aside{min-width:0;display:grid;gap:10px;align-content:start}
.aside-section{border:1px solid #edf0f5;border-radius:6px;background:#fbfcfe;padding:10px}
.aside-title{margin:0 0 7px;color:var(--muted);font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.05em}
.aside-row{display:flex;justify-content:space-between;gap:10px;padding:7px 0;border-bottom:1px solid #edf0f5;font-size:13px}
.aside-row:last-child{border-bottom:0;padding-bottom:0}
.aside-row span{color:var(--muted)}
.aside-row strong{text-align:right;min-width:0;overflow:hidden;text-overflow:ellipsis}
.legend{list-style:none;margin:0;padding:0;display:grid;gap:8px}
.legend li{display:grid;grid-template-columns:auto 1fr;gap:3px 7px;align-items:center;font-size:12px}
.legend-main{font-weight:700}
.legend-note{grid-column:2;color:var(--muted);white-space:nowrap}
body.modal-open{overflow:hidden}
.chart-modal[hidden]{display:none}
.chart-modal{position:fixed;inset:0;z-index:50;display:grid;place-items:center;padding:24px}
.chart-modal-backdrop{position:absolute;inset:0;background:rgba(15,23,42,.62)}
.chart-modal-panel{position:relative;z-index:1;width:min(96vw,1280px);max-height:92vh;background:#fff;border:1px solid var(--border);border-radius:8px;box-shadow:0 24px 72px rgba(15,23,42,.25);padding:14px}
.chart-modal-close{position:absolute;top:9px;right:9px;width:30px;height:30px;border:1px solid var(--border);border-radius:6px;background:#fff;color:var(--text);font-size:18px;font-weight:750;line-height:1;cursor:pointer}
.chart-modal-close:hover{background:#f2f4f7}
.chart-modal-content{max-height:calc(92vh - 28px);overflow:auto;padding-top:28px}
.chart-modal-content .chart{width:min(1200px,100%);margin:0 auto;cursor:crosshair}
@media (max-width:900px){
  .ring-body{grid-template-columns:1fr}
  .formula-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
}
@media (max-width:560px){
  .page{padding:14px}
  .formula-grid{grid-template-columns:1fr}
  .dominant-site{display:block}
  .dominant-site strong{display:block;margin-top:3px}
  .ring-panel summary{align-items:flex-start;flex-wrap:wrap}
  .chart-modal{padding:12px}
  .chart-modal-panel{width:100%;max-height:94vh;padding:10px}
  .chart-modal-content{max-height:calc(94vh - 22px)}
}
"""


_CHART_MODAL = """
<div id="chart-modal" class="chart-modal" hidden>
  <div class="chart-modal-backdrop" data-close-chart></div>
  <div class="chart-modal-panel" role="dialog" aria-modal="true" aria-label="Expanded chart">
    <button type="button" class="chart-modal-close" data-close-chart aria-label="Close expanded chart">x</button>
    <div id="chart-modal-content" class="chart-modal-content"></div>
  </div>
</div>
"""


_SCRIPT = """
(function () {
  var modal = document.getElementById("chart-modal");
  var content = document.getElementById("chart-modal-content");
  var closeButton = modal ? modal.querySelector(".chart-modal-close") : null;
  var lastFocus = null;

  if (!modal || !content || !closeButton) {
    return;
  }

  function focusElement(element) {
    try {
      element.focus({preventScroll: true});
    } catch (err) {
      element.focus();
    }
  }

  function openChart(chart) {
    lastFocus = document.activeElement;
    content.replaceChildren();
    var copy = chart.cloneNode(true);
    copy.classList.add("chart-expanded");
    copy.removeAttribute("tabindex");
    copy.removeAttribute("role");
    copy.removeAttribute("aria-label");
    content.appendChild(copy);
    modal.hidden = false;
    document.body.classList.add("modal-open");
    focusElement(closeButton);
  }

  function closeChart() {
    modal.hidden = true;
    content.replaceChildren();
    document.body.classList.remove("modal-open");
    if (lastFocus && typeof lastFocus.focus === "function") {
      focusElement(lastFocus);
    }
  }

  document.addEventListener("click", function (event) {
    var target = event.target;
    if (!(target instanceof Element)) {
      return;
    }
    if (target.closest("[data-close-chart]")) {
      closeChart();
      return;
    }
    if (target.closest("#chart-modal")) {
      return;
    }
    var chart = target.closest(".chart");
    if (chart) {
      openChart(chart);
    }
  });

  document.addEventListener("keydown", function (event) {
    var target = event.target;
    if (event.key === "Escape" && !modal.hidden) {
      closeChart();
      return;
    }
    if (!(target instanceof Element) || target.closest("#chart-modal")) {
      return;
    }
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    var chart = target.closest(".chart");
    if (chart) {
      event.preventDefault();
      openChart(chart);
    }
  });
})();
"""


def process(jsonl_path: Path, out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or jsonl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, records = _load(jsonl_path)
    pairs_by_ring = _pair_by_ring(records)

    body = [_summary_html(meta, pairs_by_ring, jsonl_path.name)]
    for resource in RESOURCES:
        section = _resource_section(resource, meta, pairs_by_ring)
        if section:
            body.append(section)

    out_path = out_dir / "scope_stats.html"
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>scope_stats</title><style>{_STYLE}</style></head>"
        f'<body><main class="page">{"".join(body)}</main>{_CHART_MODAL}<script>{_SCRIPT}</script></body></html>'
    )
    out_path.write_text(html)
    return [out_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="path to scope_stats.jsonl")
    parser.add_argument("--out-dir", type=Path, default=None, help="output dir for the HTML (default: alongside input)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    written = process(args.jsonl, args.out_dir)
    if written:
        for path in written:
            logger.info("wrote %s", path)
    else:
        logger.warning("no begin/end pairs found in %s", args.jsonl)


if __name__ == "__main__":
    main()
