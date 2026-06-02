#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Post-process a scope_stats.jsonl into a single self-contained HTML report.

Records carry a ``phase``: ``begin`` / ``end`` are the scope boundaries, and
``task`` is a per-task sample taken at each ``submit_task`` (carrying the
submitted ``task_id``, encoded as ``(ring << 32) | local``). Records are grouped
by ``ring`` and two complementary views are drawn per ring/resource:

1. **Task timeline** — every record in stream order (begin → task… → end) as a
   single occupancy curve, so per-task resource growth is visible at the finest
   granularity. Markers are coloured by phase (begin = green, task = blue,
   end = red); scope boundaries are thus readable without a second overlapping
   curve.
2. **Per-scope deltas** — each ``begin`` paired with the first later ``end`` of
   the same ``site`` (cpp file:line), giving one point per scope for three
   deltas:

     - scope_high_water  = end.<res>_head - begin.<res>_tail
     - real_occupancy    = end.<res>_head - end.<res>_tail
     - scope_alloc       = end.<res>_head - begin.<res>_head

Naming: the ``begin.`` / ``end.`` prefix is the *scope's* entry/exit sample;
``_head`` / ``_tail`` are the *ring's* pointers (head = allocation frontier
heap_top, tail = released boundary heap_tail). The JSON fields are still named
``<res>_start`` (tail) and ``<res>_end`` (head).

tensormap is reported as a single in-use value (no head/tail), so its per-scope
view degenerates to a single real_occupancy curve.

Output: one self-contained ``<out_dir>/scope_stats.html`` with all rings and
resources. The SVG is inlined (no matplotlib, no external JS/CDN) so the file
opens offline and is trivial to share. Per-resource max capacities are listed
once in the report header (no per-chart capacity line); the header also carries
a legend explaining what each metric means.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# resource name -> (y-axis unit label, display divisor applied after wrap)
RESOURCES = {
    "task_window": ("slots", 1),
    "heap": ("MiB", 1024 * 1024),
    "tensormap": ("entries", 1),
}


def _metrics(resource: str):
    # tensormap is reported as a single in-use value (no start/end), so the
    # three start/end deltas degenerate to one occupancy curve.
    if resource == "tensormap":
        return [("real_occupancy  (end.tensormap)", lambda b, en: en["tensormap"])]
    # JSON fields are <res>_start / <res>_end, but they are the ring's tail /
    # head pointers — not the scope's begin/end. Labels use head/tail to make
    # the "scope.begin vs ring.tail" distinction unambiguous.
    s, e = f"{resource}_start", f"{resource}_end"
    head, tail = f"{resource}_head", f"{resource}_tail"
    return [
        (f"scope_high_water  (end.{head} - begin.{tail})", lambda b, en: en[e] - b[s]),
        (f"real_occupancy  (end.{head} - end.{tail})", lambda b, en: en[e] - en[s]),
        (f"scope_alloc  (end.{head} - begin.{head})", lambda b, en: en[e] - b[e]),
    ]


def _load(jsonl_path: Path) -> tuple[dict, list[dict]]:
    lines = jsonl_path.read_text().splitlines()
    # Line 1 is run metadata; the rest are per-scope records.
    meta = json.loads(lines[0]) if lines else {}
    records = [json.loads(line) for line in lines[1:] if line.strip()]
    return meta, records


def _resource_size(meta: dict, resource: str, ring: int) -> int | None:
    """Ring capacity used for wrap-around correction. heap/task_window are
    per-ring ring buffers (``*_max`` is a list); tensormap is a scalar cap."""
    cap = meta.get(f"{resource}_max")
    if isinstance(cap, list):
        return cap[ring] if 0 <= ring < len(cap) else None
    return cap


def _wrap(delta: int, size: int | None) -> int:
    """Ring-buffer occupancy is non-negative: a negative delta means the
    allocator wrapped past the buffer end (start > end), so fold it back by
    one buffer length — mirrors ``(end + size - start) % size``."""
    if size and delta < 0:
        return delta + size
    return delta


def _pair_by_ring(records: list[dict]) -> dict[int, list[tuple[dict, dict]]]:
    """Return ring -> [(begin, end), ...]. A begin pairs with the first later
    end of the same site within that ring (FIFO per site)."""
    pending: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    pairs: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for rec in records:
        ring, site, phase = rec["ring"], rec["site"], rec["phase"]
        if phase == "begin":
            pending[ring][site].append(rec)
        elif phase == "end":
            queue = pending[ring].get(site)
            if queue:
                pairs[ring].append((queue.pop(0), rec))
            else:
                logger.warning("ring %s site %s: end without matching begin", ring, site)
    return pairs


def _ordered_by_ring(records: list[dict]) -> dict[int, list[dict]]:
    """Return ring -> records in stream order (begin / task / end interleaved).
    Drives the task-grained timeline view."""
    by_ring: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        by_ring[rec["ring"]].append(rec)
    return by_ring


def _occupancy(rec: dict, resource: str, size: int | None) -> int:
    """Single-sample live occupancy for one record (no begin/end pairing).
    Ring resources fold wrap-around; tensormap is the in-use count directly."""
    if resource == "tensormap":
        return rec["tensormap"]
    return _wrap(rec[f"{resource}_end"] - rec[f"{resource}_start"], size)


def _task_label(rec: dict) -> str:
    """Human-readable phase tag for a timeline tooltip; task records decode the
    (ring << 32) | local task_id."""
    phase = rec["phase"]
    if phase == "task":
        tid = rec.get("task_id", 0)
        return f"task r{tid >> 32}:{tid & 0xFFFFFFFF}"
    return phase


_SVG_W = 720
_SVG_H = 160
_PAD_L = 56
_PAD_R = 16
_PAD_T = 12
_PAD_B = 28


def _esc(text: str) -> str:
    """Minimal XML escaping for text/attribute content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _svg_chart(label: str, series: list[float], sites: list[str], unit: str, fmt: str) -> str:
    """One inline-SVG line chart for a single metric series. The y-range is
    derived from the data (0..peak); per-resource max capacities are shown once
    in the report header instead of as a per-chart reference line."""
    n = len(series)
    peak = max(series) if series else 0.0
    y_max = peak if peak > 0 else 1.0
    plot_w = _SVG_W - _PAD_L - _PAD_R
    plot_h = _SVG_H - _PAD_T - _PAD_B

    def sx(i: int) -> float:
        return _PAD_L + (plot_w * i / (n - 1) if n > 1 else plot_w / 2)

    def sy(v: float) -> float:
        return _PAD_T + plot_h * (1 - v / y_max)

    parts = [f'<svg viewBox="0 0 {_SVG_W} {_SVG_H}" class="chart" preserveAspectRatio="xMidYMid meet">']
    # Axes (left + bottom) and y gridlines at 0 / y_max.
    parts.append(f'<line x1="{_PAD_L}" y1="{_PAD_T}" x2="{_PAD_L}" y2="{_PAD_T + plot_h}" class="axis"/>')
    parts.append(
        f'<line x1="{_PAD_L}" y1="{_PAD_T + plot_h}" x2="{_SVG_W - _PAD_R}" y2="{_PAD_T + plot_h}" class="axis"/>'
    )
    ymax_label = f"{fmt.format(y_max)} {_esc(unit)}"
    parts.append(f'<text x="{_PAD_L - 6}" y="{sy(y_max):.1f}" class="ylab" text-anchor="end">{ymax_label}</text>')
    parts.append(f'<text x="{_PAD_L - 6}" y="{sy(0):.1f}" class="ylab" text-anchor="end">0</text>')
    parts.append(f'<text x="{_PAD_L}" y="{_SVG_H - 8}" class="alab">scope (begin/end order)</text>')
    peak_label = f"peak {fmt.format(peak)} {_esc(unit)}"
    parts.append(f'<text x="{_PAD_L + plot_w}" y="{_SVG_H - 8}" class="alab" text-anchor="end">{peak_label}</text>')

    if n > 0:
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(series))
        parts.append(f'<polyline points="{pts}" class="line"/>')
        for i, v in enumerate(series):
            title = f"#{i} {fmt.format(v)} {unit} — {sites[i]}"
            parts.append(
                f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="2.5" class="dot"><title>{_esc(title)}</title></circle>'
            )

    parts.append("</svg>")
    return f'<figure class="metric"><figcaption>{_esc(label)}</figcaption>{"".join(parts)}</figure>'


def _svg_timeline(records: list[dict], resource: str, size: int | None, unit: str, fmt: str, divisor: int) -> str:
    """One inline-SVG occupancy curve over the ring's records in stream order.
    Markers are coloured by phase so scope boundaries (begin/end) stand out
    against the per-task samples without a second overlapping curve."""
    series = [_occupancy(r, resource, size) / divisor for r in records]
    n = len(series)
    peak = max(series) if series else 0.0
    y_max = peak if peak > 0 else 1.0
    plot_w = _SVG_W - _PAD_L - _PAD_R
    plot_h = _SVG_H - _PAD_T - _PAD_B

    def sx(i: int) -> float:
        return _PAD_L + (plot_w * i / (n - 1) if n > 1 else plot_w / 2)

    def sy(v: float) -> float:
        return _PAD_T + plot_h * (1 - v / y_max)

    parts = [f'<svg viewBox="0 0 {_SVG_W} {_SVG_H}" class="chart" preserveAspectRatio="xMidYMid meet">']
    parts.append(f'<line x1="{_PAD_L}" y1="{_PAD_T}" x2="{_PAD_L}" y2="{_PAD_T + plot_h}" class="axis"/>')
    parts.append(
        f'<line x1="{_PAD_L}" y1="{_PAD_T + plot_h}" x2="{_SVG_W - _PAD_R}" y2="{_PAD_T + plot_h}" class="axis"/>'
    )
    parts.append(
        f'<text x="{_PAD_L - 6}" y="{sy(y_max):.1f}" class="ylab" text-anchor="end">'
        f"{fmt.format(y_max)} {_esc(unit)}</text>"
    )
    parts.append(f'<text x="{_PAD_L - 6}" y="{sy(0):.1f}" class="ylab" text-anchor="end">0</text>')
    parts.append(f'<text x="{_PAD_L}" y="{_SVG_H - 8}" class="alab">task</text>')
    parts.append(
        f'<text x="{_PAD_L + plot_w}" y="{_SVG_H - 8}" class="alab" text-anchor="end">'
        f"peak {fmt.format(peak)} {_esc(unit)}</text>"
    )

    if n > 0:
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(series))
        parts.append(f'<polyline points="{pts}" class="line"/>')
        for i, (v, rec) in enumerate(zip(series, records)):
            phase = rec["phase"]
            cls = {"begin": "dot-begin", "end": "dot-end"}.get(phase, "dot-task")
            r = 3.0 if phase != "task" else 2.2
            title = f"#{i} {_task_label(rec)} {fmt.format(v)} {unit} — {rec.get('site', '?')}"
            parts.append(
                f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="{r}" class="{cls}"><title>{_esc(title)}</title></circle>'
            )

    parts.append("</svg>")
    cap = "live occupancy per record (begin=green, task=blue, end=red)"
    return f'<figure class="metric"><figcaption>{cap}</figcaption>{"".join(parts)}</figure>'


def _ring_section(
    ring: int, pairs: list[tuple[dict, dict]], ordered: list[dict], resource: str, size: int | None
) -> str:
    unit, divisor = RESOURCES[resource]
    fmt = "{:.1f}" if divisor > 1 else "{:.0f}"
    timeline = _svg_timeline(ordered, resource, size, unit, fmt, divisor)
    sites = [e.get("site", "?") for _, e in pairs]
    scope_charts = []
    for label, fn in _metrics(resource):
        series = [_wrap(fn(b, e), size) / divisor for b, e in pairs]
        scope_charts.append(_svg_chart(label, series, sites, unit, fmt))
    # Two visually distinct groups so the per-task (stream-order) view and the
    # per-scope (begin/end delta) view are not mistaken for one another.
    task_group = (
        f'<div class="group group-task">'
        f'<div class="group-hd">per-task timeline — {len(ordered)} records (stream order)</div>'
        f"{timeline}</div>"
    )
    scope_group = (
        f'<div class="group group-scope">'
        f'<div class="group-hd">per-scope deltas — {len(pairs)} scope pairs (begin/end)</div>'
        f"{''.join(scope_charts)}</div>"
    )
    return f'<section class="ring"><h3>ring {ring} — {_esc(resource)}</h3>{task_group}{scope_group}</section>'


_STYLE = """
body{font-family:system-ui,Arial,sans-serif;margin:24px;color:#222}
h1{font-size:20px} h2{font-size:16px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}
h3{font-size:14px;color:#555;margin:16px 0 4px}
.group{margin:6px 0 16px;padding:6px 10px 10px;border-radius:6px;border:1px solid #e2e2e8}
.group-task{background:#f3f8ff;border-color:#cfe0fb}
.group-scope{background:#fbf8f2;border-color:#ecdcc0}
.group-hd{font-size:12px;font-weight:600;letter-spacing:.02em;margin:2px 0 6px}
.group-task .group-hd{color:#1d4ed8}
.group-scope .group-hd{color:#b45309}
.metric{margin:0 0 8px;display:inline-block;width:740px;vertical-align:top;
background:#fff;border-radius:4px;padding:4px}
figcaption{font-size:12px;color:#333;margin-bottom:2px}
.chart{width:100%;height:auto;border:1px solid #eee;background:#fff}
.axis{stroke:#999;stroke-width:1}
.line{fill:none;stroke:#2563eb;stroke-width:1.4}
.dot{fill:#2563eb}
.dot-task{fill:#2563eb}
.dot-begin{fill:#16a34a}
.dot-end{fill:#dc2626}
.ylab,.alab{font-size:10px;fill:#666}
.intro{background:#f7f7f9;border:1px solid #e2e2e8;border-radius:6px;
padding:12px 16px;margin:8px 0 20px;max-width:760px}
.intro p{margin:4px 0;font-size:13px;line-height:1.5}
.intro code{background:#ececf1;padding:1px 4px;border-radius:3px;font-size:12px}
.intro .name{font-weight:600;color:#1d4ed8}
.intro .caps{margin-top:8px;padding-top:8px;border-top:1px solid #e2e2e8}
.intro .caps b{color:#dc2626}
"""


def _capacity_summary(meta: dict) -> str:
    """One-line-per-resource max-capacity overview shown in the report header."""
    rows = []
    for resource, (unit, divisor) in RESOURCES.items():
        cap = meta.get(f"{resource}_max")
        fmt = "{:.1f}" if divisor > 1 else "{:.0f}"
        if isinstance(cap, list):
            vals = sorted(set(cap))
            if len(vals) == 1:
                txt = f"{fmt.format(vals[0] / divisor)} {unit} (all rings)"
            else:
                txt = ", ".join(f"ring{i}={fmt.format(v / divisor)}" for i, v in enumerate(cap)) + f" {unit}"
        elif cap is not None:
            txt = f"{fmt.format(cap / divisor)} {unit}"
        else:
            txt = "n/a"
        rows.append(f"<p><span class='name'>{_esc(resource)}</span> max capacity: <b>{_esc(txt)}</b></p>")
    return f"<div class='caps'>{''.join(rows)}</div>"


def _intro(meta: dict) -> str:
    return f"""
<div class="intro">
<p>Each scope samples resource usage once on entry (<code>begin</code>) and once
on exit (<code>end</code>). Naming: the <code>begin.</code> / <code>end.</code>
prefix is the <b>scope's</b> entry/exit sample; the <code>_head</code> /
<code>_tail</code> suffix is the <b>ring's</b> pointer
(head = allocation frontier heap_top, tail = released boundary heap_tail).
For ring-buffer resources, occupancy uses the wrap formula
<code>(head + cap - tail) % cap</code> (<code>tail &gt; head</code> means it
wrapped past the end). The y-range is 0..peak; per-resource max capacities are
listed below instead of as a per-chart line.</p>
<p><span class="name">scope_high_water</span> = <code>end.head - begin.tail</code> —
the highest occupancy reached during the scope's lifetime: residual not yet
released on entry + what this scope added; the actual peak it held.</p>
<p><span class="name">real_occupancy</span> = <code>end.head - end.tail</code> —
what is genuinely still occupied at the exit instant (the ring occupancy that
<code>heap_used_bytes()</code> returns); the live level on leaving.</p>
<p><span class="name">scope_alloc</span> = <code>end.head - begin.head</code> —
how far the allocation frontier (head) advanced, i.e. this scope's own net
(unreleased) allocation.</p>
<p>tensormap reports a single in-use value (no head/tail), so it shows just one
real_occupancy curve.</p>
<p>Each ring is split into two groups: a blue
<span class="name">per-task timeline</span> and an amber
<span class="name">per-scope deltas</span> block, so the stream-order view and
the begin/end-delta view are not mistaken for one another. The timeline is live
occupancy sampled at every record in stream order, so per-task growth is
visible at the finest granularity. Markers are coloured by phase —
<b style="color:#16a34a">begin</b>, <b style="color:#2563eb">task</b>,
<b style="color:#dc2626">end</b> — and a task point's tooltip decodes its
<code>(ring &lt;&lt; 32) | local</code> task_id.</p>
{_capacity_summary(meta)}
</div>
"""


def process(jsonl_path: Path, out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or jsonl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, records = _load(jsonl_path)
    pairs_by_ring = _pair_by_ring(records)
    ordered_by_ring = _ordered_by_ring(records)

    body = [f"<h1>scope_stats — {_esc(jsonl_path.name)}</h1>", _intro(meta)]
    for resource in RESOURCES:
        sections = []
        for ring in sorted(ordered_by_ring):
            ordered = ordered_by_ring[ring]
            if not ordered:
                continue
            pairs = pairs_by_ring.get(ring, [])
            sections.append(_ring_section(ring, pairs, ordered, resource, _resource_size(meta, resource, ring)))
        if sections:
            body.append(f"<h2>{_esc(resource)}</h2>{''.join(sections)}")

    out_path = out_dir / "scope_stats.html"
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>scope_stats</title><style>{_STYLE}</style></head>"
        f"<body>{''.join(body)}</body></html>"
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
