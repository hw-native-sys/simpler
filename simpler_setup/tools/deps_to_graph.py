#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
deps.json → pan-zoom HTML dependency-graph viewer.

deps.json is the host-replay-rebuilt task graph (one edge per producer→consumer
pair, complete across the full submit_task trace — superset of fanout). This
tool runs the graph through Graphviz to produce an SVG and wraps it in a
self-contained HTML page with a minimal drag-to-pan + wheel-to-zoom shim
(no JS dependency, no CDN, opens offline in any browser).

The HTML format scales to any graph the browser's SVG renderer can handle
(practically up to ~50k nodes when paired with ``--engine sfdp``); the only
gotcha is that high zoom slightly blurs text — that's a CSS-transform tradeoff
in exchange for 60fps GPU-composited pan/zoom even on huge graphs.

When ``l2_perf_records.json`` is colocated with ``deps.json``, node labels are
enriched with the per-task ``func_id`` and ``core_type`` so a node reads as
``t12 · kernel_mul · aiv`` rather than just ``t12``; nodes are colored by
core_type (AIC blue, AIV orange).

Usage:
    python -m simpler_setup.tools.deps_to_graph DEPS_JSON [-o OUT] [--engine ENGINE]
    python -m simpler_setup.tools.deps_to_graph                          # newest under outputs/
    python -m simpler_setup.tools.deps_to_graph deps.json
    python -m simpler_setup.tools.deps_to_graph deps.json --engine sfdp  # force-directed (for big graphs)

Requires Graphviz installed (``brew install graphviz`` / ``apt install graphviz``).
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _normalize_task_id(v):
    """Unsigned 64-bit task id (matches deps.json edges and l2_perf task_id)."""
    try:
        t = int(v)
    except (TypeError, ValueError):
        return None
    if t < 0:
        t &= (1 << 64) - 1
    return t


def _node_id(task_id):
    """DOT-safe node id: ``T{ring}_{local}``."""
    tid = _normalize_task_id(task_id)
    if tid is None:
        return f"T_{task_id}"
    ring = (tid >> 32) & 0xFF
    local = tid & 0xFFFFFFFF
    return f"T{ring}_{local}"


def _make_task_formatter(nodes):
    """Build a task-id → display-string formatter sized to the graph.

    If every node lives in ring 0 the display is just ``{local}`` (the local
    counter alone — no noise on workloads that never enter a manual scope).
    The moment any node is in ring ≥ 1 we switch to the explicit
    ``({ring}, {local})`` tuple for *every* node so the asymmetry is visible
    instead of hidden (you can't have ``t0`` next to ``r1t3`` and know which
    ring t0 lives in without context).
    """
    has_multi_ring = False
    for n in nodes:
        tid = _normalize_task_id(n)
        if tid is None:
            continue
        if (tid >> 32) & 0xFF != 0:
            has_multi_ring = True
            break

    def fmt(task_id):
        tid = _normalize_task_id(task_id)
        if tid is None:
            return str(task_id)
        ring = (tid >> 32) & 0xFF
        local = tid & 0xFFFFFFFF
        if has_multi_ring:
            return f"({ring}, {local})"
        return str(local)

    return fmt


def _load_deps_edges(deps_path):
    """Returns (sorted list of unique edges, sorted list of all node ids)."""
    with open(deps_path) as f:
        data = json.load(f)
    if data.get("version") != 1:
        print(f"Warning: deps.json version={data.get('version')} (expected 1)", file=sys.stderr)
    edges_raw = data.get("edges", [])
    seen = set()
    edges = []
    nodes = set()
    for e in edges_raw:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            continue
        pred = _normalize_task_id(e[0])
        succ = _normalize_task_id(e[1])
        if pred is None or succ is None:
            continue
        nodes.add(pred)
        nodes.add(succ)
        key = (pred, succ)
        if key in seen:
            continue
        seen.add(key)
        edges.append(key)
    return edges, sorted(nodes)


def _load_task_meta(deps_path, func_names=None):
    """Optional l2_perf_records.json sidecar → {task_id: {'func_id', 'core_type', ...}}.

    Mixed-kernel tasks (single submit_task that spans both AIC and AIV blocks)
    appear as multiple perf-record entries with the same ``task_id`` but
    different ``core_id`` / ``core_type``. We aggregate per ``task_id``: when
    multiple distinct ``core_type`` values are seen, the task's ``core_type``
    collapses to the sentinel ``"mix"`` (which the legend / styling table maps
    to a diamond). ``func_id`` follows the AIC entry when present, otherwise
    the first entry — mixed tasks usually have one "primary" function id.

    Returns {} if no sidecar present. ``func_names`` (optional dict) overrides
    the default ``f{func_id}`` label with a human name.
    """
    perf_path = Path(deps_path).parent / "l2_perf_records.json"
    if not perf_path.exists():
        return {}
    try:
        with perf_path.open() as f:
            perf = json.load(f)
    except (OSError, ValueError) as e:
        print(f"Warning: couldn't read {perf_path}: {e}", file=sys.stderr)
        return {}

    # First pass: collect every (core_type, func_id) tuple per task_id.
    by_tid: dict[int, list[dict]] = {}
    for task in perf.get("tasks", []):
        tid = _normalize_task_id(task.get("task_id"))
        if tid is None:
            continue
        by_tid.setdefault(tid, []).append(task)

    meta: dict[int, dict] = {}
    for tid, entries in by_tid.items():
        core_types = {e.get("core_type") for e in entries if e.get("core_type")}
        if len(core_types) > 1:
            core_type = "mix"
            # Prefer the AIC entry's func_id (cube usually carries the "main"
            # op in mixed kernels) so the label reads sensibly.
            primary = next((e for e in entries if e.get("core_type") == "aic"), entries[0])
        else:
            core_type = next(iter(core_types), None)
            primary = entries[0]
        func_id = primary.get("func_id")
        func_name = None
        if func_names and func_id is not None:
            func_name = func_names.get(str(func_id)) or func_names.get(func_id)
        meta[tid] = {
            "func_id": func_id,
            "func_name": func_name,
            "core_type": core_type,
            "core_id": primary.get("core_id"),
            "duration_us": primary.get("duration_us"),
        }
    return meta


def _label(task_id, meta, fmt_task, have_perf=False):
    base = fmt_task(task_id)
    m = meta.get(_normalize_task_id(task_id))
    if not m:
        # Node referenced by deps but absent from the perf sidecar: alloc_tensors
        # task (see emit_dot docstring). Surface that in the label so the user
        # knows the dashed-note style isn't just "missing data". When no perf
        # sidecar exists at all, we can't tell — fall back to bare id.
        return f"{base} · alloc" if have_perf else base
    parts = [base]
    fn = m.get("func_name") or (f"f{m['func_id']}" if m.get("func_id") is not None else None)
    if fn:
        parts.append(fn)
    # core_type is intentionally NOT in the label — node shape (box/ellipse) and
    # color already encode it, and the HUD legend at the top-right of the page
    # spells out the mapping. Keeps labels short on dense graphs.
    return " · ".join(parts)


# Per-core-type visual styling. AIC (cube unit) is a box; AIV (vector unit)
# is an ellipse; "mix" (single submit_task spanning both core types) is a
# diamond; "alloc" — a task that came from ``alloc_tensors`` (got a real
# task_id and shows up as a producer in deps via ``owner_task_id``, but
# never dispatched a kernel so no l2_perf record and no func_id) — is a
# dashed gray note. Distinct shape AND color so each stays readable even
# without color (B&W print, accessibility, etc.).
_CORE_STYLE = {
    "aic": {"shape": "box", "style": "rounded,filled", "fillcolor": "#66A3FF"},
    "aiv": {"shape": "ellipse", "style": "filled", "fillcolor": "#FFB366"},
    "mix": {"shape": "diamond", "style": "filled", "fillcolor": "#66CC99"},
    "alloc": {"shape": "note", "style": "filled,dashed", "fillcolor": "#EAEAEA"},
}
_DEFAULT_STYLE = {"shape": "box", "style": "rounded,filled", "fillcolor": "#E0E0E0"}


def _node_style(core_type):
    return _CORE_STYLE.get(core_type, _DEFAULT_STYLE)


def emit_dot(edges, nodes, meta, direction="LR"):
    """Graphviz DOT source. Used internally to feed the layout engine before
    wrapping the SVG in HTML.

    Nodes that appear as edge endpoints but are absent from the perf-records
    sidecar are tagged as ``alloc``: in practice these are tasks created by
    ``alloc_tensors`` (which gets a real task_id and is referenced via
    ``owner_task_id`` from downstream consumers, but never dispatches a
    kernel and therefore has no perf entry). When no perf sidecar exists at
    all (``meta`` empty) we can't disambiguate and fall back to the default
    style for every node.
    """
    fmt_task = _make_task_formatter(nodes)
    have_perf = bool(meta)
    lines = [
        "digraph deps {",
        f"  rankdir={direction};",
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [color="#888"];',
    ]
    for n in nodes:
        m = meta.get(n)
        if m:
            style = _node_style(m.get("core_type"))
        elif have_perf:
            style = _CORE_STYLE["alloc"]
        else:
            style = _DEFAULT_STYLE
        # Escape any backslash / double-quote in the label.
        label = _label(n, meta, fmt_task, have_perf=have_perf).replace("\\", "\\\\").replace('"', '\\"')
        attrs = f'label="{label}", shape={style["shape"]}, style="{style["style"]}", fillcolor="{style["fillcolor"]}"'
        lines.append(f"  {_node_id(n)} [{attrs}];")
    for pred, succ in edges:
        lines.append(f"  {_node_id(pred)} -> {_node_id(succ)};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_svg(dot_text, engine="dot"):
    """Pipe DOT through the Graphviz layout engine and return raw SVG bytes.

    Raises FileNotFoundError if the engine binary is not on PATH; RuntimeError
    if the engine returns a non-zero exit code.
    """
    if shutil.which(engine) is None:
        raise FileNotFoundError(
            f"Graphviz '{engine}' not found on PATH. Install graphviz: brew install graphviz / apt install graphviz"
        )
    proc = subprocess.run(
        [engine, "-Tsvg"],
        input=dot_text.encode(),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace")
        raise RuntimeError(f"{engine} -Tsvg failed (exit {proc.returncode}):\n{msg}")
    return proc.stdout


# Self-contained HTML wrapper. Drag-to-pan + wheel-to-zoom in vanilla JS.
# The SVG is inlined so the file is single-page and works offline. We use CSS
# transform (translate + scale) rather than mutating the SVG viewBox — that
# keeps pan/zoom on the GPU compositor for 60fps performance on large graphs.
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>deps.json — {n_nodes} nodes, {n_edges} edges</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #fafafa; font-family: -apple-system, sans-serif; }}
  #hud {{ position: fixed; top: 8px; left: 8px; background: #fff; padding: 6px 10px;
         border: 1px solid #ddd; border-radius: 4px; font-size: 12px; z-index: 10;
         box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  #hud kbd {{ font-family: ui-monospace, monospace; background: #f0f0f0;
              padding: 1px 4px; border-radius: 2px; }}
  #legend {{ position: fixed; top: 8px; right: 8px; background: #fff; padding: 6px 10px;
             border: 1px solid #ddd; border-radius: 4px; font-size: 12px; z-index: 10;
             box-shadow: 0 1px 3px rgba(0,0,0,0.08); display: flex; gap: 12px; align-items: center; }}
  #legend .swatch {{ display: inline-flex; align-items: center; gap: 4px; }}
  #legend svg {{ display: block; }}
  #stage {{ width: 100vw; height: 100vh; overflow: hidden; cursor: grab; }}
  #stage.panning {{ cursor: grabbing; }}
  #stage > svg {{ transform-origin: 0 0; transition: none; max-width: none; }}
</style>
</head>
<body>
<div id="hud">
  {n_nodes} nodes · {n_edges} edges &nbsp;|&nbsp;
  <kbd>drag</kbd> pan &nbsp; <kbd>wheel</kbd> zoom &nbsp; <kbd>f</kbd> fit &nbsp; <kbd>r</kbd> reset
</div>
<div id="legend">
  <span class="swatch">
    <svg width="18" height="14" viewBox="0 0 18 14">
      <rect x="1" y="2" width="16" height="10" rx="3" ry="3" fill="#66A3FF" stroke="#333" stroke-width="1"/>
    </svg>
    AIC (cube)
  </span>
  <span class="swatch">
    <svg width="18" height="14" viewBox="0 0 18 14">
      <ellipse cx="9" cy="7" rx="8" ry="5" fill="#FFB366" stroke="#333" stroke-width="1"/>
    </svg>
    AIV (vector)
  </span>
  <span class="swatch">
    <svg width="18" height="14" viewBox="0 0 18 14">
      <polygon points="9,1 17,7 9,13 1,7" fill="#66CC99" stroke="#333" stroke-width="1"/>
    </svg>
    mix
  </span>
  <span class="swatch">
    <svg width="18" height="14" viewBox="0 0 18 14">
      <path d="M2,2 L13,2 L16,5 L16,12 L2,12 Z" fill="#EAEAEA" stroke="#333" stroke-width="1" stroke-dasharray="2,1"/>
    </svg>
    alloc
  </span>
</div>
<div id="stage">
{svg_body}
</div>
<script>
(function () {{
  const stage = document.getElementById('stage');
  const svg = stage.querySelector('svg');
  if (!svg) return;
  // Drop fixed width/height so the SVG renders at its natural size; we control
  // scale via CSS transform.
  svg.removeAttribute('width');
  svg.removeAttribute('height');

  let scale = 1, tx = 0, ty = 0;
  const apply = () => {{ svg.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(${{scale}})`; }};

  // Wheel zoom about cursor.
  stage.addEventListener('wheel', (e) => {{
    e.preventDefault();
    const rect = stage.getBoundingClientRect();
    const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    const factor = Math.exp(-e.deltaY * 0.001);
    const newScale = Math.min(20, Math.max(0.02, scale * factor));
    // Keep the point under the cursor fixed in screen space.
    tx = cx - (cx - tx) * (newScale / scale);
    ty = cy - (cy - ty) * (newScale / scale);
    scale = newScale;
    apply();
  }}, {{ passive: false }});

  // Drag to pan.
  let dragging = false, lastX = 0, lastY = 0;
  stage.addEventListener('mousedown', (e) => {{
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    stage.classList.add('panning');
  }});
  window.addEventListener('mousemove', (e) => {{
    if (!dragging) return;
    tx += e.clientX - lastX; ty += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    apply();
  }});
  window.addEventListener('mouseup', () => {{ dragging = false; stage.classList.remove('panning'); }});

  // 'f' = fit-to-view, 'r' = reset to 1:1.
  const fit = () => {{
    const bb = svg.getBoundingClientRect();
    // bb is in current-transform coordinates, so undo the transform first.
    const naturalW = bb.width / scale, naturalH = bb.height / scale;
    const sx = stage.clientWidth / naturalW, sy = stage.clientHeight / naturalH;
    scale = Math.min(sx, sy) * 0.95;
    tx = (stage.clientWidth - naturalW * scale) / 2;
    ty = (stage.clientHeight - naturalH * scale) / 2;
    apply();
  }};
  document.addEventListener('keydown', (e) => {{
    if (e.key === 'f') fit();
    else if (e.key === 'r') {{ scale = 1; tx = 0; ty = 0; apply(); }}
  }});
  // Auto-fit on first paint.
  requestAnimationFrame(fit);
}})();
</script>
</body>
</html>
"""


def emit_html(edges, nodes, meta, direction="LR", engine="dot"):
    """Build the pan/zoom HTML page: DOT → Graphviz SVG → inline into template."""
    dot = emit_dot(edges, nodes, meta, direction=direction)
    svg_bytes = render_svg(dot, engine=engine)
    svg_text = svg_bytes.decode("utf-8", errors="replace")
    # Strip the XML prolog / DOCTYPE that graphviz emits — inlining keeps the
    # <svg> root only.
    if "<svg" in svg_text:
        svg_text = svg_text[svg_text.index("<svg") :]
    return _HTML_TEMPLATE.format(
        n_nodes=len(nodes),
        n_edges=len(edges),
        svg_body=svg_text,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _find_latest_deps_json():
    outputs = Path("outputs")
    if not outputs.is_dir():
        return None
    candidates = sorted(outputs.rglob("deps.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _load_func_names_json(path):
    """Load func_id → name mapping from a JSON file (``callable_id_to_name``
    shape, as written by ``scene_test._dump_name_map``, or a flat dict).
    Returns {} on failure.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print(f"Warning: couldn't read {path}: {e}", file=sys.stderr)
        return {}
    return data.get("callable_id_to_name") or data


def _autoload_name_map(deps_path):
    """Look for a ``name_map_*.json`` next to deps.json (written by
    ``scene_test._dump_name_map`` when the case ran with diagnostics on).
    Returns {} if no candidate exists; the newest match wins on ties.
    """
    candidates = sorted(Path(deps_path).parent.glob("name_map_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return {}
    return _load_func_names_json(candidates[-1])


def _build_parser():
    p = argparse.ArgumentParser(
        description="Render deps.json as a pan/zoom HTML dependency graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s                                    # newest deps.json under ./outputs/
  %(prog)s outputs/.../deps.json
  %(prog)s deps.json -o my_graph.html
  %(prog)s deps.json --engine sfdp            # force-directed (recommended >1000 nodes)
  %(prog)s deps.json --func-names names.json  # label nodes with kernel names
""",
    )
    p.add_argument("input", nargs="?", help="Path to deps.json (default: newest under ./outputs/).")
    p.add_argument("-o", "--output", help="Output HTML path (default: same dir as input, deps_graph.html).")
    p.add_argument(
        "--engine",
        choices=["dot", "neato", "sfdp", "fdp", "circo", "twopi"],
        default="dot",
        help="Graphviz layout engine. 'dot' for hierarchical (<500 nodes), 'sfdp' for force-directed (10k+ nodes).",
    )
    p.add_argument(
        "--direction",
        choices=["TB", "LR", "BT", "RL"],
        default="LR",
        help="Flow direction for hierarchical layouts (default: LR; ignored by sfdp/neato).",
    )
    p.add_argument(
        "--func-names",
        help="JSON with callable_id_to_name (or flat {func_id: name}) for label enrichment.",
    )
    return p


def main():
    args = _build_parser().parse_args()

    input_path = args.input or _find_latest_deps_json()
    if input_path is None:
        print("No deps.json given and no candidate found under ./outputs/.", file=sys.stderr)
        return 1
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"{input_path} not found.", file=sys.stderr)
        return 1

    edges, nodes = _load_deps_edges(input_path)
    # Explicit --func-names wins over the auto-discovered sibling file.
    func_names = _load_func_names_json(args.func_names) if args.func_names else _autoload_name_map(input_path)
    meta = _load_task_meta(input_path, func_names=func_names)
    # Some workloads (embarrassingly parallel kernels, scope-fenced iterations
    # that never reuse a tensor) produce no deps edges. Without the perf-records
    # union the graph would be empty even though tasks ran. Seed isolated nodes
    # from meta so every recorded task shows up, with its mix/aic/aiv shape.
    if meta:
        nodes = sorted(set(nodes) | set(meta.keys()))

    html = emit_html(edges, nodes, meta, direction=args.direction, engine=args.engine)

    out = Path(args.output) if args.output else input_path.parent / "deps_graph.html"
    out.write_text(html)
    print(f"Wrote {out} ({len(nodes)} nodes, {len(edges)} edges, engine={args.engine})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
