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
deps.json → text or pan-zoom HTML dependency-graph viewer.

deps.json is the host-replay-rebuilt task graph (one edge per producer→consumer
pair, complete across the full submit_task trace — superset of fanout). This
tool supports two output formats:

- text (default): a grep-friendly task index plus per-task predecessor /
  successor detail blocks
- html: Graphviz SVG wrapped in a self-contained pan/zoom HTML page

In text mode the default view shows task identity ``(ring, local)`` together
with ``kind=``, ``func_id=``, and SPMD block num when applicable. Text-mode
``func_id=`` comes only from the three-slot ``kernel_ids`` layout and preserves
``[aic,aiv0,aiv1]`` with inactive slots kept as ``-1``; ``alloc`` / ``dummy``
tasks render as ``func_id=none``. ``SPMD block num = N`` is the logical block
num captured from ``block_num``. In HTML mode SPMD nodes use a red border plus a
transparent right-side ``xN`` block num label; nodes are colored by
``core_type`` when a perf sidecar is colocated (AIC blue, AIV orange).

Usage:
    python -m simpler_setup.tools.deps_viewer DEPS_JSON
    python -m simpler_setup.tools.deps_viewer DEPS_JSON --format text
    python -m simpler_setup.tools.deps_viewer DEPS_JSON --format html --engine sfdp
    python -m simpler_setup.tools.deps_viewer DEPS_JSON --edge-mode reduced
    python -m simpler_setup.tools.deps_viewer DEPS_JSON --edge-mode omitted

``--edge-mode`` selects which edges are visible (structural transitive reduction,
purely on ``(pred, succ)`` — per-edge tensor identity is ignored, and it is
skipped with a warning if the graph contains a cycle):

- ``full`` (default) — every edge.
- ``reduced`` — the minimal edge set: drops every edge whose ordering is already
  implied by a longer path (e.g. ``A->C`` when ``A->B->C`` exists).
- ``omitted`` — only the redundant edges ``reduced`` would drop (its complement),
  useful for auditing exactly which dependencies are transitively covered.

``reduced`` and ``omitted`` print the redundant edges to stdout. Text output
emits only the selected edge set. HTML output keeps every edge in the Graphviz
layout and colors the unselected edges like the page background, preserving the
full-graph layout while drawing the selected edge set above unselected edges.

HTML output requires Graphviz installed (``brew install graphviz`` /
``apt install graphviz``). Text output does not.
"""

import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path


def _normalize_task_id(v):
    """Unsigned 64-bit task id (matches deps.json edges and l2_swimlane task_id).

    Accepts ints (legacy) and strings (current schema): deps.json emits all
    uint64 fields as quoted strings to dodge JSON-number precision loss in
    JavaScript-based consumers, since tensor_ids (FNV hashes) and buffer
    addresses routinely exceed Number.MAX_SAFE_INTEGER (2^53 - 1)."""
    try:
        t = int(v)
    except (TypeError, ValueError):
        return None
    if t < 0:
        t &= (1 << 64) - 1
    return t


# Same coercion semantics — alias so the call sites read as "this is a
# tensor_id, not a task_id". Both encode 64-bit unsigned values as JSON strings.
_normalize_tensor_id = _normalize_task_id


def _normalize_small_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


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


def _sort_task_id_key(v):
    tid = _normalize_task_id(v)
    if tid is None:
        return (1, str(v))
    return (0, tid)


def _load_deps_edges(deps_path):
    """Parse deps.json into renderer-friendly pieces.

    Returns a 5-tuple:
        edges: sorted list of unique (pred, succ) pairs — what the graph
            renders as arrows. Multiple annotated edges sharing the same
            (pred, succ) (distinct arg / source / slice) collapse to one
            arrow here.
        nodes: sorted list of all referenced task ids.
        annotations: dict[(pred, succ) -> list[dict]] of annotation rows
            (one per annotated edge), keyed in insertion order so HTML edge
            rendering can resolve per-edge tensor identities and target the
            right input port on the consumer node.
        tensor_table: dict[tensor_id -> dict] from the tensors[] block.
        task_table: dict[task_id -> dict] from the tasks[] block,
            carrying the per-arg input/output slot info that the HTML view
            renders as compartments inside each task node.
    """
    with open(deps_path) as f:
        data = json.load(f)
    edges_raw = data.get("edges", [])
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    nodes: set[int] = set()
    annotations: dict[tuple[int, int], list[dict]] = {}
    for e in edges_raw:
        if not isinstance(e, dict):
            continue
        pred = _normalize_task_id(e.get("pred"))
        succ = _normalize_task_id(e.get("succ"))
        if pred is None or succ is None:
            continue
        nodes.add(pred)
        nodes.add(succ)
        key = (pred, succ)
        if key not in seen:
            seen.add(key)
            edges.append(key)
        edge_copy = dict(e)
        if "tensor_id" in edge_copy:
            edge_copy["tensor_id"] = _normalize_tensor_id(edge_copy["tensor_id"])
        annotations.setdefault(key, []).append(edge_copy)
    tensors_raw = data.get("tensors", []) if isinstance(data.get("tensors"), list) else []
    tensor_table: dict[int, dict] = {}
    for ord_idx, t in enumerate(tensors_raw):
        if not isinstance(t, dict):
            continue
        tid = _normalize_tensor_id(t.get("tensor_id"))
        if tid is not None:
            enriched = dict(t)
            enriched["tensor_id"] = tid
            enriched["name"] = f"T{ord_idx}"
            tensor_table[tid] = enriched
    tasks_raw = data.get("tasks", []) if isinstance(data.get("tasks"), list) else []
    task_table: dict[int, dict] = {}
    for t in tasks_raw:
        if not isinstance(t, dict):
            continue
        tid = _normalize_task_id(t.get("task_id"))
        if tid is not None:
            entry = dict(t)
            args_copy = []
            for a in t.get("args", []):
                if not isinstance(a, dict):
                    continue
                a_copy = dict(a)
                if "tensor_id" in a_copy:
                    a_copy["tensor_id"] = _normalize_tensor_id(a_copy["tensor_id"])
                args_copy.append(a_copy)
            entry["args"] = args_copy
            task_table[tid] = entry
    _backfill_output_tensor_ids(task_table, annotations)
    return sorted(edges), sorted(nodes), annotations, tensor_table, task_table


def _transitive_reduction(edges, nodes):
    """DAG transitive reduction on structural ``(pred, succ)`` edges.

    An edge ``(u, v)`` is redundant when ``v`` is still reachable from ``u``
    through some other path that does not use the direct ``(u, v)`` edge — i.e.
    the dependency it expresses is already implied by a longer chain. Reduction
    is purely structural: it ignores the per-edge tensor / arg annotations, so a
    ``(u, v)`` carrying its own tensor is dropped whenever the ordering it
    encodes is transitively covered.

    Runs in ``O(V·E)``: nodes are visited in reverse topological order while a
    descendant-reachability set is accumulated per node, so each edge is tested
    with a single set membership rather than a fresh graph walk. Assumes every
    edge endpoint is present in ``nodes`` (the caller's contract).

    Returns ``(kept_edges, removed_edges, is_dag)``. ``kept_edges`` and
    ``removed_edges`` preserve the order of the input ``edges`` (so passing the
    sorted list from ``_load_deps_edges`` yields sorted results without a second
    sort). On a cyclic graph the input edges are returned unchanged with
    ``is_dag=False`` (transitive reduction is only well-defined on a DAG); the
    caller warns and emits the full graph.
    """
    succ: dict[int, set[int]] = {n: set() for n in nodes}
    indeg: dict[int, int] = {n: 0 for n in nodes}
    for u, v in edges:
        if v not in succ[u]:
            succ[u].add(v)
            indeg[v] += 1

    # Kahn topological sort doubles as the cycle check: if we can't drain every
    # node, a cycle remains and reduction would be ill-defined. Keep the drain
    # order so we can walk it in reverse (descendants before ancestors).
    remaining = dict(indeg)
    queue = deque(n for n, d in remaining.items() if d == 0)
    topo_order: list[int] = []
    while queue:
        n = queue.popleft()
        topo_order.append(n)
        for m in succ[n]:
            remaining[m] -= 1
            if remaining[m] == 0:
                queue.append(m)
    if len(topo_order) != len(succ):
        return list(edges), [], False

    # Reverse topological order guarantees every successor's reachability set is
    # already final before we process a node. For node u, an edge (u, v) is
    # redundant iff v lies in the reachability set of some *other* successor of
    # u (v is reachable via a length >= 2 path); indirect collects those.
    reach: dict[int, set[int]] = {}
    redundant: set[tuple[int, int]] = set()
    for u in reversed(topo_order):
        indirect: set[int] = set()
        for v in succ[u]:
            indirect |= reach[v]
        for v in succ[u]:
            if v in indirect:
                redundant.add((u, v))
        node_reach = set(succ[u])
        node_reach |= indirect
        reach[u] = node_reach

    # edges arrives already sorted (from _load_deps_edges); filtering preserves
    # that order, so kept/removed stay sorted without a second sort.
    kept = [e for e in edges if e not in redundant]
    removed = [e for e in edges if e in redundant]
    return kept, removed, True


def _backfill_output_tensor_ids(task_table, annotations):
    """Recover ``tensor_id`` for OUTPUT slots that the runtime hadn't
    materialized at submit time.

    Each creator-source edge tells us the producer (``pred``) created the
    tensor that the consumer (``succ``) reads. We know the consumer's
    ``tensor_id`` and ``consumer_arg_idx``; we know the producer task but
    NOT which of its OUTPUT slots produced this tensor (the captured
    DepGenRecord has a zeroed blob for OUTPUT). When a producer has
    exactly one OUTPUT slot with no tensor_id assigned, the assignment is
    unambiguous and we backfill it so the viewer can route the edge into
    the right row. When there are multiple unassigned OUTPUT slots we
    leave them alone — guessing would be worse than the body-attach
    fallback.
    """
    for (pred, _succ), rows in annotations.items():
        for row in rows:
            if row.get("source") != "creator":
                continue
            tid = row.get("tensor_id")
            if tid is None:
                continue
            pred_task = task_table.get(pred)
            if not pred_task:
                continue
            already_known = False
            unassigned = []
            for a in pred_task.get("args", []):
                if a.get("type") in ("INOUT", "OUTPUT_EXISTING"):
                    if a.get("tensor_id") == tid:
                        already_known = True
                        break
                if a.get("type") == "OUTPUT":
                    if a.get("tensor_id") is None:
                        unassigned.append(a)
                    elif a.get("tensor_id") == tid:
                        already_known = True
                        break
            if already_known:
                continue
            if len(unassigned) == 1:
                unassigned[0]["tensor_id"] = tid
                unassigned[0]["inferred"] = True


def _kernel_ids_slots(task_entry):
    """Return the raw kernel_ids list as-is, padded/truncated to exactly 3 slots."""
    if not isinstance(task_entry, dict):
        return [-1, -1, -1]
    kernel_ids = task_entry.get("kernel_ids")
    if not isinstance(kernel_ids, list):
        return [-1, -1, -1]
    slots = []
    for i in range(3):
        if i < len(kernel_ids):
            v = _normalize_small_int(kernel_ids[i])
            slots.append(v if v is not None else -1)
        else:
            slots.append(-1)
    return slots


def _merge_task_meta_with_kernel_ids(meta, task_table, func_names=None):
    merged = {task_id: dict(entry) for task_id, entry in meta.items()}
    for task_id, task_entry in task_table.items():
        slots = _kernel_ids_slots(task_entry)
        valid_ids = [i for i in slots if i >= 0]
        if not valid_ids:
            continue
        entry = merged.setdefault(task_id, {})
        if entry.get("func_id") is None:
            entry["func_id"] = valid_ids[0]
        entry["func_ids"] = valid_ids
        entry["_kernel_slots"] = slots
        if func_names:
            entry["func_labels"] = [
                func_names.get(str(slot)) or func_names.get(slot) or f"f{slot}" if slot >= 0 else "-1" for slot in slots
            ]
            # func_name = the ACTIVE slots' names, order-preserving-deduped.
            # kernel_ids is [aic, aiv0, aiv1]: an AIC-only task shows the AIC
            # name, an AIV-only task the AIV name (never the inactive slot's
            # "-1"), and a MIX task both (e.g. "paged_attention_cce_aic +
            # paged_attention_cce_aiv"). aiv0/aiv1 usually carry the same
            # func_id (one AIV kernel on two cores), so the dedup collapses
            # them to one name.
            if not entry.get("func_name"):
                active_names = []
                for fid in valid_ids:
                    nm = func_names.get(str(fid)) or func_names.get(fid) or f"f{fid}"
                    if nm not in active_names:
                        active_names.append(nm)
                entry["func_name"] = " + ".join(active_names)
        elif any(s >= 0 for s in slots):
            entry["func_labels"] = [f"f{slot}" if slot >= 0 else "-1" for slot in slots]
        if not entry.get("core_type"):
            has_aic = slots[0] >= 0
            has_aiv = slots[1] >= 0 or slots[2] >= 0
            if has_aic and has_aiv:
                entry["core_type"] = "mix"
            elif has_aic:
                entry["core_type"] = "aic"
            elif has_aiv:
                entry["core_type"] = "aiv"
    return merged


def _load_task_meta(deps_path, func_names=None):
    """Optional l2_swimlane_records.json sidecar → {task_id: {'func_id', 'core_type', ...}}.

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
    perf_path = Path(deps_path).parent / "l2_swimlane_records.json"
    if not perf_path.exists():
        return {}
    try:
        # Route through swimlane_converter.read_perf_data so v2 raw on-disk
        # JSON (aicore_tasks/aicpu_tasks flat tuples in cycle domain) gets
        # joined into the v1-shape dict this function expects. Direct
        # json.load would see no top-level `tasks` array on v2 and silently
        # return {} — leaving every node uncolored / unlabeled.
        from .swimlane_converter import read_perf_data  # noqa: PLC0415

        perf = read_perf_data(perf_path)
    except (OSError, ValueError) as e:
        print(f"Warning: couldn't read {perf_path}: {e}", file=sys.stderr)
        return {}

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


def _label(task_id, meta, task_table, fmt_task, marker=""):
    base = fmt_task(task_id)
    pfx = f"{marker} " if marker else ""
    normalized_task_id = _normalize_task_id(task_id)
    kind = _task_kind(normalized_task_id, meta, task_table)
    if kind == "alloc":
        return f"{pfx}{base} · alloc"
    if kind == "dummy":
        return f"{pfx}{base} · dummy"
    func_name = (meta.get(normalized_task_id) or {}).get("func_name")
    if func_name:
        return f"{pfx}{base} · {func_name}"
    return f"{pfx}{base}"


_CORE_STYLE = {
    "aic": {"shape": "box", "style": "rounded,filled", "fillcolor": "#66A3FF"},
    "aiv": {"shape": "ellipse", "style": "filled", "fillcolor": "#FFB366"},
    "mix": {"shape": "diamond", "style": "filled", "fillcolor": "#66CC99"},
    "alloc": {"shape": "note", "style": "filled,dashed", "fillcolor": "#EAEAEA"},
}
_DEFAULT_STYLE = {"shape": "box", "style": "rounded,filled", "fillcolor": "#E0E0E0"}


def _node_style(core_type):
    return _CORE_STYLE.get(core_type, _DEFAULT_STYLE)


def _format_dims(values):
    """Compact "[a,b,c]" for shape/offset arrays; "[]" when empty."""
    if not values:
        return "[]"
    return "[" + ",".join(str(v) for v in values) + "]"


_CORE_HEADER_COLOR = {
    "aic": "#66A3FF",
    "aiv": "#FFB366",
    "mix": "#66CC99",
    "alloc": "#EAEAEA",
}
_INPUT_BG = "#EAF2FF"
_OUTPUT_BG = "#FFF2E5"
_HEADER_FALLBACK = "#D8D8D8"
_SPMD_COLOR = "#C62828"


def _html_escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _dot_escape_label(text):
    return str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _short_tensor_label(tid, tensor_table):
    """Display name for a tensor: ``T<idx>`` from tensors[] order when known,
    else a short hex prefix of the FNV id (fallback used by edges that
    reference a tensor id with no matching tensors[] entry)."""
    if isinstance(tid, int) and tid in tensor_table:
        return tensor_table[tid].get("name") or f"T?{tid & 0xFFFF:04x}"
    if isinstance(tid, int):
        return f"T?{tid & 0xFFFF:04x}"
    return "T?"


def _arg_row_html(arg, tensor_table, side):
    """One HTML cell (rendered as a 4-line block) for an input or output arg
    slot of a task node.

    Line 1: ``arg<idx> <ARG_TYPE>[ ?] <Tname>:<dtype>``  — slot identity.
    Line 2: ``storage: <N> elems``                       — underlying buffer size.
    Line 3: ``shape: [...]``                            — slice this slot accesses.
    Line 4: ``offset: [...]``                           — slice start offset into the buffer.
    """
    idx = arg.get("idx")
    arg_type = arg.get("type", "?")
    port = f"{side}_{idx}"
    bg = _INPUT_BG if side == "in" else _OUTPUT_BG
    tid = arg.get("tensor_id")
    if tid is None:
        body = f"arg{idx} {arg_type} (alloc)"
        return f'<TR><TD ALIGN="LEFT" PORT="{port}" BGCOLOR="{bg}">{_html_escape(body)}</TD></TR>'

    tname = _short_tensor_label(tid, tensor_table)
    dtype = arg.get("dtype")
    shape = arg.get("shape")
    start_offset = arg.get("start_offset")
    strides = arg.get("strides")
    buffer_numel = None
    if isinstance(tid, int) and tid in tensor_table:
        tt = tensor_table[tid]
        buffer_numel = tt.get("buffer_numel")
        if dtype is None:
            dtype = tt.get("dtype")
    if arg.get("inferred"):
        if shape is None and buffer_numel is not None:
            shape = [int(buffer_numel)]
        if start_offset is None:
            start_offset = "0"
        if strides is None and shape:
            strides = [1] * len(shape)

    dtype_str = f":{dtype.lower()}" if isinstance(dtype, str) else ""
    inferred_mark = " ?" if arg.get("inferred") else ""
    head = f"arg{idx} {arg_type}{inferred_mark} {tname}{dtype_str}"
    storage_line = f"storage: {buffer_numel} elems" if buffer_numel is not None else "storage: ?"
    shape_line = f"shape: {_format_dims(shape) if isinstance(shape, list) else '[]'}"
    strides_line = f"strides: {_format_dims(strides) if isinstance(strides, list) else '[]'}"
    offset_line = f"start_offset: {start_offset if start_offset is not None else '?'} (elem)"

    body = (
        f'{_html_escape(head)}<BR ALIGN="LEFT"/>'
        f'<FONT POINT-SIZE="9">{_html_escape(storage_line)}<BR ALIGN="LEFT"/>'
        f'{_html_escape(shape_line)}<BR ALIGN="LEFT"/>'
        f'{_html_escape(strides_line)}<BR ALIGN="LEFT"/>'
        f'{_html_escape(offset_line)}<BR ALIGN="LEFT"/></FONT>'
    )
    return f'<TR><TD ALIGN="LEFT" PORT="{port}" BGCOLOR="{bg}">{body}</TD></TR>'


def _task_node_html(task_id, task_entry, meta_entry, tensor_table, fmt_task, marker=""):
    """Build a Graphviz HTML-like label for a task node showing:
        - input rows (top)     INPUT + INOUT slots
        - identity header      "<marker> (ring, local) · <func_name>"
        - output rows (bottom) INOUT + OUTPUT_EXISTING + OUTPUT slots
    INOUT slots appear in BOTH compartments (read-then-write semantics).
    ``marker`` is the 🔥 / ⭐ early-dispatch badge (empty for most tasks).
    """
    args = task_entry.get("args") if task_entry else None
    if not isinstance(args, list):
        args = []
    inputs = [a for a in args if a.get("type") in ("INPUT", "INOUT")]
    outputs = [a for a in args if a.get("type") in ("INOUT", "OUTPUT", "OUTPUT_EXISTING")]
    core_type = meta_entry.get("core_type") if meta_entry else None
    header_bg = _CORE_HEADER_COLOR.get(core_type if isinstance(core_type, str) else "", _HEADER_FALLBACK)
    border_attrs = f'BORDER="1" COLOR="{_SPMD_COLOR}"' if _task_block_num(task_entry) > 1 else 'BORDER="0"'

    ident = fmt_task(task_id)
    func_name = meta_entry.get("func_name") if meta_entry else None
    if func_name:
        ident = f"{ident} · {func_name}"
    if marker:
        ident = f"{marker} {ident}"

    rows = []
    for a in inputs:
        rows.append(_arg_row_html(a, tensor_table, "in"))
    rows.append(f'<TR><TD ALIGN="CENTER" BGCOLOR="{header_bg}"><B>{_html_escape(ident)}</B></TD></TR>')
    for a in outputs:
        rows.append(_arg_row_html(a, tensor_table, "out"))

    table = f'<TABLE {border_attrs} CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">' + "".join(rows) + "</TABLE>"
    return table


def _producer_output_port(pred_task_entry, edge_tensor_id):
    """Resolve the producer-side HTML port for an annotated edge when possible."""
    if not pred_task_entry or edge_tensor_id is None:
        return None
    args = pred_task_entry.get("args")
    if not isinstance(args, list):
        return None
    for a in args:
        if a.get("type") not in ("OUTPUT", "OUTPUT_EXISTING", "INOUT"):
            continue
        if a.get("tensor_id") == edge_tensor_id:
            return f"out_{a.get('idx')}"
    return None


def _task_kind(task_id, meta, task_table):
    task_entry = task_table.get(task_id)
    if isinstance(task_entry, dict):
        slots = _kernel_ids_slots(task_entry)
        if any(slot >= 0 for slot in slots):
            return "submit"
        return "dummy"
    entry = meta.get(task_id)
    if isinstance(entry, dict):
        slots = entry.get("_kernel_slots")
        if isinstance(slots, list) and any(slot >= 0 for slot in slots):
            return "submit"
    return "alloc"


def _task_func_id_label(task_id, meta, task_table):
    kind = _task_kind(task_id, meta, task_table)
    if kind in ("alloc", "dummy"):
        return "func_id=none"
    entry = meta.get(task_id)
    if entry:
        slots = entry.get("_kernel_slots")
        if isinstance(slots, list) and slots:
            return "func_id=[" + ",".join(str(slot) for slot in slots) + "]"
    return "func_id=unknown"


def _task_block_num(task_entry):
    if not isinstance(task_entry, dict):
        return 1
    block_num = _normalize_small_int(task_entry.get("block_num"))
    if block_num is None or block_num < 1:
        return 1
    return block_num


def _task_blocks_text(task_entry):
    block_num = _task_block_num(task_entry)
    if block_num <= 1:
        return ""
    return f"SPMD block num = {block_num}"


def _plain_node_attrs(task_id, meta, task_table, fmt_task, marker=""):
    meta_entry = meta.get(task_id)
    kind = _task_kind(task_id, meta, task_table)
    if kind == "submit" and meta_entry:
        style = _node_style(meta_entry.get("core_type"))
    elif kind == "alloc":
        style = _CORE_STYLE["alloc"]
    else:
        style = _DEFAULT_STYLE

    task_entry = task_table.get(task_id)
    label = _dot_escape_label(_label(task_id, meta, task_table, fmt_task, marker=marker))
    label_attr = f'label="{label}"'
    style_attr = style["style"]
    if _task_block_num(task_entry) > 1:
        extra = f', color="{_SPMD_COLOR}", penwidth=1.5'
    else:
        extra = ""
    return f'{label_attr}, shape={style["shape"]}, style="{style_attr}", fillcolor="{style["fillcolor"]}"{extra}'


def emit_text(edges, nodes, meta, deps_path, annotations=None, tensor_table=None, task_table=None):
    """Render deps.json as grep-friendly plain text.

    Output shape:
        SUMMARY
        TASK INDEX
        per-task detail blocks with FANIN / FANOUT peer lists
    """
    annotations = annotations or {}
    tensor_table = tensor_table or {}
    task_table = task_table or {}
    sorted_nodes = sorted(nodes, key=_sort_task_id_key)
    fmt_task = _make_task_formatter(sorted_nodes)
    have_perf = bool(meta)
    have_func_name_map = any(entry.get("func_name") for entry in meta.values())

    pred_map = {tid: set() for tid in sorted_nodes}
    succ_map = {tid: set() for tid in sorted_nodes}
    for pred, succ in edges:
        pred_map.setdefault(succ, set()).add(pred)
        succ_map.setdefault(pred, set()).add(succ)
    annotated_edges = sum(len(rows) for rows in annotations.values())

    lines = [
        "SUMMARY",
        f"  source_deps_json: {deps_path}",
        f"  tasks: {len(sorted_nodes)}",
        f"  unique_task_edges: {len(edges)}",
        f"  annotated_edges: {annotated_edges}",
        f"  tensors: {len(tensor_table)}",
        f"  perf_sidecar: {'yes' if have_perf else 'no'}",
        f"  func_name_map: {'yes' if have_func_name_map else 'no'}",
        "",
        "TASK INDEX",
    ]
    for task_id in sorted_nodes:
        kind = _task_kind(task_id, meta, task_table)
        func_label = _task_func_id_label(task_id, meta, task_table)
        block_label = _task_blocks_text(task_table.get(task_id))
        task_labels = " ".join(label for label in (func_label, block_label) if label)
        lines.append(
            f"TASK {fmt_task(task_id)} kind={kind} {task_labels} "
            f"fanin={len(pred_map.get(task_id, set()))} fanout={len(succ_map.get(task_id, set()))}"
        )

    for task_id in sorted_nodes:
        kind = _task_kind(task_id, meta, task_table)
        func_label = _task_func_id_label(task_id, meta, task_table)
        block_label = _task_blocks_text(task_table.get(task_id))
        task_labels = " ".join(label for label in (func_label, block_label) if label)
        lines.append("")
        lines.append(f"=== TASK {fmt_task(task_id)} kind={kind} {task_labels} ===")

        pred_peers = sorted(pred_map.get(task_id, set()), key=_sort_task_id_key)
        lines.append(f"FANIN {len(pred_peers)}")
        for pred_tid in pred_peers:
            lines.append(f"  <- {fmt_task(pred_tid)}")

        succ_peers = sorted(succ_map.get(task_id, set()), key=_sort_task_id_key)
        lines.append(f"FANOUT {len(succ_peers)}")
        for succ_tid in succ_peers:
            lines.append(f"  -> {fmt_task(succ_tid)}")

    return "\n".join(lines) + "\n"


def _task_markers(nodes, edges, meta, task_table):
    """Map task_id -> marker string for the node label.

    🔥 (fire): the task is either a flagged early-dispatch producer
        (deps.json ``early_dispatch`` — the submit had allow_early_resolve)
        or an alloc task. Alloc tasks are immediate graph sources, so they
        are treated as fire-marked producers by default.
    ⭐ (star): every one of the task's predecessors is 🔥 (and it has at
        least one), so the task is fully fed by flagged producers or alloc
        sources.
    A task can carry both.
    """
    pred_map: dict[int, set] = {}
    for pred, succ in edges:
        pred_map.setdefault(succ, set()).add(pred)

    def _fire_marked(tid):
        return _task_kind(tid, meta, task_table) == "alloc" or bool((task_table.get(tid) or {}).get("early_dispatch"))

    markers = {}
    for tid in nodes:
        fire = "🔥" if _fire_marked(tid) else ""
        preds = pred_map.get(tid, set())
        star = "⭐" if preds and all(_fire_marked(p) for p in preds) else ""
        if fire or star:
            markers[tid] = fire + star
    return markers


def emit_dot(
    edges,
    nodes,
    meta,
    direction="LR",
    annotations=None,
    tensor_table=None,
    task_table=None,
    show_tensor_info=None,
    hidden_edges=None,
):
    """Graphviz DOT source. Used internally to feed the layout engine before
    wrapping the SVG in HTML.

    Two rendering modes selected by ``show_tensor_info``:

    1. Plain mode (``task_table`` is None or empty) — bare shape/color nodes,
       bare arrows. Used when only task-level structure is available.
    2. Rich mode (``task_table`` non-empty) — every task whose args were
       captured renders as an HTML-table node with input rows above an
       identity header and output rows below. Edges terminate on the
       consumer's ``in_<arg_idx>`` port, and originate from the
       producer's ``out_<arg_idx>`` port whenever the producer slot's
       tensor_id matches the edge's tensor_id.
    """
    fmt_task = _make_task_formatter(nodes)
    annotations = annotations or {}
    tensor_table = tensor_table or {}
    task_table = task_table or {}
    hidden_edges = set(hidden_edges or ())
    show_tensor = bool(task_table) if show_tensor_info is None else bool(show_tensor_info and task_table)
    markers = _task_markers(nodes, edges, meta, task_table)
    lines = [
        "digraph deps {",
        f"  rankdir={direction};",
        # Per-edge visibility requires each logical edge to retain its complete spline.
        "  concentrate=false;",
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [color="#888"];',
    ]
    for n in nodes:
        m = meta.get(n)
        marker = markers.get(n, "")
        if show_tensor and n in task_table:
            html = _task_node_html(n, task_table.get(n), m, tensor_table, fmt_task, marker=marker)
            lines.append(f"  {_node_id(n)} [shape=none, margin=0, label=<{html}>];")
            continue
        lines.append(f"  {_node_id(n)} [{_plain_node_attrs(n, meta, task_table, fmt_task, marker=marker)}];")

    def edge_attr_str(edge_attrs):
        return (" [" + ", ".join(edge_attrs) + "]") if edge_attrs else ""

    def hidden_edge_attrs():
        return ['class="hidden-edge"', 'color="#eef2f7"', 'fontcolor="#eef2f7"']

    for pred, succ in edges:
        hidden = (pred, succ) in hidden_edges
        hidden_attrs = hidden_edge_attrs() if hidden else []
        if not show_tensor:
            lines.append(f"  {_node_id(pred)} -> {_node_id(succ)}{edge_attr_str(hidden_attrs)};")
            continue
        rows = annotations.get((pred, succ), [])
        if not rows:
            lines.append(f"  {_node_id(pred)} -> {_node_id(succ)}{edge_attr_str(hidden_attrs)};")
            continue
        for row in rows:
            arg = row.get("arg")
            tid = row.get("tensor_id")
            source = row.get("source")
            tail = ""
            head = ""
            edge_attrs = []
            if hidden:
                edge_attrs.extend(hidden_edge_attrs())
            arg_idx = _normalize_small_int(arg)
            if arg_idx is not None and arg_idx >= 0:
                head = f":in_{arg_idx}:w"
            out_port = _producer_output_port(task_table.get(pred), tid)
            if out_port:
                tail = f":{out_port}:e"
            if source == "explicit" and not hidden:
                edge_attrs.append('style="dashed"')
                edge_attrs.append('color="#B0B0B0"')
            overlap = row.get("overlap")
            if overlap and overlap != "covered" and not hidden:
                edge_attrs.append(f'label="{_html_escape(overlap)}", fontsize=8, fontcolor="#C04040"')
            attr_str = edge_attr_str(edge_attrs)
            lines.append(f"  {_node_id(pred)}{tail} -> {_node_id(succ)}{head}{attr_str};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_svg(dot_text, engine="dot"):
    """Pipe DOT through the Graphviz layout engine and return raw SVG bytes."""
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


def _spmd_badges_json(nodes, task_table):
    task_table = task_table or {}
    badges = {}
    for task_id in nodes:
        block_num = _task_block_num(task_table.get(task_id))
        if block_num > 1:
            badges[_node_id(task_id)] = block_num
    return json.dumps(badges, sort_keys=True, separators=(",", ":"))


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>deps.json — {n_nodes} nodes, {n_edges} edges</title>
<style>
  html, body {{
    margin: 0; height: 100%; background: #f8fafc;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    color: #0f172a; overflow: hidden; -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }}
  #hud, #legend {{
    position: fixed; z-index: 10; display: flex; align-items: center;
    background: rgba(255,255,255,0.94); border: 1px solid rgba(203,213,225,0.9);
    border-radius: 10px; color: #0f172a; font-size: 12px;
    box-shadow: 0 10px 26px rgba(15,23,42,0.10); backdrop-filter: blur(10px);
  }}
  #hud {{ top: 12px; left: 12px; gap: 8px; padding: 8px 12px; }}
  #hud .stat {{ font-weight: 600; color: #020617; }}
  #hud .muted {{ color: #cbd5e1; }}
  #hud .divider {{ width: 1px; height: 16px; margin: 0 4px; background: #e2e8f0; }}
  #hud kbd {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: #f1f5f9;
    border: 1px solid #cbd5e1; color: #334155; padding: 1px 5px; border-radius: 5px;
  }}
  #legend {{
    top: 12px; right: 12px; gap: 12px; padding: 8px 12px;
    flex-wrap: wrap; max-width: min(760px, calc(100vw - 24px));
  }}
  #legend .swatch {{ display: inline-flex; align-items: center; gap: 6px; color: #334155; white-space: nowrap; }}
  #legend svg {{ display: block; }}
  #stage {{ width: 100vw; height: 100vh; overflow: hidden; cursor: grab; background:
    radial-gradient(circle at 20px 20px, rgba(100,116,139,0.18) 1px, transparent 1px), #eef2f7;
    background-size: 24px 24px; }}
  #stage.panning {{ cursor: grabbing; }}
  #stage > svg {{
    transform-origin: 0 0; transition: none; max-width: none; overflow: visible;
    filter: drop-shadow(0 16px 28px rgba(15,23,42,0.10));
  }}
  #stage .node path, #stage .node ellipse, #stage .node polygon {{
    filter: drop-shadow(0 3px 4px rgba(15,23,42,0.18));
  }}
  #stage .edge path, #stage .edge polygon {{ opacity: 0.78; }}
</style>
</head>
<body>
<div id="hud">
  <span class="stat">{n_nodes} nodes</span>
  <span class="muted">·</span>
  <span class="stat">{n_edges} edges</span>
  <span class="divider"></span>
  <kbd>drag</kbd><span>pan</span>
  <kbd>wheel</kbd><span>zoom</span>
  <kbd>f</kbd><span>fit</span>
  <kbd>r</kbd><span>reset</span>
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
  <span class="swatch">
    <svg width="34" height="14" viewBox="0 0 34 14">
      <rect x="1" y="2" width="18" height="10" rx="3" ry="3" fill="none"
            stroke="#C62828" stroke-width="1"/>
      <text x="20" y="8" fill="#C62828"
            font-family="Helvetica, sans-serif" font-size="7" font-weight="700">xN</text>
    </svg>
    SPMD block num
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
  svg.removeAttribute('width');
  svg.removeAttribute('height');
  const spmdBlocks = {spmd_badges_json};
  const svgNS = 'http://www.w3.org/2000/svg';

  function svgEl(name, attrs) {{
    const el = document.createElementNS(svgNS, name);
    for (const [key, value] of Object.entries(attrs)) {{
      el.setAttribute(key, value);
    }}
    return el;
  }}

  function addSpmdBadges() {{
    for (const [nodeId, blocks] of Object.entries(spmdBlocks)) {{
      const title = Array.from(svg.querySelectorAll('g.node > title')).find((item) => item.textContent === nodeId);
      if (!title) continue;
      const group = title.parentElement;
      const graph = group ? group.parentElement : null;
      if (!group || !graph || graph.querySelector(`.spmd-badge[data-node-id="${{nodeId}}"]`)) continue;

      const box = group.getBBox();
      const text = `x${{blocks}}`;
      const x = box.x + box.width + 1;
      const y = box.y + box.height / 2 - 3;
      const badge = svgEl('g', {{ class: 'spmd-badge', 'data-node-id': nodeId }});
      badge.appendChild(svgEl('title', {{}}));
      badge.lastChild.textContent = `SPMD block num: ${{blocks}}`;

      badge.appendChild(svgEl('text', {{
        x, y, fill: '#C62828', 'font-family': 'Helvetica, sans-serif',
        'font-size': 8, 'font-weight': 700,
      }}));
      badge.lastChild.textContent = text;
      graph.appendChild(badge);
    }}
  }}
  addSpmdBadges();

  let scale = 1, tx = 0, ty = 0;
  const apply = () => {{ svg.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(${{scale}})`; }};

  stage.addEventListener('wheel', (e) => {{
    e.preventDefault();
    const rect = stage.getBoundingClientRect();
    const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    const factor = Math.exp(-e.deltaY * 0.001);
    const newScale = Math.min(20, Math.max(0.02, scale * factor));
    tx = cx - (cx - tx) * (newScale / scale);
    ty = cy - (cy - ty) * (newScale / scale);
    scale = newScale;
    apply();
  }}, {{ passive: false }});

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

  const fit = () => {{
    const bb = svg.getBoundingClientRect();
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
  requestAnimationFrame(fit);
}})();
</script>
</body>
</html>
"""


def _layer_svg_edges(svg_text):
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return svg_text

    def local_name(element):
        return element.tag.rsplit("}", 1)[-1]

    def classes(element):
        return set(element.get("class", "").split())

    graph = next(
        (element for element in root.iter() if local_name(element) == "g" and "graph" in classes(element)),
        None,
    )
    if graph is None:
        return svg_text

    edge_groups = []
    non_edge_children = []
    has_hidden_edges = False
    for child in graph:
        child_classes = classes(child)
        if local_name(child) == "g" and "edge" in child_classes:
            hidden = "hidden-edge" in child_classes
            edge_groups.append((child, hidden))
            has_hidden_edges |= hidden
        else:
            non_edge_children.append(child)
    if not has_hidden_edges:
        return svg_text

    graph[:] = non_edge_children

    namespace = root.tag[1:].partition("}")[0] if root.tag.startswith("{") else ""
    group_tag = f"{{{namespace}}}g" if namespace else "g"
    hidden_layer = ET.Element(group_tag, {"class": "edge-layer hidden-edge-layer"})
    visible_layer = ET.Element(group_tag, {"class": "edge-layer visible-edge-layer"})

    for edge, hidden in edge_groups:
        layer = hidden_layer if hidden else visible_layer
        layer.append(edge)

    insert_at = next(
        (
            index
            for index, child in enumerate(non_edge_children)
            if local_name(child) == "g" and "node" in classes(child)
        ),
        len(non_edge_children),
    )
    graph.insert(insert_at, hidden_layer)
    graph.insert(insert_at + 1, visible_layer)

    if namespace:
        ET.register_namespace("", namespace)
    return ET.tostring(root, encoding="unicode")


def emit_html(
    edges,
    nodes,
    meta,
    direction="LR",
    engine="dot",
    annotations=None,
    tensor_table=None,
    task_table=None,
    show_tensor_info=None,
    html_edge_style=None,
):
    """Build the pan/zoom HTML page: DOT → Graphviz SVG → inline into template."""
    html_edge_style = html_edge_style or {}
    hidden_edges = html_edge_style.get("hidden_edges")
    visible_edge_count = html_edge_style.get("visible_edge_count")
    if visible_edge_count is None:
        visible_edge_count = len(edges)
    dot = emit_dot(
        edges,
        nodes,
        meta,
        direction=direction,
        annotations=annotations,
        tensor_table=tensor_table,
        task_table=task_table,
        show_tensor_info=show_tensor_info,
        hidden_edges=hidden_edges,
    )
    svg_bytes = render_svg(dot, engine=engine)
    svg_text = svg_bytes.decode("utf-8", errors="replace")
    if "<svg" in svg_text:
        svg_text = svg_text[svg_text.index("<svg") :]
    svg_text = _layer_svg_edges(svg_text)
    return _HTML_TEMPLATE.format(
        n_nodes=len(nodes),
        n_edges=visible_edge_count,
        svg_body=svg_text,
        spmd_badges_json=_spmd_badges_json(nodes, task_table),
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
    """Load func_id → name mapping from a JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print(f"Warning: couldn't read {path}: {e}", file=sys.stderr)
        return {}
    return data.get("callable_id_to_name") or data


def _autoload_name_map(deps_path):
    """Look for a ``name_map_*.json`` next to deps.json."""
    candidates = sorted(Path(deps_path).parent.glob("name_map_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return {}
    return _load_func_names_json(candidates[-1])


def _build_parser():
    p = argparse.ArgumentParser(
        description="Render deps.json as text or pan/zoom HTML dependency graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s                                    # newest deps.json under ./outputs/, default text output
  %(prog)s outputs/.../deps.json
  %(prog)s deps.json --format text -o graph.txt
  %(prog)s deps.json --format html --engine sfdp
  %(prog)s deps.json --format html --show-tensor-info
  %(prog)s deps.json --edge-mode reduced      # select non-redundant edges, print what was removed
  %(prog)s deps.json --edge-mode omitted      # select transitively-implied (redundant) edges
""",
    )
    p.add_argument("input", nargs="?", help="Path to deps.json (default: newest under ./outputs/).")
    p.add_argument("-o", "--output", help="Output path (default: deps_viewer.txt for text, deps_viewer.html for html).")
    p.add_argument(
        "--format",
        choices=["text", "html"],
        default="text",
        help="Output format: text (default) or html.",
    )
    p.add_argument(
        "--edge-mode",
        choices=["full", "reduced", "omitted"],
        default="full",
        help=(
            "full (default) selects every dependency edge; reduced applies transitive reduction, selecting the "
            "minimal edge set; omitted selects only the redundant edges reduced would drop (the complement of "
            "reduced). reduced/omitted print the redundant edges to stdout, are structural (pred,succ) level, "
            "apply to both text and html, and are skipped with a warning on a cyclic graph. In html, all edges "
            "still participate in layout; unselected edges are colored as background and drawn below selected edges."
        ),
    )
    p.add_argument(
        "--engine",
        choices=["dot", "neato", "sfdp", "fdp", "circo", "twopi"],
        default="dot",
        help="Graphviz layout engine for HTML output.",
    )
    p.add_argument(
        "--direction",
        choices=["TB", "LR", "BT", "RL"],
        default="LR",
        help="Flow direction for hierarchical HTML layouts.",
    )
    p.add_argument(
        "--func-names",
        help="JSON file with callable_id_to_name (or flat {func_id: name}) for task-label enrichment.",
    )
    p.add_argument(
        "--show-tensor-info",
        action="store_true",
        help=(
            "For HTML output, render per-task input/output tensor details and route edges to specific arg ports. "
            "Default: off."
        ),
    )
    return p


def _argv_has_option(argv, name):
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv)


def _validate_args(args, argv):
    if args.format == "html":
        return 0
    html_only = ["--engine", "--direction", "--show-tensor-info"]
    for option in html_only:
        if _argv_has_option(argv, option):
            print(f"error: {option} is only valid with --format html", file=sys.stderr)
            return 2
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(argv)
    rc = _validate_args(args, argv)
    if rc != 0:
        return rc

    input_path = args.input or _find_latest_deps_json()
    if input_path is None:
        print("No deps.json given and no candidate found under ./outputs/.", file=sys.stderr)
        return 1
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"{input_path} not found.", file=sys.stderr)
        return 1

    edges, nodes, annotations, tensor_table, task_table = _load_deps_edges(input_path)
    func_names = _load_func_names_json(args.func_names) if args.func_names else _autoload_name_map(input_path)
    meta = _load_task_meta(input_path, func_names=func_names)
    meta = _merge_task_meta_with_kernel_ids(meta, task_table, func_names=func_names)
    if meta:
        nodes = sorted(set(nodes) | set(meta.keys()), key=_sort_task_id_key)

    mode_stem = "deps_viewer"
    hidden_html_edges = set()
    visible_html_edge_count = len(edges)
    if args.edge_mode in ("reduced", "omitted"):
        kept, removed, is_dag = _transitive_reduction(edges, nodes)
        if not is_dag:
            print(
                f"warning: dependency graph has a cycle; transitive reduction skipped, "
                f"emitting full graph (--edge-mode {args.edge_mode} ignored)",
                file=sys.stderr,
            )
        else:
            fmt_task = _make_task_formatter(nodes)
            # reduced keeps the minimal edge set; omitted keeps exactly the
            # redundant edges that reduced would drop (the two are complements).
            shown = kept if args.edge_mode == "reduced" else removed
            if args.edge_mode == "reduced":
                print(
                    f"Transitive reduction: removed {len(removed)} redundant edge(s) of {len(edges)} ({len(kept)} kept)"
                )
            else:
                print(f"Redundant edges only: showing {len(removed)} redundant edge(s) of {len(edges)}")
            for u, v in removed:
                print(f"  - {fmt_task(u)} -> {fmt_task(v)}")
            if args.format == "html":
                hidden_html_edges = set(removed if args.edge_mode == "reduced" else kept)
                visible_html_edge_count = len(shown)
                print(
                    f"HTML layout preserves all {len(edges)} edge(s); "
                    f"{len(hidden_html_edges)} unselected edge(s) are colored as background"
                )
            else:
                edges = shown
                # Keep only the annotations of the shown edges so annotated-edge
                # counts (emit_text summary, the final print) stay consistent with
                # the rendered edge set instead of counting the hidden ones.
                shown_set = set(shown)
                annotations = {k: v for k, v in annotations.items() if k in shown_set}
            mode_stem = f"deps_viewer_{args.edge_mode}"

    out = (
        Path(args.output)
        if args.output
        else input_path.parent / f"{mode_stem}.{'txt' if args.format == 'text' else 'html'}"
    )
    if args.format == "text":
        text = emit_text(
            edges,
            nodes,
            meta,
            input_path,
            annotations=annotations,
            tensor_table=tensor_table,
            task_table=task_table,
        )
        out.write_text(text)
        annotated_edges = sum(len(rows) for rows in annotations.values())
        print(
            f"Wrote {out} "
            f"({len(nodes)} tasks, {len(edges)} unique edges, "
            f"{annotated_edges} annotated edges, format=text)"
        )
        return 0

    html = emit_html(
        edges,
        nodes,
        meta,
        direction=args.direction,
        engine=args.engine,
        annotations=annotations,
        tensor_table=tensor_table,
        task_table=task_table,
        show_tensor_info=args.show_tensor_info,
        html_edge_style={
            "hidden_edges": hidden_html_edges,
            "visible_edge_count": visible_html_edge_count,
        },
    )
    out.write_text(html)
    if hidden_html_edges:
        print(
            f"Wrote {out} ({len(nodes)} nodes, {visible_html_edge_count} visible edges, "
            f"{len(edges)} layout edges, engine={args.engine}, format=html)"
        )
    else:
        print(f"Wrote {out} ({len(nodes)} nodes, {len(edges)} edges, engine={args.engine}, format=html)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
