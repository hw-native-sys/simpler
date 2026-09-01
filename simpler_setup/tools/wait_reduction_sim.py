#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Offline WAIT-edge reduction coverage simulator (issue #1376).

Reads one or more ``deps.json`` captures (see ``docs/dfx/dep-gen.md``),
reconstructs the global submission order from the ``tasks[]`` record order,
and measures how many redundant WAIT edges the runtime's bounded reachability
bitmap would remove — against the exact full-graph transitive reduction as
the upper bound.

Two models run over the same WAIT graph:

- **Full reduction** (upper bound): exact transitive reachability on the whole
  DAG. An edge ``p -> t`` is redundant when ``p`` reaches ``t`` through any
  other WAIT path of length >= 2. This is what a host-resident full-DAG
  reducer (e.g. the hbg Definition pass) could remove.
- **Online bitmap** (the runtime algorithm): a faithful mirror of
  ``reduce_wait_edges`` — per-task frozen ``R[t]`` bitmap over the last BL
  submissions, two-pass ``direct``/``via`` fold, ``d > BL`` window misses kept
  conservatively, ``d == BL`` direct bit without the shift.

Run for BL=64/128/256 and compare removal counts and the seq-distance CDF to
decide the production window size (issue acceptance #9).

Usage::

    python -m simpler_setup.tools.wait_reduction_sim DEPS_JSON... [--bl 64,128,256] [--json OUT.json]

Modeling notes:

- Edges are OR-accumulated per ``(pred, succ)`` pair across their records
  (same convention ``deps.json`` documents for consumers). Only pairs whose
  accumulated flags contain ``wait`` participate in reachability; the
  ``wait``-only vs ``wait|retain`` split decides whether a removal is a pure
  drop or a RETAIN-only demotion.
- ``deps.json`` predates any runtime reduction: both replay passes record the
  as-constructed edge set, so the same capture serves as input for baseline
  and comparison alike.
- ``alloc_tensors`` tasks bypass the dep_gen capture point and appear only as
  edge ``pred`` values. They are inserted into the submission order just
  before their first consumer reference, which is the earliest position the
  runtime could have submitted them; their own fanin is empty by construction.
- Captures made before the ``flags`` field existed carry only ``source``; the
  source-to-flags mapping (creator -> wait|retain, tensormap -> wait,
  explicit -> wait|retain) reconstructs the replay's conservative flags.
- ``DepGenRecord`` does not preserve the kind of an explicit dependency
  (issue #1827), so an explicit runtime WAIT-only edge appears as
  WAIT|RETAIN in ``deps.json``. Removal counts remain valid, but the report
  marks demote-vs-drop classifications involving such pairs as uncertain.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

SOURCE_FLAGS = {
    "creator": ("wait", "retain"),
    "tensormap": ("wait",),
    "explicit": ("wait", "retain"),
}


def _edge_flags(edge: dict) -> frozenset[str]:
    """Per-record edge flags, falling back to the source-derived mapping."""
    flags = edge.get("flags")
    if flags is not None:
        return frozenset(flags)
    return frozenset(SOURCE_FLAGS.get(edge.get("source", ""), ("wait", "retain")))


def load_wait_graph(
    path: Path,
) -> tuple[
    list[str],
    dict[str, int],
    dict[tuple[str, str], frozenset[str]],
    frozenset[tuple[str, str]],
]:
    """Return graph data and pairs whose RETAIN classification is uncertain.

    The submission order is the ``tasks[]`` record order; alloc tasks that
    never appear in ``tasks[]`` are inserted immediately before their first
    consumer reference.
    """
    data = json.loads(path.read_text())
    order: list[str] = [t["task_id"] for t in data.get("tasks", [])]
    all_flags: dict[tuple[str, str], set[str]] = {}
    explicit_pairs: set[tuple[str, str]] = set()
    certain_retain_pairs: set[tuple[str, str]] = set()
    for edge in data.get("edges", []):
        pair = (edge["pred"], edge["succ"])
        flags = _edge_flags(edge)
        # Every record of a pair contributes, RETAIN-only ones included: the
        # pair is wait overall if any record waits, and retain overall if any
        # record retains (OR-accumulate).
        all_flags.setdefault(pair, set()).update(flags)
        if edge.get("source") == "explicit":
            explicit_pairs.add(pair)
        elif "retain" in flags:
            certain_retain_pairs.add(pair)
    # The WAIT graph is the subset that gates readiness somewhere; a pair no
    # record waits on is not one of its edges.
    pair_flags = {pair: flags for pair, flags in all_flags.items() if "wait" in flags}
    # Insert unseen preds before their first consumer's position in `order`.
    seen = set(order)
    for pred, _succ in pair_flags:
        if pred in seen:
            continue
        # Find the earliest consumer that is itself in the recorded order.
        pos = len(order)
        for p2, s2 in pair_flags:
            if p2 == pred and s2 in seen:
                pos = min(pos, order.index(s2))
        order.insert(pos, pred)
        seen.add(pred)
    seq = {task_id: i for i, task_id in enumerate(order)}
    uncertain_retain_pairs = (explicit_pairs & pair_flags.keys()) - certain_retain_pairs
    return (
        order,
        seq,
        {pair: frozenset(f) for pair, f in pair_flags.items()},
        frozenset(uncertain_retain_pairs),
    )


def full_reduction(order: list[str], pair_flags: dict[tuple[str, str], frozenset[str]]) -> set[tuple[str, str]]:
    """Exact full-DAG transitive reduction of the WAIT graph (upper bound)."""
    nodes = set(order)
    for pred, succ in pair_flags:
        nodes.add(pred)
        nodes.add(succ)
    succ_map: dict[str, set[str]] = {n: set() for n in nodes}
    indeg: dict[str, int] = {n: 0 for n in nodes}
    for u, v in pair_flags:
        if v not in succ_map[u]:
            succ_map[u].add(v)
            indeg[v] += 1
    queue = deque(n for n, d in indeg.items() if d == 0)
    topo: list[str] = []
    while queue:
        n = queue.popleft()
        topo.append(n)
        for m in succ_map[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    if len(topo) != len(nodes):
        raise ValueError("WAIT graph contains a cycle; reduction is ill-defined")
    reach: dict[str, set[str]] = {}
    redundant: set[tuple[str, str]] = set()
    for u in reversed(topo):
        indirect: set[str] = set()
        for v in succ_map[u]:
            indirect |= reach[v]
        for v in succ_map[u]:
            if v in indirect:
                redundant.add((u, v))
        reach[u] = indirect | succ_map[u]
    return redundant


def online_bitmap(
    order: list[str],
    seq: dict[str, int],
    pair_flags: dict[tuple[str, str], frozenset[str]],
    bl: int,
) -> set[tuple[str, str]]:
    """Mirror of the runtime reduce_wait_edges at window size ``bl``."""
    mask = (1 << bl) - 1
    wait_preds: dict[str, list[str]] = {}
    for pred, succ in pair_flags:
        wait_preds.setdefault(succ, []).append(pred)
    r_bitmaps: dict[str, int] = {t: 0 for t in order}
    removed: set[tuple[str, str]] = set()
    for t in order:
        seq_t = seq[t]
        direct = 0
        via = 0
        tracked: dict[str, int] = {}
        for p in wait_preds.get(t, []):
            d = seq_t - seq[p]
            if d < 1 or d > bl:
                continue  # window miss: kept conservatively by the runtime too
            tracked[p] = d
            direct |= 1 << (d - 1)
            if d < bl:
                via |= r_bitmaps[p] << d
        # Python ints are unbounded; the mask emulates the bl-wide register so
        # bits shifted past the window drop exactly as they would in silicon.
        r_bitmaps[t] = (direct | via) & mask
        for p, d in tracked.items():
            if via & (1 << (d - 1)):
                removed.add((p, t))
    return removed


def percentile(sorted_vals: list[int], frac: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(int(frac * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def simulate(path: Path, bls: list[int]) -> dict:
    order, seq, pair_flags, uncertain_retain_pairs = load_wait_graph(path)
    wait_pairs = {p for p, f in pair_flags.items() if "wait" in f}
    redundant = full_reduction(order, pair_flags)

    pair_distances = {(p, s): seq[s] - seq[p] for (p, s) in wait_pairs}
    distances = sorted(d for d in pair_distances.values() if d >= 0)
    cross_ring_pairs = {(pred, succ) for (pred, succ) in wait_pairs if (int(pred) >> 32) != (int(succ) >> 32)}
    report: dict = {
        "file": str(path),
        "tasks": len(order),
        "wait_pairs": len(wait_pairs),
        "full_reduction_upper_bound": len(redundant),
        "full_demote_to_retain": sum(1 for pair in redundant if "retain" in pair_flags.get(pair, frozenset())),
        "full_pure_drop": sum(1 for pair in redundant if "retain" not in pair_flags.get(pair, frozenset())),
        "full_retain_classification_uncertain": len(redundant & uncertain_retain_pairs),
        "seq_distance": {
            "p50": percentile(distances, 0.50),
            "p90": percentile(distances, 0.90),
            "p99": percentile(distances, 0.99),
            "max": distances[-1] if distances else 0,
        },
        "windows": {},
    }
    for bl in bls:
        removed = online_bitmap(order, seq, pair_flags, bl)
        within = sum(1 for d in distances if d <= bl)
        window_miss_pairs = {pair for pair, d in pair_distances.items() if d < 1 or d > bl}
        redundant_within = {pair for pair in redundant if pair_distances[pair] <= bl}
        cross_ring_redundant = redundant & cross_ring_pairs
        report["windows"][str(bl)] = {
            "removed": len(removed),
            "pct_of_upper_bound": (round(100.0 * len(removed) / len(redundant), 2) if redundant else 0.0),
            "demote_to_retain": sum(1 for pair in removed if "retain" in pair_flags.get(pair, frozenset())),
            "pure_drop": sum(1 for pair in removed if "retain" not in pair_flags.get(pair, frozenset())),
            "retain_classification_uncertain": len(removed & uncertain_retain_pairs),
            "pairs_within_window": within,
            "pct_pairs_within_window": (round(100.0 * within / len(distances), 2) if distances else 0.0),
            "window_miss_wait_pairs": len(window_miss_pairs),
            "redundant_window_misses": len(redundant & window_miss_pairs),
            "bitmap_misses_within_window": len(redundant_within - removed),
            "cross_ring_wait_pairs": len(cross_ring_pairs),
            "cross_ring_redundant_wait_pairs": len(cross_ring_redundant),
            "cross_ring_removed": len(removed & cross_ring_pairs),
            "cross_ring_misses": len(cross_ring_redundant - removed),
            "estimated_readiness_fanout_nodes_removed": len(removed),
            "estimated_dep_pool_entries_removed": len(removed),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wait_reduction_sim",
        description="Measure bounded-bitmap WAIT reduction coverage against the full-DAG upper bound.",
    )
    parser.add_argument("deps_json", nargs="+", type=Path, help="deps.json capture(s)")
    parser.add_argument(
        "--bl",
        default="64,128,256",
        help="comma-separated bitmap window sizes to simulate (default 64,128,256)",
    )
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    bls = [int(x) for x in args.bl.split(",") if x.strip()]
    reports = []
    for path in args.deps_json:
        try:
            reports.append(simulate(path, bls))
        except (OSError, ValueError, KeyError) as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            return 1

    for r in reports:
        print(f"== {r['file']}")
        print(
            f"   tasks={r['tasks']}  wait_pairs={r['wait_pairs']}  "
            f"upper_bound={r['full_reduction_upper_bound']} "
            f"(demote {r['full_demote_to_retain']} / drop {r['full_pure_drop']}; "
            f"classification uncertain {r['full_retain_classification_uncertain']})"
        )
        cdf = r["seq_distance"]
        print(f"   seq distance: p50={cdf['p50']} p90={cdf['p90']} p99={cdf['p99']} max={cdf['max']}")
        for bl in bls:
            w = r["windows"][str(bl)]
            print(
                f"   BL={bl:>3}: removed={w['removed']:>5} "
                f"({w['pct_of_upper_bound']:>6.2f}% of upper bound; "
                f"demote {w['demote_to_retain']} / drop {w['pure_drop']}; "
                f"window misses {w['redundant_window_misses']}; "
                f"cross-ring misses {w['cross_ring_misses']}; "
                f"{w['pct_pairs_within_window']:.1f}% pairs within window)"
            )
        print()

    if args.json is not None:
        args.json.write_text(json.dumps(reports, indent=1))
        print(f"json report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
