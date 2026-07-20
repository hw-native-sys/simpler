# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Critical-path analysis over an L2-swimlane build_output.

Given a run directory produced by a ``--enable-l2-swimlane`` run, this
reconstructs the critical path of each device execution and writes a report
beside each discovered swimlane-records file. The report describes, for every
task on that path, its wall-clock duration, the compute time it contributes,
and the runtime-scheduling stall (wait) before it, each as a fraction of the
global makespan.

Inputs consumed per rank (auto-discovered under the given directory):
  - ``l2_swimlane_records.json`` : per (task, core-block) execution slices
  - ``deps.json``                : task dependency DAG (producer -> consumer)
  - ``name_map*.json``           : callable_id -> kernel name
  - ``merged_swimlane*.json``    : optional Perfetto Worker View source

Two critical paths are computed:
  A) Static CPM  : longest duration-weighted path in the happens-before DAG,
                   i.e. the dependency-limited latency floor (unlimited cores).
  B) Observed    : the as-executed "blame" path, walked backward from the last
                   finishing task through whichever predecessor (data dependency
                   or same-core resource) most tightly gated each task's start.
                   Its compute + stall segments tile the makespan exactly.

Usage:
    python -m simpler_setup.tools.critical_path build_output/_jit_l3_decode_fwd_<ts>
    python -m simpler_setup.tools.critical_path build_output/<case> --top 60 --stdout
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .swimlane_converter import format_task_display

INF = float("inf")

# Structured per-task L2 records file. Current runs emit ``l2_swimlane_records.json``;
# ``l2_perf_records.json`` is the name used in some docs / older runtimes. The first
# name that exists in a rank dir is used.
SWIMLANE_RECORD_NAMES = ("l2_swimlane_records.json", "l2_perf_records.json")
STATIC_TRACE_NAME = "CPM_static.json"
OBSERVED_TRACE_NAME = "CPM_observed.json"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _find_records(d: Path) -> Path | None:
    """Return the structured swimlane-records file in *d*, or None."""
    for name in SWIMLANE_RECORD_NAMES:
        if (d / name).exists():
            return d / name
    return None


def _find_name_map(d: Path) -> Path | None:
    """Return the newest sibling ``name_map*.json`` file in *d*, or None."""
    candidates = sorted(
        (path for path in d.glob("name_map*.json") if path.is_file()),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    return candidates[-1] if candidates else None


def _find_merged_swimlane(d: Path) -> Path | None:
    """Return the newest sibling ``merged_swimlane*.json`` file in *d*, or None."""
    candidates = sorted(
        (path for path in d.glob("merged_swimlane*.json") if path.is_file()),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    return candidates[-1] if candidates else None


# --------------------------------------------------------------------------- #
# Data loading and per-task graph construction                                #
# --------------------------------------------------------------------------- #
@dataclass
class RankGraph:
    """Per-task timing + happens-before / resource predecessors for one rank."""

    freq: int
    start: dict[str, int]
    end: dict[str, int]
    dur: dict[str, int]
    name: dict[str, str]
    nblock: dict[str, int]
    hb_pred: dict[str, list[str]]  # data-dependency predecessors (time-forward)
    core_prev: dict[str, tuple[str, int]]  # same-core resource predecessor (task, end)
    t0: int
    t1: int
    kept_edges: int
    label: str


def discover_rank_dirs(root: Path) -> list[Path]:
    """Return every directory under *root* that holds a full swimlane triple."""
    if root.is_file():
        root = root.parent
    dirs: set[Path] = set()
    for name in SWIMLANE_RECORD_NAMES:
        for rec in root.rglob(name):
            d = rec.parent
            if (d / "deps.json").exists() and _find_name_map(d) is not None:
                dirs.add(d)
    return sorted(dirs)


def build_graph(rank_dir: Path, root: Path, tol: int) -> RankGraph:
    rec_file = _find_records(rank_dir)
    if rec_file is None:
        raise FileNotFoundError(f"no swimlane records ({' / '.join(SWIMLANE_RECORD_NAMES)}) in {rank_dir}")
    name_map_file = _find_name_map(rank_dir)
    if name_map_file is None:
        raise FileNotFoundError(f"no name_map*.json in {rank_dir}")
    sw = _read_json(rec_file)
    deps = _read_json(rank_dir / "deps.json")
    name_map = _read_json(name_map_file).get("callable_id_to_name", {})

    freq = int(sw["metadata"]["clock_freq_hz"])
    # aicore_tasks rows are [core_id, task_token, reg_task_id, start_tick, end_tick,
    # receive_to_start_cycles] (see L2SwimlaneCollector::export_swimlane_json). Kernel
    # names come from deps.json's kernel_ids, not from these rows.
    recs = sw["aicore_tasks"]
    if not recs:
        raise RuntimeError(f"no aicore_tasks records in {rec_file} (empty/failed profiling run?)")

    start: dict[str, int] = {}
    end: dict[str, int] = {}
    nblock: collections.Counter = collections.Counter()
    core_slices: dict[int, list[tuple[int, int, str]]] = collections.defaultdict(list)
    for core, tid, _seq, st, en, _recv in recs:
        t = str(tid)
        start[t] = min(start.get(t, INF), st)
        end[t] = max(end.get(t, -INF), en)
        nblock[t] += 1
        core_slices[core].append((st, en, t))

    dur = {t: end[t] - start[t] for t in start}
    # task_id / edge endpoints are strings in deps.json but ints in the swimlane
    # records; normalise everything to str so the joins never silently miss.
    tinfo = {str(t["task_id"]): t for t in deps["tasks"]}

    def name_of(t: str) -> str:
        ks = tinfo.get(t, {}).get("kernel_ids") or []
        cid = next((k for k in ks if k is not None and k >= 0), None)
        if cid is None:
            return "unknown"
        return name_map.get(str(cid), f"cid{cid}")

    name = {t: name_of(t) for t in start}

    # Happens-before DAG: keep a data-dep edge p->s only when p is timed, finished
    # by the time s started (+tol), AND started strictly earlier.  The deps graph
    # is already an acyclic producer->consumer DAG; the strict-start condition makes
    # edge retention antisymmetric, so the retained graph is provably acyclic even
    # under tick-level ties, and every edge is time-forward for the DP / backward walk.
    hb_pred: dict[str, list[str]] = collections.defaultdict(list)
    kept = 0
    seen: set[tuple[str, str]] = set()
    for e in deps["edges"]:
        p, s = str(e["pred"]), str(e["succ"])
        if p == s or p not in start or s not in start or (p, s) in seen:
            continue
        seen.add((p, s))
        if end[p] <= start[s] + tol and start[p] < start[s]:
            hb_pred[s].append(p)
            kept += 1

    # Same-core resource predecessor: the moment the core a task first lands on was
    # freed = max end over all earlier slices on that core (running-max handles any
    # pipelined/overlapping slices correctly, not just the immediately-prior one).
    core_prev: dict[str, tuple[str, int]] = {}
    for _core, sl in core_slices.items():
        sl.sort()
        best_end, best_task = -INF, None
        for st, en, t in sl:
            if best_task is not None and best_task != t and best_end <= st + tol and st == start[t]:
                cur = core_prev.get(t)
                if cur is None or best_end > cur[1]:
                    # best_end is a real tick here (set alongside best_task); the
                    # -INF float only survives while best_task is still None.
                    core_prev[t] = (best_task, int(best_end))
            if en > best_end:
                best_end, best_task = en, t

    label = str(rank_dir.relative_to(root)) if rank_dir != root else rank_dir.name
    return RankGraph(
        freq=freq,
        start=start,
        end=end,
        dur=dur,
        name=name,
        nblock=dict(nblock),
        hb_pred=hb_pred,
        core_prev=core_prev,
        t0=min(start.values()),
        t1=max(end.values()),
        kept_edges=kept,
        label=label,
    )


# --------------------------------------------------------------------------- #
# Critical-path algorithms                                                    #
# --------------------------------------------------------------------------- #
def topo_order(g: RankGraph) -> list[str]:
    indeg = {t: 0 for t in g.start}
    succ: dict[str, list[str]] = collections.defaultdict(list)
    for s, preds in g.hb_pred.items():
        for p in preds:
            succ[p].append(s)
            indeg[s] += 1
    q = collections.deque(t for t in g.start if indeg[t] == 0)
    order: list[str] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != len(g.start):
        raise RuntimeError(f"happens-before graph is cyclic ({len(order)}/{len(g.start)})")
    return order


def static_cpm(g: RankGraph, order: list[str]) -> tuple[list[str], int]:
    """Longest duration-weighted path in the happens-before DAG."""
    finish: dict[str, int] = {}
    choice: dict[str, str | None] = {}
    for v in order:
        best_p, best_f = None, 0
        for p in g.hb_pred.get(v, ()):
            if finish[p] > best_f:
                best_f, best_p = finish[p], p
        finish[v] = best_f + g.dur[v]
        choice[v] = best_p
    sink = max(finish, key=lambda t: finish[t])
    path: list[str] = []
    v: str | None = sink
    while v is not None:
        path.append(v)
        v = choice[v]
    path.reverse()
    return path, finish[sink]


@dataclass
class Segment:
    task: str
    name: str
    dur: int  # task wall-clock span (ticks)
    compute: int  # non-overlapped compute contributed on the path (ticks)
    stall: int  # gap before this task on the path (ticks)
    kind: str  # blame class for the gap: data-wait / core-wait / front-gap


@dataclass
class RankResult:
    label: str
    freq: int
    makespan: int
    cpm_len: int
    cpm_ntasks: int
    cpm_path: list[str]
    cpm_head: list[str]
    segments: list[Segment] = field(default_factory=list)
    compute_total: int = 0
    stall_total: int = 0
    stall_by_kind: dict[str, int] = field(default_factory=dict)
    ntasks: int = 0
    kept_edges: int = 0


def observed_path(g: RankGraph, tol: int) -> list[tuple[str, str]]:
    """Backward blame walk -> list of (task, gap_kind), sink-first then reversed."""
    sink = max(g.end, key=lambda t: g.end[t])
    walk: list[tuple[str, str]] = []
    v: str | None = sink
    seen: set[str] = set()
    while v is not None and v not in seen:
        seen.add(v)
        cand: list[tuple[int, str, str]] = []
        for p in g.hb_pred.get(v, ()):
            if g.end[p] <= g.start[v] + tol and g.start[p] < g.start[v]:
                cand.append((g.end[p], p, "data-wait"))
        cp = g.core_prev.get(v)
        if cp is not None and cp[1] <= g.start[v] + tol and g.start.get(cp[0], INF) < g.start[v]:
            cand.append((cp[1], cp[0], "core-wait"))
        if not cand:
            walk.append((v, "front-gap"))
            break
        cand.sort()
        _bind_end, bp, kind = cand[-1]
        walk.append((v, kind))
        v = bp
    walk.reverse()
    return walk


def analyze_rank(g: RankGraph, tol: int) -> RankResult:
    order = topo_order(g)
    cpm_path, cpm_len = static_cpm(g, order)
    walk = observed_path(g, tol)

    # Frontier sweep -> exact tiling of [t0, end[sink]] into compute + stall.
    segments: list[Segment] = []
    stall_by_kind: collections.Counter = collections.Counter()
    compute_total = stall_total = 0
    frontier = g.t0
    for t, kind in walk:
        s, e = g.start[t], g.end[t]
        gap = max(0, s - frontier)
        eff = max(0, e - max(s, frontier))
        segments.append(Segment(task=t, name=g.name[t], dur=g.dur[t], compute=eff, stall=gap, kind=kind))
        stall_by_kind[kind] += gap
        compute_total += eff
        stall_total += gap
        frontier = max(frontier, e)

    return RankResult(
        label=g.label,
        freq=g.freq,
        makespan=g.t1 - g.t0,
        cpm_len=cpm_len,
        cpm_ntasks=len(cpm_path),
        cpm_path=cpm_path,
        cpm_head=[g.name[t] for t in cpm_path[:10]],
        segments=segments,
        compute_total=compute_total,
        stall_total=stall_total,
        stall_by_kind=dict(stall_by_kind),
        ntasks=len(g.start),
        kept_edges=g.kept_edges,
    )


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #
def _us(ticks: int, freq: int) -> float:
    return ticks / freq * 1e6


def _pct(part: int, whole: int) -> float:
    return part / whole * 100 if whole else 0.0


def render_report(results: list[RankResult], top: int) -> str:
    out: list[str] = []
    w = out.append
    w("# Critical-path report (L2 swimlane)\n")
    w(
        "Generated by `python -m simpler_setup.tools.critical_path`. "
        "Times are device wall-clock; percentages are of each rank's makespan.\n"
    )
    w("- **Static CPM path**: dependency-limited latency floor (unlimited cores).")
    w(
        "- **Observed path**: the as-executed critical path; its per-task compute + "
        "the runtime-scheduling stall before each task tile the makespan exactly."
    )
    w(
        "- **stall kinds**: `data-wait` = waiting on an upstream producer; "
        "`core-wait` = waiting for the assigned core to be freed (resource "
        "serialization); `front-gap` = launch/dispatch delay before the first task.\n"
    )

    for r in results:
        ms = r.makespan
        w(f"\n## Rank `{r.label}`\n")
        w(f"- tasks: {r.ntasks} | happens-before edges: {r.kept_edges}")
        w(f"- **makespan**: {_us(ms, r.freq) / 1e3:.3f} ms")
        w(
            f"- **static CPM path**: {_us(r.cpm_len, r.freq) / 1e3:.3f} ms "
            f"({_pct(r.cpm_len, ms):.1f}% of makespan) over {r.cpm_ntasks} tasks"
        )
        w(f"- **observed critical path**: {len(r.segments)} tasks")
        w(f"  - compute: {_us(r.compute_total, r.freq) / 1e3:.3f} ms ({_pct(r.compute_total, ms):.1f}%)")
        w(f"  - stall (runtime scheduling): {_us(r.stall_total, r.freq) / 1e3:.3f} ms ({_pct(r.stall_total, ms):.1f}%)")
        for kind, val in sorted(r.stall_by_kind.items(), key=lambda kv: -kv[1]):
            if val:
                w(f"    - {kind}: {_us(val, r.freq) / 1e3:.3f} ms ({_pct(val, ms):.1f}%)")
        check = r.compute_total + r.stall_total
        w(
            f"  - tiling check: compute+stall = {check} ticks vs makespan {ms} ticks "
            f"({'exact' if check == ms else f'diff {check - ms}'})"
        )

        # Time by kernel family on the observed path.
        fam_c: collections.Counter = collections.Counter()
        fam_s: collections.Counter = collections.Counter()
        fam_n: collections.Counter = collections.Counter()
        for seg in r.segments:
            fam = _family(seg.name)
            fam_c[fam] += seg.compute
            fam_s[fam] += seg.stall
            fam_n[fam] += 1
        w("\n### Time on the observed critical path, by kernel family\n")
        w("| kernel family | compute ms | % makespan | stall ms | # on path |")
        w("|---|---:|---:|---:|---:|")
        for fam, cval in fam_c.most_common(top):
            w(
                f"| {fam} | {_us(cval, r.freq) / 1e3:.3f} | {_pct(cval, ms):.1f}% | "
                f"{_us(fam_s[fam], r.freq) / 1e3:.3f} | {fam_n[fam]} |"
            )

        # Full per-task listing of the observed critical path.
        w(f"\n### Observed critical path — all {len(r.segments)} tasks\n")
        w("| # | kernel | task µs | compute µs (%mk) | stall µs (%mk) | stall kind |")
        w("|---:|---|---:|---:|---:|---|")
        for i, seg in enumerate(r.segments):
            w(
                f"| {i} | {seg.name} | {_us(seg.dur, r.freq):.1f} | "
                f"{_us(seg.compute, r.freq):.1f} ({_pct(seg.compute, ms):.2f}%) | "
                f"{_us(seg.stall, r.freq):.1f} ({_pct(seg.stall, ms):.2f}%) | "
                f"{seg.kind if seg.stall else ''} |"
            )
    w("")
    return "\n".join(out)


def _family(name: str) -> str:
    # Strip an optional trailing block index and/or an aic/aiv suffix in a single
    # pass, so 'gate_0_aic', 'hc_pre_fused_3_aic' and 'exp_up_mm_2' all fold to
    # their family ('gate', 'hc_pre_fused', 'exp_up_mm').
    return re.sub(r"(_\d+)?(_(?:aic|aiv))?$", "", name)


def render_summary(results: list[RankResult]) -> str:
    lines: list[str] = []
    for r in results:
        ms = r.makespan
        by_kind = sorted(r.stall_by_kind.items(), key=lambda kv: -kv[1])
        detail = ", ".join(f"{k} {_pct(v, ms):.1f}%" for k, v in by_kind if v)
        lines.append(
            f"[{r.label}] makespan={_us(ms, r.freq) / 1e3:.3f}ms  "
            f"CPM={_us(r.cpm_len, r.freq) / 1e3:.2f}ms({_pct(r.cpm_len, ms):.0f}%)  "
            f"compute={_pct(r.compute_total, ms):.1f}%  "
            f"stall={_pct(r.stall_total, ms):.1f}% ({detail})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Perfetto critical-path traces                                              #
# --------------------------------------------------------------------------- #
def _trace_events(trace_data) -> list[dict]:
    if isinstance(trace_data, dict):
        events = trace_data.get("traceEvents")
    elif isinstance(trace_data, list):
        events = trace_data
    else:
        events = None
    if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
        raise ValueError("merged swimlane JSON must contain a traceEvents list")
    return events


def _critical_trace(trace_data, critical_task_ids: set[str]):
    """Copy the merged trace and anonymize non-critical Worker View task bars."""
    events = [event.copy() for event in _trace_events(trace_data)]
    visible_task_ids = {
        str(event.get("args", {}).get("taskId"))
        for event in events
        if event.get("ph") == "X" and event.get("pid") == 4 and event.get("args", {}).get("taskId") is not None
    }
    missing = critical_task_ids - visible_task_ids
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(f"merged Worker View is missing {len(missing)} critical task(s): {sample}")

    for event in events:
        task_id = event.get("args", {}).get("taskId")
        if event.get("ph") != "X" or event.get("pid") != 4 or task_id is None or str(task_id) in critical_task_ids:
            continue
        event["name"] = f"·({format_task_display(task_id)})"

    if isinstance(trace_data, dict):
        output = trace_data.copy()
        output["traceEvents"] = events
        return output
    return events


def _write_critical_path_traces(rank_dir: Path, result: RankResult) -> list[Path]:
    merged_path = _find_merged_swimlane(rank_dir)
    if merged_path is None:
        return []
    merged_trace = _read_json(merged_path)
    paths = (
        (STATIC_TRACE_NAME, set(result.cpm_path)),
        (OBSERVED_TRACE_NAME, {segment.task for segment in result.segments}),
    )
    traces: list[tuple[Path, object]] = []
    for filename, critical_task_ids in paths:
        output_path = rank_dir / filename
        trace = _critical_trace(merged_trace, critical_task_ids)
        traces.append((output_path, trace))

    output_paths: list[Path] = []
    for output_path, trace in traces:
        output_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        output_paths.append(output_path)
    return output_paths


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reconstruct the critical path of an L2-swimlane run tree "
        "and write a compute/stall report plus Perfetto critical-path traces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "build_dir",
        type=Path,
        help="directory containing outputs/... or dfx_outputs/.../l2_swimlane_records.json artifacts",
    )
    p.add_argument(
        "--report",
        default="critical_path_report.md",
        help="report filename, written beside each discovered swimlane records file",
    )
    p.add_argument("--top", type=int, default=25, help="rows in the kernel-family table")
    p.add_argument("--tol", type=int, default=2, help="tick tolerance when deciding 'finished before' (clock ticks)")
    p.add_argument("--stdout", action="store_true", help="also print a one-line-per-rank summary to stdout")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root: Path = args.build_dir.expanduser().resolve()
    if not root.exists():
        print(f"error: path not found: {root}", file=sys.stderr)
        return 2
    # If a file (e.g. a records file) was passed, anchor discovery and the
    # relative-path labels on its parent directory.
    if root.is_file():
        root = root.parent
    report_name = Path(args.report)
    if report_name.is_absolute() or report_name.name != args.report:
        print("error: --report must be a filename, not a path", file=sys.stderr)
        return 2
    rank_dirs = discover_rank_dirs(root)
    if not rank_dirs:
        print(
            f"error: no l2_swimlane_records.json (+deps.json+name_map*.json) found "
            f"under {root}. Did the run use --enable-l2-swimlane?",
            file=sys.stderr,
        )
        return 2

    results: list[RankResult] = []
    report_paths: list[Path] = []
    trace_paths: list[Path] = []
    had_trace_errors = False
    for d in rank_dirs:
        g = build_graph(d, root, args.tol)
        result = analyze_rank(g, args.tol)
        results.append(result)
        out_path = d / args.report
        out_path.write_text(render_report([result], args.top), encoding="utf-8")
        report_paths.append(out_path)
        try:
            written_traces = _write_critical_path_traces(d, result)
        except (OSError, ValueError) as exc:
            print(
                f"error: invalid merged_swimlane*.json in {d}: {exc}",
                file=sys.stderr,
            )
            had_trace_errors = True
            continue
        if written_traces:
            trace_paths.extend(written_traces)
        else:
            print(
                f"Warning: no merged_swimlane*.json in {d}; skipped Perfetto critical-path traces",
                file=sys.stderr,
            )

    print(f"Analyzed {len(results)} rank(s) under {root}")
    print(render_summary(results))
    for out_path in report_paths:
        print(f"Report written to: {out_path}")
    for trace_path in trace_paths:
        print(f"Perfetto trace written to: {trace_path}")
    if args.stdout:
        print("\n" + render_report(results, args.top))
    return 2 if had_trace_errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
