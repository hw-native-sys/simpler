#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Conditional zero-overhead resource lower bound with SPMD common admission."""

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import struct
from collections import Counter, defaultdict, deque
from pathlib import Path

from ortools.sat.python import cp_model

QUANTUM_NS = 100


def _serial_hint(nodes, capacities, finish_lb, previous):
    if not previous or not previous.get("schedule"):
        # A serial sequence of ready tasks, with capacity-sized block waves,
        # is an explicit feasible hint; it never alters the lower-bound model.
        serial = []
        cursor = 0
        for tid in finish_lb:
            groups = defaultdict(list)
            for block, lane, kind, duration in nodes[tid]["units"]:
                groups[block].append((block, lane, kind, duration))
            wave = []
            used = Counter()

            def flush_wave(wave, begin):
                for block, lane, kind, duration in wave:
                    serial.append(
                        {
                            "task": tid,
                            "block": block,
                            "lane": lane,
                            "kind": kind,
                            "start": begin,
                            "end": begin + duration,
                            "duration": duration,
                        }
                    )
                return begin + max((u[3] for u in wave), default=0)

            for group in groups.values():
                need = Counter(u[2] for u in group)
                if any(used[k] + need[k] > capacities[k] for k in need):
                    cursor = flush_wave(wave, cursor)
                    wave = []
                    used = Counter()
                wave.extend(group)
                used.update(need)
            cursor = flush_wave(wave, cursor)
        prior_lower = previous.get("lower_ticks", 0) if previous else 0
        previous = {"schedule": serial, "lower_ticks": prior_lower, "feasible_ticks": cursor}
    return previous


def _add_hints(model, previous, units, starts, ends, finish_lb, edges, makespan):
    if previous and previous.get("schedule"):
        prior = {(u["task"], u["block"], u["lane"]): u for u in previous["schedule"]}
        hinted = set()
        for tid, block, lane, kind, duration, start, end in units:
            u = prior[tid, block, lane]
            assert u["duration"] == duration
            for var, value in [(start, u["start"]), (end, u["end"])]:
                if var.index not in hinted:
                    model.add_hint(var, value)
                    hinted.add(var.index)
        hinted_finish = {}
        incoming = defaultdict(list)
        for a, b in edges:
            incoming[b].append(a)
        for tid in finish_lb:
            selected = [u for u in previous["schedule"] if u["task"] == tid]
            begin = min((u["start"] for u in selected), default=max([hinted_finish[a] for a in incoming[tid]] + [0]))
            end = max((u["end"] for u in selected), default=begin)
            hinted_finish[tid] = end
            for var, value in [(starts[tid], begin), (ends[tid], end)]:
                if var.index not in hinted:
                    model.add_hint(var, value)
                    hinted.add(var.index)
        feasible_upper = max(hinted_finish.values(), default=0)
        model.add(makespan <= feasible_upper)
        model.add_hint(makespan, feasible_upper)
        if previous.get("lower_ticks") is not None:
            model.add(makespan >= previous["lower_ticks"])


def _check_schedule(nodes, capacities, schedule):
    # Independently check capacities, data dependencies and common-start constraints.
    events = defaultdict(list)
    per_task = defaultdict(list)
    per_block = defaultdict(list)
    for u in schedule:
        assert u["end"] - u["start"] == u["duration"]
        per_task[u["task"]].append(u)
        per_block[u["task"], u["block"]].append(u)
        if u["duration"]:
            events[u["kind"]] += [(u["start"], 1), (u["end"], -1)]
    for kind, ev in events.items():
        active = 0
        for _, delta in sorted(ev):
            active += delta
            assert 0 <= active <= capacities[kind]
        assert active == 0
    for tid, node in nodes.items():
        if node["sync"]:
            assert len({u["start"] for u in per_task[tid]}) <= 1
    for group in per_block.values():
        assert len({u["start"] for u in group}) <= 1


def _add_dependencies(model, nodes, edges, starts, ends, lower_duration):
    succ = defaultdict(set)
    indeg = dict.fromkeys(nodes, 0)
    for a, b in edges:
        assert a in nodes and b in nodes
        model.add(starts[b] >= ends[a])
        if b not in succ[a]:
            succ[a].add(b)
            indeg[b] += 1
    ready = deque(n for n in nodes if indeg[n] == 0)
    release_lb = defaultdict(int)
    finish_lb = {}
    while ready:
        n = ready.popleft()
        finish_lb[n] = release_lb[n] + lower_duration[n]
        for b in succ[n]:
            release_lb[b] = max(release_lb[b], finish_lb[n])
            indeg[b] -= 1
            if indeg[b] == 0:
                ready.append(b)
    assert len(finish_lb) == len(nodes), "cyclic graph"
    return finish_lb


def solve_model(nodes, edges, capacities, seconds=3, previous=None):
    """Return a proven lower bound and a separately validated feasible schedule."""
    if seconds <= 0 or not capacities or any(value <= 0 for value in capacities.values()):
        raise ValueError("Positive solver time and capacities are required")
    if not nodes:
        raise ValueError("The model requires at least one task")
    for node in nodes.values():
        if any(unit[2] not in capacities or unit[3] < 0 for unit in node["units"]):
            raise ValueError("Unknown engine or negative duration")
    # nodes: {task_id: {sync: bool, units: [(block, lane, kind, integer_duration)]}}
    model = cp_model.CpModel()
    horizon = max(1, sum(u[3] for n in nodes.values() for u in n["units"]))
    starts = {}
    ends = {}
    intervals = defaultdict(list)
    units = []
    demands = Counter()
    lower_duration = {}
    for tid, node in nodes.items():
        release = model.new_int_var(0, horizon, f"t{tid}_start")
        task_end = model.new_int_var(0, horizon, f"t{tid}_end")
        starts[tid] = release
        ends[tid] = task_end
        groups = {}
        unit_ends = []
        work = Counter()
        for block, lane, kind, duration in node["units"]:
            # MIX subtasks share a block admission; sync_start joins all blocks too.
            key = "cohort" if node["sync"] else ("block", block)
            if key not in groups:
                groups[key] = release if node["sync"] else model.new_int_var(0, horizon, f"t{tid}_b{block}_start")
            start = groups[key]
            model.add(start >= release)
            end = model.new_int_var(0, horizon, f"t{tid}_b{block}_l{lane}_end")
            model.add(end == start + duration)
            interval = model.new_interval_var(start, duration, end, f"t{tid}_b{block}_l{lane}")
            intervals[kind].append(interval)
            unit_ends.append(end)
            units.append((tid, block, lane, kind, duration, start, end))
            work[kind] += duration
            demands[kind] += duration
        if unit_ends:
            model.add_max_equality(task_end, unit_ends)
            model.add_min_equality(release, list(groups.values()))
        else:
            model.add(task_end == release)
        lower_duration[tid] = max(
            [u[3] for u in node["units"]] + [math.ceil(work[k] / capacities[k]) for k in capacities] + [0]
        )
    for kind, items in intervals.items():
        model.add_cumulative(items, [1] * len(items), capacities[kind])
    finish_lb = _add_dependencies(model, nodes, edges, starts, ends, lower_duration)
    simple_lb = max(list(finish_lb.values()) + [math.ceil(demands[k] / capacities[k]) for k in capacities] + [0])
    makespan = model.new_int_var(simple_lb, horizon, "makespan")
    model.add_max_equality(makespan, list(ends.values()))
    model.minimize(makespan)
    previous = _serial_hint(nodes, capacities, finish_lb, previous)
    _add_hints(model, previous, units, starts, ends, finish_lb, edges, makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 2
    solver.parameters.random_seed = 17
    status = solver.solve(model)
    status_name = solver.status_name(status)
    assert status not in (cp_model.INFEASIBLE, cp_model.MODEL_INVALID), status_name
    # Best objective bound is the proven lower bound; an incumbent alone is not an optimum.
    lower = max(
        simple_lb, math.floor(solver.best_objective_bound + 1e-7), previous.get("lower_ticks", 0) if previous else 0
    )
    upper = None
    schedule = []
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        upper = int(round(solver.objective_value))
        assert lower <= upper
        for tid, block, lane, kind, duration, start, end in units:
            schedule.append(
                {
                    "task": tid,
                    "block": block,
                    "lane": lane,
                    "kind": kind,
                    "start": solver.value(start),
                    "end": solver.value(end),
                    "duration": duration,
                }
            )
        _check_schedule(nodes, capacities, schedule)
        for a, b in edges:
            assert solver.value(starts[b]) >= solver.value(ends[a])
    if upper is None and previous and previous.get("feasible_ticks") is not None:
        upper = previous["feasible_ticks"]
        schedule = previous["schedule"]
        status_name += " (previous feasible retained)"
    return {
        "time_limit_seconds": seconds,
        "status": status_name,
        "lower_ticks": lower,
        "feasible_ticks": upper,
        "simple_relaxation_ticks": simple_lb,
        "optimal": status == cp_model.OPTIMAL,
        "wall_seconds": solver.wall_time,
        "schedule": schedule,
    }


def plan_sync(bundle):
    data = (bundle / "plan-rank0.bin").read_bytes()
    size = struct.unpack_from("<Q", data, 8)[0]
    p = 16
    result = {}
    while p < len(data):
        h = struct.unpack_from("<4IQ8i", data, p)
        if h[0] == 2 and any(k >= 0 for k in h[5:8]):
            result[h[4]] = bool(h[9])
        p += 56 + h[1] * size + 8 * (h[2] + h[3])
    assert p == len(data)
    return result


def analyze_case(bundle, capture, output, seconds=3):
    root = Path(__file__).resolve().parents[1]
    path = root / "production/ideal_bound.py"
    spec = importlib.util.spec_from_file_location("baseline_bound", path)
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    checked = helper.read_case(bundle, capture)  # coverage, DAG, metadata, validation and topology audit
    meta = json.loads((bundle / "case.json").read_text())
    flags = plan_sync(bundle)
    table = root / "workloads" / meta["flavor"] / meta["case"] / "tasks.tsv"
    original_ids = {
        int(line.split()[0]) for line in table.read_text().splitlines() if line and not line.startswith("#")
    }
    output.mkdir(parents=True, exist_ok=True)
    results = []
    signature = None
    for iteration in range(checked["warmup"], checked["warmup"] + checked["rounds"]):
        folder = capture / f"run{iteration}"
        records = json.loads((folder / "chip_swimlane_records.json").read_text())
        deps = helper.complete_flat_dag(
            json.loads((folder / "deps.json").read_text()), folder / "source-task-map.txt", original_ids, 1
        )
        mapping = {
            target: source
            for source, target in (
                map(int, line.split()) for line in (folder / "source-task-map.txt").read_text().splitlines()
            )
        }
        kinds = records["metadata"]["core_types"]
        hz = records["metadata"]["clock_freq_hz"]
        identities = [
            {
                "run_epoch": iteration + 1,
                "core_id": r[0],
                "task_id": r[1],
                "reg_task_id": r[2],
                "core_type": kinds[r[0]],
            }
            for r in records["aicore_tasks"]
        ]
        helper.attach_execution_identities(records["aicore_execution_identities"], identities, "host_build_graph")
        nodes = {
            int(t["task_id"]): {"sync": flags.get(mapping[int(t["task_id"])], False), "units": []}
            for t in deps["tasks"]
        }
        for row, ident in zip(records["aicore_tasks"], identities, strict=True):
            ticks = math.floor((row[4] - row[3]) * 1e9 / hz / QUANTUM_NS)
            nodes[int(row[1])]["units"].append((ident["block_idx"], ident["subtask_slot"], kinds[row[0]], ticks))
        for node in nodes.values():
            if node["sync"]:
                c = Counter(u[2] for u in node["units"])
                assert c["aic"] <= 32 and c["aiv"] <= 64
        edges = sorted({(int(e["pred"]), int(e["succ"])) for e in deps["edges"]})
        current = (
            [(n, nd["sync"], [(u[0], u[1], u[2]) for u in sorted(nd["units"])]) for n, nd in sorted(nodes.items())],
            edges,
        )
        if signature is None:
            signature = current
        else:
            assert current == signature
        problem = {"nodes": nodes, "edges": edges, "quantum_ns": QUANTUM_NS, "capacities": {"aic": 32, "aiv": 64}}
        problem_sha256 = hashlib.sha256(json.dumps(problem, sort_keys=True).encode()).hexdigest()
        prior_path = output / f"run{iteration}.json"
        prior = json.loads(prior_path.read_text()) if prior_path.exists() else None
        if prior and prior.get("problem_sha256") != problem_sha256:
            raise ValueError("Output contains a different problem; use a fresh output directory")
        if prior and prior["optimal"]:
            solved = prior
        else:
            solved = solve_model(nodes, edges, {"aic": 32, "aiv": 64}, seconds, previous=prior)
        envelope = (max(r[4] for r in records["aicore_tasks"]) - min(r[3] for r in records["aicore_tasks"])) * 1000 / hz
        solved["lower_bound_ms"] = solved["lower_ticks"] * QUANTUM_NS / 1e6
        solved["feasible_ms"] = (
            None if solved["feasible_ticks"] is None else solved["feasible_ticks"] * QUANTUM_NS / 1e6
        )
        solved.update(
            problem_sha256=problem_sha256,
            iteration=iteration,
            kernel_envelope_ms=envelope,
            sync_tasks=sum(n["sync"] for n in nodes.values()),
            quantum_ns=QUANTUM_NS,
        )
        assert solved["lower_bound_ms"] <= envelope + 1e-9
        (output / f"run{iteration}.json").write_text(json.dumps(solved, indent=2) + "\n")
        results.append({k: v for k, v in solved.items() if k != "schedule"})
    summary = {
        "case": meta["case"],
        "flavor": meta["flavor"],
        "device": checked["device"],
        "mean_lower_bound_ms": statistics.mean(x["lower_bound_ms"] for x in results),
        "all_optimal": all(x["optimal"] for x in results),
        "mean_feasible_ms": statistics.mean(x["feasible_ms"] for x in results)
        if all(x["feasible_ms"] is not None for x in results)
        else None,
        "samples": results,
        "model": (
            "typed-capacity, non-preemptive block intervals; MIX per-block common start; "
            "sync_start all-block common start; task-completion DAG; zero scheduler cost; "
            "cluster placement relaxed; measured per-variant L1 durations floored to 100ns"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("capture", type=Path, help="Validated main HBG L1 rank directory")
    parser.add_argument("output", type=Path, help="Model results, schedules and per-round solver bounds")
    parser.add_argument("--time-limit", type=float, default=30, help="Maximum CPU seconds per measured round")
    args = parser.parse_args()
    result = analyze_case(args.bundle.resolve(), args.capture.resolve(), args.output.resolve(), args.time_limit)
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
