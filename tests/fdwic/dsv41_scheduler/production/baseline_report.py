#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reduce paired L1 measurements and an explicit zero-scheduler-cost DAG model."""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path

from audit_graph import audit
from graph_topology import provenance

from simpler_setup.tools.hbg.execution_identity import attach_execution_identities
from simpler_setup.tools.strace_timing import parse_spans


def kernel_span_ms(records):
    if records["chip_swimlane_level"] != 1:
        raise ValueError("Both baseline metrics require the same L1 run")
    rows = records["aicore_tasks"]
    hz = records["metadata"]["clock_freq_hz"]
    if not rows or hz <= 0 or any(row[3] <= 0 or row[4] <= row[3] for row in rows):
        raise ValueError("Missing or invalid kernel interval")
    return (max(row[4] for row in rows) - min(row[3] for row in rows)) * 1000 / hz


def paired_means(ranks):
    if not ranks or not ranks[0]:
        raise ValueError("Empty rank samples")
    count = len(ranks[0])
    if any(len(rank) != count for rank in ranks):
        raise ValueError("Rank sample counts disagree")
    result = {}
    for metric in ("device_wall_ms", "aicore_span_ms", "ideal_lower_bound_ms"):
        values = [sample[metric] for rank in ranks for sample in rank]
        result[metric] = (
            None
            if any(value is None for value in values)
            else statistics.mean(max(rank[i][metric] for rank in ranks) for i in range(count))
        )
    return result


def ideal_lower_bound_ms(records, deps):
    """A lower bound under fixed measured kernel durations and zero scheduling cost.

    Caller must establish that no asynchronous/cross-rank completion is omitted.
    The DAG has task-completion edges. Per-task and global AIC/AIV work bounds
    constrain the otherwise unlimited-ready-task model; no schedule is constructed.
    """
    hz = records["metadata"]["clock_freq_hz"]
    kinds = records["metadata"]["core_types"]
    capacities = Counter(kinds)
    if hz <= 0 or capacities["aic"] == 0 or capacities["aiv"] == 0:
        raise ValueError("Missing physical worker capacity or clock frequency")
    tasks = {int(task["task_id"]): task for task in deps["tasks"]}
    if len(tasks) != len(deps["tasks"]):
        raise ValueError("Duplicate dependency task")
    blocks, total = defaultdict(list), Counter()
    for row in records["aicore_tasks"]:
        tid, core, start, end = int(row[1]), int(row[0]), row[3], row[4]
        if tid not in tasks or not 0 <= core < len(kinds) or end <= start:
            raise ValueError("Kernel interval has no matching dependency node")
        blocks[tid].append((kinds[core], end - start))
        total[kinds[core]] += end - start
    duration = {}
    for tid, task in tasks.items():
        work = Counter()
        for kind, cycles in blocks[tid]:
            work[kind] += cycles
        if any(kernel >= 0 for kernel in task["kernel_ids"]) and not blocks[tid]:
            raise ValueError("Dependency DAG contains an unmeasured kernel")
        duration[tid] = max(
            max((cycles for _, cycles in blocks[tid]), default=0),
            work["aic"] / capacities["aic"],
            work["aiv"] / capacities["aiv"],
        )
    successors = defaultdict(set)
    indegree = dict.fromkeys(tasks, 0)
    for edge in deps["edges"]:
        pred, succ = int(edge["pred"]), int(edge["succ"])
        if pred not in tasks or succ not in tasks:
            raise ValueError("Dependency DAG has an unobserved endpoint")
        if succ not in successors[pred]:
            successors[pred].add(succ)
            indegree[succ] += 1
    ready = deque(tid for tid, degree in indegree.items() if degree == 0)
    finish, release, visited = {}, defaultdict(float), 0
    while ready:
        tid = ready.popleft()
        finish[tid] = release[tid] + duration[tid]
        visited += 1
        for succ in successors[tid]:
            release[succ] = max(release[succ], finish[tid])
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
    if visited != len(tasks):
        raise ValueError("Dependency graph contains a cycle")
    cycles = max(max(finish.values(), default=0), total["aic"] / capacities["aic"], total["aiv"] / capacities["aiv"])
    return cycles * 1000 / hz


def complete_flat_dag(deps, mapping_path, original_ids, repetitions):
    mappings = [tuple(map(int, line.split())) for line in mapping_path.read_text().splitlines()]
    targets = [target for _, target in mappings]
    counts = Counter(source for source, _ in mappings if source in original_ids)
    if len(set(targets)) != len(targets) or counts != Counter(dict.fromkeys(original_ids, repetitions)):
        raise ValueError("Original task mapping is incomplete or duplicated")
    compute = {target for source, target in mappings if source in original_ids}
    allocations = set(targets) - compute
    observed = {int(task["task_id"]) for task in deps["tasks"]}
    if not compute <= observed:
        raise ValueError("Original submitted task is absent from the dependency capture")
    endpoints = {int(edge[key]) for edge in deps["edges"] for key in ("pred", "succ")}
    missing = endpoints - observed
    if not missing <= allocations:
        raise ValueError("Dependency endpoint is neither captured nor a verified Host allocation")
    return {
        **deps,
        "tasks": deps["tasks"]
        + [{"task_id": str(tid), "kernel_ids": [-1, -1, -1], "block_num": 0} for tid in sorted(missing)],
    }


def read_rank(item, bundle, folder, workload, warmup, rounds, topology):
    result = json.loads((folder / "result.json").read_text())
    if not result["passed"] or (result["warmup"], result["rounds"]) != (warmup, rounds):
        raise ValueError("Run failed numerical validation or sample contract")
    entries = re.findall(
        r"run=(\d+) scheduler_mode=(\d+) profiling=(\d+) warmup=(\d+)", (folder / "console.log").read_text()
    )
    count = warmup + rounds
    if len(entries) != count or any(
        tuple(map(int, row)) != (i, 1, 1, int(i < warmup)) for i, row in enumerate(entries)
    ):
        raise ValueError("Only complete resident L1 measurements are accepted")
    spans = [
        span
        for path in (folder / "host").glob("host.*.log")
        for span in parse_spans(path.read_text().splitlines())
        if span.name == "chip.run.runner_run.device_wall" and span.is_device
    ]
    spans.sort(key=lambda span: (span.pid, span.inv))
    if len(spans) != (count if result["snapshot_inputs"] else 1 + 2 * count):
        raise ValueError("Incomplete device-wall spans")
    shapes = [line.split() for line in workload.read_text().splitlines() if line and not line.startswith("#")]
    unavailable = "cross-rank/asynchronous completion events are absent from L1" if item["ranks"] > 1 else None
    if any(int(row[-1]) for row in shapes):
        unavailable = "defer_wait logical completion events are absent from L1"
    samples = []
    for i in range(warmup, count):
        run = folder / f"run{i}"
        audit(bundle, run, workload)
        records = json.loads((run / "chip_swimlane_records.json").read_text())
        span = spans[i if result["snapshot_inputs"] else 2 + 2 * i]
        epochs = {row["run_epoch"] for row in records["aicore_execution_identities"]["records"]}
        if epochs != {span.inv}:
            raise ValueError("Kernel and device-wall samples refer to different invocations")
        kinds = records["metadata"]["core_types"]
        tasks = [
            {
                "run_epoch": span.inv,
                "core_id": row[0],
                "reg_task_id": row[2],
                "task_id": row[1],
                "core_type": kinds[row[0]],
            }
            for row in records["aicore_tasks"]
        ]
        attach_execution_identities(records["aicore_execution_identities"], tasks, records["metadata"]["runtime"])
        device_ms, core_ms = span.dur / 1e6, kernel_span_ms(records)
        if device_ms <= 0 or core_ms > device_ms + 0.002:
            raise ValueError("Kernel envelope exceeds its device-wall interval")
        bound = None
        if unavailable is None:
            deps = (
                topology
                if topology is not None
                else complete_flat_dag(
                    json.loads((run / "deps.json").read_text()),
                    run / "source-task-map.txt",
                    {int(row[0]) for row in shapes},
                    item["repetitions"],
                )
            )
            bound = ideal_lower_bound_ms(records, deps)
            if bound > core_ms + 0.002:
                raise ValueError("The model lower bound exceeds the observed kernel envelope")
        samples.append(
            {
                "iteration": i,
                "invocation": span.inv,
                "device_wall_ms": device_ms,
                "aicore_span_ms": core_ms,
                "ideal_lower_bound_ms": bound,
            }
        )
    return {"device": result["device"], "samples": samples, "ideal_unavailable_reason": unavailable}


def summarize(bundles, captures, topology_root, matrix):
    if matrix["profiling_level"] != 1:
        raise ValueError("The retained baseline is measured at L1")
    suite = Path(__file__).resolve().parents[1]
    results = []
    for item in matrix["cases"]:
        bundle, capture = bundles / item["id"], captures / item["id"]
        topology = None
        if item["mode"] == "graph" and item["ranks"] == 1:
            topology = json.loads((topology_root / (item["id"] + ".json")).read_text())
            if topology.get("baseline_provenance") != provenance(bundle):
                raise ValueError("Graph topology comes from a different bundle or collection level")
        workload = suite / "workloads" / item["flavor"] / item["case"] / "tasks.tsv"
        ranks = [
            read_rank(item, bundle, capture / f"rank{rank}", workload, matrix["warmup"], matrix["rounds"], topology)
            for rank in range(item["ranks"])
        ]
        values = paired_means([rank["samples"] for rank in ranks])
        row = {**item, "means": values, "ranks_data": ranks}
        results.append(row)
        print(item["id"], json.dumps(values), flush=True)
    by_case = defaultdict(set)
    for row in results:
        by_case[row["case"]].add(tuple(rank["device"] for rank in row["ranks_data"]))
    if any(len(devices) != 1 for devices in by_case.values()):
        raise ValueError("Variants of one workload did not use the same device allocation")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", type=Path)
    parser.add_argument("captures", type=Path)
    parser.add_argument("topology", type=Path, help="Verified Graph kernel DAGs; structure only, no timing")
    parser.add_argument("output", type=Path)
    parser.add_argument("--matrix", type=Path, default=Path(__file__).resolve().parents[1] / "baseline_matrix.json")
    args = parser.parse_args()
    result = summarize(args.bundles, args.captures, args.topology, json.loads(args.matrix.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
