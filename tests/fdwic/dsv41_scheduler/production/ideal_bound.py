#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Conditional zero-scheduler-cost lower bound from single-device L1 captures."""

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path

from simpler_setup.tools.hbg.execution_identity import attach_execution_identities


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


def read_case(bundle, capture):
    metadata = json.loads((bundle / "case.json").read_text())
    result = json.loads((capture / "result.json").read_text())
    if metadata["device_count"] != 1 or result["ranks"] != 1:
        raise ValueError("Communication cases are excluded")
    if not result["passed"] or not result["snapshot_inputs"] or result["runtime"] != "host_build_graph":
        raise ValueError("A validated isolated HBG capture is required")
    suite = Path(__file__).resolve().parents[1]
    workload = suite / "workloads" / metadata["flavor"] / metadata["case"] / "tasks.tsv"
    rows = [line.split() for line in workload.read_text().splitlines() if line and not line.startswith("#")]
    if any(int(row[-1]) for row in rows):
        raise ValueError("Deferred completion cannot be inferred from kernel end")
    source_ids = {int(row[0]) for row in rows}
    warmup, rounds = result["warmup"], result["rounds"]
    if warmup < 1 or rounds < 1:
        raise ValueError("Warmup and measured rounds are required")
    console = re.findall(
        r"run=(\d+) scheduler_mode=(\d+) profiling=(\d+) warmup=(\d+)", (capture / "console.log").read_text()
    )
    if len(console) != warmup + rounds or any(
        tuple(map(int, row)) != (i, 1, 1, int(i < warmup)) for i, row in enumerate(console)
    ):
        raise ValueError("Expected complete HBG L1 rounds")
    samples = []
    topology = None
    for iteration in range(warmup, warmup + rounds):
        run = capture / f"run{iteration}"
        records = json.loads((run / "chip_swimlane_records.json").read_text())
        if records["chip_swimlane_level"] != 1 or records["metadata"]["runtime"] != "host_build_graph":
            raise ValueError("Expected HBG L1 trace")
        kinds = records["metadata"]["core_types"]
        if Counter(kinds) != {"aic": 32, "aiv": 64}:
            raise ValueError("Expected this fixture's 32 AIC / 64 AIV capacity")
        deps = complete_flat_dag(
            json.loads((run / "deps.json").read_text()), run / "source-task-map.txt", source_ids, 1
        )
        if any(task.get("early_dispatch", False) for task in deps["tasks"]):
            raise ValueError("Early dependency release is not modeled")
        nodes = {int(task["task_id"]): task for task in deps["tasks"]}
        signature = (
            sorted((tid, tuple(task["kernel_ids"]), task["block_num"]) for tid, task in nodes.items()),
            sorted({(int(edge["pred"]), int(edge["succ"])) for edge in deps["edges"]}),
        )
        if topology is not None and signature != topology:
            raise ValueError("Topology changed across samples")
        topology = signature
        tasks = [
            {
                "run_epoch": iteration + 1,
                "core_id": row[0],
                "task_id": row[1],
                "reg_task_id": row[2],
                "core_type": kinds[row[0]],
            }
            for row in records["aicore_tasks"]
        ]
        attach_execution_identities(records["aicore_execution_identities"], tasks, "host_build_graph")
        expected = {
            (tid, block, lane)
            for tid, task in nodes.items()
            for lane, kernel in enumerate(task["kernel_ids"])
            if kernel >= 0
            for block in range(task["block_num"])
        }
        actual = {(task["task_id"], task["block_idx"], task["subtask_slot"]) for task in tasks if task["executed"]}
        if actual != expected or len(tasks) != len(expected) or len(tasks) != metadata["source"]["executions"]:
            raise ValueError("Incomplete or extra block/lane execution coverage")
        bound = ideal_lower_bound_ms(records, deps)
        intervals = records["aicore_tasks"]
        envelope = (
            (max(row[4] for row in intervals) - min(row[3] for row in intervals))
            * 1000
            / records["metadata"]["clock_freq_hz"]
        )
        if bound > envelope + 1e-9:
            raise ValueError("Model bound exceeds the observed kernel envelope")
        samples.append({"iteration": iteration, "lower_bound_ms": bound})
    return {
        "case": metadata["case"],
        "flavor": metadata["flavor"],
        "device": result["device"],
        "source_level": 1,
        "warmup": warmup,
        "rounds": rounds,
        "callable_sha256": hashlib.sha256((bundle / "callable.bin").read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256((bundle / "plan-rank0.bin").read_bytes()).hexdigest(),
        "mean_lower_bound_ms": statistics.mean(s["lower_bound_ms"] for s in samples),
        "samples": samples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("capture", type=Path, help="One rank directory from an isolated HBG L1 run")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(read_case(args.bundle, args.capture), indent=2) + "\n")
