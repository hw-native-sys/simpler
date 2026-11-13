#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate repeated model shapes, native instances and physical execution coverage."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from simpler_setup.tools.hbg.task_id import Space, TaskId


def execution_shapes(identities, graph, shells):
    seen, tasks = set(), defaultdict(list)
    for identity in identities:
        task = TaskId.parse(identity["task_id"])
        if task is None or not identity["executed"]:
            raise ValueError("Missing execution or invalid task identity")
        if task.space() != (Space.SUB_TASK if graph else Space.GLOBAL):
            raise ValueError("Execution used the wrong submission path")
        if graph and task.parent_id() not in shells:
            raise ValueError("Execution belongs to an unsubmitted Graph")
        key = (identity["task_id"], identity["block_idx"], identity["subtask_slot"])
        if key in seen:
            raise ValueError("Duplicate task/block/lane execution")
        seen.add(key)
        tasks[task.raw].append(identity)
    actual_shapes = defaultdict(Counter)
    for raw, rows in tasks.items():
        task = TaskId(raw)
        blocks = rows[0]["block_num"]
        slots = tuple(sorted({row["subtask_slot"] for row in rows}))
        if any(row["block_num"] != blocks for row in rows):
            raise ValueError("Inconsistent block count")
        if {(row["block_idx"], row["subtask_slot"]) for row in rows} != {
            (block, slot) for block in range(blocks) for slot in slots
        }:
            raise ValueError("Incomplete block/lane coverage")
        actual_shapes[task.parent_id() if graph else 0][blocks, slots] += 1
    return actual_shapes


def audit(bundle, run, workload):
    case = json.loads((bundle / "case.json").read_text())
    config = case.get("graph_benchmark", {"mode": "flat", "repetitions": 1})
    submission = json.loads((run / "source-task-map.txt.graph.json").read_text())
    if (submission["mode"], submission["repetitions"]) != (config["mode"], config["repetitions"]):
        raise ValueError("Submission mode/repetition mismatch")
    graph = config["mode"] == "graph"
    repeats = config["repetitions"]
    shells = submission["shells"]
    if len(shells) != (repeats if graph else 0) or len(set(shells)) != len(shells):
        raise ValueError("Graph instance count or identity mismatch")
    if submission["recordings"] not in ({0, 1} if graph else {0}):
        raise ValueError("One reusable definition must not be recorded repeatedly")
    expected_shapes = Counter()
    for line in workload.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        _, _, mask, blocks, _ = line.split()
        slots = tuple(slot for slot in range(3) if int(mask) & (1 << slot))
        if slots:
            expected_shapes[int(blocks), slots] += 1
    records = json.loads((run / "chip_swimlane_records.json").read_text())
    identities = records["aicore_execution_identities"]["records"]
    expected_count = case["source"]["executions"] * repeats
    if len(identities) != expected_count or len(records["aicore_tasks"]) != expected_count:
        raise ValueError("Physical execution count differs from the original model")
    actual_shapes = execution_shapes(identities, graph, shells)
    if graph:
        if set(actual_shapes) != set(shells) or any(shape != expected_shapes for shape in actual_shapes.values()):
            raise ValueError("An instance changed original task shapes")
        deps = json.loads((run / "deps.json").read_text())
        edges = {(int(edge["pred"]), int(edge["succ"])) for edge in deps["edges"]}
        if not set(zip(shells, shells[1:])) <= edges:
            raise ValueError("Missing serial Graph completion dependency")
    elif actual_shapes[0] != Counter({key: value * repeats for key, value in expected_shapes.items()}):
        raise ValueError("Flat repetition changed original task shapes")
    return {
        "mode": config["mode"],
        "repetitions": repeats,
        "executions": expected_count,
        "instances": len(shells),
        "recordings": submission["recordings"],
        "passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("workload", type=Path, help="Original workloads/<flavor>/<case>/tasks.tsv")
    args = parser.parse_args()
    print(json.dumps(audit(args.bundle, args.run, args.workload)))
