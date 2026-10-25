# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""CSA graph and observable execution obligations; no invented block identities."""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def project_graph(deps):
    if deps.get("runtime") != "host_build_graph":
        raise ValueError("CSA requires a host_build_graph dependency capture")
    refs = {int(t["task_id"]) for t in deps["tasks"]}
    if len(refs) != len(deps["tasks"]):
        raise ValueError("duplicate CSA logical task")
    parents = defaultdict(set)
    host_ready = set()
    for edge in deps["edges"]:
        pred, succ = int(edge["pred"]), int(edge["succ"])
        if succ in refs:
            if pred in refs:
                parents[succ].add(pred)
            else:
                host_ready.add(pred)
    nodes = []
    for task in sorted(deps["tasks"], key=lambda t: int(t["task_id"])):
        ref = int(task["task_id"])
        nodes.append(
            {
                "task_id": ref,
                "kernel_ids": task["kernel_ids"],
                "block_num": task["block_num"],
                "early_dispatch": task["early_dispatch"],
                "deps": sorted(parents[ref]),
                "args": [
                    {key: arg[key] for key in ("idx", "type", "dtype", "shape", "start_offset", "strides")}
                    for arg in task["args"]
                    if "tensor_id" in arg
                ],
            }
        )
    return {"nodes": nodes, "host_ready_refs": sorted(host_ready)}


def frozen_graph():
    return json.loads(Path(__file__).with_name("graph_contract.json").read_text())


def validate_graph(deps):
    projected = project_graph(deps)
    if projected != frozen_graph()["graph"]:
        raise ValueError("CSA frozen graph changed: kernels, blocks, dependency edges or tensor views differ")
    return projected


def _audit_block_identity(tasks, rows):
    identity_fields = {"block_idx", "block_num", "subtask_slot", "executed"}
    has_identity = any(identity_fields.intersection(row) for row in rows)
    if has_identity:
        expected_identities = {
            (ref, block, slot)
            for ref, task in tasks.items()
            for block in range(task["block_num"])
            for slot, kernel in enumerate(task["kernel_ids"])
            if kernel >= 0
        }
        actual_identities = set()
        for row in rows:
            if not identity_fields.issubset(row):
                raise ValueError("CSA execution identity is partially missing")
            block, blocks, slot = (row[field] for field in ("block_idx", "block_num", "subtask_slot"))
            ref = row["task_id"]
            if (
                any(type(value) is not int for value in (block, blocks, slot))
                or row["executed"] is not True
                or blocks != tasks[ref]["block_num"]
                or not 0 <= block < blocks
                or not 0 <= slot < 3
                or ("aic" if slot == 0 else "aiv") != row["core_type"]
            ):
                raise ValueError("CSA execution identity contradicts the graph")
            identity = (ref, block, slot)
            if identity not in expected_identities or identity in actual_identities:
                raise ValueError("CSA execution identity is duplicated or unexpected")
            actual_identities.add(identity)
        if actual_identities != expected_identities:
            raise ValueError("CSA execution identity coverage is incomplete")
    return has_identity


def audit_execution(deps, rows):
    tasks = {int(t["task_id"]): t for t in deps["tasks"]}
    if len(tasks) != len(deps["tasks"]):
        raise ValueError("duplicate CSA logical task")
    expected = Counter()
    for ref, task in tasks.items():
        blocks = task["block_num"]
        if type(blocks) is not int or blocks < 1:
            raise ValueError("invalid CSA logical block count")
        for slot, kernel in enumerate(task["kernel_ids"]):
            if kernel >= 0:
                expected[(ref, "aic" if slot == 0 else "aiv")] += blocks
    actual = Counter((row["task_id"], row["core_type"]) for row in rows)
    if actual != expected:
        raise ValueError("CSA per-task AIC/AIV execution counts differ")
    epochs = {row["run_epoch"] for row in rows}
    if len(epochs) != 1 or any(type(e) is not int or e <= 0 for e in epochs):
        raise ValueError("CSA capture must contain one positive run epoch")
    has_identity = _audit_block_identity(tasks, rows)
    starts, ends = {}, {}
    for row in rows:
        start, end = row["start_time_us"], row["end_time_us"]
        if not math.isfinite(start) or not math.isfinite(end) or start > end:
            raise ValueError("invalid CSA kernel interval")
        ref = row["task_id"]
        starts[ref] = min(starts.get(ref, start), start)
        ends[ref] = max(ends.get(ref, end), end)
    parents = defaultdict(set)
    for edge in deps.get("edges", []):
        pred, succ = int(edge["pred"]), int(edge["succ"])
        if pred in tasks and succ in tasks:
            parents[succ].add(pred)

    resolved = {}
    visiting = set()

    def execution_predecessors(ref):
        if ref in visiting:
            raise ValueError("cyclic CSA dependency")
        if ref in resolved:
            return resolved[ref]
        visiting.add(ref)
        result = set()
        for pred in parents[ref]:
            if pred in ends:
                result.add(pred)
            result.update(execution_predecessors(pred))
        visiting.remove(ref)
        resolved[ref] = result
        return result

    for succ in tasks:
        predecessors = execution_predecessors(succ)
        if succ in starts:
            for pred in predecessors:
                if ends[pred] > starts[succ]:
                    raise ValueError(f"CSA dependency started early: {pred} -> {succ}")
    return {
        "execution_epoch": next(iter(epochs)),
        "per_kind_executions": dict(Counter(row["core_type"] for row in rows)),
        "dependency_edges_checked": sum(len(v) for v in parents.values()),
        "block_identity_verified": has_identity,
    }
