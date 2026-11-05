# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Test-side projection of this archived Perfetto capture; not a native protocol trace."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from simpler_setup.tools.tmr.task_id import TaskId

from .types import ContractViolation, require

FIXTURE = Path(__file__).resolve().parents[3] / "fdwic/pa_hbg/spmd/fixtures/cla_workload.json"
SOURCE_SHA256 = "a33e497915b499f2ee13b9ae8182f4cb358d62106f8d358bbae9bc67fb30d747"


def _integer(value):
    require(type(value) is int and value >= 0, "CAPTURE_INTEGER")
    return value


def _ns(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "CAPTURE_TIME")
    return round(value * 1000)


def _task(event):
    args = event["args"]
    return _integer(args["taskId"] if "taskId" in args else args["task_id"])


def _hint_ids(value):
    match = re.fullmatch(r"(\d+): \[(.*)\]", value)
    if match is None:
        raise ContractViolation("CAPTURE_HINT")
    names = match[2].split(", ") if match[2] else []
    require(len(names) == int(match[1]) and len(names) == len(set(names)), "CAPTURE_HINT")
    return set(names)


def _dependency_pairs(events, slices):
    pending = defaultdict(list)
    for event in events:
        if event.get("ph") in ("s", "f") and event["name"] in ("dependency", "hb_violation"):
            pending[_integer(event["id"])].append(event)
    edges, counts, annotations = set(), defaultdict(set), []
    for flow_id, pair in sorted(pending.items()):
        require(len(pair) == 2 and {e["ph"] for e in pair} == {"s", "f"}, "CAPTURE_FLOW_PAIR")
        start, end = sorted(pair, key=lambda e: e["ph"] == "f")
        require(start["name"] == end["name"], "CAPTURE_FLOW_PAIR")
        bound = []
        for event in (start, end):
            require(event["bind_id"] in slices, "CAPTURE_BIND")
            bar = slices[event["bind_id"]]
            require(event["pid"] == bar["pid"] and event["tid"] == bar["tid"], "CAPTURE_BIND")
            bound.append(_task(bar))
        pred, succ = bound
        for field, task in (("output_task_count", pred), ("input_task_count", succ)):
            value = _integer(start[field])
            require(value > 0 and end[field] == value, "CAPTURE_BLOCK_COUNT")
            counts[task].add(value)
        edges.add((pred, succ))
        if start["name"] == "hb_violation":
            source_bar = slices[start["bind_id"]]
            annotations.append(
                {
                    "flow_id": flow_id,
                    "pid": start["pid"],
                    "pred": pred,
                    "succ": succ,
                    "src_bar": start["bind_id"],
                    "dst_bar": end["bind_id"],
                    "src_start_ns": _ns(source_bar["ts"]),
                    "src_end_ns": _ns(source_bar["ts"] + source_bar["dur"]),
                    "dst_start_ns": _ns(end["ts"]),
                }
            )
    require(bool(edges), "CAPTURE_NO_DEPENDENCIES")
    require(all(len(values) == 1 for values in counts.values()), "CAPTURE_BLOCK_COUNT")
    return edges, {task: next(iter(values)) for task, values in counts.items()}, annotations


def _kernel_row(event, lanes):
    args = event["args"]
    task = _task(event)
    hint = re.fullmatch(r"Task:(r\d+t\d+), FuncId:(\d+), CoreId:(\d+)", args["event-hint"])
    if hint is None:
        raise ContractViolation("CAPTURE_HINT")
    require(hint[1] == TaskId(task).display(), "CAPTURE_HINT")
    lane = lanes[(event["pid"], event["tid"])]
    kind = "aic" if lane.startswith("AIC_") else "aiv"
    require(lane.startswith(("AIC_", "AIV_")), "CAPTURE_LANE")
    receive, duration = _ns(event["ts"]), _ns(event["dur"])
    setup, kernel = _ns(args["local_setup_us"]), _ns(args["kernel-duration-us"])
    require(kernel > 0 and abs(duration - setup - kernel) <= 1, "CAPTURE_TIME")
    return {
        "event_id": _integer(event["id"]),
        "task": task,
        "core": int(hint[3]),
        "kind": kind,
        "func": int(hint[2]),
        "receive": receive,
        "start": receive + setup,
        "end": receive + duration,
    }


def _extract_workload(document):
    """Validate observed counts/edges and retain unknown attributes as unknown."""
    require(document["metadata"]["runtime"] == "tensormap_and_ringbuffer", "CAPTURE_RUNTIME")
    events = document["traceEvents"]
    require(isinstance(events, list), "CAPTURE_SCHEMA")
    processes = {e["args"]["name"]: e["pid"] for e in events if e.get("name") == "process_name"}
    require(all(name in processes for name in ("Worker View", "Scheduler View", "AICPU Orchestrator")), "CAPTURE_VIEWS")
    worker_pid, scheduler_pid = processes["Worker View"], processes["Scheduler View"]
    lanes = {(e["pid"], e["tid"]): e["args"]["name"] for e in events if e.get("name") == "thread_name"}
    bars = [e for e in events if e.get("ph") == "X" and e.get("cat") == "event" and "id" in e]
    slices = {e["id"]: e for e in bars}
    require(len(slices) == len(bars), "CAPTURE_DUPLICATE_SLICE")
    epochs = {_integer(e["args"]["run_epoch"]) for e in bars}
    require(len(epochs) == 1, "CAPTURE_EPOCH")
    epoch = next(iter(epochs))
    submits = [e for e in events if e.get("cat") == "orchestrator" and e.get("ph") == "X"]
    require(all(e["args"]["run_epoch"] == epoch for e in submits), "CAPTURE_EPOCH")
    submits.sort(key=lambda e: _integer(e["args"]["submit_idx"]))
    require([e["args"]["submit_idx"] for e in submits] == list(range(len(submits))), "CAPTURE_SUBMIT")
    task_refs = {_task(e): i for i, e in enumerate(submits)}
    require(len(task_refs) == len(submits), "CAPTURE_SUBMIT")
    edges, block_counts, annotations = _dependency_pairs(events, slices)
    require(all(a in task_refs and b in task_refs and task_refs[a] < task_refs[b] for a, b in edges), "CAPTURE_DAG")
    groups, services = defaultdict(list), {}
    for event in bars:
        if event["pid"] != worker_pid:
            continue
        if "taskId" in event["args"]:
            groups[_task(event)].append(_kernel_row(event, lanes))
        else:
            task = _task(event)
            require(task not in services and event["args"]["phase"] in ("alloc", "dummy_task"), "CAPTURE_SERVICE")
            services[task] = event
    require(
        not groups.keys() & services.keys() and groups.keys() | services.keys() == task_refs.keys(), "CAPTURE_COVERAGE"
    )

    mirror_keys = []
    for event in bars:
        if event["pid"] != scheduler_pid:
            continue
        hint = re.fullmatch(r"Task:(r\d+t\d+), FuncId:(\d+), CoreId:(\d+)", event["args"]["event-hint"])
        if hint is None:
            raise ContractViolation("CAPTURE_HINT")
        require(hint[1] == TaskId(_task(event)).display(), "CAPTURE_HINT")
        mirror_keys.append((_task(event), int(hint[3])))
    kernel_keys = [(t, row["core"]) for t, rows in groups.items() for row in rows]
    # This archive has no repeated task/core occurrence; other captures need an actual dispatch identity.
    require(len(set(kernel_keys)) == len(kernel_keys), "CAPTURE_AMBIGUOUS_OCCURRENCE")
    require(Counter(mirror_keys) == Counter(kernel_keys), "CAPTURE_MIRROR")
    fanins, fanouts = defaultdict(set), defaultdict(set)
    for pred, succ in edges:
        fanins[succ].add(pred)
        fanouts[pred].add(succ)
    nodes = []
    for event in submits:
        task = _task(event)
        rows = groups.get(task, [])
        base = {
            "label": TaskId(task).display(),
            "task_id": task,
            "submit_index": task_refs[task],
            "deps": sorted(task_refs[p] for p in fanins[task]),
        }
        if not rows:
            kind = "alloc" if services[task]["args"]["phase"] == "alloc" else "dummy"
            nodes.append({**base, "kind": kind, "blocks": 1, "canonical_mask": 0, "executions": 0})
            continue
        require(task in block_counts, "CAPTURE_BLOCK_COUNT")
        blocks = block_counts[task]
        by_kind = Counter(row["kind"] for row in rows)
        require(
            set(by_kind) <= {"aic", "aiv"} and all(count == blocks for count in by_kind.values()), "CAPTURE_COVERAGE"
        )
        mask = (1 if by_kind["aic"] else 0) | (2 if by_kind["aiv"] else 0)
        funcs = {kind: sorted({r["func"] for r in rows if r["kind"] == kind}) for kind in sorted(by_kind)}
        require(all(len(values) == 1 for values in funcs.values()), "CAPTURE_FUNCTION")
        for row in rows:
            args = slices[row["event_id"]]["args"]
            require(_hint_ids(args["fanin-hint"]) == {TaskId(p).display() for p in fanins[task]}, "CAPTURE_EDGE_HINT")
            require(_hint_ids(args["fanout-hint"]) == {TaskId(p).display() for p in fanouts[task]}, "CAPTURE_EDGE_HINT")
        durations = [row["end"] - row["start"] for row in rows]
        nodes.append(
            {
                **base,
                "kind": "kernel",
                "blocks": blocks,
                "canonical_mask": mask,
                "executions": len(rows),
                "functions": funcs,
                "kernel_window_ns": [min(r["start"] for r in rows), max(r["end"] for r in rows)],
                "kernel_duration_range_ns": [min(durations), max(durations)],
            }
        )

    observations = [
        [r["event_id"], task_refs[t], r["core"], r["kind"], r["func"], r["receive"], r["start"], r["end"]]
        for t, rows in groups.items()
        for r in rows
    ]
    # No block_idx, subtask_slot, sync_start or physical cluster relation is reconstructed.
    result = {
        "schema": "cla-workload-v1",
        "source_runtime": document["metadata"]["runtime"],
        "run_epoch": epoch,
        "nodes": nodes,
        "observations": sorted(observations),
        "scheduler_hb_annotations": annotations,
        "summary": {
            "events": len(events),
            "kernel_tasks": len(groups),
            "executions": len(kernel_keys),
            "aic": sum(r["kind"] == "aic" for rows in groups.values() for r in rows),
            "aiv": sum(r["kind"] == "aiv" for rows in groups.values() for r in rows),
            "alloc": sum(n["kind"] == "alloc" for n in nodes),
            "dummy": sum(n["kind"] == "dummy" for n in nodes),
            "edges": len(edges),
            "worker_lanes": len({r["core"] for rows in groups.values() for r in rows}),
        },
    }
    result["empty_queue_with_active_kernels"] = []
    for index, event in enumerate(events):
        if event.get("cat") != "queue" or any(event["args"].values()):
            continue
        tick = _ns(event["ts"])
        active = [row[0] for row in observations if row[6] <= tick < row[7]]
        if active:
            result["empty_queue_with_active_kernels"].append(
                {"event_index": index, "time_ns": tick, "active_event_ids": sorted(active)}
            )
            break
    verify_worker_dependencies(result)
    return result


def extract_workload(document):
    try:
        return _extract_workload(document)
    except (KeyError, TypeError, AttributeError) as error:
        raise ContractViolation("CAPTURE_SCHEMA") from error


def verify_worker_dependencies(workload):
    """Use all real kernel ends; neither representative arrows nor Scheduler finish is TaskDone."""
    rows = defaultdict(list)
    require(
        len({row[0] for row in workload["observations"]}) == len(workload["observations"]), "CAPTURE_DUPLICATE_SLICE"
    )
    for row in workload["observations"]:
        rows[row[1]].append(row)
    for node in workload["nodes"]:
        ref = node["submit_index"]
        if node["kind"] != "kernel":
            continue
        require(len(rows[ref]) == node["executions"], "CAPTURE_COVERAGE")
        start = min(row[6] for row in rows[ref])
        pending, visited = list(node["deps"]), set()
        while pending:
            predecessor = pending.pop()
            require(type(predecessor) is int and 0 <= predecessor < ref, "CAPTURE_DAG")
            if predecessor in visited:
                continue
            visited.add(predecessor)
            if rows[predecessor]:
                require(max(row[7] for row in rows[predecessor]) <= start, "CAPTURE_EARLY_CONSUMER")
            else:
                pending.extend(workload["nodes"][predecessor]["deps"])


def load_fixture():
    result = json.loads(FIXTURE.read_text())
    require(result["source_sha256"] == SOURCE_SHA256, "CAPTURE_PROVENANCE")
    return result
