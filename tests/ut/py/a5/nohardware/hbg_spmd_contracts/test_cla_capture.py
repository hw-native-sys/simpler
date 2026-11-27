# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Trace-derived regression witnesses plus destructive in-memory mutations."""

import copy
from collections import Counter

import pytest

from .cla_trace import extract_workload, load_fixture, verify_worker_dependencies
from .types import ContractViolation


def test_real_capture_inventory_does_not_double_count_scheduler_mirrors():
    profile = load_fixture()
    assert profile["summary"] == {
        "events": 7893,
        "kernel_tasks": 56,
        "executions": 736,
        "aic": 176,
        "aiv": 560,
        "alloc": 43,
        "dummy": 2,
        "edges": 199,
        "worker_lanes": 96,
    }
    assert len(profile["nodes"]) == 101
    assert len(profile["observations"]) == 736
    assert Counter(n["blocks"] for n in profile["nodes"] if n["kind"] == "kernel") == {
        1: 6,
        2: 6,
        4: 13,
        5: 2,
        6: 1,
        8: 9,
        10: 1,
        12: 8,
        20: 1,
        24: 1,
        32: 8,
    }
    assert sum(len(n["deps"]) for n in profile["nodes"]) == 199
    assert sum(n["executions"] for n in profile["nodes"]) == 736


def test_ring_local_ids_and_function_ids_are_not_task_identity():
    nodes = load_fixture()["nodes"]
    by_label = {n["label"]: n for n in nodes}
    assert len({by_label[label]["task_id"] for label in ("r1t0", "r2t0", "r3t0")}) == 3
    producer, consumer = by_label["r2t4"], by_label["r1t13"]
    assert producer["task_id"] > consumer["task_id"]
    assert producer["submit_index"] < consumer["submit_index"]
    assert producer["submit_index"] in consumer["deps"]
    repeated = [by_label[f"r3t{i}"] for i in (18, 21, 24, 27, 30, 33, 36, 39)]
    assert len({n["task_id"] for n in repeated}) == 8
    assert len({tuple(n["functions"]["aic"]) for n in repeated}) == 1


@pytest.mark.parametrize(
    "label,low,high",
    [
        ("r2t17", 454, 23512),
        ("r2t20", 642, 40082),
        ("r3t10", 146011, 199657),
    ],
)
def test_measured_tails_are_preserved_as_evidence_not_device_performance_thresholds(label, low, high):
    node = next(n for n in load_fixture()["nodes"] if n["label"] == label)
    assert node["blocks"] == 32
    assert node["kernel_duration_range_ns"] == [low, high]


@pytest.mark.parametrize("pred,succ", [("r2t16", "r2t17"), ("r2t20", "r3t12"), ("r2t20", "r3t14")])
def test_consumer_after_representative_anchor_but_before_last_return_is_rejected(pred, succ):
    profile = load_fixture()
    by_label = {n["label"]: n for n in profile["nodes"]}
    p, q = by_label[pred]["submit_index"], by_label[succ]["submit_index"]
    rows = profile["observations"]
    producers = [r for r in rows if r[1] == p]
    anchor = min(producers, key=lambda r: (r[5], r[2]))
    last = max(r[7] for r in producers)
    assert anchor[7] < last
    wrong_start = (anchor[7] + last) // 2
    consumer = min((r for r in rows if r[1] == q), key=lambda r: r[6])
    consumer[6] = wrong_start
    consumer[5] = wrong_start - 1
    with pytest.raises(ContractViolation, match="CAPTURE_EARLY_CONSUMER"):
        verify_worker_dependencies(profile)


def test_scheduler_hb_annotation_does_not_imply_worker_dependency_violation():
    profile = load_fixture()
    (annotation,) = profile["scheduler_hb_annotations"]
    assert annotation["flow_id"] == 322 and annotation["pid"] == 3
    assert annotation["dst_start_ns"] < annotation["src_end_ns"]
    nodes = {n["task_id"]: n for n in profile["nodes"]}
    pred, succ = nodes[annotation["pred"]], nodes[annotation["succ"]]
    assert pred["label"] == "r2t3" and succ["label"] == "r2t4"
    assert pred["kernel_window_ns"][1] == 318861 < succ["kernel_window_ns"][0] == 340895
    verify_worker_dependencies(profile)


def test_dummy_projection_keeps_its_kernel_fanin_in_dependency_validation():
    profile = load_fixture()
    nodes = {n["label"]: n for n in profile["nodes"]}
    producer = nodes["r1t33"]["submit_index"]
    consumer = nodes["r3t12"]
    assert producer not in consumer["deps"]
    assert producer in nodes["r1t34"]["deps"] and nodes["r1t34"]["submit_index"] in consumer["deps"]
    next(row for row in profile["observations"] if row[1] == producer)[7] = consumer["kernel_window_ns"][0] + 1
    with pytest.raises(ContractViolation, match="CAPTURE_EARLY_CONSUMER"):
        verify_worker_dependencies(profile)


def test_real_empty_shared_queue_has_twenty_one_active_kernel_occurrences():
    profile = load_fixture()
    (witness,) = profile["empty_queue_with_active_kernels"]
    assert witness["time_ns"] == 367338 and len(witness["active_event_ids"]) == 21
    rows = {r[0]: r for r in profile["observations"]}
    assert all(rows[e][6] <= witness["time_ns"] < rows[e][7] for e in witness["active_event_ids"])


def tiny_capture():
    events = [
        {"name": "process_name", "args": {"name": name}, "pid": pid, "ph": "M"}
        for name, pid in (("Worker View", 4), ("Scheduler View", 3), ("AICPU Orchestrator", 1))
    ]
    for core in range(2):
        events.append({"name": "thread_name", "pid": 4, "tid": core, "ph": "M", "args": {"name": f"AIV_{core}"}})
    for task in range(2):
        events.append(
            {"cat": "orchestrator", "ph": "X", "pid": 1, "args": {"task_id": task, "submit_idx": task, "run_epoch": 1}}
        )
        for core in range(2):
            hint = f"Task:r0t{task}, FuncId:2, CoreId:{core}"
            args = {
                "taskId": task,
                "run_epoch": 1,
                "event-hint": hint,
                "fanin-hint": "0: []" if task == 0 else "1: [r0t0]",
                "fanout-hint": "1: [r0t1]" if task == 0 else "0: []",
                "local_setup_us": 1,
                "kernel-duration-us": 2,
            }
            events.append(
                {
                    "id": task * 2 + core,
                    "cat": "event",
                    "ph": "X",
                    "pid": 4,
                    "tid": core,
                    "ts": 10 * task,
                    "dur": 3,
                    "args": args,
                }
            )
            events.append(
                {
                    "id": 4 + task * 2 + core,
                    "cat": "event",
                    "ph": "X",
                    "pid": 3,
                    "tid": core,
                    "ts": 10 * task,
                    "dur": 4,
                    "args": copy.deepcopy(args),
                }
            )
    for phase, bound in (("s", 0), ("f", 2)):
        events.append(
            {
                "name": "dependency",
                "cat": "flow",
                "ph": phase,
                "id": 0,
                "pid": 4,
                "tid": 0,
                "bind_id": bound,
                "input_task_count": 2,
                "output_task_count": 2,
            }
        )
    return {"metadata": {"runtime": "tensormap_and_ringbuffer"}, "traceEvents": events}


def test_projection_reader_accepts_a_minimal_complete_capture():
    assert extract_workload(tiny_capture())["summary"]["executions"] == 4


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("runtime", "CAPTURE_RUNTIME"),
        ("missing_epoch", "CAPTURE_SCHEMA"),
        ("duplicate_slice", "CAPTURE_DUPLICATE_SLICE"),
        ("drop_mirror", "CAPTURE_MIRROR"),
        ("drop_execution", "CAPTURE_COVERAGE"),
        ("split_epoch", "CAPTURE_EPOCH"),
        ("nan_duration", "CAPTURE_TIME"),
        ("negative_setup", "CAPTURE_TIME"),
        ("half_flow", "CAPTURE_FLOW_PAIR"),
        ("wrong_bind", "CAPTURE_BIND"),
        ("wrong_count", "CAPTURE_BLOCK_COUNT"),
        ("wrong_hint", "CAPTURE_EDGE_HINT"),
        ("context_from_label", "CAPTURE_HINT"),
        ("duplicate_occurrence", "CAPTURE_AMBIGUOUS_OCCURRENCE"),
    ],
)
def test_bad_source_evidence_fails_before_it_can_define_a_test(mutation, code):
    doc = tiny_capture()
    events = doc["traceEvents"]
    bar = next(e for e in events if e.get("id") == 0 and e.get("ph") == "X")
    flow = next(e for e in events if e.get("ph") == "s")
    if mutation == "runtime":
        doc["metadata"]["runtime"] = "host_build_graph"
    elif mutation == "missing_epoch":
        del bar["args"]["run_epoch"]
    elif mutation == "duplicate_slice":
        events.append(copy.deepcopy(bar))
    elif mutation == "drop_mirror":
        events.remove(next(e for e in events if e.get("id") == 4 and e.get("ph") == "X"))
    elif mutation == "drop_execution":
        events[:] = [e for e in events if not (e.get("ph") == "X" and e.get("id") in (1, 5))]
    elif mutation == "split_epoch":
        bar["args"]["run_epoch"] = 2
    elif mutation == "nan_duration":
        bar["dur"] = float("nan")
    elif mutation == "negative_setup":
        bar["args"]["local_setup_us"] = -1
    elif mutation == "half_flow":
        events.remove(flow)
    elif mutation == "wrong_bind":
        flow["bind_id"] = 999
    elif mutation == "wrong_count":
        flow["output_task_count"] = 3
    elif mutation == "wrong_hint":
        bar["args"]["fanout-hint"] = "0: []"
    elif mutation == "context_from_label":
        bar["args"]["event-hint"] = "Task:r1t0, FuncId:2, CoreId:0"
    else:
        next(e for e in events if e.get("id") == 1 and e.get("ph") == "X")["args"]["event-hint"] = bar["args"][
            "event-hint"
        ]
    with pytest.raises(ContractViolation, match=code):
        extract_workload(doc)
