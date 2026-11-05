# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit resident identities must survive the real converter without inference."""

import copy
import json

import pytest
from _capture_builder import Capture

from simpler_setup.tools.swimlane_converter import generate_chrome_trace_json, read_perf_data


def capture(level=2):
    builder = Capture(runtime="host_build_graph", level=level, core_types=("aic", "aiv"), scheduler_producer="aicore")
    records = []
    for block in range(2):
        builder.task(task_id=7, core_id=1, start=100 + block * 30, end=120 + block * 30, run_epoch=0, reg_task_id=block)
        records.append(
            dict(
                run_epoch=0,
                core_id=1,
                reg_task_id=block,
                task_id=7,
                block_idx=block,
                block_num=2,
                subtask_slot=1,
                pending_slot=block,
                dispatch_generation=block + 1,
                order_key=0,
                group_seq=None,
                cluster_index=0,
                cluster_lane=1,
                executed=True,
            )
        )
    result = builder.build()
    result["aicore_execution_identities"] = dict(schema="hbg-execution-identity-v1", producer="aicore", records=records)
    return result


def read(tmp_path, raw):
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(raw))
    return read_perf_data(path)["tasks"]


@pytest.mark.parametrize("level", (1, 2, 3, 4))
def test_explicit_identity_survives_without_deriving_block_from_record_index(tmp_path, level):
    raw = capture(level)
    raw["aicore_execution_identities"]["records"].reverse()
    rows = read(tmp_path, raw)
    assert [(r["block_idx"], r["subtask_slot"], r["block_num"]) for r in rows] == [(0, 1, 2), (1, 1, 2)]
    assert all(r["order_key"] == 0 and r["group_seq"] is None and r["executed"] is True for r in rows)


def test_historical_capture_does_not_acquire_invented_identity(tmp_path):
    raw = capture()
    del raw["aicore_execution_identities"]
    assert all("block_idx" not in r for r in read(tmp_path, raw))


@pytest.mark.parametrize("field", tuple(capture()["aicore_execution_identities"]["records"][0]))
def test_every_identity_field_is_required(tmp_path, field):
    raw = capture()
    del raw["aicore_execution_identities"]["records"][0][field]
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_epoch", 1),
        ("core_id", 0),
        ("reg_task_id", 100),
        ("task_id", 8),
        ("block_idx", -1),
        ("block_idx", 2),
        ("block_idx", True),
        ("block_num", 0),
        ("subtask_slot", 0),
        ("subtask_slot", 3),
        ("pending_slot", 2),
        ("dispatch_generation", 0),
        ("dispatch_generation", 1 << 32),
        ("order_key", -1),
        ("order_key", True),
        ("group_seq", 0),
        ("group_seq", 1 << 64),
        ("cluster_lane", 0),
        ("cluster_lane", 3),
        ("cluster_index", -1),
        ("executed", 1),
    ],
)
def test_malformed_or_misjoined_identity_is_not_ignored(tmp_path, field, value):
    raw = capture()
    raw["aicore_execution_identities"]["records"][0][field] = value
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)


@pytest.mark.parametrize("mutation", ("missing", "extra", "duplicate_join", "duplicate_execution", "unknown_field"))
def test_identity_is_a_bijection_with_rows_and_executions(tmp_path, mutation):
    raw = capture()
    records = raw["aicore_execution_identities"]["records"]
    if mutation == "missing":
        records.pop()
    elif mutation in ("extra", "duplicate_join"):
        records.append(copy.deepcopy(records[0]))
        if mutation == "extra":
            records[-1]["reg_task_id"] = 3
    elif mutation == "duplicate_execution":
        records[1]["block_idx"] = 0
    else:
        records[0]["guessed"] = True
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)


@pytest.mark.parametrize("field,value", [("schema", "future-v2"), ("producer", "aicpu"), ("records", None)])
def test_unknown_or_incomplete_identity_envelope_is_rejected(tmp_path, field, value):
    raw = capture()
    raw["aicore_execution_identities"][field] = value
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)


def test_nonexecuting_predicate_is_marked_not_mistaken_for_a_kernel(tmp_path):
    raw = capture()
    record = raw["aicore_execution_identities"]["records"][0]
    record["executed"] = False
    assert read(tmp_path, raw)[0]["executed"] is False


@pytest.mark.parametrize("pid", (3, 4))
def test_perfetto_task_views_preserve_explicit_identity(tmp_path, pid):
    rows = read(tmp_path, capture())
    output = tmp_path / "trace.json"
    generate_chrome_trace_json(rows, str(output), runtime_name="host_build_graph")
    events = [
        event
        for event in json.loads(output.read_text())["traceEvents"]
        if event.get("pid") == pid and event.get("cat") == "event" and event.get("ph") == "X"
    ]
    assert len(events) == 2
    assert {event["args"]["block_idx"] for event in events} == {0, 1}
    assert all(event["args"]["subtask_slot"] == 1 and event["args"]["order_key"] == 0 for event in events)


@pytest.mark.parametrize("epoch", (0, (1 << 64) - 1))
def test_epoch_extremes_are_literal_identity_not_absence(tmp_path, epoch):
    raw = capture()
    for row in raw["aicore_tasks"]:
        row[6] = epoch
    for row in raw["scheduler_tasks"]["records"]:
        row[4] = epoch
    for record in raw["aicore_execution_identities"]["records"]:
        record["run_epoch"] = epoch
    assert all(row["run_epoch"] == epoch for row in read(tmp_path, raw))


@pytest.mark.parametrize("field,value", [("order_key", 1), ("cluster_index", 1), ("dispatch_generation", 1)])
def test_same_task_worker_or_slot_cannot_carry_inconsistent_identity(tmp_path, field, value):
    raw = capture()
    records = raw["aicore_execution_identities"]["records"]
    records[1][field] = value
    if field == "dispatch_generation":
        records[1]["pending_slot"] = records[0]["pending_slot"]
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)


@pytest.mark.parametrize("mask", (3, 5, 6, 7))
def test_each_mix_mask_keeps_fixed_lane_and_group_identity(tmp_path, mask):
    builder = Capture(runtime="host_build_graph", core_types=("aic", "aiv", "aiv"), scheduler_producer="aicore")
    records = []
    for slot in range(3):
        if not mask & (1 << slot):
            continue
        builder.task(task_id=7, core_id=slot, start=100, end=120, run_epoch=1, reg_task_id=slot)
        records.append(
            dict(
                run_epoch=1,
                core_id=slot,
                reg_task_id=slot,
                task_id=7,
                block_idx=0,
                block_num=1,
                subtask_slot=slot,
                pending_slot=0,
                dispatch_generation=1,
                order_key=0,
                group_seq=9,
                cluster_index=0,
                cluster_lane=slot,
                executed=True,
            )
        )
    raw = builder.build()
    raw["aicore_execution_identities"] = dict(schema="hbg-execution-identity-v1", producer="aicore", records=records)
    rows = read(tmp_path, raw)
    assert {r["subtask_slot"] for r in rows} == {slot for slot in range(3) if mask & (1 << slot)}
    assert all(r["group_seq"] == 9 for r in rows)
    vector = next(r for r in records if r["subtask_slot"] != 0)
    vector["cluster_lane"] = 3 - vector["cluster_lane"]
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)


def test_identical_record_ids_and_generations_from_distinct_epochs_do_not_collide(tmp_path):
    raw = capture()
    later = copy.deepcopy(raw)
    for row in later["aicore_tasks"]:
        row[6] = 1
    for row in later["scheduler_tasks"]["records"]:
        row[4] = 1
    for row in later["aicore_execution_identities"]["records"]:
        row["run_epoch"] = 1
    raw["aicore_tasks"] += later["aicore_tasks"]
    raw["scheduler_tasks"]["records"] += later["scheduler_tasks"]["records"]
    raw["aicore_execution_identities"]["records"] += later["aicore_execution_identities"]["records"]
    assert len(read(tmp_path, raw)) == 4


@pytest.mark.parametrize("runtime", ("tensormap_and_ringbuffer",))
def test_runtime_cannot_reinterpret_hbg_identity(tmp_path, runtime):
    raw = capture()
    raw["metadata"]["runtime"] = runtime
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)


def test_null_identity_section_is_not_an_absent_historical_section(tmp_path):
    raw = capture()
    raw["aicore_execution_identities"] = None
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)


@pytest.mark.parametrize("mutation", ("cluster", "sequence", "missing_one", "missing_all"))
def test_same_mix_block_cannot_join_different_groups(tmp_path, mutation):
    raw = capture()
    records = raw["aicore_execution_identities"]["records"]
    for index, record in enumerate(records):
        record.update(core_id=index, block_idx=0, block_num=1, subtask_slot=index, cluster_lane=index, group_seq=7)
        raw["aicore_tasks"][index][0] = index
        raw["scheduler_tasks"]["records"][index][0] = index
    read(tmp_path, raw)
    if mutation == "cluster":
        records[1]["cluster_index"] = 1
    elif mutation == "sequence":
        records[1]["group_seq"] = 8
    elif mutation == "missing_one":
        records[1]["group_seq"] = None
    else:
        for record in records:
            record["group_seq"] = None
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)


def test_multiblock_identity_cannot_omit_its_admission_order(tmp_path):
    raw = capture()
    for record in raw["aicore_execution_identities"]["records"]:
        record["order_key"] = None
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)
