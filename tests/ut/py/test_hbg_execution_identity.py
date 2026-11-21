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
from _native_protocol_builder import ProtocolBuilder

from simpler_setup.tools.hbg.schedule_timing import _lane_name
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
                order_key=0,
                group_seq=None,
                group_kind=0,
                cluster_index=0,
                cluster_lane=1,
                executed=True,
            )
        )
    result = builder.build()
    result["aicore_execution_identities"] = dict(
        schema="hbg-execution-identity", version=2, producer="aicore", records=records
    )
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


def test_l2_lane_names_sort_ordinary_slots_then_each_proxy_with_its_mix_members():
    # Perfetto's JSON process tracks are ordered by their lowercase display name.
    ordinary = [(role, slot) for role in (0, 1, 2) for slot in (0, 1)]
    mix = [(role, slot) for slot in (4, 5) for role in (3, 0, 1, 2)]
    expected = [(cluster, role, slot) for cluster in (0, 1, 10) for role, slot in ordinary + mix]
    assert sorted(reversed(expected), key=lambda track: _lane_name(track).lower()) == expected


def test_l2_checkpoints_render_one_proxy_and_separate_member_slots(tmp_path):
    raw = capture()
    section = raw["aicore_execution_identities"]
    section["version"] = 3
    for lane, (record, kernel, scheduler) in enumerate(
        zip(section["records"], raw["aicore_tasks"], raw["scheduler_tasks"]["records"])
    ):
        record.update(
            core_id=lane,
            block_idx=0,
            block_num=1,
            group_kind=2,
            group_seq=1,
            subtask_slot=lane,
            cluster_lane=lane,
            pending_slot=4,
        )
        kernel[0] = scheduler[0] = lane
        kernel[5] = 2
        scheduler[2] = kernel[3] - 4
        start = kernel[3]
        record["schedule_cycles"] = [start - 14, start - 10, start - 6, kernel[4] + 3]
    section["records"][0]["pending_fanin"] = [2, 92]
    section["records"][1]["schedule_cycles"] += [70, 75]
    rows = read(tmp_path, raw)
    output = tmp_path / "trace.json"
    for row in rows:
        row["func_id"] = row["cluster_lane"]
    generate_chrome_trace_json(
        rows,
        str(output),
        func_id_to_name={"0": "proj_aic", "1": "proj_aiv"},
        runtime_name="host_build_graph",
        scheduler_task_producer="aicore",
        chip_swimlane_level=2,
    )
    events = json.loads(output.read_bytes())["traceEvents"]
    schedule = [event for event in events if event.get("cat") == "HBG L2"]
    labels = {e["id2"]["local"]: e["name"] for e in schedule if e["ph"] == "n"}
    assert set(labels.values()) == {"C00 / MIX0/AIC", "C00 / MIX0/AIV0", "C00 / MIX0 (AIV0 proxy)"}
    assert all("tid" not in e and e["pid"] == 3 for e in schedule)
    starts = [e for e in schedule if e["ph"] == "b"]
    (handoff,) = [e for e in starts if e["name"].startswith("SSBUF arm")]
    assert set(handoff["args"]["member_observation_us_after_arm"]) == {"AIC", "AIV0"}
    end = next(e for e in schedule[schedule.index(handoff) + 1 :] if e["id2"] == handoff["id2"])
    assert end["ph"] == "e"
    assert end["ts"] - handoff["ts"] == pytest.approx(
        rows[1]["schedule_checkpoints"][0] - rows[1]["mix_proxy_timing"][1]
    )
    assert not any(e["name"].startswith(("EXECUTE", "Kernel execution")) for e in starts)
    workers = [e for e in events if e["pid"] == 4 and e["ph"] == "X"]
    assert len(workers) == 2
    assert {e["name"] for e in workers} == {"t7-proj"}
    assert {e["args"]["correlation_id"] for e in workers} == {"spmd/7"}
    assert {e["args"]["kernel_slice_label"] for e in workers} == {"proj_aic(t7)", "proj_aiv(t7)"}
    binds = [e for e in starts if e["name"].startswith("MIX member bind")]
    assert {labels[e["id2"]["local"]] for e in binds} == {"C00 / MIX0/AIC", "C00 / MIX0/AIV0"}
    (wait,) = [e for e in starts if e["name"].startswith("Predecessor pending")]
    assert wait["args"]["first_pending_task_id"] == 2
    assert [e["dur"] for e in events if e["pid"] == 4 and e["ph"] == "X"] == pytest.approx(
        [r["duration_us"] for r in rows]
    )

    section["records"][0]["schedule_cycles"][1] = 1
    with pytest.raises(ValueError, match="out-of-order"):
        read(tmp_path, raw)


def test_l3_preserves_historical_ordinary_source_ids(tmp_path):
    raw = capture(3)
    for record in raw["aicore_execution_identities"]["records"]:
        record["pending_slot"] += 2
    assert [row["pending_slot"] for row in read(tmp_path, raw)] == [2, 3]


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
        ("pending_slot", 4),
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


@pytest.mark.parametrize("field,value", [("order_key", 1), ("cluster_index", 1), ("block_idx", 0)])
def test_same_task_worker_or_slot_cannot_carry_inconsistent_identity(tmp_path, field, value):
    raw = capture()
    records = raw["aicore_execution_identities"]["records"]
    records[1][field] = value
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)


@pytest.mark.parametrize("mask", (3, 5, 6, 7))
@pytest.mark.parametrize("bank", (4, 5))
def test_each_mix_mask_keeps_fixed_lane_and_group_identity(tmp_path, mask, bank):
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
                pending_slot=bank,
                order_key=0,
                group_seq=9,
                group_kind=2,
                cluster_index=0,
                cluster_lane=slot,
                executed=True,
            )
        )
    raw = builder.build()
    raw["aicore_execution_identities"] = dict(
        schema="hbg-execution-identity", version=2, producer="aicore", records=records
    )
    rows = read(tmp_path, raw)
    assert {r["subtask_slot"] for r in rows} == {slot for slot in range(3) if mask & (1 << slot)}
    assert all(r["group_seq"] == 9 and r["pending_slot"] == bank for r in rows)
    vector = next(r for r in records if r["subtask_slot"] != 0)
    vector["cluster_lane"] = 3 - vector["cluster_lane"]
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)


def test_identical_record_ids_from_distinct_epochs_do_not_collide(tmp_path):
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


def two_lane_mix_capture():
    raw = capture()
    records = raw["aicore_execution_identities"]["records"]
    for index, record in enumerate(records):
        record.update(
            core_id=index,
            block_idx=0,
            block_num=1,
            subtask_slot=index,
            cluster_lane=index,
            group_seq=7,
            group_kind=2,
            pending_slot=4,
        )
        raw["aicore_tasks"][index][0] = index
        raw["scheduler_tasks"]["records"][index][0] = index
    return raw


@pytest.mark.parametrize("mutation", ("cluster", "sequence", "bank", "missing_one", "missing_all"))
def test_same_mix_block_cannot_join_different_groups(tmp_path, mutation):
    raw = two_lane_mix_capture()
    records = raw["aicore_execution_identities"]["records"]
    read(tmp_path, raw)
    if mutation == "cluster":
        records[1]["cluster_index"] = 1
    elif mutation == "sequence":
        records[1]["group_seq"] = 9
    elif mutation == "bank":
        records[1]["pending_slot"] = 5
    elif mutation == "missing_one":
        records[1]["group_seq"] = None
    else:
        for record in records:
            record["group_seq"] = None
    expected = (
        "Mix block spans different groups" if mutation in ("cluster", "sequence", "bank") else "group source mismatch"
    )
    with pytest.raises(ValueError, match=expected):
        read(tmp_path, raw)


def test_multiblock_self_execution_requires_a_stable_rank(tmp_path):
    raw = capture()
    for record in raw["aicore_execution_identities"]["records"]:
        record["order_key"] = None
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)


@pytest.mark.parametrize("fault", (None, "source", "ordered", "generation"))
def test_ordinary_cluster_identity_requires_its_explicit_source(tmp_path, fault):
    raw = two_lane_mix_capture()
    records = raw["aicore_execution_identities"]["records"]
    if fault is None:
        assert all(row["group_kind"] == 2 and row["order_key"] == 0 for row in read(tmp_path, raw))
        return
    if fault == "source":
        records[0]["group_kind"] = 1
    elif fault == "ordered":
        records[0]["order_key"] = None
    else:
        records[0]["group_seq"] = 8
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)


@pytest.mark.parametrize("kind,rank", ((1, 0), (2, None)))
def test_unversioned_group_members_can_use_different_local_pending_slots(tmp_path, kind, rank):
    raw = two_lane_mix_capture()
    section = raw["aicore_execution_identities"]
    del section["version"]
    for slot, record in enumerate(section["records"]):
        record.update(group_kind=kind, order_key=rank, pending_slot=slot)
    assert [row["pending_slot"] for row in read(tmp_path, raw)] == [0, 1]


@pytest.mark.parametrize("kind,rank,seq", ((0, None, None), (1, 0, 2), (2, None, 3)))
def test_unversioned_capture_retains_historical_source_and_nullable_rank(tmp_path, kind, rank, seq):
    raw = capture()
    section = raw["aicore_execution_identities"]
    del section["version"]
    for record in section["records"]:
        record.update(group_kind=kind, order_key=rank, group_seq=seq)
    rows = read(tmp_path, raw)
    assert all(row["order_key"] == rank and row["group_kind"] == kind and row["group_seq"] == seq for row in rows)


def test_unversioned_capture_cannot_combine_legacy_rank_and_ranked_slots(tmp_path):
    raw = capture()
    section = raw["aicore_execution_identities"]
    del section["version"]
    for record in section["records"]:
        record["order_key"] = None
    section["records"][1]["pending_slot"] = 2
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)


@pytest.mark.parametrize("version", (None, False, True, 2.0, 4))
def test_explicit_identity_version_must_be_supported(tmp_path, version):
    raw = capture()
    raw["aicore_execution_identities"]["version"] = version
    with pytest.raises(ValueError, match="identity"):
        read(tmp_path, raw)


def test_ranked_protocol_rejects_unversioned_null_rank(tmp_path):
    protocol, records = ProtocolBuilder([(2, 1, ())]).run()
    builder = Capture(
        runtime="host_build_graph", level=4, core_types=("aic", "aiv", "aiv") * 2, scheduler_producer="aicore"
    )
    for execution in protocol["executions"]:
        builder.task(
            task_id=execution["task_ref"],
            core_id=execution["worker_index"],
            reg_task_id=execution["index"],
            run_epoch=protocol["run_epoch"],
            start=execution["entry"][0],
            end=execution["visible"][0],
        )
    raw = builder.build()
    raw["aicore_execution_identities"] = dict(schema="hbg-execution-identity", producer="aicore", records=records)
    raw["aicore_protocol"] = protocol
    assert read(tmp_path, raw)[0]["order_key"] == 0
    records[0]["order_key"] = None
    with pytest.raises(ValueError, match="native protocol: execution identity group source mismatch"):
        read(tmp_path, raw)


def test_native_l2_spmd_slice_preserves_worker_label_and_task_color():
    from simpler_setup.tools.l2_pftrace import encode_l2_trace  # noqa: PLC0415

    proto = pytest.importorskip("perfetto.protos.perfetto.trace.perfetto_trace_pb2")
    name = "t7-proj_spmd"
    events = [{"ph": "M", "pid": 4, "name": "process_name", "args": {"name": "Worker View"}}]
    for tid, member in enumerate(("proj_aic", "proj_aiv")):
        events.append({"ph": "M", "pid": 4, "tid": tid, "name": "thread_name", "args": {"name": member}})
        events.append(
            {
                "ph": "X",
                "pid": 4,
                "tid": tid,
                "cat": "event",
                "name": name,
                "ts": 1,
                "dur": 2,
                "args": {"correlation_id": "spmd/7", "kernel_slice_label": member},
            }
        )
    trace = proto.Trace.FromString(encode_l2_trace(events))
    begins = [p.track_event for p in trace.packet if p.track_event.type == proto.TrackEvent.TYPE_SLICE_BEGIN]
    assert [(e.name, e.correlation_id_str) for e in begins] == [(name, "spmd/7")] * 2
    assert {a.string_value for e in begins for a in e.debug_annotations if a.name == "kernel_slice_label"} == {
        "proj_aic",
        "proj_aiv",
    }
