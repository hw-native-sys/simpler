# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import copy
import json

import pytest
from _capture_builder import Capture
from _native_protocol_builder import ProtocolBuilder

from simpler_setup.tools.hbg.native_causality import audit_native_protocol
from simpler_setup.tools.hbg.native_protocol import STAMPS, _event_records, audit_protocol_sequences
from simpler_setup.tools.swimlane_converter import generate_chrome_trace_json, read_perf_data


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
@pytest.mark.parametrize("blocks", (1, 2, 7))
def test_complete_wire_protocol_across_pure_mix_and_waves(mask, blocks):
    raw, identities = ProtocolBuilder([(mask, blocks, ())]).run()
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


def test_host_completed_service_and_kernel_dependencies_remain_distinct():
    raw, identities = ProtocolBuilder([(None, 0, ()), (1, 1, (0,)), (0, 1, (1,)), (6, 2, (2,))]).run()
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


@pytest.mark.parametrize("shapes", ([], [(None, 0, ())], [(0, 1, ())]))
def test_empty_host_only_and_service_only_runs_do_not_invent_executions(shapes):
    raw, identities = ProtocolBuilder(shapes).run()
    result = audit_native_protocol(raw, identities)
    assert result["protocol_verified"] is True
    assert result["executions"] == result["blocks"] == 0
    assert result["device_logical_tasks"] == sum(mask is not None for mask, _, _ in shapes)


def test_swimlane_core_id_joins_compact_worker_not_hardware_id():
    raw, identities = ProtocolBuilder([(2, 1, ())]).run()
    worker = raw["executions"][0]["worker_index"]
    assert identities[0]["core_id"] == worker != raw["workers"][worker]["core_id"]
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


def test_fast_member_enters_independent_b_before_remote_a_task_done():
    builder = ProtocolBuilder([(1, 2, ()), (1, 1, ())])
    builder.admit(0)
    fast = builder.claim(0, 0, 0)
    slow = builder.claim(0, 1, 3)
    builder.execute(0, 0, 0, 0, owner=fast)
    builder.close(0, 0)
    later = builder.execute(1, 0, 0, 0)
    builder.execute(0, 1, 0, 3, owner=slow)
    done = builder.actor(0, 3, "completion")
    builder.close(0, 3)
    assert later["entry"][0] < done["before"][0]
    assert audit_native_protocol(builder.raw, builder.identities)["protocol_verified"] is True


def test_nonmember_may_execute_while_all_a_members_still_have_obligations():
    builder = ProtocolBuilder([(1, 2, ()), (2, 1, ())])
    builder.admit(0, worker=1)
    builder.execute(1, 0, 1, 1)
    for block, worker in enumerate((0, 3)):
        claimed = builder.claim(0, block, worker)
        builder.execute(0, block, 0, worker, owner=claimed)
    builder.actor(0, 3, "completion")
    builder.close(0)
    assert audit_native_protocol(builder.raw, builder.identities)["protocol_verified"] is True
    builder.raw["executions"][0]["permit_tail"] = 0
    assert audit_protocol_sequences(builder.raw)["sequence_integrity_verified"] is True
    with pytest.raises(ValueError, match="causal cycle"):
        audit_native_protocol(builder.raw, builder.identities)


def test_independent_b_can_be_admitted_before_smaller_host_task_ref_a():
    builder = ProtocolBuilder([(1, 2, ()), (1, 2, ())])
    raw, identities = builder.run(order=(1, 0))
    assert raw["admission_log"] == [1, 0]
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


def test_per_worker_clock_offsets_are_not_a_global_execution_order():
    raw, identities = ProtocolBuilder([(1, 4, ()), (7, 4, (0,))]).run()
    for record, kind, _ in _event_records(raw):
        offset = (10 - record["worker_index"]) * 1000000
        for field in STAMPS[kind].split():
            if record[field][1]:
                record[field][0] += offset
        if kind == "execution":
            record["publication_before_cycles"] += offset
            record["publication_after_cycles"] += offset
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


def test_ordinary_payload_rebuild_and_predicate_skip_do_not_invent_kernel_events():
    builder = ProtocolBuilder([(2, 1, ())])
    builder.raw["tasks"][0]["flags"] |= 64
    row = builder.execute(0, 0, 1, 1, skip=True)
    row["publication_count"] = 3
    assert row["returned"] == row["visible"] == [0, 0]
    assert audit_native_protocol(builder.raw, builder.identities)["protocol_verified"] is True


def test_publisher_after_stamp_need_not_precede_remote_observation():
    raw, identities = ProtocolBuilder([(7, 1, ())]).run()
    actor = raw["task_events"][0]["admission"]
    threshold = actor["after"][0]
    for record, kind, _ in _event_records(raw):
        if record["worker_index"] != actor["worker_index"]:
            continue
        for field in STAMPS[kind].split():
            if record[field][1] and record[field][0] >= threshold:
                record[field][0] += 1000000
        if kind == "execution":
            record["publication_before_cycles"] += 1000000
            record["publication_after_cycles"] += 1000000
    assert raw["members"][1]["observed"][0] < actor["after"][0]
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


def resequence(raw):
    streams = {worker["index"]: [] for worker in raw["workers"]}
    for record, kind, _ in _event_records(raw):
        for field in STAMPS[kind].split():
            if record[field][1]:
                streams[record["worker_index"]].append(record[field])
    for worker, stamps in streams.items():
        stamps.sort(key=lambda stamp: stamp[0])
        for sequence, stamp in enumerate(stamps, 1):
            stamp[1] = sequence
        raw["workers"][worker]["next_sequence"] = len(stamps)


def test_three_irrevocable_claims_cannot_hide_behind_two_later_reused_slots():
    builder = ProtocolBuilder([(1, 3, ())], clusters=1)
    builder.admit(0)
    claims = [builder.claim(0, block, 0) for block in range(3)]
    for block, claim in enumerate(claims):
        builder.execute(0, block, 0, 0, owner=claim)
    builder.actor(0, 0, "completion")
    builder.close(0)
    assert audit_protocol_sequences(builder.raw)["sequence_integrity_verified"] is True
    with pytest.raises(ValueError, match="protocol"):
        audit_native_protocol(builder.raw, builder.identities)


def test_two_irrevocable_claims_fit_the_two_payload_slots():
    builder = ProtocolBuilder([(1, 2, ())], clusters=1)
    builder.admit(0)
    claims = [builder.claim(0, block, 0) for block in range(2)]
    for block, claim in enumerate(claims):
        builder.execute(0, block, 0, 0, owner=claim)
    builder.actor(0, 0, "completion")
    builder.close(0)
    assert audit_native_protocol(builder.raw, builder.identities)["protocol_verified"] is True


def test_direct_refill_payload_can_precede_previous_trace_cleanup():
    raw, identities = ProtocolBuilder([(1, 1, ()), (1, 1, (0,))]).run()
    next_execution = raw["executions"][1]
    previous_done = raw["task_events"][0]["completion"]["after"][0]
    next_execution.update(
        ready_source=2,
        publication_mode=1,
        publication_before_cycles=previous_done + 1,
        publication_after_cycles=previous_done + 2,
    )
    assert next_execution["publication_after_cycles"] < raw["executions"][0]["trace_published"][0]
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


@pytest.mark.parametrize("epoch", (0, (1 << 64) - 1))
def test_epoch_extremes_and_reordered_wire_rows_preserve_causality(epoch):
    raw, identities = ProtocolBuilder([(5, 2, ()), (2, 3, (0,))], epoch=epoch).run()
    for table in ("tasks", "workers", "task_controls", "admissions", "task_events", "members", "blocks", "executions"):
        raw[table].reverse()
    for block in raw["blocks"]:
        block["lanes"].reverse()
    identities.reverse()
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


def test_dependency_cycle_is_rejected_even_with_complete_counts_and_sequences():
    raw, identities = ProtocolBuilder([(0, 1, ()), (2, 1, (0,))]).run()
    predecessor = raw["task_events"][0]["completion"]
    end = raw["executions"][0]["trace_published"][0]
    predecessor["before"][0], predecessor["after"][0] = end + 10, end + 20
    resequence(raw)
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"] is True
    with pytest.raises(ValueError, match="causal cycle"):
        audit_native_protocol(raw, identities)


def test_gm_release_cannot_precede_active_lane_retirement():
    raw, identities = ProtocolBuilder([(7, 1, ())]).run()
    lane = raw["blocks"][0]["lanes"][1]
    lane["release_before"][0] = lane["arrivals_observed"][0] + 1
    lane["release_after"][0] = lane["arrivals_observed"][0] + 2
    resequence(raw)
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"] is True
    with pytest.raises(ValueError, match="protocol"):
        audit_native_protocol(raw, identities)


def wire_capture():
    protocol, identities = ProtocolBuilder([(1, 2, ()), (6, 1, (0,))]).run()
    capture = Capture(
        runtime="host_build_graph",
        level=4,
        clock_freq_hz=1000000000,
        core_types=("aic", "aiv", "aiv") * 2,
        scheduler_producer="aicore",
    )
    for row in protocol["executions"]:
        capture.task(
            task_id=row["task_ref"],
            core_id=row["worker_index"],
            reg_task_id=row["index"],
            run_epoch=protocol["run_epoch"],
            start=row["entry"][0],
            end=row["visible"][0],
        )
    raw = capture.build()
    raw["aicore_protocol"] = protocol
    raw["aicore_execution_identities"] = dict(schema="hbg-execution-identity-v1", producer="aicore", records=identities)
    return raw


def test_real_converter_preserves_native_audit_in_perfetto_metadata(tmp_path):
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(wire_capture()))
    parsed = read_perf_data(path)
    assert parsed["protocol_audit"]["protocol_verified"] is True
    assert parsed["native_protocol"]["schema"] == "hbg-native-protocol-v1"
    trace = generate_chrome_trace_json(
        parsed["tasks"], None, runtime_name="host_build_graph", timeline_metadata=parsed["timeline_metadata"]
    )
    assert trace["metadata"]["hbg_protocol_audit"] == parsed["protocol_audit"]


def test_historical_capture_is_readable_but_has_no_native_proof(tmp_path):
    raw = wire_capture()
    del raw["aicore_protocol"]
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(raw))
    assert "protocol_audit" not in read_perf_data(path)


@pytest.mark.parametrize("mutation", ("bad_ack", "unknown_schema", "missing_identity", "wrong_level"))
def test_converter_refuses_corrupt_present_protocol(tmp_path, mutation):
    raw = wire_capture()
    if mutation == "bad_ack":
        raw["aicore_protocol"]["blocks"][-1]["lanes"][0]["arrived_seq"][0] = 0
    elif mutation == "unknown_schema":
        raw["aicore_protocol"]["schema"] = "future"
    elif mutation == "missing_identity":
        del raw["aicore_execution_identities"]
    else:
        raw["chip_swimlane_level"] = 3
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="protocol"):
        read_perf_data(path)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_member",
        "wrong_member",
        "false_close",
        "pending_close",
        "wrong_ticket",
        "wrong_owner",
        "wrong_block",
        "bad_payload",
        "wrong_permit",
        "wrong_slot",
        "wrong_generation",
        "wrong_seal",
        "stale_ack",
        "missing_lane",
        "bad_placement",
        "wrong_order",
        "wrong_topology",
        "wrong_done",
        "wrong_control",
        "extra_execution",
        "wrong_identity",
    ),
)
def test_counts_and_complete_sequences_cannot_hide_protocol_corruption(mutation):
    raw, identities = ProtocolBuilder([(1, 2, ()), (6, 2, (0,)), (2, 1, (1,))]).run()
    pure = raw["blocks"][0]
    mixed = raw["blocks"][2]
    execution = raw["executions"][0]
    edits = {
        "wrong_member": (raw["members"][0], "task_ref", 1),
        "false_close": (raw["members"][0], "closed_cursor", 1),
        "pending_close": (raw["members"][0], "pending_mask", 1),
        "wrong_ticket": (pure["owner"], "completion_ticket", 1),
        "wrong_owner": (pure["owner"], "cluster_index", 1),
        "wrong_block": (pure["owner"], "block_idx", 1),
        "bad_payload": (execution, "publication_count", 2),
        "wrong_permit": (execution, "order_key", 1),
        "wrong_slot": (execution, "subtask_slot", 1),
        "wrong_generation": (execution, "generation", 0),
        "wrong_seal": (raw["admissions"][1]["close_seals"], 0, raw["admissions"][1]["close_seals"][0] + 1),
        "stale_ack": (mixed["lanes"][1]["arrived_seq"], 2, 0),
        "missing_lane": (mixed["lanes"][0], "resource_mask", 6),
        "bad_placement": (mixed["lanes"][1], "lane_subtasks", [255, 2, 1]),
        "wrong_order": (raw, "admission_log", raw["admission_log"][::-1]),
        "wrong_topology": (raw["workers"][0], "cluster_workers", [0, 1, 4]),
        "wrong_done": (raw["task_events"][0]["completion"], "flags", 1),
        "wrong_control": (raw["task_controls"][0], "completed_blocks", 1),
        "wrong_identity": (identities[0], "block_idx", 1),
    }
    if mutation == "missing_member":
        raw["admissions"][0]["member_words"][0] &= ~1
        raw["admissions"][0]["member_count"] -= 1
    elif mutation == "extra_execution":
        identities.append(copy.deepcopy(identities[0]))
    else:
        target, field, value = edits[mutation]
        target[field] = value
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"] is True
    with pytest.raises(ValueError, match="protocol"):
        audit_native_protocol(raw, identities)
