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

from simpler_setup.tools.hbg.native_causality import MIX, audit_native_protocol
from simpler_setup.tools.hbg.native_protocol import (
    STAMPS,
    _event_records,
    audit_protocol_sequences,
    read_native_protocol,
)
from simpler_setup.tools.swimlane_converter import generate_chrome_trace_json, read_perf_data


@pytest.mark.parametrize(
    "mask,blocks,source,publication",
    ((1, 1, 0, 0), (2, 1, 0, 0), (1, 1, 2, 0), (1, 1, 2, 1), (1, 2, 0, 0), (7, 1, 3, 0)),
    ids=("aic_pool", "aiv_pool", "direct", "direct_refill", "spmd_distributed", "mix_cooperative"),
)
def test_current_ready_sources_preserve_ordinary_and_cooperative_causality(mask, blocks, source, publication):
    raw, identities = ProtocolBuilder([(mask, blocks, ())]).run()
    for execution in raw["executions"]:
        execution.update(ready_source=source, publication_mode=publication)
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


@pytest.mark.parametrize("mask,blocks", ((1, 1), (2, 1), (1, 2), (7, 1)))
def test_current_ready_sources_reject_removed_stolen_source(mask, blocks):
    raw, identities = ProtocolBuilder([(mask, blocks, ())]).run()
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True
    raw["executions"][0]["ready_source"] = 1
    with pytest.raises(ValueError, match="invalid source/publication mode"):
        audit_native_protocol(raw, identities)


def _claim_seal_capture(mask, blocks, *, sync=False, sealed=True, clusters=2):
    raw, identities = ProtocolBuilder([(mask, blocks, ())], sync_tasks=(0,) if sync else (), clusters=clusters).run()
    return _set_claim_seal_observation(raw, identities, sealed=sealed)


def _set_claim_seal_observation(raw, identities, *, sealed=True):
    tasks = {task["task_ref"]: task for task in raw["tasks"]}
    for member in raw["members"]:
        task = tasks[member["task_ref"]]
        blocks = task["block_num"]
        use = sealed and not (task["flags"] & MIX and member["worker_index"] % 3) and member["closed_cursor"] == blocks
        member.update(claim_seal_seen=blocks if use else 0, claim_cursor_seen=blocks if use else 0)
    return raw, identities


def _resource_scoped_capture(masks=(3, 5, 3, 7, 6), clusters=2, blocks=3):
    raw, identities = ProtocolBuilder([(mask, blocks, ()) for mask in masks], clusters=clusters).run()
    _set_claim_seal_observation(raw, identities)
    raw["schema"] = "hbg-native-protocol-v20"
    return raw, identities


@pytest.mark.parametrize("clusters", (1, 2, 22, 32))
@pytest.mark.parametrize("blocks", (1, 2, 3, 8))
@pytest.mark.parametrize("masks", ((3,), (5,), (7,), (6,), (3, 5, 3, 7, 6)))
def test_current_resource_scope_preserves_all_clusters_and_lane_predecessors(clusters, blocks, masks):
    raw, identities = _resource_scoped_capture(masks, clusters, blocks)
    result = audit_native_protocol(raw, identities)
    assert result["protocol_verified"] is True
    assert "graph_verified" not in result


@pytest.mark.parametrize("value", (None, [], [0, 0], [0, 0, 0, 0], [True, 0, 0], [-1, 0, 0], [1 << 64, 0, 0]))
def test_current_requires_exact_typed_lane_predecessors(value):
    raw, _ = _resource_scoped_capture((3,))
    assert read_native_protocol(raw) == raw
    lane = raw["blocks"][0]["lanes"][0]
    if value is None:
        del lane["previous_lane_seq"]
    else:
        lane["previous_lane_seq"] = value
    with pytest.raises(ValueError, match="protocol"):
        read_native_protocol(raw)


@pytest.mark.parametrize("mutation", ("missing_lane", "extra_lane", "wrong_resources", "stale_ack", "wrong_previous"))
def test_current_resource_scope_rejects_wrong_membership_and_stale_group_proof(mutation):
    raw, identities = _resource_scoped_capture()
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True
    group = next(row for row in raw["blocks"] if row["owner"]["task_ref"] == 3)
    lane = group["lanes"][1]
    if mutation == "missing_lane":
        group["lanes"].pop()
    elif mutation == "extra_lane":
        first = raw["blocks"][0]
        extra = copy.deepcopy(first["lanes"][1])
        extra["index"] = extra["lane"] = 2
        first["lanes"].append(extra)
    elif mutation == "wrong_resources":
        lane["resource_mask"] = 3
    elif mutation == "stale_ack":
        lane["arrived_seq"][2] -= 1
    else:
        lane["previous_lane_seq"][1] -= 1
    with pytest.raises(ValueError):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
@pytest.mark.parametrize("blocks,sync", ((1, False), (2, False), (16, False), (2, True)))
@pytest.mark.parametrize("sealed", (False, True))
def test_current_close_layout_preserves_claim_seal_and_sync_start_proof(mask, blocks, sync, sealed):
    raw, identities = _claim_seal_capture(mask, blocks, sync=sync, sealed=sealed)
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


@pytest.mark.parametrize("clusters", (2, 22))
def test_current_close_seal_rows_do_not_alias(clusters):
    raw, identities = ProtocolBuilder(
        [(3, 2, ()), (2, 1, ()), (6, 2, ()), (7, 2, ())], clusters=clusters, sync_tasks=(0, 2, 3)
    ).run()
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True
    mixed = [row for row in raw["admissions"] if row["close_seals"]]
    mixed[1]["close_seals_offset"] = mixed[0]["close_seals_offset"] + clusters * 8
    with pytest.raises(ValueError, match="aliased or wrong task close seals"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("fault", ("alias", "interior", "swap", "unaligned", "gap", "claim_seal"))
def test_current_close_seal_rows_and_claim_seal_evidence_are_both_mandatory(fault):
    raw, identities = ProtocolBuilder([(3, 2, ()), (6, 2, ()), (7, 2, ())], sync_tasks=(0, 1, 2)).run()
    raw, identities = _set_claim_seal_observation(raw, identities)
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True
    mixed = [row for row in raw["admissions"] if row["close_seals"]]
    if fault == "alias":
        mixed[1]["close_seals_offset"] = mixed[0]["close_seals_offset"]
    elif fault == "interior":
        mixed[1]["close_seals_offset"] = mixed[0]["close_seals_offset"] + 64
    elif fault == "swap":
        mixed[1]["close_seals_offset"], mixed[2]["close_seals_offset"] = (
            mixed[2]["close_seals_offset"],
            mixed[1]["close_seals_offset"],
        )
    elif fault == "unaligned":
        for row in mixed:
            row["close_seals_offset"] += 8
    elif fault == "gap":
        mixed[2]["close_seals_offset"] += 128
    else:
        raw["task_controls"][0]["claim_seal"] = 0
    with pytest.raises(ValueError, match="native protocol"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("mask", (1, 2, 3, 5, 6, 7))
@pytest.mark.parametrize("blocks", (1, 2, 3, 16))
@pytest.mark.parametrize("sealed", (False, True))
def test_current_claim_seal_and_strong_fallback_keep_the_complete_causal_chain(mask, blocks, sealed):
    raw, identities = _claim_seal_capture(mask, blocks, sealed=sealed)
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


@pytest.mark.parametrize("mask", (1, 2, 3, 5, 6, 7))
@pytest.mark.parametrize("sealed", (False, True))
def test_current_sync_start_partial_cursor_closure_remains_valid(mask, sealed):
    raw, identities = _claim_seal_capture(mask, 2, sync=True, sealed=sealed)
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


@pytest.mark.parametrize("mask", (1, 2, 3, 5, 6, 7))
@pytest.mark.parametrize(
    "fault", ("missing", "duplicate", "publisher", "value", "early", "epoch", "witness", "terminal")
)
def test_current_rejects_invalid_claim_seal_proof(mask, fault):
    raw, identities = _claim_seal_capture(mask, 3, sync=True, clusters=3)
    last = next(block for block in raw["blocks"] if block["claim"]["confirmed_count"] == 3)
    certificate = last["claim"]
    if fault == "missing":
        last["claim"] = None
    elif fault == "duplicate":
        certificate["confirmed_count"] = 1
    elif fault == "publisher":
        certificate["worker_index"] = (certificate["worker_index"] + 1) % raw["worker_count"]
    elif fault == "value":
        certificate["seal_value"] = 2
    elif fault == "early":
        last["owner"]["claim_after"], certificate["published_before"] = (
            certificate["published_before"],
            last["owner"]["claim_after"],
        )
    elif fault == "epoch":
        certificate["run_epoch"] += 1
    elif fault == "witness":
        member = next(member for member in raw["members"] if member["claim_seal_seen"])
        member["claim_cursor_seen"] = 2
    elif fault == "terminal":
        raw["task_controls"][0]["claim_seal"] = 2
    with pytest.raises(ValueError, match="native protocol"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("mask", (1, 2, 4))
@pytest.mark.parametrize("partial_retirement", (False, True))
def test_pure_trace_retirement_follows_completion_contribution(mask, partial_retirement):
    raw, identities = ProtocolBuilder([(mask, 2, ())]).run()
    if partial_retirement:
        execution = raw["executions"][0]
        execution["trace_published"], execution["contribution_before"] = (
            execution["contribution_before"],
            execution["trace_published"],
        )
        with pytest.raises(ValueError, match="invalid local order"):
            audit_native_protocol(raw, identities)
    else:
        assert audit_native_protocol(raw, identities)["protocol_verified"] is True


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
@pytest.mark.parametrize("blocks", (2, 3))
@pytest.mark.parametrize("tail", (False, True))
def test_partial_sync_terminal_tickets_accept_bounded_tail(mask, blocks, tail):
    positions = 8 if mask in (2, 4) else 4
    raw, identities = ProtocolBuilder([(mask, blocks, ())], clusters=4, sync_tasks=(0,)).run()
    raw["task_controls"][0]["claim_tickets"] = positions if tail else blocks
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
def test_current_full_sync_start_remains_ticket_free(mask):
    blocks = 4 if mask in (2, 4) else 2
    raw, identities = ProtocolBuilder([(mask, blocks, ())], sync_tasks=(0,)).run()
    raw["schema"] = "hbg-native-protocol-v20"
    raw["task_controls"][0].update(claim_tickets=0)
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True
    raw["task_controls"][0]["claim_tickets"] = 1
    with pytest.raises(ValueError, match="ticket"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
@pytest.mark.parametrize("blocks", (1, 2, 3, 7, 9, 16))
@pytest.mark.parametrize("tail", (False, True))
def test_ordinary_work_has_no_sync_claim_tickets(mask, blocks, tail):
    raw, identities = ProtocolBuilder([(mask, blocks, ())]).run()
    raw["schema"] = "hbg-native-protocol-v20"
    raw["task_controls"][0]["claim_tickets"] = int(tail)
    if tail:
        with pytest.raises(ValueError, match="ticket"):
            audit_native_protocol(raw, identities)
    else:
        assert audit_native_protocol(raw, identities)["protocol_verified"] is True


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
def test_current_partial_sync_start_bounds_tickets_by_eligible_positions(mask):
    blocks = 2
    positions = 8 if mask in (2, 4) else 4
    raw, identities = ProtocolBuilder([(mask, blocks, ())], clusters=4, sync_tasks=(0,)).run()
    raw["schema"] = "hbg-native-protocol-v20"
    raw["task_controls"][0]["claim_tickets"] = positions
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True
    for invalid in (blocks - 1, positions + 1):
        raw["task_controls"][0]["claim_tickets"] = invalid
        with pytest.raises(ValueError, match="ticket"):
            audit_native_protocol(raw, identities)


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
@pytest.mark.parametrize("fault", ("missing", "short", "excess", "bool", "negative", "wrap"))
def test_current_small_claims_reject_invalid_raw_ticket_values(mask, fault):
    positions = 4 if mask in (2, 4) else 2
    raw, identities = ProtocolBuilder([(mask, 2, ())]).run()
    raw["schema"] = "hbg-native-protocol-v20"
    if fault == "missing":
        del raw["task_controls"][0]["claim_tickets"]
    else:
        raw["task_controls"][0]["claim_tickets"] = {
            "short": 1,
            "excess": 3 + positions,
            "bool": True,
            "negative": -1,
            "wrap": (1 << 64) - 1,
        }[fault]
    with pytest.raises(ValueError, match="protocol"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
@pytest.mark.parametrize("delta", (-1, 1))
@pytest.mark.parametrize("counter", ("next_block", "completed_subtasks"))
def test_current_valid_raw_tickets_do_not_excuse_wrong_valid_or_completion_counts(mask, delta, counter):
    raw, identities = ProtocolBuilder([(mask, 2, ())]).run()
    raw["schema"] = "hbg-native-protocol-v20"
    raw["task_controls"][0]["claim_tickets"] = 0
    raw["task_controls"][0][counter] += delta
    with pytest.raises(ValueError, match="terminal .*counter"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
@pytest.mark.parametrize("blocks", (1, 2, 7))
def test_complete_wire_protocol_across_pure_mix_and_waves(mask, blocks):
    raw, identities = ProtocolBuilder([(mask, blocks, ())]).run()
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
def test_sync_start_uses_distinct_positions_within_capacity(mask):
    blocks = 4 if mask in (2, 4) else 2
    raw, identities = ProtocolBuilder([(mask, blocks, ())], sync_tasks=(0,)).run()
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
def test_sync_start_capacity_is_not_pending_slot_count(mask):
    blocks = 5 if mask in (2, 4) else 3
    raw, identities = ProtocolBuilder([(mask, blocks, ())], sync_tasks=(0,)).run()
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"] is True
    with pytest.raises(ValueError, match="sync_start.*capacity"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
def test_sync_start_rejects_two_blocks_on_one_position_even_if_first_returned(mask):
    builder = ProtocolBuilder([(mask, 2, ())], sync_tasks=(0,))
    builder.admit(0)
    worker = 0 if mask & 1 or mask == 6 else 1
    for block in range(2):
        if mask.bit_count() > 1:
            builder.mix_block(0, block, 0)
        else:
            owner = builder.claim(0, block, worker)
            builder.execute(0, block, mask.bit_length() - 1, worker, owner=owner)
    builder.close(0)
    assert audit_protocol_sequences(builder.raw)["sequence_integrity_verified"] is True
    with pytest.raises(ValueError, match="sync_start.*position"):
        audit_native_protocol(builder.raw, builder.identities)


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
def test_sync_start_local_close_and_next_task_can_precede_remote_claim(mask):
    builder = ProtocolBuilder([(mask, 2, ()), (mask, 2, ())], sync_tasks=(0, 1))
    builder.admit(0)
    builder.admit(1, observe_members=False)
    mixed = mask.bit_count() > 1
    for block, cluster in enumerate((0, 1)):
        workers = [cluster * 3 + lane for lane in range(3)] if mixed else [cluster * 3 + int(mask != 1)]
        for task in (0, 1):
            if task == 1:
                for member in builder.raw["members"]:
                    if member["task_ref"] == task and member["worker_index"] in workers:
                        builder.stamp(member, "observed")
            if mixed:
                builder.mix_block(task, block, cluster)
            else:
                owner = builder.claim(task, block, workers[0])
                builder.execute(task, block, mask.bit_length() - 1, workers[0], owner=owner)
            for worker in workers:
                builder.close(task, worker)
    for task in (0, 1):
        for member in builder.raw["members"]:
            if member["task_ref"] != task or member["closed"][1]:
                continue
            if not member["observed"][1]:
                builder.stamp(member, "observed")
            builder.close(task, member["worker_index"])
    assert audit_native_protocol(builder.raw, builder.identities)["protocol_verified"] is True
    fast = next(row for row in builder.raw["members"] if row["closed_cursor"] == 1)
    fast["closed_cursor"] = 0
    with pytest.raises(ValueError, match="close cursor"):
        audit_native_protocol(builder.raw, builder.identities)


@pytest.mark.parametrize("mask", (2, 3, 5, 6, 7))
def test_sync_start_zero_work_member_cannot_close_before_cursor_exhaustion(mask):
    raw, identities = ProtocolBuilder([(mask, 2, ())], clusters=3, sync_tasks=(0,)).run()
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True
    idle = next(row for row in raw["members"] if row["worker_index"] == (6 if mask != 2 else 4))
    idle["closed_cursor"] = 1
    with pytest.raises(ValueError, match="close cursor"):
        audit_native_protocol(raw, identities)


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
    builder = ProtocolBuilder([(1, 2, ()), (1, 1, ())], sync_tasks=(0,))
    builder.admit(0)
    fast = builder.claim(0, 0, 0)
    slow = builder.claim(0, 1, 3)
    builder.execute(0, 0, 0, 0, owner=fast)
    builder.close(0, 0)
    later = builder.execute(1, 0, 0, 0)
    builder.execute(0, 1, 0, 3, owner=slow)
    done = next(row for row in builder.raw["task_events"] if row["index"] == 0)["completion"]
    builder.close(0, 3)
    assert later["entry"][0] < done["before"][0]
    assert audit_native_protocol(builder.raw, builder.identities)["protocol_verified"] is True


def test_nonmember_may_execute_while_all_a_members_still_have_obligations():
    builder = ProtocolBuilder([(1, 2, ()), (2, 1, ())], sync_tasks=(0,))
    builder.admit(0, worker=1)
    builder.execute(1, 0, 1, 1)
    for block, worker in enumerate((0, 3)):
        claimed = builder.claim(0, block, worker)
        builder.execute(0, block, 0, worker, owner=claimed)
    builder.close(0)
    assert audit_native_protocol(builder.raw, builder.identities)["protocol_verified"] is True
    builder.raw["executions"][0]["permit_tail"] = 0
    assert audit_protocol_sequences(builder.raw)["sequence_integrity_verified"] is True
    with pytest.raises(ValueError, match="causal cycle"):
        audit_native_protocol(builder.raw, builder.identities)


def test_independent_b_can_be_admitted_before_smaller_host_task_ref_a():
    builder = ProtocolBuilder([(1, 2, ()), (1, 2, ())], sync_tasks=(0, 1))
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
    raw, identities = ProtocolBuilder([(7, 2, ())], sync_tasks=(0,)).run()
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
        if kind == "actor" and record["flags"] & (1 << 30):
            continue
        for field in STAMPS[kind].split():
            if record[field][1]:
                streams[record["worker_index"]].append(record[field])
    for worker, stamps in streams.items():
        stamps.sort(key=lambda stamp: stamp[0])
        for sequence, stamp in enumerate(stamps, 1):
            stamp[1] = sequence
        raw["workers"][worker]["next_sequence"] = len(stamps)
    for event in raw["task_events"]:
        done = event["completion"]
        if done is not None and done["flags"] & (1 << 30):
            last = max(
                (row for row in raw["executions"] if row["task_ref"] == event["index"]),
                key=lambda row: row["contribution_ticket"],
            )
            done["before"] = last["contribution_before"].copy()
            done["after"] = last["contribution_after"].copy()


def test_ordinary_slot_reuse_requires_prior_trace_retirement():
    raw, identities = ProtocolBuilder([(1, 3, ())], clusters=1).run()
    first, _, third = raw["executions"]
    third["publication_before_cycles"] = first["publication_before_cycles"] + 1
    third["publication_after_cycles"] = first["publication_before_cycles"] + 2
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"] is True
    with pytest.raises(ValueError, match="slot before trace retirement"):
        audit_native_protocol(raw, identities)


def test_ordinary_blocks_reuse_two_slots_after_retirement():
    raw, identities = ProtocolBuilder([(1, 5, ())], clusters=1).run()
    assert {row["pending_slot"] for row in raw["executions"]} == {0, 1}
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


def test_direct_refill_payload_follows_previous_trace_and_task_done():
    raw, identities = ProtocolBuilder([(1, 1, ()), (1, 1, (0,))]).run()
    next_execution = raw["executions"][1]
    previous_done = raw["task_events"][0]["completion"]["after"][0]
    previous_trace = raw["executions"][0]["trace_published"][0]
    next_execution.update(
        ready_source=2,
        publication_mode=1,
        publication_before_cycles=previous_trace + 1,
        publication_after_cycles=previous_trace + 2,
    )
    assert previous_done < previous_trace < next_execution["publication_before_cycles"]
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


def test_cross_core_successor_can_enter_before_producer_trace_publication():
    raw, identities = ProtocolBuilder([(1, 1, ()), (2, 1, (0,))]).run()
    producer, consumer = raw["executions"]
    assert producer["worker_index"] != consumer["worker_index"]
    producer["trace_published"][0] = consumer["entry"][0] + 1
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True
    producer["trace_published"] = [0, 0]
    with pytest.raises(ValueError, match="native protocol"):
        audit_native_protocol(raw, identities)


def test_synchronous_done_is_the_last_contribution_without_extra_sequence_events():
    raw, identities = ProtocolBuilder([(1, 1, ())]).run()
    raw["schema"] = "hbg-native-protocol-v20"
    execution = raw["executions"][0]
    done = raw["task_events"][0]["completion"]
    assert done["flags"] & (1 << 30)
    assert done["before"] == execution["contribution_before"]
    assert done["after"] == execution["contribution_after"]
    assert raw["workers"][0]["next_sequence"] == 8
    result = audit_native_protocol(raw, identities)
    assert result["protocol_verified"] and result["event_count"] == 8


@pytest.mark.parametrize(
    "fault",
    (
        "deferred",
        "admission",
        "missing_flag",
        "wrong_task",
        "wrong_epoch",
        "zero_sequence",
        "different_interval",
        "two_last",
    ),
)
def test_completion_alias_must_identify_exactly_one_physical_return_before_sequence_deduplication(fault):
    raw, _ = ProtocolBuilder([(1, 2, ())], sync_tasks=(0,)).run()
    done = raw["task_events"][0]["completion"]
    last = raw["executions"][-1]
    if fault == "deferred":
        done["flags"] |= 1 << 31
    elif fault == "admission":
        raw["task_events"][0]["admission"]["flags"] |= 1 << 30
    elif fault == "missing_flag":
        done["flags"] &= ~(1 << 30)
    elif fault == "wrong_task":
        done["task_ref"] = 1
    elif fault == "wrong_epoch":
        last["run_epoch"] += 1
    elif fault == "zero_sequence":
        done["before"] = last["contribution_before"] = [0, 0]
    elif fault == "different_interval":
        done["after"][0] += 1
    else:
        raw["executions"][0]["contribution_ticket"] = last["contribution_ticket"]
    assert read_native_protocol(raw) == raw
    with pytest.raises(ValueError, match="alias|duplicate sequence"):
        audit_protocol_sequences(raw)


@pytest.mark.parametrize("field,value", (("canonical_total", 0), ("pending_compute", 1), ("completion_debt", 1)))
def test_terminal_virtual_gate_requires_its_own_completed_contribution(field, value):
    raw, identities = ProtocolBuilder([(0, 1, ())]).run()
    raw["task_controls"][0][field] = value
    with pytest.raises(ValueError, match="canonical completion gate"):
        audit_native_protocol(raw, identities)


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
    raw["aicore_execution_identities"] = dict(schema="hbg-execution-identity-v2", producer="aicore", records=identities)
    return raw


def test_real_converter_preserves_native_audit_in_perfetto_metadata(tmp_path):
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(wire_capture()))
    parsed = read_perf_data(path)
    assert parsed["protocol_audit"]["protocol_verified"] is True
    assert parsed["native_protocol"]["schema"] == "hbg-native-protocol-v20"
    trace = generate_chrome_trace_json(
        parsed["tasks"], None, runtime_name="host_build_graph", timeline_metadata=parsed["timeline_metadata"]
    )
    assert trace["metadata"]["hbg_protocol_audit"] == parsed["protocol_audit"]


def test_capture_without_protocol_has_no_native_proof(tmp_path):
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
        "wrong_pending_slot",
        "wrong_seal",
        "stale_ack",
        "missing_lane",
        "bad_placement",
        "wrong_order",
        "wrong_topology",
        "wrong_control",
        "extra_execution",
        "wrong_identity",
    ),
)
def test_counts_and_complete_sequences_cannot_hide_protocol_corruption(mutation):
    raw, identities = ProtocolBuilder([(1, 2, ()), (6, 2, (0,)), (2, 1, (1,))], sync_tasks=(0, 1)).run()
    pure = raw["blocks"][0]
    mixed = raw["blocks"][2]
    execution = raw["executions"][0]
    edits = {
        "wrong_member": (raw["members"][0], "task_ref", 1),
        "false_close": (raw["members"][0], "closed_cursor", 0),
        "pending_close": (raw["members"][0], "pending_mask", 1),
        "wrong_ticket": (pure["owner"], "completion_ticket", 1),
        "wrong_owner": (pure["owner"], "cluster_index", 1),
        "wrong_block": (pure["owner"], "block_idx", 1),
        "bad_payload": (execution, "publication_count", 2),
        "wrong_permit": (execution, "order_key", 1),
        "wrong_slot": (execution, "subtask_slot", 1),
        "wrong_pending_slot": (execution, "pending_slot", 2),
        "wrong_seal": (raw["admissions"][1]["close_seals"], 0, raw["admissions"][1]["close_seals"][0] + 1),
        "stale_ack": (mixed["lanes"][1]["arrived_seq"], 2, 0),
        "missing_lane": (mixed["lanes"][0], "resource_mask", 6),
        "bad_placement": (mixed["lanes"][1], "lane_subtasks", [255, 2, 1]),
        "wrong_order": (raw, "admission_log", raw["admission_log"][::-1]),
        "wrong_topology": (raw["workers"][0], "cluster_workers", [0, 1, 4]),
        "wrong_control": (raw["task_controls"][0], "completed_subtasks", 1),
        "wrong_identity": (identities[0], "block_idx", 1),
    }
    if mutation == "missing_member":
        raw["admissions"][0]["member_count"] -= 1
    elif mutation == "extra_execution":
        identities.append(copy.deepcopy(identities[0]))
    else:
        target, field, value = edits[mutation]
        target[field] = value
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"] is True
    with pytest.raises(ValueError, match="protocol"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
@pytest.mark.parametrize("blocks", (1, 2))
def test_deferred_finish_can_move_to_another_worker_after_execution_release(mask, blocks):
    raw, identities = ProtocolBuilder([(mask, blocks, ())], deferred_tasks=(0,)).run()
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True
    raw["task_events"][0]["completion"]["flags"] &= ~(1 << 31)
    with pytest.raises(ValueError, match="TaskDone"):
        audit_native_protocol(raw, identities)


def test_deferred_finish_flag_cannot_relabel_admission_or_inline_control():
    raw, identities = ProtocolBuilder([(7, 2, ())], sync_tasks=(0,)).run()
    raw["task_events"][0]["admission"]["flags"] |= 1 << 31
    with pytest.raises(ValueError, match="identity"):
        audit_native_protocol(raw, identities)
    raw, identities = ProtocolBuilder([(0, 1, ())]).run()
    raw["task_events"][0]["completion"]["flags"] |= 1 << 31
    with pytest.raises(ValueError, match="deferred"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("blocks", ((2,) * 6, (2, 3, 4, 3)))
def test_ordinary_variable_grids_have_no_admission_membership(blocks):
    shapes = [(mask, count, ()) for count in blocks for mask in (1, 2)]
    builder = ProtocolBuilder(shapes, clusters=6)
    raw, identities = builder.run()
    _set_claim_seal_observation(raw, identities)
    raw.update(schema="hbg-native-protocol-v20", close_seal_stride=64)
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    assert raw["cooperative_task_count"] == 0
    assert raw["admissions"] == raw["members"] == raw["blocks"] == []
    assert len(raw["executions"]) == sum(count for _, count, _ in shapes)
    raw, identities = ProtocolBuilder([(7, 2, ())], clusters=6, sync_tasks=(0,)).run()
    raw["admissions"][0]["member_rank_offset"] = 1
    with pytest.raises(ValueError, match="missing or unknown fields"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("mutation", (None, "record_alignment", "record_alias", "seal_alignment", "seal_alias"))
def test_compact_admission_and_seals_accept_64_only_alignment_and_reject_bad_regions(mutation):
    raw, identities = ProtocolBuilder([(7, 2, ()), (7, 2, (0,))], clusters=2, sync_tasks=(0, 1)).run()
    _set_claim_seal_observation(raw, identities)
    raw.update(schema="hbg-native-protocol-v20", close_seal_stride=64)
    for index, row in enumerate(raw["admissions"]):
        row["record_offset"] += 64
        row["close_seals_offset"] = 8192 + 64 + index * raw["cluster_count"] * 64
    assert raw["admissions"][0]["record_offset"] % 128 == 64
    assert raw["admissions"][0]["close_seals_offset"] % 128 == 64
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    if mutation is None:
        return
    first, second = raw["admissions"]
    if mutation == "record_alignment":
        first["record_offset"] += 8
    elif mutation == "record_alias":
        second["record_offset"] = first["record_offset"]
    elif mutation == "seal_alignment":
        first["close_seals_offset"] += 8
    else:
        second["close_seals_offset"] = first["close_seals_offset"]
    with pytest.raises(ValueError, match="record|seal"):
        audit_native_protocol(raw, identities)


def test_current_cooperative_capture_rejects_independent_flag():
    raw, identities = ProtocolBuilder([(1, 2, ())]).run()
    _set_claim_seal_observation(raw, identities)
    raw.update(schema="hbg-native-protocol-v20", close_seal_stride=64)
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    raw["tasks"][0]["flags"] |= 128
    raw["task_events"][0]["completion"]["flags"] |= 128
    with pytest.raises(ValueError, match="unknown task flags"):
        audit_native_protocol(raw, identities)


def _subtask_capture(shapes=((3, 2, ()), (2, 1, (0,))), *, deferred=()):
    raw, identities = ProtocolBuilder(
        shapes,
        clusters=2,
        deferred_tasks=deferred,
    ).run()
    _set_claim_seal_observation(raw, identities)
    raw["schema"] = "hbg-native-protocol-v20"
    return raw, identities


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
@pytest.mark.parametrize("blocks", (1, 2, 5))
@pytest.mark.parametrize("deferred", ((), (0,)))
def test_current_counts_actual_subtasks_and_accepts_last_lane_done_before_mix_retirement(mask, blocks, deferred):
    raw, identities = _subtask_capture(((mask, blocks, ()), (2, 1, (0,))), deferred=deferred)
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True
    assert raw["task_controls"][0]["completed_subtasks"] == blocks * mask.bit_count()
    if mask.bit_count() > 1:
        last = max((e for e in raw["executions"] if e["task_ref"] == 0), key=lambda e: e["contribution_ticket"])
        done = raw["task_events"][0]["completion"]
        if not deferred:
            assert done["worker_index"] == last["worker_index"]
        assert done["after"][0] < raw["blocks"][-1]["owner"]["completion_before"][0]


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate_ticket",
        "missing_ticket",
        "foreign_contributor",
        "block_count",
        "wrong_done_owner",
        "done_before_last",
        "retirement_ticket",
        "missing_stamp",
        "old_counter_name",
    ),
)
def test_current_rejects_broken_subtask_completion_proof(mutation):
    raw, identities = _subtask_capture()
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True
    executions = [e for e in raw["executions"] if e["task_ref"] == 0]
    last = max(executions, key=lambda e: e["contribution_ticket"])
    done = raw["task_events"][0]["completion"]
    if mutation == "duplicate_ticket":
        last["contribution_ticket"] = 0
    elif mutation == "missing_ticket":
        last["contribution_ticket"] += 1
    elif mutation == "foreign_contributor":
        last["contribution_worker"] = 0
    elif mutation == "block_count":
        raw["task_controls"][0]["completed_subtasks"] = 2
    elif mutation == "wrong_done_owner":
        done["worker_index"] = 0
    elif mutation == "done_before_last":
        last["contribution_after"], done["before"] = done["before"], last["contribution_after"]
    elif mutation == "retirement_ticket":
        raw["blocks"][0]["owner"]["completion_ticket"] = 0
    elif mutation == "missing_stamp":
        del last["contribution_before"]
    else:
        control = raw["task_controls"][0]
        control["completed_blocks"] = control.pop("completed_subtasks")
    with pytest.raises(ValueError):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("fault", (None, "hold", "frontier", "tail", "stage", "source", "identity"))
def test_ordinary_cluster_rendezvous_retains_sync_arbitration_proof(fault):
    raw, identities = ProtocolBuilder([(1, 2, ()), (6, 1, (0,)), (7, 2, (1,))], sync_tasks=(0, 2)).run()
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True
    group = next(block for block in raw["blocks"] if block["owner"]["task_ref"] == 1)
    owner = group["owner"]
    assert owner["sync_tail"] == 1 and owner["order_key"] == (1 << 64) - 1
    execution = next(row for row in raw["executions"] if row["task_ref"] == 1)
    if fault is None:
        assert raw["cooperative_task_count"] == 2 and raw["sync_mix_task_count"] == 1
        return
    if fault == "hold":
        owner["sync_hold"][0] = 1
    elif fault == "frontier":
        owner["sync_frontier"][0] = 0
    elif fault == "tail":
        execution["permit_tail"] = 0
    elif fault == "stage":
        owner["staged"] = [0, 0]
    elif fault == "source":
        execution["flags"] = (execution["flags"] & ~128) | 8
    else:
        next(row for row in identities if row["task_id"] == 1)["group_kind"] = 1
    with pytest.raises(ValueError, match="protocol"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("mask", (3, 5, 6, 7))
def test_future_mix_reservation_precedes_fanin_but_claim_follows_staging(mask):
    builder = ProtocolBuilder([(1, 1, ()), (mask, 1, (0,))], clusters=1)
    reserved = builder.claim(1, 0, 0)
    assert reserved["owner"]["claim_before"] == reserved["owner"]["claim_after"] == [0, 0]
    builder.execute(0, 0, 0, 0)
    group = builder.mix_block(1, 0, 0, reserved=reserved)
    done = next(row for row in builder.raw["task_events"] if row["index"] == 0)["completion"]
    assert done["after"][0] < group["owner"]["staged"][0] < group["owner"]["claim_before"][0]
    assert audit_native_protocol(builder.raw, builder.identities)["protocol_verified"]


@pytest.mark.parametrize("sync", (False, True))
def test_mix_cannot_stage_or_admit_before_its_fanin_completes(sync):
    count = 2 if sync else 1
    builder = ProtocolBuilder([(1, 1, ()), (3, count, (0,))], clusters=2, sync_tasks=(1,) if sync else ())
    if sync:
        builder.admit(1)
    for block in range(count):
        builder.mix_block(1, block, block)
    if sync:
        builder.close(1)
    builder.execute(0, 0, 0, 0)
    assert audit_protocol_sequences(builder.raw)["sequence_integrity_verified"]
    with pytest.raises(ValueError, match="causal cycle"):
        audit_native_protocol(builder.raw, builder.identities)


def test_ordinary_mix_final_owner_intervals_need_not_follow_fresh_block_ordinals():
    builder = ProtocolBuilder([(3, 2, ())], clusters=1)
    builder.mix_block(0, 1, 0)
    builder.mix_block(0, 0, 0)
    assert audit_native_protocol(builder.raw, builder.identities)["protocol_verified"]


@pytest.mark.parametrize("boundary", ("claim_before", "claim_after", "publish_before"))
def test_ordinary_mix_stage_precedes_claim_and_publication(boundary):
    raw, identities = ProtocolBuilder([(3, 1, ())]).run()
    owner = raw["blocks"][0]["owner"]
    owner["staged"], owner[boundary] = owner[boundary], owner["staged"]
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"]
    with pytest.raises(ValueError, match="protocol"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("mask", (1, 2, 4))
def test_single_ordinary_execution_has_one_fresh_ticket(mask):
    raw, identities = ProtocolBuilder([(mask, 1, ())]).run()
    assert raw["task_controls"][0]["next_block"] == 1
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    raw["task_controls"][0]["next_block"] = 0
    with pytest.raises(ValueError, match="terminal block counters"):
        audit_native_protocol(raw, identities)


def test_every_ordinary_spmd_block_must_wait_for_fanin_not_only_block_zero():
    builder = ProtocolBuilder([(1, 1, ()), (1, 2, (0,))], clusters=1)
    builder.execute(1, 1, 0, 0)
    builder.execute(0, 0, 0, 0)
    builder.execute(1, 0, 0, 0)
    assert audit_protocol_sequences(builder.raw)["sequence_integrity_verified"]
    with pytest.raises(ValueError, match="causal cycle"):
        audit_native_protocol(builder.raw, builder.identities)


def test_early_claim_timestamps_cannot_be_attached_to_later_protocol_sequences():
    builder = ProtocolBuilder([(1, 1, ()), (3, 1, (0,))], clusters=1)
    reserved = builder.claim(1, 0, 0)
    predecessor = builder.execute(0, 0, 0, 0)
    builder.mix_block(1, 0, 0, reserved=reserved)
    owner = reserved["owner"]
    for claim, execution in (("claim_before", "permit"), ("claim_after", "entry")):
        owner[claim][0] = predecessor[execution][0]
    with pytest.raises(ValueError, match="backwards clock"):
        audit_native_protocol(builder.raw, builder.identities)
