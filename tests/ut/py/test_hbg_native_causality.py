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

from simpler_setup.tools.hbg.native_causality import MIX, NativeAudit, audit_native_protocol, point
from simpler_setup.tools.hbg.native_protocol import (
    CONTROL_NOTIFY_ACCOUNTING,
    COUNTER_ACCOUNTING,
    STAMPS,
    _event_records,
    audit_protocol_sequences,
    read_native_protocol,
    stamp_worker,
)
from simpler_setup.tools.swimlane_converter import generate_chrome_trace_json, read_perf_data


@pytest.mark.parametrize(
    "mask,blocks,source",
    ((1, 1, 0), (2, 1, 0), (1, 2, 0), (7, 1, 3)),
    ids=("aic_catalog", "aiv_catalog", "spmd_distributed", "mix_cooperative"),
)
def test_current_ready_sources_preserve_ordinary_and_cooperative_causality(mask, blocks, source):
    raw, identities = ProtocolBuilder([(mask, blocks, ())]).run()
    for execution in raw["executions"]:
        execution["ready_source"] = source
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


@pytest.mark.parametrize("source", (1, 2))
@pytest.mark.parametrize("mask,blocks", ((1, 1), (2, 1), (1, 2), (7, 1)))
def test_current_ready_sources_reject_unproduced_sources(mask, blocks, source):
    raw, identities = ProtocolBuilder([(mask, blocks, ())]).run()
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True
    raw["executions"][0]["ready_source"] = source
    with pytest.raises(ValueError, match="invalid ready source"):
        audit_native_protocol(raw, identities)


def _resource_scoped_capture(masks=(3, 5, 3, 7, 6), clusters=2, blocks=3):
    raw, identities = ProtocolBuilder([(mask, blocks, ()) for mask in masks], clusters=clusters).run()
    raw["schema"] = "hbg-native-protocol"
    return raw, identities


@pytest.mark.parametrize("clusters", (1, 2, 22, 32))
@pytest.mark.parametrize("blocks", (1, 2, 3, 8))
@pytest.mark.parametrize("masks", ((3,), (5,), (7,), (6,), (3, 5, 3, 7, 6)))
def test_current_resource_scope_preserves_all_clusters_and_lane_predecessors(clusters, blocks, masks):
    raw, identities = _resource_scoped_capture(masks, clusters, blocks)
    result = audit_native_protocol(raw, identities)
    assert result["protocol_verified"] is True
    assert "graph_verified" not in result


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
        lane["group_seq"] += 2
    with pytest.raises(ValueError):
        audit_native_protocol(raw, identities)


def test_noncomputing_aiv0_publishes_for_empty_and_active_clusters():
    raw, identities = ProtocolBuilder([(5, 1, ()), (2, 2, ()), (6, 2, ())], sync_tasks=(1, 2)).run()
    first = raw["blocks"][0]["owner"]
    assert first["worker_index"] == 1 and first["completion_worker"] == 0
    assert [row["lane"] for row in raw["blocks"][0]["lanes"]] == [0, 2]
    assert {e["cluster_index"] for e in raw["floor_publications"] if e["prior_mix_ref"] == 0} == {0, 1}
    assert raw["tasks"][1]["required_floor"] == [0, 1, 1]
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    raw["tasks"][1]["required_floor"][1] = 0
    with pytest.raises(ValueError, match="threshold|proxy"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize(
    "fault", ("missing_publisher", "stale_at_commit", "stale_at_permit", "missing_reverse", "false_disjoint")
)
def test_irreversible_commit_and_later_entry_consume_separate_floor_evidence(fault):
    raw, ids = ProtocolBuilder([(6, 1, ()), (2, 2, ()), (5, 2, ())], sync_tasks=(1, 2)).run()
    assert audit_native_protocol(raw, ids)["protocol_verified"]
    if fault == "missing_publisher":
        raw["floor_publications"][0]["prior_mix_ref"] = (1 << 32) - 1
    elif fault == "stale_at_commit":
        next(member for member in raw["members"] if member["task_ref"] == 1)["observed_mix_floor"] = 0
    elif fault == "stale_at_permit":
        execution = next(e for e in raw["executions"] if e["task_ref"] == 1)
        execution["observed_mix_floor"] = 0
        assert audit_protocol_sequences(raw)["sequence_integrity_verified"]
    elif fault == "missing_reverse":
        next(member for member in raw["members"] if member["task_ref"] == 2 and member["worker_index"] % 3 == 2)[
            "observed_self_floor"
        ] = 0
    else:
        row = next(member for member in raw["members"] if member["task_ref"] == 2 and member["worker_index"] % 3 == 0)
        row["observed_mix_floor"] = 1
    with pytest.raises(ValueError, match="floor|threshold|opposite"):
        audit_native_protocol(raw, ids)


@pytest.mark.parametrize("fault", ["claim_before_floor", "commit_before_claim"])
def test_pure_sync_claim_observes_floor_before_committing(fault):
    raw, ids = ProtocolBuilder([(5, 1, ()), (2, 2, ())], sync_tasks=(1,)).run()
    assert audit_native_protocol(raw, ids)["protocol_verified"]
    owner = next(b["owner"] for b in raw["blocks"] if b["owner"]["task_ref"] == 1)
    member = next(m for m in raw["members"] if m["worker_index"] == owner["worker_index"])
    floor = next(f for f in raw["floor_publications"] if f["worker_index"] == owner["worker_index"])
    if fault == "claim_before_floor":
        owner["claim_before"][0] = floor["before"][0] - 1
    else:
        member["committed"][0] = owner["claim_after"][0] - 1
    resequence(raw)
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"]
    with pytest.raises(ValueError, match="causal cycle|commitment before block claim"):
        audit_native_protocol(raw, ids)


@pytest.mark.parametrize("mask", [3, 5], ids=["member", "accountant"])
def test_lower_rank_self_permit_must_stay_outside_active_mix_arrival_to_fin(mask):
    raw, ids = ProtocolBuilder([(1, 1, ()), (mask, 1, ())]).run()
    assert audit_native_protocol(raw, ids)["protocol_verified"]
    lane = raw["blocks"][0]["lanes"][0]
    pure = raw["executions"][0]
    assert pure["worker_index"] == lane["worker_index"] and pure["order_key"] < lane["order_key"]
    for delta, field in enumerate(("descriptor_ready", "arrive_before", "arrive_after"), 1):
        lane[field][0] = pure["permit"][0] - 4 + delta
    resequence(raw)
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"]
    with pytest.raises(ValueError, match="self work during active MIX"):
        audit_native_protocol(raw, ids)


@pytest.mark.parametrize("copy", ["execution", "member"])
def test_mix_lane_repeats_its_local_floor_observation_at_commit_arrival_and_permit(copy):
    raw, ids = ProtocolBuilder([(1, 2, ()), (3, 2, ()), (None, 0, ())], sync_tasks=(0, 1)).run()
    assert audit_native_protocol(raw, ids)["protocol_verified"]
    table = raw["members"] if copy == "member" else raw["executions"]
    row = next(r for r in table if r["task_ref"] == 1 and r["worker_index"] == 0)
    row["observed_self_floor"] += 1
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"]
    with pytest.raises(ValueError, match="inconsistent MIX self-floor observation"):
        audit_native_protocol(raw, ids)


def test_mix_reads_share_the_nonmember_seal_prefix():
    count = 10
    b = ProtocolBuilder([(1, 2, ())] * count + [(3, 1, ())] * count, clusters=3, sync_tasks=range(count))
    b.run(order=range(count))
    for task in range(count, 2 * count):
        b.mix_block(task, 0, 2)
    audit = NativeAudit(b.raw, b.identities)
    assert audit.audit()["protocol_verified"]
    # The third cluster owns no pure block. Its MIX reads all need the same
    # ten seals, including the earliest ones from other clocks.
    for task in range(count):
        seal = audit.claim_seals[task]
        published = next(
            row["claim"] for row in b.raw["blocks"] if row["owner"]["task_ref"] == task and row["claim"]["seal_value"]
        )
        successors = audit.causal.edges[seal] - {audit.causal.key(point(published, "published_after"))}
        assert len(successors) == 1
    # The earliest seal must still precede the last MIX arrival. The clocks of
    # the two workers alone impose no such order.
    last_arrival = b.raw["blocks"][-1]["lanes"][0]
    audit.causal.add(point(last_arrival, "arrive_before"), audit.claim_seals[0])
    with pytest.raises(ValueError, match="causal cycle"):
        audit.causal.validate()


def test_floor_rows_remain_readable_in_dense_and_older_sparse_layouts():
    raw, ids = ProtocolBuilder([(1, 2, ()), (3, 1, ()), (2, 2, ()), (5, 1, ())], sync_tasks=(0, 2)).run()
    clusters = raw["cluster_count"]
    assert raw["floor_count"] == (raw["mix_task_count"] + 1) * clusters
    assert {r["index"] for r in raw["floor_publications"]} == set(range(raw["floor_count"]))
    assert audit_native_protocol(raw, ids)["protocol_verified"]
    raw["floor_count"] = (raw["cooperative_task_count"] + 1) * clusters
    with pytest.raises(ValueError, match="misbound MIX floor"):
        audit_native_protocol(raw, ids)
    for event in raw["floor_publications"]:
        prior = event["prior_mix_ref"]
        row = 0 if prior == (1 << 32) - 1 else 1 + raw["tasks"][prior]["cooperative_index"]
        event["index"] = row * clusters + event["cluster_index"]
    assert audit_native_protocol(raw, ids)["protocol_verified"]


def test_ordinary_mix_arrival_cannot_rely_on_a_later_permit_proof():
    raw, ids = ProtocolBuilder([(1, 2, ()), (5, 1, ())], sync_tasks=(0,)).run()
    assert audit_native_protocol(raw, ids)["protocol_verified"]
    group = next(b for b in raw["blocks"] if b["owner"]["task_ref"] == 1)
    lane = next(row for row in group["lanes"] if row["lane"] == 0)
    assert lane["observed_self_floor"] > 0
    lane["observed_self_floor"] = 0
    with pytest.raises(ValueError, match="threshold"):
        audit_native_protocol(raw, ids)


def test_distinct_block_ids_do_not_imply_a_global_order_of_atomic_claims():
    b = ProtocolBuilder([(1, 2, ())], sync_tasks=(0,))
    last_id = b.claim(0, 1, 3)
    first_id = b.claim(0, 0, 0)
    b.execute(0, 1, 0, 3, owner=last_id)
    b.execute(0, 0, 0, 0, owner=first_id)
    assert first_id["claim"]["seal_value"] == 2 and last_id["claim"]["confirmed_count"] == 0
    assert audit_native_protocol(b.raw, b.identities)["protocol_verified"]


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
@pytest.mark.parametrize("blocks", (1, 2, 7))
def test_complete_wire_protocol_across_pure_mix_and_waves(mask, blocks):
    raw, identities = ProtocolBuilder([(mask, blocks, ())]).run()
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
def test_sync_start_uses_distinct_positions_within_capacity(mask):
    blocks = 4 if mask in (2, 4) else 2
    raw, identities = ProtocolBuilder([(mask, blocks, ())], sync_tasks=(0,)).run()
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True
    if mask == 4:
        assert {e["cluster_lane"] for e in raw["executions"]} == {1, 2}
        assert {e["subtask_slot"] for e in raw["executions"]} == {2}


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
def test_sync_start_capacity_is_not_pending_slot_count(mask):
    blocks = 4 if mask in (2, 4) else 2
    raw, identities = ProtocolBuilder([(mask, blocks, ())], sync_tasks=(0,), deferred_tasks=(0,)).run()
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    raw["tasks"][0]["block_num"] += 1
    with pytest.raises(ValueError, match="sync_start distinct position capacity exceeded"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
def test_sync_start_rejects_two_blocks_on_one_position_even_if_first_returned(mask):
    builder = ProtocolBuilder([(mask, 2, ())], sync_tasks=(0,))
    worker = 0 if mask & 1 or mask == 6 else 1
    for block in range(2):
        if bin(mask).count("1") > 1:
            builder.mix_block(0, block, 0)
        else:
            owner = builder.claim(0, block, worker)
            builder.execute(0, block, mask.bit_length() - 1, worker, owner=owner)
    first, second = builder.raw["executions"][0], builder.raw["executions"][bin(mask).count("1")]
    assert first["trace_published"][0] < second["permit"][0]
    # The builder retired the first member row again on the second return. Keep
    # that single commitment for each physical position and rebuild its sequence.
    builder.raw["members"] = [row for row in builder.raw["members"] if row["retired"][1]]
    resequence(builder.raw)
    assert audit_protocol_sequences(builder.raw)["sequence_integrity_verified"]
    with pytest.raises(ValueError, match="sync_start assigned two blocks to one physical worker"):
        audit_native_protocol(builder.raw, builder.identities)


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
    fast = builder.claim(0, 0, 0)
    slow = builder.claim(0, 1, 3)
    builder.execute(0, 0, 0, 0, owner=fast)
    later = builder.execute(1, 0, 0, 0)
    builder.execute(0, 1, 0, 3, owner=slow)
    done = next(row for row in builder.raw["task_events"] if row["index"] == 0)["completion"]
    assert later["entry"][0] < done["before"][0]
    assert audit_native_protocol(builder.raw, builder.identities)["protocol_verified"] is True


def test_nonmember_may_execute_while_all_a_members_still_have_obligations():
    builder = ProtocolBuilder([(1, 2, ()), (2, 1, ())], sync_tasks=(0,))
    builder.execute(1, 0, 1, 1)
    for block, worker in enumerate((0, 3)):
        claimed = builder.claim(0, block, worker)
        builder.execute(0, block, 0, worker, owner=claimed)
    assert audit_native_protocol(builder.raw, builder.identities)["protocol_verified"] is True
    builder.raw["executions"][0]["observed_mix_floor"] = 1
    assert audit_protocol_sequences(builder.raw)["sequence_integrity_verified"] is True
    with pytest.raises(ValueError, match="opposite-source read"):
        audit_native_protocol(builder.raw, builder.identities)


def test_control_notification_join_does_not_use_the_accounting_cas_as_ready_time():
    builder = ProtocolBuilder([(1, 1, ()), (0, 1, (0,)), (1, 1, (1,))])
    raw, identities = builder.run()
    actor = raw["task_events"][1]["completion"]
    actor["flags"] |= CONTROL_NOTIFY_ACCOUNTING
    actor["before"][0] = builder.clock + 10
    actor["after"][0] = builder.clock + 20
    resequence(raw)
    assert audit_native_protocol(raw, identities)["protocol_verified"]


def test_control_notification_flag_cannot_relabel_a_kernel_completion():
    raw, identities = ProtocolBuilder([(1, 1, ())]).run()
    raw["task_events"][0]["completion"]["flags"] |= CONTROL_NOTIFY_ACCOUNTING
    with pytest.raises(ValueError, match="control notification accounting|completion alias"):
        audit_native_protocol(raw, identities)


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
    assert row["returned"] == row["visible"] == [0, 0]
    assert audit_native_protocol(builder.raw, builder.identities)["protocol_verified"] is True


def test_publisher_after_stamp_need_not_precede_remote_observation():
    raw, identities = ProtocolBuilder([(7, 2, ())], sync_tasks=(0,)).run()
    owner = raw["blocks"][0]["owner"]
    threshold, publisher = owner["publish_after"][0], owner["worker_index"]
    for record, kind, _ in _event_records(raw):
        for field in STAMPS[kind].split():
            if stamp_worker(record, field) == publisher and record[field][1] and record[field][0] >= threshold:
                record[field][0] += 1000000
        if kind == "execution" and record["worker_index"] == publisher:
            record["publication_before_cycles"] += 1000000
            record["publication_after_cycles"] += 1000000
    assert raw["blocks"][0]["lanes"][0]["descriptor_ready"][0] < owner["publish_after"][0]
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True


def resequence(raw):
    streams = {worker["index"]: [] for worker in raw["workers"]}
    for record, kind, _ in _event_records(raw):
        if kind == "actor" and record["flags"] & ((1 << 29) | (1 << 30)):
            continue
        for field in STAMPS[kind].split():
            if record[field][1]:
                streams[stamp_worker(record, field)].append(record[field])
    for worker, stamps in streams.items():
        stamps.sort(key=lambda stamp: stamp[0])
        for sequence, stamp in enumerate(stamps, 1):
            stamp[1] = sequence
        raw["workers"][worker]["next_sequence"] = len(stamps)
    for event in raw["task_events"]:
        done = event["completion"]
        if done is not None and done["flags"] & (1 << 29):
            last = max(
                (row["owner"] for row in raw["blocks"] if row["owner"]["task_ref"] == event["index"]),
                key=lambda row: row["completion_ticket"],
            )
            done["before"] = last["completion_before"].copy()
            done["after"] = last["completion_after"].copy()
        elif done is not None and done["flags"] & (1 << 30):
            last = max(
                (row for row in raw["executions"] if row["task_ref"] == event["index"]),
                key=lambda row: row["contribution_ticket"],
            )
            done["before"] = last["contribution_before"].copy()
            done["after"] = last["contribution_after"].copy()


def test_ordinary_slot_reuse_requires_prior_trace_retirement():
    raw, identities = ProtocolBuilder([(1, 5, ())], clusters=1).run()
    first, *_, reused = raw["executions"]
    reused["publication_before_cycles"] = first["publication_before_cycles"] + 1
    reused["publication_after_cycles"] = first["publication_before_cycles"] + 2
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"] is True
    with pytest.raises(ValueError, match="slot before trace retirement"):
        audit_native_protocol(raw, identities)


def test_ordinary_blocks_reuse_four_original_records_after_retirement():
    raw, identities = ProtocolBuilder([(1, 5, ())], clusters=1).run()
    assert {row["pending_slot"] for row in raw["executions"]} == {0, 1, 2, 3}
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
    raw["schema"] = "hbg-native-protocol"
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
        "foreign_owner",
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
    elif fault == "foreign_owner":
        done["worker_index"] = (last["worker_index"] + 1) % raw["worker_count"]
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
    for table in (
        "tasks",
        "workers",
        "task_controls",
        "floor_publications",
        "task_events",
        "members",
        "blocks",
        "executions",
    ):
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
    raw["aicore_execution_identities"] = dict(schema="hbg-execution-identity", producer="aicore", records=identities)
    return raw


def test_real_converter_preserves_native_audit_in_perfetto_metadata(tmp_path):
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(wire_capture()))
    parsed = read_perf_data(path)
    assert parsed["protocol_audit"]["protocol_verified"] is True
    assert parsed["native_protocol"]["schema"] == "hbg-native-protocol"
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
        lane = raw["aicore_protocol"]["blocks"][-1]["lanes"][0]
        lane["arrived_seq"][lane["lane"]] = 0
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
        "wrong_frontier",
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
        "false_close": (raw["members"][0], "observed_mix_floor", len(raw["tasks"]) + 1),
        "pending_close": (raw["members"][0], "observed_self_floor", 1),
        "wrong_ticket": (pure["owner"], "completion_ticket", 1),
        "wrong_owner": (pure["owner"], "cluster_index", 1),
        "wrong_block": (pure["owner"], "block_idx", 1),
        "bad_payload": (execution, "publication_count", 2),
        "wrong_permit": (execution, "order_key", 1),
        "wrong_slot": (execution, "subtask_slot", 1),
        "wrong_pending_slot": (execution, "pending_slot", 6),
        "wrong_frontier": (raw["floor_publications"][-1], "floor", len(raw["tasks"]) + 1),
        "stale_ack": (mixed["lanes"][1]["arrived_seq"], 2, 0),
        "missing_lane": (mixed["lanes"][0], "resource_mask", 2),
        "bad_placement": (mixed["lanes"][1], "lane_subtasks", [255, 2, 1]),
        "wrong_order": (raw["tasks"][0], "schedule_rank", raw["tasks"][1]["schedule_rank"]),
        "wrong_topology": (raw["workers"][0], "cluster_workers", [0, 1, 4]),
        "wrong_control": (raw["task_controls"][0], "completed_subtasks", 1),
        "wrong_identity": (identities[0], "block_idx", 1),
    }
    if mutation == "missing_member":
        raw["members"][0]["task_ref"] = len(raw["tasks"])
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


def test_deferred_finish_flag_cannot_relabel_inline_control():
    raw, identities = ProtocolBuilder([(0, 1, ())]).run()
    raw["task_events"][0]["completion"]["flags"] |= 1 << 31
    with pytest.raises(ValueError, match="deferred"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("blocks", ((2,) * 6, (2, 3, 4, 3)))
def test_pure_ordinary_grids_have_no_cooperative_membership(blocks):
    shapes = [(mask, count, ()) for count in blocks for mask in (1, 2)]
    builder = ProtocolBuilder(shapes, clusters=6)
    raw, identities = builder.run()
    raw.update(schema="hbg-native-protocol")
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    assert raw["cooperative_task_count"] == 0
    assert raw["floor_publications"] == raw["members"] == raw["blocks"] == []
    assert len(raw["executions"]) == sum(count for _, count, _ in shapes)
    raw, identities = ProtocolBuilder([(7, 2, ())], clusters=6, sync_tasks=(0,)).run()
    raw["members"][0]["member_rank_offset"] = 1
    with pytest.raises(ValueError, match="missing or unknown fields"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("flag", (128,))
def test_current_cooperative_capture_rejects_independent_flag(flag):
    raw, identities = ProtocolBuilder([(1, 2, ())]).run()
    raw.update(schema="hbg-native-protocol")
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    raw["tasks"][0]["flags"] |= flag
    raw["task_events"][0]["completion"]["flags"] |= flag
    with pytest.raises(ValueError, match="unknown task flags"):
        audit_native_protocol(raw, identities)


def _subtask_capture(shapes=((3, 2, ()), (2, 1, (0,))), *, deferred=()):
    raw, identities = ProtocolBuilder(
        shapes,
        clusters=2,
        deferred_tasks=deferred,
    ).run()
    raw["schema"] = "hbg-native-protocol"
    return raw, identities


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
@pytest.mark.parametrize("blocks", (1, 2, 5))
@pytest.mark.parametrize("deferred", ((), (0,)))
def test_current_counts_actual_subtasks_and_aliases_mix_group_contribution(mask, blocks, deferred):
    raw, identities = _subtask_capture(((mask, blocks, ()), (2, 1, (0,))), deferred=deferred)
    assert audit_native_protocol(raw, identities)["protocol_verified"] is True
    assert raw["task_controls"][0]["completed_subtasks"] == blocks * bin(mask).count("1")
    if bin(mask).count("1") > 1:
        last = max(
            (b["owner"] for b in raw["blocks"] if b["owner"]["task_ref"] == 0),
            key=lambda owner: owner["completion_ticket"],
        )
        done = raw["task_events"][0]["completion"]
        if not deferred:
            assert done["worker_index"] == last["completion_worker"]
            assert done["flags"] & (1 << 29)
            assert done["before"] == last["completion_before"]
            assert done["after"] == last["completion_after"]
        else:
            assert done["before"][0] > last["completion_after"][0]


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
    raw, identities = _subtask_capture(((1, 2, ()), (2, 1, (0,))))
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
        raw["task_controls"][0]["completed_subtasks"] = 1
    elif mutation == "wrong_done_owner":
        done["worker_index"] = 0
    elif mutation == "done_before_last":
        last["contribution_after"], done["before"] = done["before"], last["contribution_after"]
    elif mutation == "retirement_ticket":
        last["contribution_ticket"] = (1 << 64) - 1
    elif mutation == "missing_stamp":
        del last["contribution_before"]
    else:
        control = raw["task_controls"][0]
        control["completed_blocks"] = control.pop("completed_subtasks")
    with pytest.raises(ValueError):
        audit_native_protocol(raw, identities)


def test_ordinary_mix_final_owner_intervals_need_not_follow_fresh_block_ordinals():
    builder = ProtocolBuilder([(3, 2, ())], clusters=1)
    builder.mix_block(0, 1, 0)
    builder.mix_block(0, 0, 0)
    assert audit_native_protocol(builder.raw, builder.identities)["protocol_verified"]


@pytest.mark.parametrize("mask", (1, 2, 4))
def test_single_ordinary_execution_has_one_fresh_ticket(mask):
    raw, identities = ProtocolBuilder([(mask, 1, ())]).run()
    assert raw["task_controls"][0]["next_block"] == 1
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    raw["task_controls"][0]["next_block"] = 0
    with pytest.raises(ValueError, match="terminal block counters"):
        audit_native_protocol(raw, identities)


def test_mix_handshake_precedes_final_payload_publication():
    raw, identities = ProtocolBuilder([(5, 1, ())]).run()
    lane = raw["blocks"][0]["lanes"][1]
    execution = raw["executions"][1]
    observed = lane["arrivals_observed"][0]
    assert lane["worker_index"] == execution["worker_index"]
    execution["publication_before_cycles"] = observed
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    execution["publication_before_cycles"] = observed - 1
    with pytest.raises(ValueError, match="MIX payload publication before local group handshake"):
        audit_native_protocol(raw, identities)


def test_mix_generations_follow_rank_even_for_independent_tasks():
    raw, identities = ProtocolBuilder([(3, 2, ()), (5, 1, ())], clusters=1).run(order=(1, 0))
    # Both floors are published in rank order after both descriptors completed.
    for index, floor in enumerate(sorted(raw["floor_publications"], key=lambda row: row["floor"])):
        floor["before"][0] = 100000 + 20 * index
        floor["after"][0] = floor["before"][0] + 10
    resequence(raw)
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"]
    with pytest.raises(ValueError, match="decreasing MIX rank"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("mixed", [False, True], ids=["pure-payload", "mix-arrival"])
def test_fanin_must_precede_first_use_even_when_permit_is_later(mixed):
    raw, identities = ProtocolBuilder([(0, 1, ()), (3 if mixed else 2, 1, (0,))], clusters=1).run()
    done = raw["task_events"][0]["completion"]
    if mixed:
        lane = raw["blocks"][0]["lanes"][1]
        assert lane["worker_index"] == done["worker_index"]
        boundary = lane["arrive_before"][0]
    else:
        execution = raw["executions"][0]
        assert execution["worker_index"] == done["worker_index"]
        boundary = execution["publication_before_cycles"]
    done["before"][0], done["after"][0] = boundary - 2, boundary - 1
    resequence(raw)
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    done["before"][0], done["after"][0] = boundary + 1, boundary + 2
    resequence(raw)
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"]
    with pytest.raises(ValueError, match="causal cycle"):
        audit_native_protocol(raw, identities)


def test_only_latest_local_mix_retirement_needs_an_edge_to_each_pure_read():
    pairs = 12
    raw, identities = ProtocolBuilder([(3, 1, ()), (1, 1, ())] * pairs, clusters=1).run()
    audit = NativeAudit(raw, identities)
    assert audit.audit()["protocol_verified"]
    reads = {audit.causal.key(point(e, "permit")) for e in raw["executions"] if not e["group_seq"]}
    retirements = [audit.causal.key(point(b["lanes"][0], "release_after")) for b in raw["blocks"]]
    assert sum(len(audit.causal.edges[r] & reads) for r in retirements) == pairs


@pytest.mark.parametrize("pure_mask", [1, 2], ids=["member", "accountant"])
def test_pure_read_waits_for_latest_local_retirement_across_equal_rank_blocks(pure_mask):
    raw, identities = ProtocolBuilder([(3, 2, ()), (pure_mask, 1, ())], clusters=1).run()
    # The second block retires later even when its wire row appears first.
    raw["blocks"].reverse()
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    last = raw["blocks"][0]
    execution = raw["executions"][-1]
    if pure_mask == 1:
        retired = last["lanes"][0]["release_after"]
    else:
        retired = last["owner"]["completion_after"]
    retired[0] = execution["permit"][0] + 1
    resequence(raw)
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"]
    with pytest.raises(ValueError, match="self execution crossed an earlier active MIX obligation"):
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


def test_current_protocol_rejects_removed_refill_publication_field():
    raw, identities = ProtocolBuilder([(1, 1, ())]).run()
    raw["executions"][0]["publication_mode"] = 0
    with pytest.raises(ValueError, match="unknown fields"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("fault", ("duplicate_ticket", "wrong_owner", "lane_contribution", "before_release"))
def test_mix_group_completion_preserves_unique_credit_and_physical_retirement(fault):
    raw, identities = ProtocolBuilder([(6, 2, ()), (2, 1, (0,))]).run()
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    first, last = [b["owner"] for b in raw["blocks"]]
    done = raw["task_events"][0]["completion"]
    if fault == "duplicate_ticket":
        last["completion_ticket"] = first["completion_ticket"]
    elif fault == "wrong_owner":
        done["worker_index"] += 1
    elif fault == "lane_contribution":
        raw["executions"][0]["contribution_worker"] = 1
    else:
        last["completion_before"], last["releases_observed"] = (last["releases_observed"], last["completion_before"])
        done["before"] = last["completion_before"].copy()
    with pytest.raises(ValueError):
        audit_native_protocol(raw, identities)


def _counter_capture(*, mask=1, separate=False, successor=False, sync=False, async_mask=None):
    shapes: list[tuple[int, int, tuple[int, ...]]] = [(mask, 2, ())]
    if successor:
        shapes.append((1, 1, (0,)))
    builder = ProtocolBuilder(shapes, sync_tasks=(0,) if sync else ())
    raw, identities = builder.run()
    threshold = 0
    assert len(raw["tasks"]) == len(raw["task_controls"]) == len(raw["task_events"])
    for task, control, actors in zip(raw["tasks"], raw["task_controls"], raw["task_events"]):
        task_async_mask = task["active_mask"] if async_mask is None else task["active_mask"] & async_mask
        threshold += task["block_num"] * (bin(task["active_mask"]).count("1") + bin(task_async_mask).count("1"))
        control.update(completion_backend=1, counter_index=0, counter_threshold=threshold, async_mask=task_async_mask)
        done = actors["completion"]
        done["flags"] = task["flags"] | COUNTER_ACCOUNTING | ((1 << 31) if separate else 0)
        when = builder.clock + 100 + 4 * task["task_ref"]
        done.update(before=[when, 1], after=[when + 1, 1])
    for control in raw["task_controls"]:
        control["counter_limit"] = threshold
    raw["completion_counters"] = [dict(index=0, value=threshold, valid=1, limit=threshold)]
    for block in raw["blocks"]:
        owner = block["owner"]
        mixed = bool(raw["tasks"][owner["task_ref"]]["flags"] & MIX)
        may_defer = bool(raw["task_controls"][owner["task_ref"]]["async_mask"])
        owner.update(
            completion_ticket=(1 << 64) - 1,
            completion_credits=bin(owner["active_mask"]).count("1")
            if mixed
            else (1 if separate or not may_defer else 2),
        )
    for execution in raw["executions"]:
        task = raw["tasks"][execution["task_ref"]]
        mixed = bool(task["flags"] & MIX)
        may_defer = bool(raw["task_controls"][execution["task_ref"]]["async_mask"] & (1 << execution["subtask_slot"]))
        if not mixed:
            execution.update(
                contribution_ticket=(1 << 64) - 1, contribution_credits=1 if separate or not may_defer else 2
            )
        if may_defer and (mixed or separate):
            when = execution["completion_consumed"][0] + 1
            raw["condition_contributions"].append(
                dict(
                    index=execution["index"],
                    run_epoch=raw["run_epoch"],
                    task_ref=execution["task_ref"],
                    worker_index=execution["worker_index"],
                    block_idx=execution["block_idx"],
                    subtask_slot=execution["subtask_slot"],
                    source=1 if separate else 0,
                    credits=1,
                    before=[when, 1],
                    after=[when + 1, 1],
                )
            )
    resequence(raw)
    return raw, identities


@pytest.mark.parametrize(
    ("mask", "separate", "sync"),
    [(1, False, False), (1, True, False), (1, False, True), (3, False, False), (6, True, True)],
)
def test_counter_credits_preserve_physical_and_condition_causality_without_return_tickets(mask, separate, sync):
    raw, identities = _counter_capture(mask=mask, separate=separate, sync=sync)
    assert audit_native_protocol(raw, identities)["protocol_verified"]


def test_counter_fixture_accepts_python39_zip_signature(monkeypatch):
    standard_zip = zip

    def zip_without_keywords(*iterables):
        return standard_zip(*iterables)

    monkeypatch.setitem(globals(), "zip", zip_without_keywords)
    raw, identities = _counter_capture(successor=True)
    assert audit_native_protocol(raw, identities)["protocol_verified"]


def test_counter_successor_does_not_wait_for_deferred_gm_accounting():
    raw, identities = _counter_capture(successor=True)
    assert raw["executions"][-1]["entry"][0] < raw["task_events"][0]["completion"]["before"][0]
    assert audit_native_protocol(raw, identities)["protocol_verified"]


@pytest.mark.parametrize(("mask", "async_mask"), [(1, 0), (3, 0), (7, 6)])
def test_counter_sync_lanes_need_only_physical_credits(mask, async_mask):
    raw, identities = _counter_capture(mask=mask, async_mask=async_mask)
    assert raw["completion_counters"][0]["limit"] == 2 * (bin(mask).count("1") + bin(async_mask).count("1"))
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    raw["task_controls"][0]["async_mask"] ^= 1
    with pytest.raises(ValueError, match="counter thresholds"):
        audit_native_protocol(raw, identities)


def test_counter_publication_can_be_observed_before_the_publishers_after_sample():
    raw, identities = _counter_capture(successor=True)
    publisher, successor = raw["executions"][1:]
    assert publisher["worker_index"] != successor["worker_index"]
    publisher["contribution_after"][0] = successor["entry"][0] + 1
    publisher["trace_published"][0] = successor["entry"][0] + 2
    resequence(raw)
    assert audit_native_protocol(raw, identities)["protocol_verified"]


@pytest.mark.parametrize(
    "fault",
    [
        "credits",
        "ticket",
        "missing_condition",
        "late_condition",
        "incomplete",
        "ancestor",
        "threshold",
        "accounting_alias",
    ],
)
def test_counter_capture_rejects_missing_or_fabricated_completion_proof(fault):
    raw, identities = _counter_capture(separate=True, successor=True)
    if fault == "credits":
        raw["executions"][0]["contribution_credits"] = 2
    elif fault == "ticket":
        raw["executions"][0]["contribution_ticket"] = 0
    elif fault == "missing_condition":
        raw["condition_contributions"].pop(0)
    elif fault == "late_condition":
        condition = raw["condition_contributions"][0]
        execution = raw["executions"][-1]
        condition["worker_index"] = execution["worker_index"]
        condition["before"][0] = execution["trace_published"][0] + 1
        condition["after"][0] = execution["trace_published"][0] + 2
    elif fault == "incomplete":
        raw["completion_counters"][0]["value"] -= 1
    elif fault == "ancestor":
        raw["tasks"][1]["dependencies"] = []
    elif fault == "threshold":
        raw["task_controls"][0]["counter_threshold"] += 2
    else:
        raw["task_events"][0]["completion"]["flags"] |= 1 << 30
    resequence(raw)
    with pytest.raises(ValueError):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("mask", (1, 3), ids=("ordinary", "mix"))
def test_counter_check_cannot_contribute_before_its_kernel_returns(mask):
    raw, identities = _counter_capture(mask=mask, separate=True)
    condition = raw["condition_contributions"][0]
    execution = raw["executions"][condition["index"]]
    condition["before"][0] = execution["entry"][0] + 1
    condition["after"][0] = execution["returned"][0] - 1
    resequence(raw)
    # Exercise the condition contract directly. The full audit also rejects
    # same-worker scheduler events inside a kernel for an independent reason.
    audit = NativeAudit(raw, identities)
    with pytest.raises(ValueError, match="counter condition precedes its execution return"):
        audit.counter_condition(condition["index"], ("ready", 0), raw["tasks"][0])
