# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Synthetic async evidence checks do not qualify a native producer or hardware."""

import copy
from typing import Any

import pytest
from _native_protocol_builder import ProtocolBuilder
from test_hbg_native_causality import _add_claim_seal_certificates, _isolated_close_seals, resequence
from test_hbg_native_graph import native_capture

from simpler_setup.tools.hbg.native_async import summarize_async_protocol
from simpler_setup.tools.hbg.native_causality import audit_native_protocol
from simpler_setup.tools.hbg.native_protocol import (
    STAMPS,
    _event_records,
    audit_protocol_sequences,
    read_native_protocol,
)


def add_async(raw, identities, refs=(0,), *, empty=False, ready_first=True):
    if raw["schema"] != "hbg-native-protocol-v9":
        _add_claim_seal_certificates(raw, identities)
        _isolated_close_seals(raw)
        raw["schema"] = "hbg-native-protocol-v8"
    for record, kind, _ in _event_records(raw):
        for field in STAMPS[kind].split():
            record[field][0] *= 10
        if kind == "execution":
            record["publication_before_cycles"] *= 10
            record["publication_after_cycles"] *= 10
    for block in raw["blocks"]:
        claim = block["claim"]
        if claim is not None and claim["published_before"][1]:
            claim["published_before"][0] += 1
            claim["published_after"][0] += 2
    raw["schema"] = "hbg-native-protocol-v11" if "node_map" in raw else "hbg-native-protocol-v10"
    asynchronous: dict[str, Any] = dict(group_count=0, finish_count=0, groups=[], releases=[], finishes=[])
    raw["async"] = asynchronous

    def stamp(worker, cycles):
        return dict(run_epoch=raw["run_epoch"], worker_index=worker, at=[cycles, 1])

    for ref in refs:
        executions = [row for row in raw["executions"] if row["task_ref"] == ref]
        done = next(row["completion"] for row in raw["task_events"] if row["index"] == ref)
        original_done = done["before"][0]
        task_end = max(row["trace_published"][0] for row in executions)
        done["worker_index"] = (done["worker_index"] + 1) % raw["worker_count"]
        done["before"] = [max(original_done, task_end) + 20, 1]
        done["after"] = [done["before"][0] + 10, 1]
        indices = []
        for execution in executions:
            index = len(asynchronous["groups"])
            indices.append(index)
            end = execution["trace_published"][0]
            previous = "visible" if execution["flags"] & 2 else "entry"
            asynchronous["groups"].append(
                dict(
                    index=index,
                    run_epoch=raw["run_epoch"],
                    task_ref=ref,
                    block_idx=execution["block_idx"],
                    subtask_slot=execution["subtask_slot"],
                    execution_index=execution["index"],
                    dispatch_generation=execution["generation"],
                    lease_word=8 if empty else (7 << 3) | 3,
                    capacity=4,
                    registered_count=0 if empty else 1,
                    conditions=[] if empty else [dict(index=0, type=0, binding_index=0, owner_generation=None)],
                    state="EMPTY" if empty else "READY",
                    sealed=stamp(execution["worker_index"], execution[previous][0] + 1),
                    terminal=None if empty else stamp(done["worker_index"], end + (2 if ready_first else 8)),
                    observations=0 if empty else 1,
                    pending_count=0 if empty else 2,
                    slice_count=0 if empty else 3,
                    active_cycles=0 if empty else 17,
                    failure_code=0,
                )
            )
        for block in sorted({row["block_idx"] for row in executions}):
            block_executions = [row for row in executions if row["block_idx"] == block]
            end = max(row["trace_published"][0] for row in block_executions)
            owner = next(
                (
                    row["owner"]
                    for row in raw["blocks"]
                    if row["owner"]["task_ref"] == ref and row["owner"]["block_idx"] == block
                ),
                None,
            )
            worker = owner["worker_index"] if owner else block_executions[0]["worker_index"]
            if owner:
                end = max(end, owner["completion_after"][0])
            asynchronous["releases"].append(
                dict(
                    task_ref=ref,
                    block_idx=block,
                    run_epoch=raw["run_epoch"],
                    publication=dict(
                        run_epoch=raw["run_epoch"], worker_index=worker, before=[end + 4, 1], after=[end + 5, 1]
                    ),
                )
            )
        asynchronous["finishes"].append(
            dict(
                index=len(asynchronous["finishes"]),
                run_epoch=raw["run_epoch"],
                task_ref=ref,
                lease_word=(3 << 3) | 3,
                state="DRAINED",
                group_indices=indices,
                drained=stamp(done["worker_index"], done["after"][0] + 10000),
                waiter_count=2,
                pending_count=0,
                slice_count=2,
                active_cycles=11,
                failure_code=0,
            )
        )
    asynchronous["group_count"] = len(asynchronous["groups"])
    asynchronous["finish_count"] = len(asynchronous["finishes"])
    resequence(raw)
    return raw, identities


def async_capture(*, mask=2, blocks=1, empty=False, ready_first=True):
    raw, identities = ProtocolBuilder([(mask, blocks, ())], resource_scoped=True).run()
    return add_async(raw, identities, empty=empty, ready_first=ready_first)


@pytest.mark.parametrize("mask,blocks", ((2, 1), (1, 2), (3, 1), (6, 2), (7, 2)))
@pytest.mark.parametrize("empty", (False, True))
@pytest.mark.parametrize("ready_first", (False, True))
def test_async_join_handles_empty_mix_spmd_and_either_ready_release_order(mask, blocks, empty, ready_first):
    raw, identities = async_capture(mask=mask, blocks=blocks, empty=empty, ready_first=ready_first)
    before = copy.deepcopy(raw)
    result = audit_native_protocol(raw, identities)
    assert result["protocol_verified"] and result["async_tasks"] == 1
    assert len(raw["async"]["releases"]) == blocks
    assert raw == before


def test_async_without_binding_side_table_keeps_completion_causality():
    raw, identities = async_capture()
    for group in raw["async"]["groups"]:
        for condition in group["conditions"]:
            condition["binding_index"] = None
            condition["owner_generation"] = None
    read_native_protocol(raw)
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    raw["async"]["groups"][0]["terminal"] = None
    with pytest.raises(ValueError):
        audit_native_protocol(raw, identities)


def test_async_graph_uses_existing_public_ids_and_consumer_can_start_before_drain():
    raw, identities = native_capture()
    add_async(raw, identities, refs=(2,))
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    summary = summarize_async_protocol(raw)
    assert summary is not None
    assert summary["groups"][0]["task_id"] == (1 << 62) | (1 << 32)
    assert summary["groups"][0]["pending_count"] == 2
    assert summary["groups"][0]["active_cycles"] == 17
    assert raw["executions"][-1]["entry"][0] < raw["async"]["finishes"][0]["drained"]["at"][0]


def test_empty_predicate_skip_never_invents_return_or_visible_events():
    builder = ProtocolBuilder([(2, 1, ())])
    builder.raw["tasks"][0]["flags"] |= 64
    execution = builder.execute(0, 0, 1, 1, skip=True)
    raw, identities = add_async(builder.raw, builder.identities, empty=True)
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    assert execution["returned"] == execution["visible"] == [0, 0]


def test_remote_done_can_precede_release_after_with_incomparable_worker_clocks():
    raw, identities = async_capture()
    release = raw["async"]["releases"][0]["publication"]
    done = raw["task_events"][0]["completion"]
    release["after"][0] = done["after"][0] + 500
    resequence(raw)
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    for record, kind, _ in _event_records(raw):
        offset = 100000 * (1 + record["worker_index"])
        for field in STAMPS[kind].split():
            if record[field][1]:
                record[field][0] += offset
        if kind == "execution":
            record["publication_before_cycles"] += offset
            record["publication_after_cycles"] += offset
    assert audit_native_protocol(raw, identities)["protocol_verified"]


@pytest.mark.parametrize("state", ("UNSEALED", "PENDING", "FAILED"))
def test_incomplete_and_failed_states_are_readable_but_not_success(state):
    raw, identities = async_capture()
    group = raw["async"]["groups"][0]
    group.update(state=state, terminal=None, failure_code=-7 if state == "FAILED" else 0)
    if state == "UNSEALED":
        group["sealed"] = None
    resequence(raw)
    assert read_native_protocol(raw) == raw
    summary = summarize_async_protocol(raw)
    assert summary is not None
    assert summary["group_states"] == {state: 1}
    with pytest.raises(ValueError, match="async"):
        audit_native_protocol(raw, identities)


def test_partial_release_keeps_before_sequence_without_claiming_complete_publication():
    raw, identities = async_capture()
    release = raw["async"]["releases"][0]["publication"]
    release["after"] = None
    resequence(raw)
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"]
    summary = summarize_async_protocol(raw)
    assert summary is not None
    assert summary["execution_releases_recorded"] == 0
    with pytest.raises(ValueError, match="incomplete block release"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("table,count", (("groups", "group_count"), ("finishes", "finish_count")))
def test_dense_immutable_tables_reject_truncation(table, count):
    raw, _ = async_capture()
    raw["async"][table].clear()
    assert raw["async"][count] == 1
    with pytest.raises(ValueError, match="table length"):
        read_native_protocol(raw)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_release",
        "duplicate_release",
        "wrong_dispatch",
        "wrong_execution",
        "wrong_block",
        "wrong_epoch",
        "missing_finish_groups",
        "duplicate_finish_groups",
        "failed_finish",
        "pending_finish_batch",
        "wrong_lease",
        "unsealed",
        "missing_terminal",
        "empty_with_conditions",
        "pending_exceeds_slices",
        "unknown_type",
        "unbound",
        "missing_release_point",
        "drained_before_done",
        "ready_after_done",
        "seal_before_visible",
        "seal_after_consumed",
        "release_before_trace",
        "duplicate_group",
    ),
)
def test_complete_sequences_do_not_hide_async_protocol_corruption(mutation):
    raw, identities = async_capture()
    group = raw["async"]["groups"][0]
    finish = raw["async"]["finishes"][0]
    release = raw["async"]["releases"][0]
    done = raw["task_events"][0]["completion"]
    execution = raw["executions"][0]
    edits = {
        "wrong_dispatch": (group, "dispatch_generation", 55),
        "wrong_execution": (group, "execution_index", 10),
        "wrong_block": (group, "block_idx", 1),
        "wrong_epoch": (group, "run_epoch", 9),
        "missing_finish_groups": (finish, "group_indices", []),
        "duplicate_finish_groups": (finish, "group_indices", [0, 0]),
        "failed_finish": (finish, "state", "FAILED"),
        "pending_finish_batch": (finish, "pending_count", 1),
        "wrong_lease": (group, "lease_word", 9),
        "unsealed": (group, "sealed", None),
        "missing_terminal": (group, "terminal", None),
        "empty_with_conditions": (group, "state", "EMPTY"),
        "pending_exceeds_slices": (group, "pending_count", 4),
        "unknown_type": (group["conditions"][0], "type", 9),
        "unbound": (group["conditions"][0], "binding_index", (1 << 64) - 1),
        "missing_release_point": (release, "publication", None),
        "drained_before_done": (finish["drained"]["at"], 0, done["before"][0] - 1),
        "ready_after_done": (group["terminal"]["at"], 0, done["after"][0] + 1),
        "seal_before_visible": (group["sealed"]["at"], 0, execution["entry"][0] - 1),
        "seal_after_consumed": (group["sealed"]["at"], 0, execution["trace_published"][0] + 1),
        "release_before_trace": (release["publication"]["before"], 0, execution["entry"][0] - 1),
    }
    if mutation == "missing_release":
        raw["async"]["releases"].clear()
    elif mutation == "duplicate_release":
        duplicate = copy.deepcopy(release)
        duplicate["publication"] = None
        raw["async"]["releases"].append(duplicate)
    elif mutation == "duplicate_group":
        duplicate = copy.deepcopy(group)
        duplicate.update(index=1, sealed=None, terminal=None)
        raw["async"]["groups"].append(duplicate)
        raw["async"]["group_count"] = 2
    else:
        target, key, value = edits[mutation]
        target[key] = value
    resequence(raw)
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"]
    with pytest.raises(ValueError, match="protocol"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("value", (0, True, -1, 1 << 64, "1"))
def test_allocation_generation_is_positive_or_explicitly_unavailable(value):
    raw, _ = async_capture()
    raw["async"]["groups"][0]["conditions"][0]["owner_generation"] = value
    with pytest.raises(ValueError, match="owner_generation"):
        read_native_protocol(raw)


@pytest.mark.parametrize("path", ("root", "async", "group", "condition", "point", "finish", "release", "interval"))
def test_new_schema_rejects_unknown_fields_at_every_level(path):
    raw, _ = async_capture()
    asynchronous = raw["async"]
    group, finish, release = asynchronous["groups"][0], asynchronous["finishes"][0], asynchronous["releases"][0]
    target = {
        "root": raw,
        "async": asynchronous,
        "group": group,
        "condition": group["conditions"][0],
        "point": group["sealed"],
        "finish": finish,
        "release": release,
        "interval": release["publication"],
    }[path]
    target["invented"] = 0
    with pytest.raises(ValueError, match="unknown fields"):
        read_native_protocol(raw)


@pytest.mark.parametrize("version", range(1, 10))
def test_legacy_schema_never_accepts_async_extension(version):
    raw, _ = async_capture()
    raw["schema"] = f"hbg-native-protocol-v{version}"
    with pytest.raises(ValueError, match="unknown fields"):
        read_native_protocol(raw)


def test_ordinary_task_worker_assumption_is_preserved_without_async_coverage():
    raw, identities = ProtocolBuilder([(2, 1, ())]).run()
    _add_claim_seal_certificates(raw, identities)
    _isolated_close_seals(raw)
    raw.update(schema="hbg-native-protocol-v10")
    raw["async"] = dict(group_count=0, finish_count=0, groups=[], releases=[], finishes=[])
    raw["task_events"][0]["completion"]["worker_index"] = 2
    resequence(raw)
    with pytest.raises(ValueError, match="ordinary TaskDone wrong worker"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("version", range(1, 10))
def test_async_field_is_rejected_on_an_otherwise_valid_legacy_schema(version):
    if version == 9:
        raw, _ = native_capture()
    else:
        raw, _ = ProtocolBuilder([(2, 1, ())]).run()
        raw["schema"] = f"hbg-native-protocol-v{version}"
        control = raw["task_controls"][0]
        if version >= 2:
            control["claim_tickets"] = 0
        if version >= 5:
            control["admission_claim_lock"] = 0
        if version >= 6:
            control["claim_seal"] = 0
        if version >= 7:
            raw["close_seal_stride"] = 128
    assert read_native_protocol(raw) == raw
    raw["async"] = dict(group_count=0, finish_count=0, groups=[], releases=[], finishes=[])
    with pytest.raises(ValueError, match="unknown fields"):
        read_native_protocol(raw)


@pytest.mark.parametrize("table", ("groups", "finishes"))
def test_missing_required_async_row_fields_are_rejected(table):
    original, _ = async_capture()
    for field in original["async"][table][0]:
        raw = copy.deepcopy(original)
        del raw["async"][table][0][field]
        with pytest.raises(ValueError, match="protocol"):
            read_native_protocol(raw)


@pytest.mark.parametrize("value", (None, True, -1, 1 << 64, "1"))
@pytest.mark.parametrize("field", ("capacity", "registered_count", "observations", "pending_count", "active_cycles"))
def test_async_counters_are_typed_wire_integers(value, field):
    raw, _ = async_capture()
    raw["async"]["groups"][0][field] = value
    with pytest.raises(ValueError, match="invalid integer"):
        read_native_protocol(raw)


@pytest.mark.parametrize("field", ("sealed", "terminal"))
def test_dropped_recorded_async_point_is_a_sequence_hole(field):
    raw, _ = async_capture()
    raw["async"]["groups"][0][field] = None
    with pytest.raises(ValueError, match="lost or unexpected events"):
        audit_protocol_sequences(raw)


def test_pending_slice_aggregation_does_not_consume_unrecorded_sequence_ids():
    raw, identities = async_capture()
    group = raw["async"]["groups"][0]
    before = audit_protocol_sequences(raw)["event_count"]
    group.update(pending_count=100, slice_count=101, active_cycles=91)
    assert audit_protocol_sequences(raw)["event_count"] == before
    assert audit_native_protocol(raw, identities)["protocol_verified"]


@pytest.mark.parametrize(
    "mutation", ("zero_stamp", "partial_stamp", "duplicate_stamp", "foreign_epoch", "foreign_worker")
)
def test_async_points_never_fabricate_unavailable_or_foreign_evidence(mutation):
    raw, _ = async_capture()
    sealed = raw["async"]["groups"][0]["sealed"]
    if mutation == "zero_stamp":
        sealed["at"] = [0, 0]
    elif mutation == "partial_stamp":
        sealed["at"][1] = 0
    elif mutation == "duplicate_stamp":
        sealed["at"][1] = raw["executions"][0]["entry"][1]
    elif mutation == "foreign_epoch":
        sealed["run_epoch"] = 1
    else:
        sealed["worker_index"] = raw["worker_count"]
    with pytest.raises(ValueError, match="protocol"):
        audit_protocol_sequences(raw)


def test_different_conditions_may_share_a_binding_within_execution_scope():
    raw, identities = async_capture()
    group = raw["async"]["groups"][0]
    group["conditions"].append(dict(index=1, type=1, binding_index=0, owner_generation=None))
    group.update(registered_count=2, observations=2)
    assert audit_native_protocol(raw, identities)["protocol_verified"]


def test_lease_generation_is_independent_of_dispatch_and_positive_resource_generation():
    raw, identities = async_capture()
    group = raw["async"]["groups"][0]
    assert group["lease_word"] >> 3 != group["dispatch_generation"]
    group["conditions"][0]["owner_generation"] = 27
    assert audit_native_protocol(raw, identities)["protocol_verified"]


@pytest.mark.parametrize("mask", (2, 7))
def test_release_publisher_must_own_the_actual_logical_block(mask):
    raw, identities = async_capture(mask=mask)
    publication = raw["async"]["releases"][0]["publication"]
    publication["worker_index"] = (publication["worker_index"] + 1) % raw["worker_count"]
    resequence(raw)
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"]
    with pytest.raises(ValueError, match="release has another owner"):
        audit_native_protocol(raw, identities)


def test_skipped_kernel_cannot_claim_registered_ready_conditions():
    builder = ProtocolBuilder([(2, 1, ())])
    builder.raw["tasks"][0]["flags"] |= 64
    builder.execute(0, 0, 1, 1, skip=True)
    raw, identities = add_async(builder.raw, builder.identities)
    with pytest.raises(ValueError, match="skipped kernel has registered conditions"):
        audit_native_protocol(raw, identities)


def test_new_schema_with_no_async_tasks_keeps_ordinary_success_contract():
    raw, identities = ProtocolBuilder([(2, 1, ())]).run()
    add_async(raw, identities, refs=())
    assert audit_native_protocol(raw, identities)["protocol_verified"]
    summary = summarize_async_protocol(raw)
    assert summary is not None
    assert summary["groups"] == summary["finishes"] == []


def test_finish_pending_batch_cannot_exceed_its_native_capacity():
    raw, _ = async_capture()
    finish = raw["async"]["finishes"][0]
    finish.update(state="PENDING", pending_count=8, drained=None)
    assert read_native_protocol(raw) == raw
    finish["pending_count"] = 9
    with pytest.raises(ValueError, match=r"async.finishes\[0\].pending_count out of range: 9"):
        read_native_protocol(raw)
