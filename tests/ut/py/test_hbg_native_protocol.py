# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native protocol parsing and local sequence integrity are not a causal proof."""

import copy

import pytest

from simpler_setup.tools.hbg.native_protocol import audit_protocol_sequences, read_native_protocol


def capture(epoch=0):
    return {
        "schema": "hbg-native-protocol-v20",
        "producer": "aicore",
        "run_epoch": epoch,
        "task_count": 1,
        "worker_count": 3,
        "trace_count": 1,
        "member_count": 0,
        "cooperative_task_count": 0,
        "cluster_count": 1,
        "mix_task_count": 0,
        "sync_mix_task_count": 0,
        "expected_task_count": 1,
        "inline_completed_count": 0,
        "resolved_task_count": 1,
        "close_seal_stride": 64,
        "active_worker_count": 3,
        "scheduler_count": 3,
        "scheduler_error": 0,
        "error_claimed": 0,
        "commit_tail": 0,
        "commit_lock": 0,
        "first_error": dict.fromkeys(
            (
                "task_ref core_id core_type site run_epoch order_key block_idx "
                "group_seq worker_index cluster_lane identity_fields"
            ).split(),
            0,
        ),
        "tasks": [
            dict(
                task_ref=0, flags=33, active_mask=2, block_num=1, trace_base=0, kernel_ids=[65535] * 3, dependencies=[]
            )
        ],
        "workers": [
            dict(
                index=index,
                worker_index=index,
                context_worker_index=index,
                run_epoch=epoch,
                next_sequence=2 if index == 0 else 0,
                error_flags=0,
                finished=1,
                core_id=index,
                core_type=int(index != 0),
                cluster_index=0,
                cluster_workers=[0, 1, 2],
            )
            for index in range(3)
        ],
        "admission_log": [],
        "admissions": [],
        "task_controls": [
            dict(
                task_ref=0,
                state=2,
                canonical_total=1,
                pending_compute=0,
                completion_debt=0,
                next_block=0,
                completed_subtasks=0,
                admission_state=0,
                claim_tickets=0,
                claim_seal=0,
                parked_count=0,
            )
        ],
        "task_events": [
            dict(
                index=0,
                admission=None,
                completion=dict(
                    before=[0, 1],
                    after=[5, 2],
                    run_epoch=epoch,
                    task_ref=0,
                    order_key=(1 << 64) - 1,
                    worker_index=0,
                    flags=33,
                ),
            )
        ],
        "members": [],
        "blocks": [],
        "executions": [],
    }


@pytest.mark.parametrize("epoch", (0, (1 << 64) - 1))
def test_literal_epochs_and_cycle_zero_survive_without_claiming_causality(epoch):
    raw = capture(epoch)
    before = copy.deepcopy(raw)
    assert read_native_protocol(raw) == raw
    result = audit_protocol_sequences(raw)
    assert result == {"sequence_integrity_verified": True, "event_count": 2, "worker_count": 3}
    assert "protocol_verified" not in result
    assert raw == before


@pytest.mark.parametrize("stride", (None, 0, 8, 63, 65, 128, True, 64.0, "64", -1, 1 << 64))
def test_compact_schema_requires_64_byte_seals_and_omits_unused_execution_counter(stride):
    raw = capture()
    raw.update(schema="hbg-native-protocol-v20", close_seal_stride=64)
    raw["task_controls"][0].update(claim_tickets=0, claim_seal=0)
    assert read_native_protocol(raw) == raw
    raw["executed_task_count"] = 0
    with pytest.raises(ValueError, match="unknown fields"):
        read_native_protocol(raw)
    del raw["executed_task_count"]
    if stride is None:
        del raw["close_seal_stride"]
    else:
        raw["close_seal_stride"] = stride
    with pytest.raises(ValueError, match="close_seal_stride|missing or unknown fields"):
        read_native_protocol(raw)


@pytest.mark.parametrize("field", tuple(capture()))
def test_every_root_field_is_required(field):
    raw = capture()
    del raw[field]
    with pytest.raises(ValueError, match="protocol"):
        read_native_protocol(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "future-v2"),
        ("schema", "hbg-native-protocol-v12"),
        ("schema", "hbg-native-protocol-v13"),
        ("producer", "aicpu"),
        ("worker_count", True),
        ("run_epoch", -1),
        ("trace_count", 1 << 64),
        ("workers", {}),
        ("task_count", 2),
        ("worker_count", 4),
        ("cooperative_task_count", 1),
    ],
)
def test_bad_envelope_or_dense_table_is_rejected(field, value):
    raw = capture()
    raw[field] = value
    with pytest.raises(ValueError, match="protocol"):
        read_native_protocol(raw)


@pytest.mark.parametrize("mutation", ("duplicate", "outside", "unknown", "missing", "bad_stamp", "bool_stamp"))
def test_sparse_records_and_actor_schema_are_exact(mutation):
    raw = capture()
    row = raw["task_events"][0]
    if mutation == "duplicate":
        raw["task_events"].append(copy.deepcopy(row))
    elif mutation == "outside":
        row["index"] = 1
    elif mutation == "unknown":
        row["completion"]["guessed"] = 7
    elif mutation == "missing":
        del row["completion"]["after"]
    elif mutation == "bad_stamp":
        row["completion"]["after"] = [5]
    else:
        row["completion"]["after"] = [5, True]
    with pytest.raises(ValueError, match="protocol"):
        read_native_protocol(raw)


@pytest.mark.parametrize(
    "mutation",
    (
        "hole",
        "duplicate",
        "truncated",
        "backwards",
        "partial",
        "foreign",
        "bad_worker",
        "error",
        "unfinished",
        "scheduler_error",
    ),
)
def test_sequence_check_refuses_partial_stale_or_lost_evidence(mutation):
    raw = capture()
    actor = raw["task_events"][0]["completion"]
    if mutation == "hole":
        actor["after"][1] = 3
        raw["workers"][0]["next_sequence"] = 3
    elif mutation == "duplicate":
        actor["after"][1] = 1
    elif mutation == "truncated":
        raw["workers"][0]["next_sequence"] = 3
    elif mutation == "backwards":
        actor["before"][0] = 6
    elif mutation == "partial":
        actor["after"][1] = 0
    elif mutation == "foreign":
        actor["run_epoch"] = 1
    elif mutation == "bad_worker":
        actor["worker_index"] = 3
    elif mutation == "error":
        raw["workers"][0]["error_flags"] = 1
    elif mutation == "unfinished":
        raw["workers"][2]["finished"] = 0
    else:
        raw["scheduler_error"] = 1
    with pytest.raises(ValueError, match="protocol"):
        audit_protocol_sequences(raw)


def test_parser_preserves_error_and_incomplete_evidence_for_diagnostics():
    raw = capture()
    raw["workers"][0]["error_flags"] = 1
    raw["task_events"][0]["completion"]["after"] = [0, 0]
    raw["task_controls"][0].update(state=-1, canonical_total=0, pending_compute=65535, completion_debt=(1 << 32) - 1)
    assert read_native_protocol(raw) == raw
    with pytest.raises(ValueError, match="protocol"):
        audit_protocol_sequences(raw)


@pytest.mark.parametrize("version", (*range(1, 20), 22, 23))
def test_noncurrent_protocol_versions_are_rejected(version):
    raw = capture()
    raw["schema"] = f"hbg-native-protocol-v{version}"
    with pytest.raises(ValueError, match="unsupported schema"):
        read_native_protocol(raw)


@pytest.mark.parametrize(
    "field", ("claim_tickets", "claim_seal", "parked_count", "canonical_total", "pending_compute", "completion_debt")
)
@pytest.mark.parametrize("value", (None, True, -1, 1 << 64, "0"))
def test_current_control_words_are_mandatory_uint64(field, value):
    raw = capture()
    if value is None:
        del raw["task_controls"][0][field]
    else:
        raw["task_controls"][0][field] = value
    with pytest.raises(ValueError, match="fields|integer"):
        read_native_protocol(raw)
