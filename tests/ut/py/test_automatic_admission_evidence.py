#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reject stale mailbox capacity and join native epochs to exact sampled leases."""

from types import SimpleNamespace

import pytest

from tests.st import automatic_admission_support as support


@pytest.mark.parametrize("state,generation", [(support._TASK_DONE, 3), (support._IDLE, 4)])
def test_completion_capacity_survives_parent_entry_release(state, generation):
    identity = (1, 7, 0, 9, 11, 0, 0, 1)
    row = (identity, state, support._TASK_ACCEPTED, generation, 2)
    assert support.completed_capacity(row, 7)
    assert not support.completed_capacity(row, 8)


@pytest.mark.parametrize(
    "state,accepted,generation,capacity",
    [
        (support._IDLE, support._TASK_ACCEPTED, 5, 2),
        (support._IDLE, 0, 6, 2),
        (support._IDLE, support._TASK_ACCEPTED, 0, 2),
        (support._TASK_LAUNCHED, support._TASK_ACCEPTED, 5, 2),
        (support._TASK_DONE, support._TASK_ACCEPTED, 5, 1),
    ],
)
def test_completion_capacity_rejects_unproven_or_reused_entries(state, accepted, generation, capacity):
    identity = (1, 7, 0, 9, 11, 0, 0, 1)
    assert not support.completed_capacity((identity, state, accepted, generation, capacity), 7)


def test_native_trace_preserves_retry_epochs_and_distinct_slot_generations(monkeypatch, tmp_path):
    monkeypatch.setattr(support.evidence, "_child_log_directory", lambda: str(tmp_path))
    trace = support.NativeRunTrace(SimpleNamespace(_chip_pids=[]))
    for dispatch, generation, epoch in [(11, 9, 101), (11, 9, 102), (12, 10, 103)]:
        span = SimpleNamespace(
            inv=epoch,
            attrs=f"dispatch_id={dispatch} slot_id=0 generation={generation} run_epoch={epoch}",
        )
        trace._note_identity(23, span)
    assert trace.epochs == {(11, 0, 9): {(23, 101), (23, 102)}, (12, 0, 10): {(23, 103)}}
    assert trace.run_of_invocation == {(23, 101): (11, 0), (23, 102): (11, 0), (23, 103): (12, 0)}
    assert (11, 0, 10) not in trace.epochs
    with pytest.raises(AssertionError):
        trace._note_identity(23, SimpleNamespace(inv=104, attrs="dispatch_id=13 slot_id=0 generation=11 run_epoch=0"))
