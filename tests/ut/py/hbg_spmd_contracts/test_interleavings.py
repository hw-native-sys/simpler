# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Complete small-state exploration and unsafe-policy counterexamples."""

import itertools

import pytest

from .explorer import Workload, explore


def cohort(name, members=(0, 1), dependencies=(), mapping=None):
    mapping = members if mapping is None else mapping
    return Workload(name, members, len(mapping), dependencies, True, mapping)


def test_both_global_admission_orders_allow_overlap_without_deadlock():
    result = explore((cohort("A"), cohort("B")), 2)
    assert result.deadlocks == 0 and result.terminals == 2
    assert result.overlap


def test_real_dependency_forbids_overlap():
    result = explore((cohort("A"), cohort("B", dependencies=(0,))), 2)
    assert result.deadlocks == 0 and result.terminals == 1
    assert not result.overlap


def test_b_runs_while_a_is_waiting_for_a_real_predecessor():
    result = explore((cohort("A", dependencies=(2,)), cohort("B"), Workload("P", (2,), 1)), 3)
    assert result.deadlocks == 0 and result.b_before_unready_a


def test_nonparticipant_does_not_wait_for_global_completion():
    result = explore((cohort("A"), Workload("B", (2,), 1)), 3, fixed_admission=True)
    assert result.deadlocks == 0 and result.overlap


@pytest.mark.parametrize("blocks", [1, 2, 3, 4, 5, 6])
def test_dynamic_waves_exhaustively_preserve_progress(blocks):
    result = explore((Workload("A", (0, 1), blocks), cohort("B")), 2, fixed_admission=True)
    assert result.deadlocks == 0 and result.terminals > 0
    assert result.overlap


@pytest.mark.parametrize("members", list(itertools.permutations(((0, 1), (1, 2), (0, 2)))))
def test_every_order_of_partially_overlapping_cohorts_progresses(members):
    tasks = tuple(cohort(name, participants) for name, participants in zip(("A", "B", "C"), members))
    result = explore(tasks, 3)
    assert result.deadlocks == 0 and result.terminals == 6


def test_unsafe_per_core_task_selection_has_a_deadlock_witness():
    result = explore((cohort("A"), cohort("B")), 2, ordered=False)
    assert result.deadlocks > 0
    assert any("enter:A" in item for item in result.deadlock_witness)
    assert any("enter:B" in item for item in result.deadlock_witness)


def test_two_cores_cannot_host_three_mutually_waiting_blocks():
    result = explore((cohort("A", mapping=(0, 0, 1)),), 2)
    assert result.deadlocks > 0 and result.terminals == 0
    assert result.deadlock_witness


def test_leaving_after_one_block_can_abandon_unclaimed_work():
    tasks = (Workload("A", (0, 1), 6), cohort("B"))
    result = explore(tasks, 2, early_close=True, fixed_admission=True)
    assert result.deadlocks > 0 and result.deadlock_witness


def test_state_budget_exhaustion_is_not_reported_as_a_pass():
    with pytest.raises(RuntimeError, match="incomplete exploration"):
        explore((cohort("A"), cohort("B")), 2, limit=3)


def test_mixed_channels_can_deadlock_even_when_each_local_channel_is_fifo():
    entered_a = {"C0", "V00", "C1"}
    members_a = {"C0", "V00", "C1", "V10"}
    entered_b = {"V10", "V01", "V11"}
    members_b = {"V00", "V01", "V10", "V11"}
    running = {w: "A" for w in entered_a} | {w: "B" for w in entered_b}
    assert len(running) == 6
    assert members_a - entered_a == {"V10"}
    assert members_b - entered_b == {"V00"}
    assert running["V10"] == "B" and running["V00"] == "A"
