# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Deterministic positive and negative schedules for the SPMD contracts."""

from dataclasses import replace

import pytest

from .helpers import close_task_members, execute_block, make_protocol, run_task
from .types import MAX_EPOCH, MAX_GENERATION, ContractViolation, Task


def test_ready_b_can_commit_before_unready_a_without_task_ref_order():
    model = make_protocol(Task(0, 1, 1), Task(1, 2, 1, deps=(0,)), Task(2, 2, 2))
    with pytest.raises(ContractViolation, match="DATA_NOT_READY"):
        model.prepare_admission(1)
    model.admit(2)
    run_task(model, 2)
    run_task(model, 0)
    run_task(model, 1)
    assert model.admissions == [2, 0, 1]
    assert model.can_exit()


def test_admission_preparation_does_not_publish_a_hole_or_hold_unrelated_b():
    model = make_protocol(Task(0, 2, 1), Task(1, 2, 2))
    model.prepare_admission(0)
    assert not model.admissions and model.oldest(0) is None
    model.admit(1)
    model.commit_admission(0)
    assert model.admissions == [1, 0]


def test_duplicate_admission_and_capacity_are_not_silent():
    model = make_protocol(Task(0, 2, 1), Task(1, 2, 2), max_admissions=1)
    model.admit(0)
    with pytest.raises(ContractViolation, match="DUPLICATE_ADMISSION"):
        model.admit(0)
    with pytest.raises(ContractViolation, match="ADMISSION_CAPACITY"):
        model.admit(1)
    assert model.admissions == [0]


@pytest.mark.parametrize("source", ["ordinary", "mix", "direct_refill"])
def test_all_execution_sources_obey_the_same_local_obligation(source):
    model = make_protocol(Task(0, 2, 1), Task(1, 2, 1))
    model.admit(0)
    model.admit(1)
    with pytest.raises(ContractViolation, match="ORDER"):
        model.claim(1, 0)
    a0, a1 = model.claim(0, 0), model.claim(0, 3)
    execute_block(model, a0)
    model.close(0, 0)
    b0 = model.claim(1, 0)
    model.publish(b0)
    model.start(b0, 0, source)
    assert not model.state(0).done
    assert a1 is not None and a1.block == 1 and model.running[0] == b0


def test_true_dependency_cannot_be_released_by_first_block():
    model = make_protocol(Task(0, 2, 1), Task(1, 1, 2, deps=(0,)))
    model.admit(0)
    a0, a1 = model.claim(0, 0), model.claim(0, 3)
    execute_block(model, a0)
    assert model.state(0).completed == 1 and model.state(0).releases == 0
    with pytest.raises(ContractViolation, match="DATA_NOT_READY"):
        model.admit(1)
    execute_block(model, a1)
    close_task_members(model, 0)
    run_task(model, 1)
    assert model.state(0).releases == 1


def test_nonparticipant_vector_can_run_while_cube_task_is_incomplete():
    model = make_protocol(Task(0, 2, 1), Task(1, 2, 2))
    model.admit(0)
    model.admit(1)
    b = model.claim(1, 1)
    execute_block(model, b)
    assert model.state(0).completed == 0
    assert model.oldest(1) == 1 and model.oldest(0) == 0


def test_partial_member_sets_use_projection_not_global_minimum_frontier():
    model = make_protocol(Task(0, 1, 2), Task(1, 1, 2))
    model.admit(0, (1, 2))
    model.admit(1, (2, 4))
    b = model.claim(1, 4)
    execute_block(model, b)
    with pytest.raises(ContractViolation, match="ORDER"):
        model.claim(1, 2)


def test_done_one_block_is_not_done_with_a_still_open_claim_range():
    model = make_protocol(Task(0, 6, 1), Task(1, 2, 1))
    model.admit(0)
    model.admit(1)
    lease = model.claim(0, 0)
    execute_block(model, lease)
    with pytest.raises(ContractViolation, match="UNCLAIMED_BLOCKS"):
        model.close(0, 0)
    with pytest.raises(ContractViolation, match="ORDER"):
        model.claim(1, 0)


def test_claimed_but_unpublished_work_is_not_local_done():
    model = make_protocol(Task(0, 2, 1))
    model.admit(0)
    model.claim(0, 0)
    model.claim(0, 3)
    with pytest.raises(ContractViolation, match="LOCAL_INFLIGHT"):
        model.close(0, 0)


def test_exhausted_cursor_is_saturating_and_does_not_mean_complete():
    model = make_protocol(Task(0, 1, 1))
    model.admit(0)
    lease = model.claim(0, 0)
    assert model.claim(0, 3) is None
    assert model.state(0).next_block == 1 and not model.state(0).done
    model.close(0, 3)
    execute_block(model, lease)
    model.close(0, 0)
    assert model.can_exit()


@pytest.mark.parametrize("mask", [1, 2, 3, 5, 6, 7])
def test_sync_owner_cannot_claim_two_blocks_even_after_early_return(mask):
    model = make_protocol(Task(0, 2, mask, sync_start=True))
    model.admit(0)
    owner = 1 if mask == 2 else 0
    first = model.claim(0, owner)
    execute_block(model, first)
    with pytest.raises(ContractViolation, match="SYNC_RECLAIM"):
        model.claim(0, owner)


@pytest.mark.parametrize("mask", [3, 5, 6, 7])
def test_mix_claim_before_publication_cannot_be_bypassed(mask):
    model = make_protocol(Task(0, 2, mask, sync_start=True), Task(1, 4, 2, sync_start=True))
    model.admit(0)
    model.admit(1)
    model.claim(0, 0)
    model.claim(0, 3)
    assert model.state(0).next_block == 2
    with pytest.raises(ContractViolation, match="CLUSTER_NOT_CLOSED"):
        model.close(0, 4)
    with pytest.raises(ContractViolation, match="ORDER"):
        model.claim(1, 4)
    assert not model.running


@pytest.mark.parametrize("mask", [3, 5, 6, 7])
def test_local_mix_group_completion_does_not_wait_other_clusters(mask):
    model = make_protocol(Task(0, 2, mask), Task(1, 1, 2))
    model.admit(0)
    model.admit(1)
    local, remote = model.claim(0, 0), model.claim(0, 3)
    execute_block(model, local, "mix")
    model.close_cluster(0, 0)
    for worker in (0, 1, 2):
        model.close(0, worker)
    b = model.claim(1, 1)
    execute_block(model, b)
    assert not model.state(0).done and not model._block(remote)[1].published


@pytest.mark.parametrize("mask", [3, 5, 6, 7])
def test_mix_requires_every_local_resource_lane_to_arrive(mask):
    model = make_protocol(Task(0, 1, mask), clusters=1)
    model.admit(0)
    lease = model.claim(0, 0)
    model.publish(lease)
    for worker in (0, 1):
        model.arrive(lease, worker)
    first_executor = next(iter(model._block(lease)[1].mapping.values()))
    with pytest.raises(ContractViolation, match="RESOURCE_NOT_READY"):
        model.start(lease, first_executor, "mix")


def test_mask6_coordinator_is_control_only_but_cannot_leave_before_group_completion():
    model = make_protocol(Task(0, 1, 6), clusters=1)
    model.admit(0)
    lease = model.claim(0, 0)
    model.publish(lease)
    for worker in (0, 1, 2):
        model.arrive(lease, worker)
    with pytest.raises(ContractViolation, match="EXECUTION_MEMBER"):
        model.start(lease, 0)
    with pytest.raises(ContractViolation, match="LOCAL_INFLIGHT"):
        model.close_cluster(0, 0)
    assert not [e for e in model.events if e["kind"] == "start"]


def test_mix_cannot_pair_distinct_blocks_from_the_same_task():
    model = make_protocol(Task(0, 2, 7))
    model.admit(0)
    left, right = model.claim(0, 0), model.claim(0, 3)
    model.publish(left)
    model.publish(right)
    with pytest.raises(ContractViolation, match="EXECUTION_MEMBER"):
        model.start(right, 1, "mix")


@pytest.mark.parametrize("missing", ["return", "visibility", "communication"])
def test_completion_requires_kernel_output_and_communication_contract(missing):
    model = make_protocol(Task(0, 1, 1))
    model.admit(0)
    lease = model.claim(0, 0)
    model.publish(lease)
    model.start(lease, 0)
    if missing != "return":
        model.return_kernel(lease, 0)
    if missing == "communication":
        model.publish_output(lease, 0)
    with pytest.raises(ContractViolation, match="VISIBILITY"):
        model.retire_lane(lease, 0, communication_done=missing != "communication")
    assert model.state(0).completed == 0


def test_duplicate_lane_block_and_task_completions_are_rejected():
    model = make_protocol(Task(0, 1, 1))
    model.admit(0)
    lease = model.claim(0, 0)
    execute_block(model, lease)
    with pytest.raises(ContractViolation, match="DUPLICATE_COMPLETION"):
        model.retire_lane(lease, 0)
    with pytest.raises(ContractViolation, match="DUPLICATE_BLOCK_COMPLETION"):
        model.complete_block(lease)
    assert model.state(0).completed == 1 and model.state(0).releases == 1


@pytest.mark.parametrize("field,value", [("epoch", 0), ("generation", 999), ("owner", 3)])
def test_stale_or_forged_lease_cannot_publish(field, value):
    model = make_protocol(Task(0, 1, 1))
    model.admit(0)
    lease = model.claim(0, 0)
    assert lease is not None
    with pytest.raises(ContractViolation, match="STALE_"):
        model.publish(replace(lease, **{field: value}))


def test_generation_wrap_is_rejected_without_consuming_a_block():
    model = make_protocol(Task(0, 1, 1))
    model.admit(0)
    model.generation = MAX_GENERATION
    with pytest.raises(ContractViolation, match="GENERATION_OVERFLOW"):
        model.claim(0, 0)
    assert model.state(0).next_block == 0


def test_temporary_empty_admission_log_is_not_end_of_run():
    model = make_protocol(Task(0, 1, 1))
    assert not model.admissions and not model.can_exit()
    model.prepare_admission(0)
    assert not model.can_exit()


def test_failure_does_not_fake_completion_or_allow_new_kernel_entries():
    model = make_protocol(Task(0, 1, 1), Task(1, 1, 2, deps=(0,)))
    model.admit(0)
    lease = model.claim(0, 0)
    model.publish(lease)
    model.start(lease, 0)
    model.fail("injected")
    with pytest.raises(ContractViolation, match="NOT_QUIESCENT"):
        model.confirm_quiescent()
    model.return_kernel(lease, 0)
    model.publish_output(lease, 0)
    with pytest.raises(ContractViolation, match="RUN_FAILED"):
        model.retire_lane(lease, 0)
    assert not model.state(0).done and not model.ready(1)
    model.confirm_quiescent()
    model.reset(2)
    with pytest.raises(ContractViolation, match="STALE_EPOCH"):
        model.publish(lease)


def test_reset_requires_all_local_obligations_not_only_task_done():
    model = make_protocol(Task(0, 1, 1))
    model.admit(0)
    execute_block(model, model.claim(0, 0))
    with pytest.raises(ContractViolation, match="RUN_NOT_DRAINED"):
        model.reset(2)
    close_task_members(model, 0)
    model.reset(2, (Task(0, 3, 2),))
    run_task(model, 0)
    assert model.can_exit()
    with pytest.raises(ContractViolation, match="EPOCH"):
        model.reset(2)
    with pytest.raises(ContractViolation, match="EPOCH"):
        model.reset(MAX_EPOCH + 1)
