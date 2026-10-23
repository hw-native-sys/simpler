# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shape, capacity, identity, topology and ABI boundary contracts."""

import itertools

import pytest

from .helpers import make_protocol, run_task
from .model import Protocol
from .types import MAX_SUBTASKS, ContractViolation, Task, Topology, Worker, trace_index


@pytest.mark.parametrize("mask", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("blocks", [1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 96, 128, 512])
def test_valid_block_counts_preserve_logical_shape(mask, blocks):
    task = Task(0, blocks, mask)
    assert task.blocks == blocks
    assert task.mask == (2 if mask == 4 else mask)
    assert len(task.slots) == bin(mask).count("1")


@pytest.mark.parametrize("mask", [1, 2, 3, 5, 6, 7])
@pytest.mark.parametrize("blocks", [1, 2, 7, 33, 65, 512])
def test_ordinary_tasks_run_in_waves_without_core_capacity_limit(mask, blocks):
    model = make_protocol(Task(0, blocks, mask), clusters=2)
    run_task(model, 0)
    state = model.state(0)
    starts = [e for e in model.events if e["kind"] == "start"]
    assert len(starts) == blocks * len(state.spec.slots)
    assert {(e["block"], e["slot"]) for e in starts} == set(itertools.product(range(blocks), state.spec.slots))
    assert all(e["block_num"] == blocks for e in starts)
    assert state.completed == blocks and state.releases == 1
    assert model.can_exit()


@pytest.mark.parametrize("blocks", [0, -1, MAX_SUBTASKS + 1, True, 1.5])
def test_invalid_block_count_is_rejected_before_narrowing(blocks):
    with pytest.raises(ContractViolation, match="BLOCK_COUNT"):
        Task(0, blocks, 1)


@pytest.mark.parametrize("mask", [0, -1, 8, True, 1.5])
def test_invalid_execution_mask_is_rejected(mask):
    with pytest.raises(ContractViolation, match="ACTIVE_MASK"):
        Task(0, 2, mask)


@pytest.mark.parametrize("mask", [1, 2, 3, 5, 6, 7])
def test_host_signed_16_bit_subtask_limit_is_stricter_than_resident_unsigned_check(mask):
    slots = bin(mask).count("1")
    last_valid = MAX_SUBTASKS // slots
    assert Task(0, last_valid, mask).blocks == last_valid
    with pytest.raises(ContractViolation, match="BLOCK_COUNT|SUBTASK_CAPACITY"):
        Task(0, last_valid + 1, mask)


@pytest.mark.parametrize("mask,capacity", [(1, 2), (2, 4), (3, 2), (5, 2), (6, 2), (7, 2)])
def test_sync_start_capacity_is_per_shape(mask, capacity):
    model = make_protocol(Task(0, capacity, mask, sync_start=True))
    run_task(model, 0)
    assert model.can_exit()
    rejected = make_protocol(Task(0, capacity + 1, mask, sync_start=True))
    with pytest.raises(ContractViolation, match="CORESIDENCY_CAPACITY"):
        rejected.admit(0)
    assert not rejected.admissions
    assert not rejected.state(0).blocks


def test_sync_start_one_block_remains_invalid():
    with pytest.raises(ContractViolation, match="SYNC_SINGLE_BLOCK"):
        Task(0, 1, 1, sync_start=True)


@pytest.mark.parametrize("mask", [1, 2, 3, 5, 6, 7])
def test_sparse_worker_ids_are_not_array_indices(mask):
    topology = Topology.clusters(2, stride=17, offset=101)
    model = Protocol(topology, (Task(0, 7, mask),))
    run_task(model, 0)
    assert model.can_exit()
    assert all(e["worker"] in topology.by_id for e in model.events if e["kind"] == "start")


@pytest.mark.parametrize(
    "workers,code",
    [
        ([], "EMPTY_TOPOLOGY"),
        ([Worker(0, 0, 0), Worker(0, 1, 0)], "DUPLICATE_WORKER"),
        ([Worker(0, 0, 0), Worker(1, 0, 0)], "DUPLICATE_POSITION"),
        ([Worker(-1, 0, 0)], "WORKER_ID"),
        ([Worker(0, 0, 3)], "WORKER_ID"),
    ],
)
def test_invalid_topology_is_rejected(workers, code):
    with pytest.raises(ContractViolation, match=code):
        Topology(workers)


@pytest.mark.parametrize("missing_lane", [0, 1, 2])
def test_mix_rejects_incomplete_physical_cluster(missing_lane):
    topology = Topology(w for w in Topology.clusters(1).workers if w.lane != missing_lane)
    model = Protocol(topology, (Task(0, 2, 7),))
    with pytest.raises(ContractViolation, match="INCOMPLETE_CLUSTER"):
        model.admit(0)


@pytest.mark.parametrize("members,code", [([], "MEMBERS"), ([999], "MEMBERS"), ([0], "INCOMPLETE_CLUSTER")])
def test_mix_members_are_frozen_as_complete_resource_groups(members, code):
    model = make_protocol(Task(0, 2, 7))
    with pytest.raises(ContractViolation, match=code):
        model.admit(0, members)


@pytest.mark.parametrize("mask,wrong_member", [(1, 1), (2, 0)])
def test_no_cross_type_claim_population(mask, wrong_member):
    model = make_protocol(Task(0, 2, mask))
    with pytest.raises(ContractViolation, match="WRONG_CORE_TYPE"):
        model.admit(0, (wrong_member,))


@pytest.mark.parametrize("mask", [1, 2, 3, 5, 6, 7])
def test_trace_indices_are_dense_unique_and_include_block(mask):
    task = Task(0, 7, mask)
    indices = [trace_index(19, task, b, s) for b in range(task.blocks) for s in task.slots]
    assert indices == list(range(19, 19 + 7 * len(task.slots)))


@pytest.mark.parametrize("block,slot", [(-1, 0), (2, 0), (0, 1), (0, 3)])
def test_trace_identity_bounds(block, slot):
    with pytest.raises(ContractViolation, match="TRACE_IDENTITY"):
        trace_index(0, Task(0, 2, 1), block, slot)


def test_trace_prefix_overflow():
    with pytest.raises(ContractViolation, match="TRACE_OVERFLOW"):
        trace_index((1 << 32) - 1, Task(0, 2, 1), 1, 0)


@pytest.mark.parametrize(
    "tasks",
    [
        (Task(0, 1, 1), Task(0, 1, 2)),
        (Task(0, 1, 1, deps=(1,)), Task(1, 1, 1)),
        (Task(2, 1, 1, deps=(0,)),),
    ],
)
def test_invalid_task_graph_is_rejected(tasks):
    with pytest.raises(ContractViolation, match="DUPLICATE_TASK|FANIN"):
        Protocol(Topology.clusters(1), tasks)


def test_no_kernel_service_waits_for_real_dependency():
    model = make_protocol(Task(0, 2, 1), Task(1, 1, 0, deps=(0,), service=True))
    with pytest.raises(ContractViolation, match="DATA_NOT_READY"):
        model.complete_service(1)
    run_task(model, 0)
    model.complete_service(1)
    assert model.state(1).done and model.state(1).releases == 1
    assert len([e for e in model.events if e["kind"] == "start"]) == 2
    assert model.can_exit()


def test_empty_graph_is_terminal_without_kernel_work():
    assert Protocol(Topology.clusters(1), ()).can_exit()


@pytest.mark.parametrize("dependency", [True, "0", -1, 1.0])
def test_dependency_refs_reject_numeric_aliases_and_negative_values(dependency):
    with pytest.raises(ContractViolation, match="FANIN"):
        Task(2, 1, 1, deps=(dependency,))


def test_member_ids_reject_boolean_aliases():
    model = make_protocol(Task(0, 2, 1))
    with pytest.raises(ContractViolation, match="MEMBERS"):
        model.admit(0, (False, 3))
