# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scheduling obligations derived from CLA shapes and joins, not a replay of measured time."""

import random

import pytest

from .cla_trace import load_fixture
from .helpers import close_task_members, execute_block, make_protocol, run_task
from .test_graphs import check_complete
from .types import ContractViolation, Task


def projected_tasks():
    return tuple(
        Task(n["submit_index"], n["blocks"], n["canonical_mask"], tuple(n["deps"]), service=n["kind"] != "kernel")
        for n in load_fixture()["nodes"]
    )


@pytest.mark.parametrize("clusters", [1, 2, 4, 32])
@pytest.mark.parametrize("seed", range(4))
def test_entire_observed_dag_runs_with_different_capacities_and_ready_orders(clusters, seed):
    model = make_protocol(*projected_tasks(), clusters=clusters)
    rng = random.Random(seed)
    pending = set(model.tasks)
    while pending:
        ready = [t for t in sorted(pending) if model.ready(t)]
        assert ready
        chosen = rng.choice(ready)
        run_task(model, chosen)
        pending.remove(chosen)
    assert sum(e["kind"] == "start" for e in model.events) == 736
    assert sum(e["kind"] == "task_done" for e in model.events) == 101
    check_complete(model)


@pytest.mark.parametrize("slow_block", [0, 15, 31])
@pytest.mark.parametrize("shape", ["r2t17", "r2t20"])
def test_thirty_one_returns_cannot_release_a_real_thirty_two_block_dependency(shape, slow_block):
    node = next(n for n in load_fixture()["nodes"] if n["label"] == shape)
    model = make_protocol(Task(0, node["blocks"], node["canonical_mask"]), Task(1, 32, 3, deps=(0,)), clusters=16)
    model.admit(0)
    owners = sorted(model.state(0).members)
    leases = [model.claim(0, owners[b]) for b in range(32)]
    for b, lease in enumerate(leases):
        assert lease is not None
        if b != slow_block:
            execute_block(model, lease)
            model.close(0, lease.owner)
    assert model.state(0).completed == 31 and model.state(0).releases == 0
    with pytest.raises(ContractViolation, match="DATA_NOT_READY"):
        model.admit(1)
    assert not model.can_exit()
    slow_lease = leases[slow_block]
    assert slow_lease is not None
    execute_block(model, slow_lease)
    model.close(0, slow_lease.owner)
    run_task(model, 1)
    check_complete(model)


@pytest.mark.parametrize("slow_branch", range(8))
def test_final_mix_join_waits_for_each_of_eight_repeated_function_branches(slow_branch):
    tasks = [Task(i, 4, 3) for i in range(8)]
    tasks.append(Task(8, 32, 3, deps=tuple(range(8))))
    model = make_protocol(*tasks, clusters=4)
    for branch in range(8):
        if branch != slow_branch:
            run_task(model, branch)
    model.admit(slow_branch)
    leases = [model.claim(slow_branch, 3 * i) for i in range(4)]
    for lease in leases[:-1]:
        execute_block(model, lease)
    assert not model.ready(8)
    with pytest.raises(ContractViolation, match="DATA_NOT_READY"):
        model.admit(8)
    execute_block(model, leases[-1])
    close_task_members(model, slow_branch)
    run_task(model, 8)
    check_complete(model)


def test_two_sparse_attention_branches_must_both_release_dummy_and_eight_consumers():
    tasks = [Task(0, 12, 3), Task(1, 12, 3), Task(2, 1, 0, deps=(0, 1), service=True)]
    tasks += [Task(i, 8, 2, deps=(0, 1)) for i in range(3, 11)]
    model = make_protocol(*tasks, clusters=24)
    model.admit(0, range(36))
    model.admit(1, range(36, 72))
    a = [model.claim(0, 3 * i) for i in range(12)]
    b = [model.claim(1, 3 * i) for i in range(12, 24)]
    for lease in b:
        execute_block(model, lease)
    close_task_members(model, 1)
    assert not model.state(0).done and model.state(1).done
    assert not any(model.ready(ref) for ref in range(2, 11))
    for lease in a:
        execute_block(model, lease)
    close_task_members(model, 0)
    for ref in range(2, 11):
        run_task(model, ref)
    assert sum(e["kind"] == "start" and e["task"] == 2 for e in model.events) == 0
    check_complete(model)


def test_thirty_two_cube_returns_are_not_thirty_two_finished_mix_blocks():
    model = make_protocol(Task(0, 32, 3), Task(1, 32, 2, deps=(0,)), Task(2, 1, 1), clusters=32)
    model.admit(0)
    model.admit(2, (0,))
    leases = [model.claim(0, 3 * cluster) for cluster in range(32)]
    for lease in leases:
        assert lease is not None
        model.publish(lease)
        block = model.state(0).blocks[lease.block]
        for worker in sorted(block.resources):
            model.arrive(lease, worker)
        for worker in block.mapping.values():
            model.start(lease, worker, "mix")
        cube = block.mapping[0]
        model.return_kernel(lease, cube)
        model.publish_output(lease, cube)
        model.retire_lane(lease, cube)
    assert model.state(0).completed == 0
    with pytest.raises(ContractViolation, match="INCOMPLETE_BLOCK"):
        model.complete_block(leases[0])
    for lease in leases[:-1]:
        assert lease is not None
        vector = model.state(0).blocks[lease.block].mapping[1]
        model.return_kernel(lease, vector)
        model.publish_output(lease, vector)
        model.retire_lane(lease, vector)
        model.complete_block(lease)
        model.close_cluster(0, lease.owner)
        for w in model.state(0).blocks[lease.block].resources:
            model.close(0, w)
    # The fast cluster advances; the last Mix block still owns a running AIV.
    independent = model.claim(2, 0)
    execute_block(model, independent)
    model.close(2, 0)
    assert model.state(0).completed == 31 and not model.ready(1)
    tail = leases[-1]
    assert tail is not None
    vector = model.state(0).blocks[31].mapping[1]
    model.return_kernel(tail, vector)
    model.publish_output(tail, vector)
    model.retire_lane(tail, vector)
    model.complete_block(tail)
    model.close_cluster(0, tail.owner)
    for worker in model.state(0).blocks[31].resources:
        model.close(0, worker)
    run_task(model, 1)
    check_complete(model)
