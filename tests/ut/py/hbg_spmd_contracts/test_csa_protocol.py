# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Actual CSA DAG obligations on the executable protocol, separate from A5 numerical tests."""

import random
from collections import Counter

import pytest

from examples.a5.host_build_graph.deepseek_v4_pro_csa import graph_contract

from .helpers import close_task_members, execute_block, make_protocol, run_task
from .test_graphs import check_complete
from .types import ContractViolation, Task

NODES = graph_contract.frozen_graph()["graph"]["nodes"]


def spec(node, deps=None, sync=False):
    mask = sum(1 << lane for lane, kernel in enumerate(node["kernel_ids"]) if kernel >= 0)
    return Task(
        node["task_id"],
        node["block_num"],
        mask,
        tuple(node["deps"] if deps is None else deps),
        sync_start=sync,
        service=mask == 0,
    )


def by_ref(ref):
    return next(node for node in NODES if node["task_id"] == ref)


def test_real_csa_graph_cannot_be_replaced_by_the_smaller_cla_capture():
    tasks = [spec(node) for node in NODES]
    assert len(tasks) == 81
    assert sum(t.service for t in tasks) == 2
    assert sum(t.blocks > 1 for t in tasks) == 60
    assert max(t.blocks for t in tasks) == 512
    assert sum(t.blocks * len(t.slots) for t in tasks) == 2308
    assert Counter(t.mask for t in tasks) == {0: 2, 1: 3, 2: 50, 3: 26}
    assert by_ref(23)["block_num"] == 512
    assert by_ref(56)["block_num"] == 256
    assert len(by_ref(105)["deps"]) == 17  # 16 pack branches plus the direct input dependency.


@pytest.mark.parametrize("clusters", [1, 4, 32])
@pytest.mark.parametrize("seed", range(4))
def test_full_csa_graph_at_small_and_full_capacity_with_ready_order_permutations(clusters, seed):
    model = make_protocol(*(spec(node) for node in NODES), clusters=clusters)
    pending = set(model.tasks)
    rng = random.Random(seed)
    while pending:
        ready = [ref for ref in sorted(pending) if model.ready(ref)]
        assert ready, "CSA dependency propagation stalled"
        ref = rng.choice(ready)
        run_task(model, ref)
        pending.remove(ref)
    check_complete(model)
    starts = [event for event in model.events if event["kind"] == "start"]
    assert len(starts) == 2308
    assert Counter(e["slot"] for e in starts) == {0: 864, 1: 1444}


@pytest.mark.parametrize("slow_block", [0, 31, 32, 255, 256, 510, 511])
def test_512_mix_blocks_do_not_release_qkv_after_only_511_completions(slow_block):
    model = make_protocol(spec(by_ref(23), ()), spec(by_ref(24), (23,)), clusters=2)
    model.admit(23)
    tail = None
    for index in range(512):
        lease = model.claim(23, 0 if index == slow_block else 3)
        if index == slow_block:
            tail = lease
        else:
            execute_block(model, lease, "mix")
    assert model.state(23).completed == 511
    assert model.state(23).releases == 0
    assert not model.can_exit()
    with pytest.raises(ContractViolation, match="DATA_NOT_READY"):
        model.admit(24)
    execute_block(model, tail, "mix")
    close_task_members(model, 23)
    run_task(model, 24)
    check_complete(model)


@pytest.mark.parametrize("tail_lane", [0, 1])
def test_last_mix_lane_and_output_visibility_gate_qkv_fanout(tail_lane):
    model = make_protocol(spec(by_ref(23), ()), spec(by_ref(24), (23,)), clusters=1)
    model.admit(23)
    for _ in range(511):
        execute_block(model, model.claim(23, 0), "mix")
    lease = model.claim(23, 0)
    model.publish(lease)
    block = model.state(23).blocks[511]
    for worker in block.resources:
        model.arrive(lease, worker)
    for lane, worker in block.mapping.items():
        model.start(lease, worker, "mix")
        model.return_kernel(lease, worker)
        if lane != tail_lane:
            model.publish_output(lease, worker)
            model.retire_lane(lease, worker)
    with pytest.raises(ContractViolation, match="INCOMPLETE_BLOCK"):
        model.complete_block(lease)
    assert not model.ready(24)
    worker = block.mapping[tail_lane]
    model.publish_output(lease, worker)
    model.retire_lane(lease, worker)
    model.complete_block(lease)
    close_task_members(model, 23)
    run_task(model, 24)
    check_complete(model)


@pytest.mark.parametrize(
    "consumer,parents,slow_branch",
    [(56, tuple(range(48, 56)), branch) for branch in range(8)]
    + [(105, tuple(range(59, 105, 3)), branch) for branch in range(16)],
)
def test_every_merge_and_projection_branch_must_complete_before_real_join(consumer, parents, slow_branch):
    tasks = [spec(by_ref(ref), ()) for ref in parents] + [spec(by_ref(consumer), parents)]
    model = make_protocol(*tasks, clusters=2)
    slow = parents[slow_branch]
    for ref in parents:
        if ref != slow:
            run_task(model, ref)
    model.admit(slow)
    for _ in range(model.state(slow).spec.blocks - 1):
        execute_block(model, model.claim(slow, 1))
    assert not model.ready(consumer)
    with pytest.raises(ContractViolation, match="DATA_NOT_READY"):
        model.admit(consumer)
    execute_block(model, model.claim(slow, 1))
    close_task_members(model, slow)
    run_task(model, consumer)
    check_complete(model)


@pytest.mark.parametrize("ref", [23, 24, 32, 56, 105])
def test_oversubscribed_ordinary_csa_task_cannot_be_relabelled_sync_start(ref):
    task = spec(by_ref(ref), (), sync=True)
    model = make_protocol(task, clusters=32)
    with pytest.raises(ContractViolation, match="CORESIDENCY_CAPACITY"):
        model.admit(ref)


@pytest.mark.parametrize("ref", [14, 27])
def test_csa_phase_fences_remain_dependency_only_nodes(ref):
    node = by_ref(ref)
    model = make_protocol(*(spec(by_ref(p), ()) for p in node["deps"]), spec(node), clusters=2)
    for p in node["deps"][:-1]:
        run_task(model, p)
    assert not model.ready(ref)
    run_task(model, node["deps"][-1])
    run_task(model, ref)
    assert not any(e["kind"] == "start" and e["task"] == ref for e in model.events)
    check_complete(model)
