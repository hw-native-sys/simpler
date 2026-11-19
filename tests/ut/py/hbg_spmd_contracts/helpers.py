# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Deterministic schedules shared by contract tests."""

from .model import Protocol
from .types import Topology


def make_protocol(*tasks, clusters=2, **kwargs):
    return Protocol(Topology.clusters(clusters), tasks, **kwargs)


def execute_block(model, lease, source="ordinary"):
    state, block = model._block(lease)
    model.publish(lease)
    if state.spec.mixed:
        for worker in sorted(block.resources):
            model.arrive(lease, worker)
    for worker in block.mapping.values():
        model.start(lease, worker, source)
    for worker in block.mapping.values():
        model.return_kernel(lease, worker)
        model.publish_output(lease, worker)
        model.retire_lane(lease, worker)
    model.complete_block(lease)


def close_task_members(model, ref):
    state = model.state(ref)
    if state.spec.mixed:
        for worker in sorted(state.members):
            if model.topology.by_id[worker].lane == 0:
                model.close_cluster(ref, worker)
    for worker in sorted(state.members):
        model.close(ref, worker)


def run_task(model, ref):
    state = model.state(ref)
    if state.spec.service:
        model.complete_service(ref)
        return
    if state.order < 0:
        model.admit(ref)
    owners = [w for w in sorted(state.members) if not state.spec.mixed or model.topology.by_id[w].lane == 0]
    index = 0
    while state.next_block < state.spec.blocks:
        owner = owners[index % len(owners)]
        lease = model.claim(ref, owner)
        execute_block(model, lease, "mix" if state.spec.mixed else "ordinary")
        index += 1
    close_task_members(model, ref)
