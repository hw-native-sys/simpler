# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test-side identities and limits; these are not production runtime APIs."""

from __future__ import annotations

from dataclasses import dataclass

MAX_SUBTASKS = (1 << 15) - 1
MAX_GENERATION = (1 << 32) - 1
MAX_EPOCH = (1 << 64) - 1


class ContractViolation(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def require(condition, code):
    if not condition:
        raise ContractViolation(code)


@dataclass(frozen=True)
class Worker:
    worker_id: int
    cluster: int
    lane: int

    @property
    def kind(self):
        return "C" if self.lane == 0 else "V"


@dataclass(frozen=True)
class Task:
    """Protocol metadata; sync_start is the stored flag, not a public LaunchSpec request."""

    ref: int
    blocks: int
    mask: int
    deps: tuple = ()
    sync_start: bool = False
    service: bool = False

    def __post_init__(self):
        require(type(self.ref) is int and self.ref >= 0, "TASK_REF")
        require(type(self.sync_start) is bool and type(self.service) is bool, "TASK_FLAGS")
        object.__setattr__(self, "deps", tuple(self.deps))
        require(type(self.blocks) is int and 0 < self.blocks <= MAX_SUBTASKS, "BLOCK_COUNT")
        require(type(self.mask) is int and 0 <= self.mask <= 7, "ACTIVE_MASK")
        if self.mask == 4:
            object.__setattr__(self, "mask", 2)
        if self.service:
            require(self.mask == 0 and self.blocks == 1 and not self.sync_start, "SERVICE_SHAPE")
        else:
            require(self.mask != 0, "ACTIVE_MASK")
            require(self.blocks * len(self.slots) <= MAX_SUBTASKS, "SUBTASK_CAPACITY")
            require(not self.sync_start or self.blocks > 1, "SYNC_SINGLE_BLOCK")
        require(all(type(dep) is int and dep >= 0 for dep in self.deps), "FANIN")
        require(len(set(self.deps)) == len(self.deps) and self.ref not in self.deps, "FANIN")

    @property
    def slots(self):
        return tuple(slot for slot in range(3) if self.mask & (1 << slot))

    @property
    def mixed(self):
        return len(self.slots) > 1


class Topology:
    def __init__(self, workers):
        self.workers = tuple(workers)
        require(bool(self.workers), "EMPTY_TOPOLOGY")
        require(
            all(
                type(w.worker_id) is int
                and w.worker_id >= 0
                and type(w.cluster) is int
                and w.cluster >= 0
                and type(w.lane) is int
                and w.lane in (0, 1, 2)
                for w in self.workers
            ),
            "WORKER_ID",
        )
        self.by_id = {w.worker_id: w for w in self.workers}
        require(len(self.by_id) == len(self.workers), "DUPLICATE_WORKER")
        self.by_position = {(w.cluster, w.lane): w.worker_id for w in self.workers}
        require(len(self.by_position) == len(self.workers), "DUPLICATE_POSITION")

    @classmethod
    def clusters(cls, count, stride=1, offset=0):
        return cls(Worker(offset + (3 * c + lane) * stride, c, lane) for c in range(count) for lane in range(3))

    def default_members(self, task):
        require(not task.service, "SERVICE_HAS_NO_MEMBERS")
        if not task.mixed:
            kind = "C" if task.mask == 1 else "V"
            return frozenset(w.worker_id for w in self.workers if w.kind == kind)
        clusters = {w.cluster for w in self.workers}
        require(all((c, lane) in self.by_position for c in clusters for lane in range(3)), "INCOMPLETE_CLUSTER")
        return frozenset(self.by_id)

    def validate_members(self, task, members):
        require(all(type(worker) is int for worker in members), "MEMBERS")
        require(bool(members) and set(members) <= self.by_id.keys(), "MEMBERS")
        if task.mixed:
            clusters = {self.by_id[w].cluster for w in members}
            require(
                all(self.by_position.get((c, lane)) in members for c in clusters for lane in range(3)),
                "INCOMPLETE_CLUSTER",
            )
            capacity = len(clusters)
        else:
            kind = "C" if task.mask == 1 else "V"
            require(all(self.by_id[w].kind == kind for w in members), "WRONG_CORE_TYPE")
            capacity = len(members)
        require(not task.sync_start or task.blocks <= capacity, "CORESIDENCY_CAPACITY")

    def mapping(self, task, owner):
        require(owner in self.by_id, "WORKER_ID")
        if task.mixed:
            w = self.by_id[owner]
            require(w.lane == 0, "MIX_COORDINATOR")
            return {slot: self.by_position[(w.cluster, slot)] for slot in task.slots}
        return {task.slots[0]: owner}

    def resource_members(self, task, owner):
        if not task.mixed:
            return frozenset((owner,))
        cluster = self.by_id[owner].cluster
        return frozenset(self.by_position[(cluster, lane)] for lane in range(3))


@dataclass(frozen=True)
class Lease:
    epoch: int
    task: int
    block: int
    owner: int
    generation: int


def trace_index(base, task, block, slot):
    require(type(base) is int and base >= 0, "TRACE_BASE")
    require(
        type(block) is int and type(slot) is int and 0 <= block < task.blocks and slot in task.slots, "TRACE_IDENTITY"
    )
    result = base + block * len(task.slots) + task.slots.index(slot)
    require(result <= (1 << 32) - 1, "TRACE_OVERFLOW")
    return result
