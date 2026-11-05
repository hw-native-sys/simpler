# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Executable safety specification, not a simulation of A5 caches or timing."""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import MAX_EPOCH, MAX_GENERATION, Lease, require


@dataclass
class BlockState:
    lease: Lease
    mapping: dict
    resources: frozenset
    published: bool = False
    arrived: set = field(default_factory=set)
    started: set = field(default_factory=set)
    returned: set = field(default_factory=set)
    visible: set = field(default_factory=set)
    retired: set = field(default_factory=set)
    done: bool = False


@dataclass
class TaskState:
    spec: object
    members: frozenset = frozenset()
    order: int = -1
    next_block: int = 0
    blocks: dict = field(default_factory=dict)
    sync_owners: set = field(default_factory=set)
    closed_clusters: set = field(default_factory=set)
    completed: int = 0
    releases: int = 0
    done: bool = False
    admission_claimed: bool = False


class Protocol:
    def __init__(self, topology, tasks, epoch=1, max_admissions=None):
        require(type(epoch) is int and 0 < epoch <= MAX_EPOCH, "EPOCH")
        specs = tuple(tasks)
        require(len({t.ref for t in specs}) == len(specs), "DUPLICATE_TASK")
        refs = {t.ref for t in specs}
        for task in specs:
            require(set(task.deps) <= refs and all(d < task.ref for d in task.deps), "FANIN")
        self.topology = topology
        self.tasks = {t.ref: TaskState(t) for t in specs}
        self.epoch = epoch
        self.max_admissions = len(specs) if max_admissions is None else max_admissions
        self.admissions = []
        self.closed = {w: set() for w in topology.by_id}
        self.running = {}
        self.events = []
        self.generation = 0
        self.failed = False
        self.quiescent = False

    def _event(self, kind, **fields):
        self.events.append({"kind": kind, "epoch": self.epoch, "tick": len(self.events) + 1, **fields})

    def _active(self):
        require(not self.failed, "RUN_FAILED")

    def state(self, ref):
        require(ref in self.tasks, "TASK_REF")
        return self.tasks[ref]

    def ready(self, ref):
        return all(self.state(d).done for d in self.state(ref).spec.deps)

    def prepare_admission(self, ref):
        self._active()
        state = self.state(ref)
        require(not state.spec.service, "SERVICE_HAS_NO_MEMBERS")
        require(self.ready(ref), "DATA_NOT_READY")
        require(not state.admission_claimed and state.order < 0, "DUPLICATE_ADMISSION")
        state.admission_claimed = True
        self._event("prepare_admission", task=ref)

    def commit_admission(self, ref, members=None):
        self._active()
        state = self.state(ref)
        require(state.admission_claimed and state.order < 0, "ADMISSION_STATE")
        chosen = self.topology.default_members(state.spec) if members is None else frozenset(members)
        self.topology.validate_members(state.spec, chosen)
        require(len(self.admissions) < self.max_admissions, "ADMISSION_CAPACITY")
        state.members = chosen
        state.order = len(self.admissions)
        self.admissions.append(ref)
        self._event("admit", task=ref, order=state.order, members=sorted(chosen))

    def admit(self, ref, members=None):
        self.prepare_admission(ref)
        self.commit_admission(ref, members)

    def oldest(self, worker):
        require(worker in self.closed, "WORKER_ID")
        return next(
            (ref for ref in self.admissions if worker in self.state(ref).members and ref not in self.closed[worker]),
            None,
        )

    def _admissible(self, ref, worker):
        require(self.oldest(worker) == ref, "ORDER")
        require(worker not in self.running, "WORKER_BUSY")

    def _block(self, lease):
        require(lease.epoch == self.epoch, "STALE_EPOCH")
        state = self.state(lease.task)
        require(lease.block in state.blocks, "UNKNOWN_LEASE")
        block = state.blocks[lease.block]
        require(block.lease == lease, "STALE_GENERATION")
        return state, block

    def claim(self, ref, owner):
        self._active()
        state = self.state(ref)
        require(state.order >= 0 and owner in state.members, "NOT_ADMITTED")
        self._admissible(ref, owner)
        if state.spec.mixed:
            require(self.topology.by_id[owner].lane == 0, "MIX_COORDINATOR")
        if state.spec.sync_start:
            require(owner not in state.sync_owners, "SYNC_RECLAIM")
        outstanding = [b for b in state.blocks.values() if b.lease.owner == owner and not b.done]
        require(len(outstanding) < (1 if state.spec.mixed else 2), "PENDING_CAPACITY")
        if state.next_block == state.spec.blocks:
            return None
        require(self.generation < MAX_GENERATION, "GENERATION_OVERFLOW")
        mapping = self.topology.mapping(state.spec, owner)
        require(set(mapping.values()) <= state.members, "MEMBERS")
        self.generation += 1
        lease = Lease(self.epoch, ref, state.next_block, owner, self.generation)
        state.blocks[lease.block] = BlockState(lease, mapping, self.topology.resource_members(state.spec, owner))
        state.next_block += 1
        if state.spec.sync_start:
            state.sync_owners.add(owner)
        self._event(
            "claim",
            task=ref,
            block=lease.block,
            owner=owner,
            generation=lease.generation,
            mapping=sorted(mapping.items()),
            resources=sorted(state.blocks[lease.block].resources),
        )
        return lease

    def publish(self, lease, publisher=None):
        self._active()
        state, block = self._block(lease)
        require(publisher is None or publisher == lease.owner, "PUBLISH_OWNER")
        require(not block.published, "DUPLICATE_PUBLICATION")
        block.published = True
        self._event("publish", task=lease.task, block=lease.block, order=state.order)

    def arrive(self, lease, worker):
        self._active()
        _, block = self._block(lease)
        require(block.published, "UNPUBLISHED")
        require(worker in block.resources, "RESOURCE_MEMBER")
        self._admissible(lease.task, worker)
        require(worker not in block.arrived, "DUPLICATE_ARRIVAL")
        block.arrived.add(worker)
        self._event("arrive", task=lease.task, block=lease.block, worker=worker)

    def start(self, lease, worker, source="ordinary"):
        self._active()
        state, block = self._block(lease)
        require(source in ("ordinary", "mix", "direct_refill"), "SOURCE")
        require(block.published, "UNPUBLISHED")
        require(worker in block.mapping.values(), "EXECUTION_MEMBER")
        self._admissible(lease.task, worker)
        require(not state.spec.mixed or block.arrived == block.resources, "RESOURCE_NOT_READY")
        require(worker not in block.started, "DUPLICATE_EXECUTION")
        block.started.add(worker)
        self.running[worker] = lease
        slot = next(s for s, w in block.mapping.items() if w == worker)
        self._event(
            "start",
            task=lease.task,
            block=lease.block,
            slot=slot,
            worker=worker,
            order=state.order,
            generation=lease.generation,
            block_num=state.spec.blocks,
            source=source,
        )

    def return_kernel(self, lease, worker):
        _, block = self._block(lease)
        require(self.running.get(worker) == lease, "NOT_RUNNING")
        self.running.pop(worker)
        block.returned.add(worker)
        self._event("return", task=lease.task, block=lease.block, worker=worker)

    def publish_output(self, lease, worker):
        _, block = self._block(lease)
        require(worker in block.returned, "KERNEL_NOT_RETURNED")
        require(worker not in block.visible, "DUPLICATE_VISIBILITY")
        block.visible.add(worker)
        self._event("visible", task=lease.task, block=lease.block, worker=worker)

    def retire_lane(self, lease, worker, communication_done=True):
        self._active()
        _, block = self._block(lease)
        require(worker in block.visible and communication_done, "VISIBILITY")
        require(worker not in block.retired, "DUPLICATE_COMPLETION")
        block.retired.add(worker)
        self._event("retire", task=lease.task, block=lease.block, worker=worker, communication_done=True)

    def complete_block(self, lease, owner=None):
        self._active()
        state, block = self._block(lease)
        require(owner is None or owner == lease.owner, "COMPLETE_OWNER")
        require(block.retired == set(block.mapping.values()), "INCOMPLETE_BLOCK")
        require(not block.done, "DUPLICATE_BLOCK_COMPLETION")
        require(state.completed < state.spec.blocks, "COMPLETION_OVERRUN")
        block.done = True
        state.completed += 1
        self._event("block_done", task=lease.task, block=lease.block)
        if state.completed == state.spec.blocks:
            self._complete_task(state)

    def _complete_task(self, state):
        require(not state.done and state.releases == 0, "DUPLICATE_TASK_COMPLETION")
        state.done = True
        state.releases += 1
        self._event("task_done", task=state.spec.ref)

    def complete_service(self, ref):
        self._active()
        state = self.state(ref)
        require(state.spec.service, "NOT_SERVICE")
        require(self.ready(ref), "DATA_NOT_READY")
        self._complete_task(state)

    def close_cluster(self, ref, coordinator):
        self._active()
        state = self.state(ref)
        require(state.spec.mixed and coordinator in state.members, "MIX_COORDINATOR")
        worker = self.topology.by_id[coordinator]
        require(worker.lane == 0, "MIX_COORDINATOR")
        self._admissible(ref, coordinator)
        require(worker.cluster not in state.closed_clusters, "DUPLICATE_CLUSTER_CLOSE")
        exhausted = state.next_block == state.spec.blocks
        one_sync = state.spec.sync_start and coordinator in state.sync_owners
        require(exhausted or one_sync, "UNCLAIMED_BLOCKS")
        require(all(b.done for b in state.blocks.values() if b.lease.owner == coordinator), "LOCAL_INFLIGHT")
        state.closed_clusters.add(worker.cluster)
        self._event("cluster_close", task=ref, cluster=worker.cluster)

    def close(self, ref, worker):
        self._active()
        state = self.state(ref)
        require(worker in state.members, "MEMBERS")
        self._admissible(ref, worker)
        if state.spec.mixed:
            require(self.topology.by_id[worker].cluster in state.closed_clusters, "CLUSTER_NOT_CLOSED")
        else:
            one_sync = state.spec.sync_start and worker in state.sync_owners
            require(state.next_block == state.spec.blocks or one_sync, "UNCLAIMED_BLOCKS")
            require(all(b.done for b in state.blocks.values() if b.lease.owner == worker), "LOCAL_INFLIGHT")
        self.closed[worker].add(ref)
        self._event("local_done", task=ref, worker=worker)

    def can_exit(self):
        return (
            not self.failed
            and not self.running
            and all(s.done for s in self.tasks.values())
            and all(ref in self.closed[w] for ref in self.admissions for w in self.state(ref).members)
        )

    def fail(self, reason):
        self.failed = True
        self._event("error", reason=reason)

    def confirm_quiescent(self):
        require(self.failed and not self.running, "NOT_QUIESCENT")
        self.quiescent = True

    def reset(self, epoch, tasks=None):
        require(self.can_exit() or (self.failed and self.quiescent), "RUN_NOT_DRAINED")
        require(self.epoch < epoch <= MAX_EPOCH, "EPOCH")
        specs = tuple(s.spec for s in self.tasks.values()) if tasks is None else tasks
        self.__init__(self.topology, specs, epoch, self.max_admissions)
