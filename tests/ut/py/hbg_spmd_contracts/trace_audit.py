# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Strict oracle for a proposed normalized trace, not an existing swimlane ABI."""

from __future__ import annotations

from .types import MAX_EPOCH, MAX_GENERATION, ContractViolation, require

SCHEMA = "hbg-spmd-contract-v1"


class TraceAudit:
    def __init__(self, topology, tasks, epoch):
        self.epoch = epoch
        self.topology = topology
        specs = tuple(tasks)
        self.tasks = {task.ref: task for task in specs}
        require(len(self.tasks) == len(specs), "TRACE_TASKS")
        require(all(set(t.deps) <= self.tasks.keys() and all(d < t.ref for d in t.deps) for t in specs), "TRACE_TASKS")
        self.admissions = {}
        self.members = {}
        self.claims = {}
        self.published = set()
        self.arrived = {}
        self.started = set()
        self.returned = set()
        self.visible = set()
        self.retired = set()
        self.block_done = set()
        self.task_done = set()
        self.local_done = set()
        self.cluster_done = set()
        self.running = {}
        self.tick = 0

    def _task(self, event):
        require(type(event["task"]) is int and event["task"] in self.tasks, "TRACE_TASK")
        return self.tasks[event["task"]]

    def _key(self, event):
        task = self._task(event)
        block = event["block"]
        require(type(block) is int and 0 <= block < task.blocks, "TRACE_BLOCK")
        return task.ref, block

    def _execution(self, event):
        key = self._key(event)
        require(key in self.claims, "TRACE_UNCLAIMED")
        mapping = self.claims[key]["mapping"]
        worker = event["worker"]
        require(type(worker) is int, "TRACE_SCHEMA")
        require(worker in mapping.values(), "TRACE_EXECUTION_MEMBER")
        slot = next(slot for slot, assigned in mapping.items() if assigned == worker)
        return key + (slot,), worker

    def _frontier(self, task, worker):
        require(type(worker) is int, "TRACE_SCHEMA")
        require(task in self.members and worker in self.members[task], "TRACE_MEMBER")
        require((task, worker) not in self.local_done, "TRACE_LOCAL_CLOSED")
        require(
            not any(
                worker in self.members[prior] and (prior, worker) not in self.local_done
                for prior, order in self.admissions.items()
                if order < self.admissions[task]
            ),
            "TRACE_ORDER",
        )

    def consume(self, event):
        require(type(event["epoch"]) is int and event["epoch"] == self.epoch, "TRACE_EPOCH")
        require(type(event["tick"]) is int and event["tick"] > self.tick, "TRACE_TIME")
        self.tick = event["tick"]
        handler = getattr(self, "_on_" + event["kind"], None)
        if handler is None:
            raise ContractViolation("TRACE_EVENT")
        handler(event)

    def _on_prepare_admission(self, event):
        self._task(event)

    def _on_admit(self, event):
        task = self._task(event)
        require(task.ref not in self.admissions and not task.service, "TRACE_ADMISSION")
        require(set(task.deps) <= self.task_done, "TRACE_DEPENDENCY")
        require(type(event["order"]) is int and event["order"] == len(self.admissions), "TRACE_ADMISSION_GAP")
        members = frozenset(event["members"])
        require(len(members) == len(event["members"]), "TRACE_MEMBER")
        self.topology.validate_members(task, members)
        self.admissions[task.ref] = event["order"]
        self.members[task.ref] = members

    def _on_claim(self, event):
        task = self._task(event)
        key = self._key(event)
        owner = event["owner"]
        self._frontier(task.ref, owner)
        require(owner not in self.running, "TRACE_BUSY")
        require(key not in self.claims, "TRACE_DUPLICATE_CLAIM")
        require(type(event["generation"]) is int and 0 < event["generation"] <= MAX_GENERATION, "TRACE_GENERATION")
        mapping = dict(event["mapping"])
        require(all(type(slot) is int and type(worker) is int for slot, worker in mapping.items()), "TRACE_MAPPING")
        require(len(mapping) == len(event["mapping"]) and set(mapping) == set(task.slots), "TRACE_MAPPING")
        require(len(set(mapping.values())) == len(mapping), "TRACE_MAPPING")
        require(set(mapping.values()) <= self.members[task.ref], "TRACE_MAPPING")
        require(
            all(self.topology.by_id[w].kind == ("C" if slot == 0 else "V") for slot, w in mapping.items()),
            "TRACE_MAPPING",
        )
        resources = frozenset(event["resources"])
        require(all(type(worker) is int for worker in resources), "TRACE_RESOURCES")
        require(len(resources) == len(event["resources"]), "TRACE_RESOURCES")
        require(resources == self.topology.resource_members(task, owner), "TRACE_RESOURCES")
        if task.mixed:
            cluster = self.topology.by_id[owner].cluster
            require(self.topology.by_id[owner].lane == 0, "TRACE_MAPPING")
            require(all(self.topology.by_id[w].cluster == cluster for w in mapping.values()), "TRACE_MAPPING")
            if task.mask & 6 == 6:
                require(all(self.topology.by_id[mapping[s]].lane == s for s in (1, 2)), "TRACE_MAPPING")
        else:
            require(set(mapping.values()) == {owner}, "TRACE_MAPPING")
        if task.sync_start:
            require(
                not any(t == task.ref and claim["owner"] == owner for (t, _), claim in self.claims.items()),
                "TRACE_SYNC_RECLAIM",
            )
        self.claims[key] = {
            "owner": owner,
            "mapping": mapping,
            "resources": resources,
            "generation": event["generation"],
        }
        self.arrived[key] = set()

    def _on_publish(self, event):
        key = self._key(event)
        require(key in self.claims and key not in self.published, "TRACE_PUBLICATION")
        require(event["order"] == self.admissions[key[0]], "TRACE_ORDER")
        self.published.add(key)

    def _on_arrive(self, event):
        key = self._key(event)
        require(key in self.published, "TRACE_UNPUBLISHED")
        worker = event["worker"]
        self._frontier(key[0], worker)
        require(worker not in self.running, "TRACE_BUSY")
        require(worker in self.claims[key]["resources"] and worker not in self.arrived[key], "TRACE_ARRIVAL")
        self.arrived[key].add(worker)

    def _on_start(self, event):
        require(
            all(type(event[field]) is int for field in ("slot", "block_num", "generation", "order")), "TRACE_SCHEMA"
        )
        key = self._key(event)
        identity, worker = self._execution(event)
        self._frontier(key[0], worker)
        require(key in self.published, "TRACE_UNPUBLISHED")
        require(identity not in self.started, "TRACE_DUPLICATE_EXECUTION")
        require(worker not in self.running, "TRACE_BUSY")
        task = self.tasks[key[0]]
        claim = self.claims[key]
        require(event["slot"] == identity[2] and event["block_num"] == task.blocks, "TRACE_CONTEXT")
        require(event["generation"] == claim["generation"], "TRACE_GENERATION")
        require(event["order"] == self.admissions[key[0]], "TRACE_ORDER")
        require(event["source"] in ("ordinary", "mix", "direct_refill"), "TRACE_SOURCE")
        require(not task.mixed or self.arrived[key] == claim["resources"], "TRACE_RESOURCE_NOT_READY")
        self.started.add(identity)
        self.running[worker] = identity

    def _on_return(self, event):
        identity, worker = self._execution(event)
        require(self.running.get(worker) == identity, "TRACE_NOT_RUNNING")
        self.running.pop(worker)
        self.returned.add(identity)

    def _on_visible(self, event):
        identity, _ = self._execution(event)
        require(identity in self.returned and identity not in self.visible, "TRACE_VISIBILITY")
        self.visible.add(identity)

    def _on_retire(self, event):
        identity, _ = self._execution(event)
        require(identity in self.visible and event["communication_done"] is True, "TRACE_VISIBILITY")
        require(identity not in self.retired, "TRACE_DUPLICATE_COMPLETION")
        self.retired.add(identity)

    def _on_block_done(self, event):
        key = self._key(event)
        task = self._task(event)
        require(key in self.claims, "TRACE_UNCLAIMED")
        require(key not in self.block_done, "TRACE_DUPLICATE_BLOCK")
        require(all(key + (slot,) in self.retired for slot in task.slots), "TRACE_INCOMPLETE_BLOCK")
        self.block_done.add(key)

    def _on_task_done(self, event):
        task = self._task(event)
        require(task.ref not in self.task_done, "TRACE_DUPLICATE_TASK")
        require(set(task.deps) <= self.task_done, "TRACE_DEPENDENCY")
        require(task.service or all((task.ref, b) in self.block_done for b in range(task.blocks)), "TRACE_EARLY_DONE")
        self.task_done.add(task.ref)

    def _owned_done(self, ref, owner):
        return all(
            key in self.block_done for key, claim in self.claims.items() if key[0] == ref and claim["owner"] == owner
        )

    def _claim_closed(self, ref, owner):
        task = self.tasks[ref]
        own = [key for key, claim in self.claims.items() if key[0] == ref and claim["owner"] == owner]
        return (task.sync_start and len(own) == 1) or sum(key[0] == ref for key in self.claims) == task.blocks

    def _on_cluster_close(self, event):
        task = self._task(event)
        cluster = event["cluster"]
        require(type(cluster) is int, "TRACE_SCHEMA")
        owner = self.topology.by_position.get((cluster, 0))
        require(task.mixed and owner in self.members[task.ref], "TRACE_CLUSTER")
        self._frontier(task.ref, owner)
        require((task.ref, cluster) not in self.cluster_done, "TRACE_CLUSTER")
        require(self._claim_closed(task.ref, owner) and self._owned_done(task.ref, owner), "TRACE_LOCAL_INFLIGHT")
        self.cluster_done.add((task.ref, cluster))

    def _on_local_done(self, event):
        task = self._task(event)
        worker = event["worker"]
        self._frontier(task.ref, worker)
        require(worker not in self.running, "TRACE_BUSY")
        if task.mixed:
            require((task.ref, self.topology.by_id[worker].cluster) in self.cluster_done, "TRACE_CLUSTER")
        else:
            require(self._claim_closed(task.ref, worker) and self._owned_done(task.ref, worker), "TRACE_LOCAL_INFLIGHT")
        self.local_done.add((task.ref, worker))

    def finish(self):
        expected = {
            (t.ref, b, slot) for t in self.tasks.values() if not t.service for b in range(t.blocks) for slot in t.slots
        }
        require(self.started == expected and self.retired == expected, "TRACE_COVERAGE")
        require(self.task_done == self.tasks.keys() and not self.running, "TRACE_INCOMPLETE")
        require(
            all((t, w) in self.local_done for t, members in self.members.items() for w in members), "TRACE_INCOMPLETE"
        )
        return {"tasks": len(self.tasks), "blocks": len(self.block_done), "executions": len(expected)}


def audit_trace(topology, tasks, payload, expected_epoch, expected_producer="aicore"):
    try:
        require(type(expected_epoch) is int and 0 < expected_epoch <= MAX_EPOCH, "TRACE_EPOCH")
        require(type(payload["epoch"]) is int and payload["epoch"] == expected_epoch, "TRACE_EPOCH")
        require(
            payload["schema"] == SCHEMA
            and payload["complete"] is True
            and type(payload["dropped"]) is int
            and payload["dropped"] == 0,
            "TRACE_SCHEMA",
        )
        require(payload["producer"] == expected_producer, "TRACE_PRODUCER")
        audit = TraceAudit(topology, tasks, expected_epoch)
        for event in payload["events"]:
            audit.consume(event)
        return audit.finish()
    except (KeyError, TypeError, AttributeError) as error:
        raise ContractViolation("TRACE_SCHEMA") from error
