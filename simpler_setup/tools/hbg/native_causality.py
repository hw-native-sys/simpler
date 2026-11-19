# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Audit native publication intervals, observations and per-member obligations."""

from bisect import bisect_right
from collections import defaultdict

from .execution_identity import FIELDS, _validate_record, valid_source_index
from .native_order import CausalOrder
from .native_protocol import (
    COMPLETION_ALIAS_FLAGS,
    COMPLETION_FROM_CONTRIBUTION,
    COMPLETION_FROM_GROUP,
    CONTROL_NOTIFY_ACCOUNTING,
    COUNTER_ACCOUNTING,
    DEFERRED_FINISH,
    _completion_aliases,
    _error,
    audit_protocol_sequences,
    has_native_graph,
    stamp_worker,
)

MAX = (1 << 64) - 1
EXECUTABLE, MIX, SPMD, SYNC, INLINE, PREDICATE = (1, 4, 8, 16, 32, 64)
ASYNC_INPUT = 1 << 1
ENTRY, EXECUTED, COOPERATIVE, PUBLISHED = (1, 2, 4, 64)
CLUSTER_GROUP = 1 << 7


def require(condition, message):
    if not condition:
        _error(message)


def point(record, field):
    cycles, sequence = record[field]
    require(sequence != 0, f"missing {field} for task {record.get('task_ref')}")
    return stamp_worker(record, field), cycles, sequence


class NativeAudit:
    def __init__(self, section, identities):
        self.raw = section
        self.node_map = {}
        self.control_nodes = set()
        self.sequence_result = audit_protocol_sequences(section)
        self.completion_aliases = _completion_aliases(section)
        self.causal = CausalOrder(section)
        self.before = self.causal.before
        self.identities = identities
        self.tasks = {row["task_ref"]: row for row in section["tasks"]}
        self.workers = {row["index"]: row for row in section["workers"]}
        self.controls = {row["task_ref"]: row for row in section["task_controls"]}
        self.actors = {row["index"]: row for row in section["task_events"]}
        self.members = {row["index"]: row for row in section["members"]}
        self.blocks = {row["index"]: row for row in section["blocks"]}
        self.executions = {row["index"]: row for row in section["executions"]}
        self.conditions = {row["index"]: row for row in section["condition_contributions"]}
        self.used_conditions = set()
        self.ranks = {ref: task["schedule_rank"] for ref, task in self.tasks.items()}
        self.ordered = {ref for ref, task in self.tasks.items() if task["flags"] & (MIX | SYNC)}
        self.floor_points = {}
        self.opposite_reads = defaultdict(list)
        self.mix_reads = []
        self.expected_executions = {}
        self.expected_blocks = {}
        self.positions = {}
        self.task_members = {}
        self.task_blocks = defaultdict(list)
        self.done_points = {}
        self.claim_points = {}
        self.claim_seals = {}
        self.payload_points = {}

    def chain(self, record, *fields):
        points = [point(record, field) for field in fields]
        for left, right in zip(points, points[1:]):
            self.before(left, right, f"invalid local order {fields} for task {record['task_ref']}")

    def ordered_intervals(self, records, first, last):
        prior = None
        for record in records:
            key = (first, record["task_ref"], record["block_idx"])
            if first == "contribution_before":
                key += (record["subtask_slot"],)
            current = self.causal.bounded(key, point(record, first), point(record, last))
            if prior is not None:
                self.causal.add(prior, current)
            prior = current
            if first == "claim_before":
                self.claim_points[(record["task_ref"], record["block_idx"])] = current

    def topology(self):
        count = self.raw["worker_count"]
        clusters = self.raw["cluster_count"]
        require(count > 0 and count == clusters * 3, "incomplete worker topology")
        require(self.raw["active_worker_count"] == count, "inactive scheduler topology")
        cores = set()
        for index, worker in self.workers.items():
            peers = worker["cluster_workers"]
            cluster = worker["cluster_index"]
            require(worker["core_id"] >= 0 and worker["core_id"] not in cores, "duplicate or unknown physical core")
            cores.add(worker["core_id"])
            require(cluster < clusters and len(set(peers)) == 3 and index in peers, "invalid cluster membership")
            for lane, peer in enumerate(peers):
                require(peer in self.workers, "cluster names an absent worker")
                other = self.workers[peer]
                require(
                    other["cluster_index"] == cluster and other["cluster_workers"] == peers, "split cluster topology"
                )
                require(other["core_type"] == int(lane != 0), "cluster lane has wrong core type")
            self.positions[index] = cluster, peers.index(index)
        require({w["cluster_index"] for w in self.workers.values()} == set(range(clusters)), "missing cluster")

    def graph(self):
        if has_native_graph(self.raw):
            from .native_graph import audit_graph_expansion  # noqa: PLC0415

            self.node_map = audit_graph_expansion(self.raw)
            self.control_nodes = {ref for ref, node in self.node_map.items() if node["kind"] in (2, 3, 4)}
        prefix = 0
        trace_indices = set()
        for ref, task in sorted(self.tasks.items()):
            flags, mask, blocks = task["flags"], task["active_mask"], task["block_num"]
            deps = task["dependencies"]
            require(len(set(deps)) == len(deps) and all(0 <= dep < ref for dep in deps), "invalid frozen dependencies")
            require(
                flags & ~(EXECUTABLE | ASYNC_INPUT | MIX | SPMD | SYNC | INLINE | PREDICATE) == 0,
                "unknown task flags",
            )
            if ref in self.control_nodes:
                require(mask == 0 and blocks == 1 and flags == 0, "invalid native control metadata")
                continue
            if flags == 0:
                require(
                    mask == blocks == task["trace_base"] == 0 and task["kernel_ids"] == [0, 0, 0],
                    "bad Host-completed task",
                )
                continue
            require(flags & EXECUTABLE, "invalid executable task metadata")
            if self.node_map:
                prefix = task["trace_base"]
            require(task["trace_base"] == prefix <= (1 << 32) - 1, "noncontiguous frozen trace prefix")
            if flags & INLINE:
                require(
                    mask == 2
                    and blocks == 1
                    and task["kernel_ids"] == [65535] * 3
                    and not flags & (MIX | SPMD | SYNC | PREDICATE),
                    "invalid inline service",
                )
                continue
            active = bin(mask).count("1")
            require(0 < mask < 8 and 0 < blocks * active <= 32767, "invalid frozen task shape")
            require(bool(flags & MIX) == (active > 1) and bool(flags & SPMD) == (blocks > 1), "shape/flags mismatch")
            if flags & SYNC:
                positions = self.raw["cluster_count"] * (2 if mask in (2, 4) else 1)
                require(1 < blocks <= positions, "sync_start distinct position capacity exceeded or N=1")
            require(not flags & PREDICATE or (active == blocks == 1), "unsupported cooperative predicate")
            for slot, kernel in enumerate(task["kernel_ids"]):
                require(0 <= kernel < 65535 if mask & (1 << slot) else kernel == 65535, "invalid frozen kernel mapping")
            for block in range(blocks):
                base = prefix + block * active
                if flags & (MIX | SYNC):
                    self.expected_blocks[base] = ref, block
                for offset, slot in enumerate(slot for slot in range(3) if mask & (1 << slot)):
                    require(base + offset not in trace_indices, "overlapping execution trace ranges")
                    trace_indices.add(base + offset)
                    self.expected_executions[base + offset] = ref, block, slot
            prefix += blocks * active
        if self.node_map:
            prefix = len(trace_indices)
            require(trace_indices == set(range(prefix)), "noncontiguous native trace ranges")
        require(
            prefix == self.raw["trace_count"] or (prefix == 0 and self.raw["trace_count"] == len(self.tasks)),
            "trace capacity mismatch",
        )
        device = {ref for ref, task in self.tasks.items() if task["flags"] & EXECUTABLE} | self.control_nodes
        require(set(self.actors) == device, "missing or extraneous logical completion record")
        for ref in device:
            done = self.actors[ref]["completion"]
            require(done is not None, "missing TaskDone publication")
            if done["flags"] & CONTROL_NOTIFY_ACCOUNTING:
                require(
                    (self.tasks[ref]["flags"] & INLINE or ref in self.control_nodes)
                    and self.controls[ref]["completion_backend"] == 0
                    and not done["flags"] & (COUNTER_ACCOUNTING | DEFERRED_FINISH | COMPLETION_ALIAS_FLAGS),
                    "control notification accounting on a non-control task or conflicting source",
                )
                # No record identifies the earliest SET among multiple publishers.
                # Keep the logical fanin join untimed; the actor retains CAS time.
                self.done_points[ref] = ("ControlReady", ref)
                continue
            if self.controls[ref]["completion_backend"] == 1:
                require(
                    done["flags"] & COUNTER_ACCOUNTING and not done["flags"] & COMPLETION_ALIAS_FLAGS,
                    "counter completion must identify accounting, not a last-writer alias",
                )
                self.done_points[ref] = ("CounterReady", ref)
                continue
            require(not done["flags"] & COUNTER_ACCOUNTING, "GM completion has counter accounting actor")
            source = self.completion_aliases.get(ref)
            name = (
                ("TaskDone", ref)
                if source is None
                else ("completion_before", ref, source["block_idx"])
                if done["flags"] & COMPLETION_FROM_GROUP
                else ("contribution_before", ref, source["block_idx"], source["subtask_slot"])
            )
            self.done_points[ref] = self.causal.bounded(name, point(done, "before"), point(done, "after"))
        require(
            self.raw["expected_task_count"] == len(self.tasks) and self.raw["resolved_task_count"] == len(device),
            "logical completion count mismatch",
        )
        require(self.raw["inline_completed_count"] == len(self.tasks) - len(device), "Host completion count mismatch")
        require(set(self.executions) == set(self.expected_executions), "missing or extraneous execution")
        require(set(self.blocks) == set(self.expected_blocks), "missing or extraneous block owner")

    def task_order(self):
        require(set(self.ranks.values()) == set(range(len(self.tasks))), "schedule ranks are not a bijection")
        require(len(self.ordered) == self.raw["cooperative_task_count"], "cooperative task count mismatch")
        mix_tasks = sorted((ref for ref in self.ordered if self.tasks[ref]["flags"] & MIX), key=self.ranks.__getitem__)
        require(len(mix_tasks) == self.raw["mix_task_count"], "MIX task count mismatch")
        require(
            sum(bool(self.tasks[ref]["flags"] & SYNC) for ref in mix_tasks) == self.raw["sync_mix_task_count"],
            "sync MIX count mismatch",
        )
        mix, sync, compact = [0] * 3, [0] * 3, 0
        for ref, rank in sorted(self.ranks.items(), key=lambda item: item[1]):
            task = self.tasks[ref]
            require(all(self.ranks[dep] < rank for dep in task["dependencies"]), "dependency violates schedule rank")
            flags, active = task["flags"], task["active_mask"]
            cooperative = ref in self.ordered
            require(
                task["cooperative_index"] == (compact if cooperative else (1 << 32) - 1),
                "incorrect cooperative trace index",
            )
            compact += int(cooperative)
            mask = active | 2 if flags & MIX else (6 if active in (2, 4) else active)
            if not flags & EXECUTABLE or flags & INLINE:
                mask = 0
            expected = [0] * 3
            for lane in range(3):
                if mask & (1 << lane):
                    expected[lane] = sync[lane] if flags & MIX else mix[lane]
                    if flags & MIX:
                        mix[lane] = rank + 1
                    elif flags & SYNC:
                        sync[lane] = rank + 1
            require(task["required_floor"] == expected, "invalid conflict threshold or proxy participation")
        self.mix_by_rank = mix_tasks
        self.floor_publications()

    def floor_publications(self):
        clusters = self.raw["cluster_count"]
        none = (1 << 32) - 1
        refs = self.mix_by_rank
        next_floor = {
            ref: self.ranks[refs[i + 1]] if i + 1 < len(refs) else len(self.tasks) for i, ref in enumerate(refs)
        }
        if refs and self.ranks[refs[0]] > 0:
            next_floor[none] = self.ranks[refs[0]]
        dense = self.raw["floor_count"] == (len(refs) + 1) * clusters
        rows = {ref: i + 1 if dense else 1 + self.tasks[ref]["cooperative_index"] for i, ref in enumerate(refs)}
        previous = {}
        for event in sorted(self.raw["floor_publications"], key=lambda e: (e["cluster_index"], e["floor"])):
            ref, floor, cluster, worker = (
                event[k] for k in ("prior_mix_ref", "floor", "cluster_index", "worker_index")
            )
            require(
                ref in next_floor and next_floor[ref] == floor and self.positions.get(worker) == (cluster, 1),
                "invalid MIX floor writer or boundary",
            )
            row = 0 if ref == none else rows[ref]
            require(
                event["index"] == row * clusters + cluster and (cluster, floor) not in self.floor_points,
                "duplicate or misbound MIX floor publication",
            )
            if cluster in previous:
                self.before(point(previous[cluster], "after"), point(event, "before"), "nonmonotonic MIX publication")
            previous[cluster] = event
            self.floor_points[cluster, floor] = self.causal.bounded(
                ("MIX floor", cluster, floor), point(event, "before"), point(event, "after")
            )
        require(len(self.floor_points) == len(next_floor) * clusters, "incomplete cluster floor publications")

    def record_identity(self, record, ref, *, completion=False):
        flags = self.tasks[ref]["flags"]
        if completion:
            flags |= record["flags"] & (
                DEFERRED_FINISH | COMPLETION_ALIAS_FLAGS | COUNTER_ACCOUNTING | CONTROL_NOTIFY_ACCOUNTING
            )
        require(
            record["task_ref"] == ref
            and record["run_epoch"] == self.raw["run_epoch"]
            and record["flags"] == flags
            and record["order_key"] == self.ranks[ref],
            "task/order/epoch/flags identity mismatch",
        )

    def actor(self, actor, ref, *, completion=False):
        self.record_identity(actor, ref, completion=completion)
        require(actor["worker_index"] in self.workers, "unknown task actor")
        self.chain(actor, "before", "after")

    def check_opposite_source(self, ref, worker, record, when):
        cluster, lane = self.positions[worker]
        task = self.tasks[ref]
        mixed = bool(task["flags"] & MIX)
        required = task["required_floor"][lane]
        mix, own = record["observed_mix_floor"], record["observed_self_floor"]
        observed, other = (own, mix) if mixed else (mix, own)
        require(
            other == 0 and ((required == observed == 0) or (0 < required <= observed <= len(self.tasks))),
            "unproven rank threshold or fabricated opposite-source read",
        )
        if not required:
            return
        if mixed:
            self.opposite_reads[worker].append((required, when))
        else:
            require((cluster, observed) in self.floor_points, "observed MIX floor has no publisher")
            self.causal.add(self.floor_points[cluster, observed], when)
            self.mix_reads.append((ref, worker, when))

    def membership(self):
        count = len(self.workers)
        for e in self.executions.values():
            ref, worker = e["task_ref"], e["worker_index"]
            require(ref in self.tasks and worker in self.workers, "invalid execution task or worker")
            if not self.tasks[ref]["flags"] & SYNC:
                continue
            members = self.task_members.setdefault(ref, set())
            require(worker not in members, "sync_start assigned two blocks to one physical worker")
            members.add(worker)
        expected = {
            self.tasks[ref]["cooperative_index"] * count + worker
            for ref, members in self.task_members.items()
            for worker in members
        }
        require(set(self.members) == expected, "missing or extraneous commitment on a physical sync member")
        by_worker = defaultdict(list)
        for member in self.members.values():
            ref, worker = member["task_ref"], member["worker_index"]
            require(
                ref in self.task_members
                and worker in self.task_members[ref]
                and member["order_key"] == self.ranks[ref]
                and member["run_epoch"] == self.raw["run_epoch"]
                and member["index"] == self.tasks[ref]["cooperative_index"] * count + worker,
                "invalid member owner, epoch, or rank",
            )
            self.chain(member, "committed", "retired")
            if self.tasks[ref]["flags"] & MIX:
                self.check_opposite_source(ref, worker, member, point(member, "committed"))
            by_worker[worker].append(member)
        for events in by_worker.values():
            events.sort(key=lambda e: e["order_key"])
            for earlier, later in zip(events, events[1:]):
                self.before(
                    point(earlier, "retired"), point(later, "committed"), "conflicting member overtook earlier work"
                )

    def member(self, ref, worker):
        return self.members[self.tasks[ref]["cooperative_index"] * len(self.workers) + worker]

    def execution(self):
        slots = defaultdict(list)
        cores = defaultdict(list)
        for index, execution in self.executions.items():
            ref, block, slot = self.expected_executions[index]
            task = self.tasks[ref]
            worker = execution["worker_index"]
            require(
                (execution["task_ref"], execution["block_idx"], execution["subtask_slot"], execution["block_num"])
                == (ref, block, slot, task["block_num"]),
                "execution context mismatch",
            )
            require(
                worker in self.workers and self.workers[worker]["core_type"] == int(slot != 0),
                "execution worker/type mismatch",
            )
            require(
                (execution["cluster_index"], execution["cluster_lane"]) == self.positions[worker],
                "execution topology mismatch",
            )
            require(
                valid_source_index(
                    execution["pending_slot"],
                    2 if task["flags"] & MIX else 0,
                ),
                "invalid execution slot",
            )
            require(
                execution["publication_count"] == 1
                and execution["publication_before_cycles"]
                <= execution["publication_after_cycles"]
                <= execution["permit"][0],
                "invalid final payload publication interval",
            )
            published_before = self.causal.snapshot(
                ("payload_before", index), worker, execution["publication_before_cycles"]
            )
            published_after = self.causal.snapshot(
                ("payload_after", index), worker, execution["publication_after_cycles"]
            )
            self.causal.add(published_before, published_after)
            self.causal.add(published_after, point(execution, "permit"))
            self.payload_points[index] = published_before, published_after
            require(
                execution["ready_source"] in (0, 3),
                "invalid ready source",
            )
            self.chain(execution, "permit", "entry", "completion_consumed", "trace_published")
            if task["flags"] & MIX:
                require(
                    execution["contribution_before"] == execution["contribution_after"] == [0, 0]
                    and execution["contribution_ticket"]
                    == execution["contribution_worker"]
                    == execution["contribution_credits"]
                    == 0,
                    "Mix lane fabricated an individual completion contribution",
                )
            else:
                require(execution["contribution_worker"] == worker, "foreign subtask completion contributor")
                if self.controls[ref]["completion_backend"] == 0:
                    require(execution["contribution_credits"] == 0, "GM contribution carries counter credits")
                self.chain(
                    execution, "completion_consumed", "contribution_before", "contribution_after", "trace_published"
                )
            if execution["flags"] & EXECUTED:
                self.chain(execution, "entry", "returned", "visible", "completion_consumed")
                require(
                    execution["returned"][1] == execution["entry"][1] + 1
                    and execution["visible"][1] == execution["returned"][1] + 1,
                    "scheduler events appeared inside a kernel",
                )
            else:
                require(
                    task["flags"] & PREDICATE and execution["returned"] == execution["visible"] == [0, 0],
                    "false kernel or invalid predicate skip",
                )
            self.permit(execution, task)
            slots[(worker, execution["pending_slot"])].append(execution)
            cores[worker].append(execution)
        self.slot_lifetimes(slots, cores)
        self.execution_identities()

    def permit(self, execution, task):
        ref, worker = task["task_ref"], execution["worker_index"]
        is_mix, sync = bool(task["flags"] & MIX), bool(task["flags"] & SYNC)
        flags = (
            ENTRY
            | PUBLISHED
            | (execution["flags"] & EXECUTED)
            | (CLUSTER_GROUP if is_mix else 0)
            | (COOPERATIVE if sync else 0)
        )
        require(
            execution["order_key"] == self.ranks[ref]
            and execution["ready_source"] == (3 if is_mix else 0)
            and execution["flags"] == flags
            and execution["publication_count"] == 1
            and (
                0 < execution["group_seq"] < MAX and execution["group_seq"] % 2 == 1
                if is_mix
                else execution["group_seq"] == 0
            ),
            "invalid owner source, generation, rank, or permit flags",
        )
        self.check_opposite_source(ref, worker, execution, point(execution, "permit"))
        if sync:
            member = self.member(ref, worker)
            self.before(
                point(member, "committed"), point(execution, "permit"), "execution without sync member commitment"
            )
            self.before(
                point(execution, "completion_consumed"), point(member, "retired"), "member retired before return"
            )

    def slot_lifetimes(self, slots, cores):
        for executions in slots.values():
            executions.sort(key=lambda row: row["permit"][1])
            for previous, current in zip(executions, executions[1:]):
                require(
                    previous["trace_published"][0] <= current["publication_before_cycles"],
                    "payload overwrote a slot before trace retirement",
                )
                self.causal.add(point(previous, "trace_published"), self.payload_points[current["index"]][0])
                self.before(
                    point(previous, "completion_consumed"), point(current, "permit"), "slot reused before retirement"
                )
                if current["flags"] & COOPERATIVE and not current["flags"] & CLUSTER_GROUP:
                    owner = self.blocks[current["index"]]["owner"]
                    require(owner is not None, "missing irreversible claim")
                    self.before(
                        point(previous, "completion_consumed"),
                        point(owner, "claim_before"),
                        "pure slot claimed again before prior work retired",
                    )
        for executions in cores.values():
            executions.sort(key=lambda row: row["entry"][1])
            for previous, current in zip(executions, executions[1:]):
                self.before(
                    point(previous, "trace_published"),
                    point(current, "permit"),
                    "one Scalar ran overlapping kernels",
                )

    def execution_identities(self):
        require(isinstance(self.identities, list), "missing execution identity rows")
        found = set()
        for identity in self.identities:
            require(isinstance(identity, dict) and FIELDS <= identity.keys(), "incomplete execution identity")
            try:
                _validate_record({field: identity[field] for field in FIELDS})
            except ValueError as error:
                _error(str(error))
            index = identity["reg_task_id"]
            require(index in self.executions and index not in found, "execution identity coverage mismatch")
            found.add(index)
            execution = self.executions[index]
            # The existing swimlane core_id is the compact worker index, not the hardware topology ID.
            expected = dict(
                run_epoch=self.raw["run_epoch"],
                core_id=execution["worker_index"],
                reg_task_id=index,
                task_id=self.node_map[execution["task_ref"]]["public_task_id"]
                if self.node_map
                else execution["task_ref"],
                block_idx=execution["block_idx"],
                block_num=execution["block_num"],
                subtask_slot=execution["subtask_slot"],
                pending_slot=execution["pending_slot"],
                order_key=execution["order_key"],
                group_seq=execution["group_seq"] if execution["flags"] & CLUSTER_GROUP else None,
                group_kind=2 if execution["flags"] & CLUSTER_GROUP else 0,
                cluster_index=execution["cluster_index"],
                cluster_lane=execution["cluster_lane"],
                executed=bool(execution["flags"] & EXECUTED),
            )
            require(
                all(identity[field] == value for field, value in expected.items()),
                "protocol/execution identity join mismatch",
            )
        require(found == self.executions.keys(), "missing explicit execution identities")

    def block_owners(self):
        for index, block in self.blocks.items():
            ref, block_idx = self.expected_blocks[index]
            owner = block["owner"]
            require(owner is not None, "missing block owner")
            task = self.tasks[ref]
            self.record_identity(owner, ref)
            worker = owner["worker_index"]
            require(
                owner["block_idx"] == block_idx
                and owner["block_num"] == task["block_num"]
                and owner["active_mask"] == task["active_mask"]
                and owner["cluster_index"] == self.positions[worker][0],
                "block identity/owner shape mismatch",
            )
            self.chain(owner, "claim_before", "claim_after", "completion_before", "completion_after")
            self.task_blocks[ref].append(block)
            if task["flags"] & MIX:
                continue
            require(
                block["lanes"] == []
                and owner["group_seq"] == 0
                and owner["released_seq"] == [0, 0, 0]
                and all(owner[name] == [0, 0] for name in ("publish_before", "publish_after", "releases_observed")),
                "pure block has fake Mix events",
            )
            execution = self.executions[index]
            require(
                execution["worker_index"] == worker == owner["completion_worker"],
                "pure block migrated away from its claimant",
            )
            member = self.member(ref, worker)
            self.check_opposite_source(ref, worker, member, point(owner, "claim_before"))
            self.before(point(owner, "claim_after"), point(member, "committed"), "commitment before block claim")
            require(
                owner["claim_after"][0] <= execution["publication_before_cycles"], "pure payload published before claim"
            )
            self.before(
                point(execution, "visible"), point(owner, "completion_before"), "block completed before outputs visible"
            )
            self.before(
                point(execution, "completion_consumed"),
                point(owner, "completion_before"),
                "pure completion join before local retirement",
            )
            self.before(
                point(owner, "completion_after"),
                point(execution, "trace_published"),
                "pure trace published before completion contribution",
            )
        for ref, blocks in self.task_blocks.items():
            width = bin(self.tasks[ref]["active_mask"]).count("1")
            if self.controls[ref]["completion_backend"] == 1:
                require(
                    all(b["owner"]["completion_ticket"] == MAX for b in blocks),
                    "counter block fabricated a returned completion ticket",
                )
            else:
                by_ticket = sorted(blocks, key=lambda block: block["owner"]["completion_ticket"])
                require(
                    [b["owner"]["completion_ticket"] for b in by_ticket]
                    == list(range(0, self.tasks[ref]["block_num"] * width, width))
                    and all(b["owner"]["completion_credits"] == 0 for b in blocks),
                    "duplicate or missing block completion ticket",
                )
                self.ordered_intervals([b["owner"] for b in by_ticket], "completion_before", "completion_after")
            self.claim_certificates(ref, blocks)
            if self.tasks[ref]["flags"] & SYNC:
                owners = [b["owner"]["worker_index"] for b in blocks]
                require(len(set(owners)) == len(owners), "sync_start reused an execution position")

    def claim_certificates(self, ref, blocks):
        count = self.tasks[ref]["block_num"]
        seal = None
        claims = []
        for block in blocks:
            cert, owner = block["claim"], block["owner"]
            require(
                cert is not None
                and cert["run_epoch"] == self.raw["run_epoch"]
                and cert["worker_index"] == owner["worker_index"]
                and cert["block_idx"] == owner["block_idx"],
                "missing claim or foreign claim certificate",
            )
            allocation = self.causal.bounded(
                ("claim", ref, owner["block_idx"]), point(owner, "claim_before"), point(owner, "claim_after")
            )
            claims.append(allocation)
            if cert["seal_value"]:
                require(
                    seal is None and cert["confirmed_count"] == cert["seal_value"] == count,
                    "duplicate or incorrect final supply seal",
                )
                self.before(point(owner, "claim_after"), point(cert, "published_before"), "seal before winning claim")
                seal = self.causal.bounded(
                    ("seal", ref), point(cert, "published_before"), point(cert, "published_after")
                )
            else:
                require(
                    cert["confirmed_count"] == 0 and cert["published_before"] == cert["published_after"] == [0, 0],
                    "nonterminal claim reported a supply seal",
                )
        require(seal is not None, "missing last-claim supply seal")
        self.claim_seals[ref] = seal
        for allocation in claims:
            self.causal.add(allocation, seal)

    def verify_opposite_proofs(self):
        # Each worker needs the local retirement of a cohort it joined, or the
        # supply seal of a cohort it did not join. Build this conjunction once
        # per prefix and reuse it across MIX reads from the same worker.
        cohorts = ([], [])
        for ref in sorted(self.ordered, key=self.ranks.__getitem__):
            if not self.tasks[ref]["flags"] & MIX:
                cohorts[int(self.tasks[ref]["active_mask"] != 1)].append(ref)
        for worker, reads in self.opposite_reads.items():
            source = iter(cohorts[self.workers[worker]["core_type"]])
            earlier = next(source, None)
            prefix = None
            for required, when in sorted(reads, key=lambda read: read[0]):
                while earlier is not None and self.ranks[earlier] < required:
                    retired = (
                        point(self.member(earlier, worker), "retired")
                        if worker in self.task_members[earlier]
                        else self.claim_seals[earlier]
                    )
                    joined = ("sync prefix", worker, earlier)
                    self.causal.add(retired, joined)
                    if prefix is not None:
                        self.causal.add(prefix, joined)
                    prefix = joined
                    earlier = next(source, None)
                if prefix is not None:
                    self.causal.add(prefix, when)
        self.verify_mix_retirements()
        for event in self.raw["floor_publications"]:
            ref, cluster = event["prior_mix_ref"], event["cluster_index"]
            if ref == (1 << 32) - 1:
                continue
            groups = [b["owner"] for b in self.task_blocks[ref] if b["owner"]["cluster_index"] == cluster]
            for group in groups:
                self.before(
                    point(group, "publish_after"), point(event, "before"), "MIX floor covers unpublished local group"
                )
            if not groups or not self.tasks[ref]["flags"] & SYNC:
                self.causal.add(self.claim_seals[ref], point(event, "before"))

    def verify_mix_retirements(self):
        # A worker's sequenced events are totally ordered. Each read only needs
        # the latest retirement on that worker among MIX tasks of lower rank.
        latest = {}
        older_tasks = iter(self.mix_by_rank)
        older = next(older_tasks, None)
        for ref, worker, when in sorted(self.mix_reads, key=lambda read: self.ranks[read[0]]):
            while older is not None and self.ranks[older] < self.ranks[ref]:
                for block in self.task_blocks[older]:
                    owner = block["owner"]
                    for lane in block["lanes"]:
                        member = lane["worker_index"]
                        retired = (
                            point(owner, "completion_after")
                            if member == owner["completion_worker"]
                            else point(lane, "release_after")
                        )
                        if member not in latest or latest[member][2] < retired[2]:
                            latest[member] = retired
                older = next(older_tasks, None)
            if worker in latest:
                self.before(latest[worker], when, "self execution crossed an earlier active MIX obligation")

    def verify_self_exclusion(self):
        # The worker cannot claim pure sync work or enter any pure kernel while
        # it is in ARRIVED or RUNNING, regardless of the pure task's rank.
        if not self.mix_by_rank:
            return
        intervals = defaultdict(lambda: ([], []))
        for ref in self.mix_by_rank:
            for block in self.task_blocks[ref]:
                for lane in block["lanes"]:
                    starts, finishes = intervals[lane["worker_index"]]
                    starts.append(lane["arrive_before"][1])
                    finishes.append(lane["release_after"][1])
        for starts, finishes in intervals.values():
            starts.sort()
            finishes.sort()

        def check(worker, sequence):
            if worker in intervals:
                starts, finishes = intervals[worker]
                require(
                    bisect_right(starts, sequence) == bisect_right(finishes, sequence),
                    "self work during active MIX arrival-to-FIN interval",
                )

        for block in self.blocks.values():
            if not block["lanes"]:
                owner = block["owner"]
                check(owner["worker_index"], owner["claim_before"][1])
        for execution in self.executions.values():
            if not execution["group_seq"]:
                check(execution["worker_index"], execution["permit"][1])

    def completions(self):
        for ref, task in self.tasks.items():
            control = self.controls[ref]
            require(control["state"] == 2, "invalid terminal task state")
            require(
                control["next_block"]
                == (task["block_num"] if task["flags"] & EXECUTABLE and not task["flags"] & INLINE else 0),
                "terminal block counters mismatch",
            )
            total = (
                task["block_num"] * bin(task["active_mask"]).count("1")
                if task["flags"] & EXECUTABLE and not task["flags"] & INLINE
                else 0
            )
            require(control["completed_subtasks"] == total, "terminal subtask counter mismatch")
            require(
                control["canonical_total"] == (total or 1)
                and control["pending_compute"] == 0
                and control["completion_debt"] == 0,
                "invalid terminal canonical completion gate",
            )
            require(control["claim_seal"] == (task["block_num"] if total else 0), "invalid terminal claim seal")
            if not task["flags"] & EXECUTABLE and ref not in self.control_nodes:
                continue
            done = self.actors[ref]["completion"]
            require(done is not None, "missing TaskDone")
            self.actor(done, ref, completion=True)
            if task["flags"] & EXECUTABLE and not task["flags"] & INLINE:
                self.subtask_completion(task, done)
            else:
                require(
                    not done["flags"] & (DEFERRED_FINISH | COMPLETION_ALIAS_FLAGS),
                    "deferred or contribution completion on an inline/control task",
                )
            self.dependencies(task, done)

    def subtask_completion(self, task, done):
        if self.controls[task["task_ref"]]["completion_backend"] == 1:
            self.counter_completion(task, done)
            return
        if task["flags"] & MIX:
            last = max(self.task_blocks[task["task_ref"]], key=lambda block: block["owner"]["completion_ticket"])[
                "owner"
            ]
            if done["flags"] & DEFERRED_FINISH:
                self.before(
                    point(last, "completion_after"), point(done, "before"), "TaskDone before last group contribution"
                )
            else:
                require(done["flags"] & COMPLETION_FROM_GROUP, "TaskDone must alias its last group contribution")
            return
        count = task["block_num"] * bin(task["active_mask"]).count("1")
        executions = [self.executions[index] for index in range(task["trace_base"], task["trace_base"] + count)]
        by_ticket = sorted(executions, key=lambda execution: execution["contribution_ticket"])
        require(
            [execution["contribution_ticket"] for execution in by_ticket] == list(range(count)),
            "duplicate or missing subtask completion ticket",
        )
        self.ordered_intervals(by_ticket, "contribution_before", "contribution_after")
        last = by_ticket[-1]
        if done["flags"] & DEFERRED_FINISH:
            self.before(
                point(last, "contribution_after"), point(done, "before"), "TaskDone before last subtask contribution"
            )
        else:
            require(done["flags"] & COMPLETION_FROM_CONTRIBUTION, "TaskDone must alias its last subtask contribution")
            self.before(point(done, "after"), point(last, "trace_published"), "last subtask trace preceded TaskDone")

    def counter_completion(self, task, accounting):
        ref, width = task["task_ref"], bin(task["active_mask"]).count("1")
        count, ready = task["block_num"] * width, self.done_points[ref]
        if task["flags"] & MIX:
            for block in self.task_blocks[ref]:
                owner = block["owner"]
                require(owner["completion_credits"] == width, "invalid counter group physical credits")
                publication = self.causal.bounded(
                    ("counter_return", ref, owner["block_idx"]),
                    point(owner, "completion_before"),
                    point(owner, "completion_after"),
                )
                self.causal.add(publication, ready)
        for index in range(task["trace_base"], task["trace_base"] + count):
            execution = self.executions[index]
            may_defer = bool(self.controls[ref]["async_mask"] & (1 << execution["subtask_slot"]))
            if not task["flags"] & MIX:
                require(
                    execution["contribution_ticket"] == MAX
                    and execution["contribution_credits"] in ((1, 2) if may_defer else (1,)),
                    "counter execution fabricated a ticket or invalid credits",
                )
                publication = self.causal.bounded(
                    ("counter_return", ref, execution["block_idx"]),
                    point(execution, "contribution_before"),
                    point(execution, "contribution_after"),
                )
                self.causal.add(publication, ready)
                if index in self.blocks:
                    owner = self.blocks[index]["owner"]
                    require(
                        owner["completion_credits"] == execution["contribution_credits"],
                        "pure block counter credits disagree with its execution",
                    )
                    self.causal.add(point(owner, "completion_before"), publication)
                    self.causal.add(publication, point(owner, "completion_after"))
                if execution["contribution_credits"] == 2:
                    require(index not in self.conditions, "combined ADD2 duplicated a condition contribution")
                    continue
            if not may_defer:
                require(index not in self.conditions, "synchronous execution has a condition contribution")
                continue
            self.counter_condition(index, ready, task)
        self.causal.add(ready, point(accounting, "before"))

    def counter_condition(self, index, ready, task):
        require(index in self.conditions, "missing counter condition contribution")
        condition, execution = self.conditions[index], self.executions[index]
        require(
            condition["task_ref"] == task["task_ref"]
            and condition["block_idx"] == execution["block_idx"]
            and condition["subtask_slot"] == execution["subtask_slot"]
            and condition["credits"] == 1
            and condition["source"] in (0, 1),
            "invalid counter condition identity or credits",
        )
        if condition["source"] == 0:
            require(
                task["flags"] & MIX and condition["worker_index"] == execution["worker_index"],
                "empty counter condition has a foreign owner or non-Mix task",
            )
        self.before(point(execution, "entry"), point(condition, "before"), "counter condition precedes its execution")
        if condition["source"] == 1:
            self.before(
                point(execution, "returned"),
                point(condition, "before"),
                "counter condition precedes its execution return",
            )
        publication = self.causal.bounded(
            ("counter_condition", index), point(condition, "before"), point(condition, "after")
        )
        self.causal.add(publication, ready)
        self.used_conditions.add(index)

    def dependencies(self, task, done):
        ref = task["task_ref"]
        if task["flags"] & INLINE or ref in self.control_nodes:
            if done["flags"] & CONTROL_NOTIFY_ACCOUNTING:
                earliest = self.done_points[ref]
                self.causal.add(earliest, point(done, "before"))
            else:
                earliest = point(done, "before")
            boundaries = (earliest,)
        elif task["flags"] & MIX:
            boundaries = (point(lane, "arrive_before") for block in self.task_blocks[ref] for lane in block["lanes"])
        else:
            start = task["trace_base"]
            boundaries = (self.payload_points[index][0] for index in range(start, start + task["block_num"]))
        predecessors = [self.done_points[dep] for dep in task["dependencies"] if dep in self.done_points]
        for boundary in boundaries:
            for predecessor in predecessors:
                self.causal.add(predecessor, boundary)

    def audit(self):
        self.topology()
        self.graph()
        self.task_order()
        self.membership()
        self.execution()
        self.block_owners()
        from .native_mix_audit import audit_mix  # noqa: PLC0415

        audit_mix(self)
        self.verify_opposite_proofs()
        self.verify_self_exclusion()
        self.completions()
        require(set(self.conditions) == self.used_conditions, "counter condition coverage mismatch")
        require(
            all(row["valid"] == 1 and row["value"] == row["limit"] for row in self.raw["completion_counters"]),
            "counter observation is missing or not terminal",
        )
        self.causal.validate()
        return dict(
            self.sequence_result,
            protocol_verified=True,
            device_logical_tasks=len(self.actors),
            executions=len(self.executions),
            cooperative_tasks=len(self.ordered),
            member_obligations=len(self.members),
            blocks=len(self.blocks),
        )


def audit_native_protocol(section, identities):
    """Require a complete native protocol and a bijection with explicit kernel identities."""
    return NativeAudit(section, identities).audit()
