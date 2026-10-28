# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Audit native publication intervals, observations and per-member obligations."""

from collections import defaultdict

from .execution_identity import FIELDS, _validate_record
from .native_order import CausalOrder
from .native_protocol import _error, audit_protocol_sequences

MAX = (1 << 64) - 1
EXECUTABLE, FANIN, MIX, SPMD, SYNC, INLINE, PREDICATE = (1, 2, 4, 8, 16, 32, 64)
ENTRY, EXECUTED, COOPERATIVE, GM_GROUP, PERMIT_TAIL, PERMIT_NO_LOG, PUBLISHED = (1, 2, 4, 8, 16, 32, 64)


def require(condition, message):
    if not condition:
        _error(message)


def point(record, field):
    cycles, sequence = record[field]
    require(sequence != 0, f"missing {field} for task {record.get('task_ref')}")
    return record["worker_index"], cycles, sequence


class NativeAudit:
    def __init__(self, section, identities):
        self.raw = section
        self.sequence_result = audit_protocol_sequences(section)
        self.causal = CausalOrder(section)
        self.before = self.causal.before
        self.identities = identities
        self.tasks = {row["task_ref"]: row for row in section["tasks"]}
        self.workers = {row["index"]: row for row in section["workers"]}
        self.controls = {row["task_ref"]: row for row in section["task_controls"]}
        self.actors = {row["index"]: row for row in section["task_events"]}
        self.admissions = {row["index"]: row for row in section["admissions"]}
        self.members = {row["index"]: row for row in section["members"]}
        self.blocks = {row["index"]: row for row in section["blocks"]}
        self.executions = {row["index"]: row for row in section["executions"]}
        self.orders = {ref: order for order, ref in enumerate(section["admission_log"])}
        self.expected_executions = {}
        self.expected_blocks = {}
        self.positions = {}
        self.task_members = {}
        self.task_blocks = defaultdict(list)
        self.admission_points = {}
        self.done_points = {}
        self.claim_points = {}
        self.close_reads = {}
        self.payload_points = {}

    def chain(self, record, *fields):
        points = [point(record, field) for field in fields]
        for left, right in zip(points, points[1:], strict=False):
            self.before(left, right, f"invalid local order {fields} for task {record['task_ref']}")

    def ordered_intervals(self, records, first, last):
        prior = None
        for record in records:
            key = (first, record["task_ref"], record["block_idx"])
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
        require(self.raw["active_worker_count"] == count == self.raw["scheduler_count"], "inactive scheduler topology")
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
        prefix = 0
        for ref, task in sorted(self.tasks.items()):
            flags, mask, blocks = task["flags"], task["active_mask"], task["block_num"]
            deps = task["dependencies"]
            require(len(set(deps)) == len(deps) and all(0 <= dep < ref for dep in deps), "invalid frozen dependencies")
            require(flags < 128, "unknown task flags")
            if flags == 0:
                require(
                    mask == blocks == task["trace_base"] == 0 and task["kernel_ids"] == [0, 0, 0],
                    "bad Host-completed task",
                )
                continue
            require(flags & EXECUTABLE and bool(flags & FANIN) == bool(deps), "invalid executable task metadata")
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
            active = mask.bit_count()
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
                if flags & (MIX | SPMD):
                    self.expected_blocks[base] = ref, block
                for offset, slot in enumerate(slot for slot in range(3) if mask & (1 << slot)):
                    self.expected_executions[base + offset] = ref, block, slot
            prefix += blocks * active
        require(
            prefix == self.raw["trace_count"] or (prefix == 0 and self.raw["trace_count"] == len(self.tasks)),
            "trace capacity mismatch",
        )
        device = {ref for ref, task in self.tasks.items() if task["flags"] & EXECUTABLE}
        require(set(self.actors) == device, "missing or extraneous logical completion record")
        for ref in device:
            done = self.actors[ref]["completion"]
            require(done is not None, "missing TaskDone publication")
            self.done_points[ref] = self.causal.bounded(("TaskDone", ref), point(done, "before"), point(done, "after"))
        require(
            self.raw["expected_task_count"] == len(self.tasks) and self.raw["resolved_task_count"] == len(device),
            "logical completion count mismatch",
        )
        require(self.raw["inline_completed_count"] == len(self.tasks) - len(device), "Host completion count mismatch")
        require(set(self.executions) == set(self.expected_executions), "missing or extraneous execution")
        require(set(self.blocks) == set(self.expected_blocks), "missing or extraneous block owner")

    def admission(self):
        cooperative = {ref for ref, task in self.tasks.items() if task["flags"] & (MIX | SPMD)}
        log = self.raw["admission_log"]
        require(len(log) == len(cooperative) and set(log) == cooperative, "admission log is not a task bijection")
        require(
            self.raw["commit_tail"] == len(log) and self.raw["commit_lock"] == 0, "uncommitted admission transaction"
        )
        require(
            self.raw["mix_task_count"] == sum(bool(task["flags"] & MIX) for task in self.tasks.values()),
            "Mix task count mismatch",
        )
        prior = None
        for ref in log:
            task, record = self.tasks[ref], self.admissions[ref]
            self.record_identity(record, ref)
            require(
                record["block_num"] == task["block_num"] and record["active_mask"] == task["active_mask"],
                "admission shape mismatch",
            )
            members = {
                index
                for index, worker in self.workers.items()
                if task["flags"] & MIX or (worker["core_type"] == 0) == bool(task["active_mask"] & 1)
            }
            words = [0] * ((len(self.workers) + 63) // 64)
            for member in members:
                words[member // 64] |= 1 << (member % 64)
            require(
                record["member_words"] == words and record["member_count"] == len(members),
                "frozen admission members mismatch",
            )
            self.task_members[ref] = members
            actor = self.actors[ref]["admission"]
            require(actor is not None, "missing admission publication")
            self.actor(actor, ref)
            self.admission_points[ref] = self.causal.bounded(
                ("admission", ref), point(actor, "before"), point(actor, "after")
            )
            if prior is not None:
                self.before(point(prior, "after"), point(actor, "before"), "overlapping serialized admission commits")
            prior = actor
        self.admission_storage(cooperative)

    def admission_storage(self, cooperative):
        if not self.admissions:
            return
        stride = ((64 + ((len(self.workers) + 63) // 64) * 8 + 127) // 128) * 128
        first = self.admissions[0]["member_words_offset"]
        require(first >= 64 and first % 128 == 64, "misaligned admission bitmap")
        seals = []
        for ref, record in sorted(self.admissions.items()):
            require(record["member_words_offset"] == first + ref * stride, "wrong task bitmap binding")
            if ref not in cooperative:
                require(
                    record["run_epoch"] == 0
                    and record["task_ref"] == -1
                    and record["order_key"] == MAX
                    and record["block_num"] == record["member_count"] == record["active_mask"] == record["flags"] == 0
                    and not any(record["member_words"]),
                    "ordinary task has a cooperative admission",
                )
            if self.tasks[ref]["flags"] & MIX:
                require(len(record["close_seals"]) == self.raw["cluster_count"], "missing cluster close seals")
                seals.append(record["close_seals_offset"])
            else:
                require(
                    record["close_seals_offset"] == 0 and record["close_seals"] == [], "non-Mix task has close seals"
                )
        if seals:
            require(seals[0] > 0 and seals[0] % 128 == 0, "invalid close seal base")
            require(
                seals == [seals[0] + index * self.raw["cluster_count"] * 8 for index in range(len(seals))],
                "aliased or wrong task close seals",
            )

    def record_identity(self, record, ref):
        require(
            record["task_ref"] == ref
            and record["run_epoch"] == self.raw["run_epoch"]
            and record["flags"] == self.tasks[ref]["flags"]
            and record["order_key"] == self.orders.get(ref, MAX),
            "task/order/epoch/flags identity mismatch",
        )

    def actor(self, actor, ref):
        self.record_identity(actor, ref)
        require(actor["worker_index"] in self.workers, "unknown task actor")
        self.chain(actor, "before", "after")

    def membership(self):
        count = len(self.workers)
        expected = {
            self.orders[ref] * count + worker for ref, members in self.task_members.items() for worker in members
        }
        require(set(self.members) == expected, "member observation/closure coverage mismatch")
        frontiers = defaultdict(list)
        for index, member in self.members.items():
            order, worker = divmod(index, count)
            ref = self.raw["admission_log"][order]
            self.record_identity(member, ref)
            require(
                member["worker_index"] == worker and member["pending_mask"] == 0,
                "member closed with pending work or wrong owner",
            )
            require(order < member["observed_tail"] <= len(self.orders), "invalid observed admission prefix")
            self.chain(member, "observed", "closed")
            publication = self.admission_points[self.raw["admission_log"][member["observed_tail"] - 1]]
            self.causal.add(publication, point(member, "observed"))
            if not self.tasks[ref]["flags"] & MIX:
                require(
                    member["close_seal"] == member["last_group_seq"] == 0,
                    "pure member closed with a Mix seal",
                )
            frontiers[worker].append(member)
        for members in frontiers.values():
            members.sort(key=lambda row: row["order_key"])
            for previous, current in zip(members, members[1:], strict=False):
                self.before(
                    point(previous, "closed"), point(current, "observed"), "member overtook an earlier obligation"
                )

    def member(self, ref, worker):
        require(worker in self.task_members[ref], "execution/owner is not an admitted member")
        return self.members[self.orders[ref] * len(self.workers) + worker]

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
                execution["pending_slot"] < 2 and 0 < execution["generation"] < 1 << 32,
                "invalid execution slot/generation",
            )
            require(
                0 < execution["publication_count"] < 1 << 32
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
                execution["ready_source"] in (0, 1, 2, 3)
                and execution["publication_mode"] in (0, 1)
                and (execution["publication_mode"] == 0 or execution["ready_source"] == 2),
                "invalid source/publication mode",
            )
            self.chain(execution, "permit", "entry", "completion_consumed", "trace_published")
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
        cooperative, mixed = bool(task["flags"] & (MIX | SPMD)), bool(task["flags"] & MIX)
        flags = ENTRY | PUBLISHED | (EXECUTED if execution["flags"] & EXECUTED else 0)
        if cooperative:
            flags |= COOPERATIVE | (GM_GROUP if mixed else 0)
            require(
                execution["order_key"] == self.orders[ref]
                and execution["permit_tail"] == MAX
                and execution["publication_count"] == 1
                and execution["ready_source"] == 3
                and execution["publication_mode"] == 0,
                "invalid cooperative permit or republished payload",
            )
            member = self.member(ref, worker)
            self.before(
                point(member, "observed"), point(execution, "permit"), "execution before observing its obligation"
            )
            self.before(point(execution, "trace_published"), point(member, "closed"), "execution after member closure")
        else:
            flags |= PERMIT_TAIL if self.orders else PERMIT_NO_LOG
            require(
                execution["order_key"] == MAX and execution["ready_source"] != 3,
                "ordinary execution used a cooperative identity",
            )
            tail = execution["permit_tail"]
            require(tail <= len(self.orders), "ordinary permit names a future prefix")
            permit = point(execution, "permit")
            read = ("permit_read", execution["index"])
            self.causal.add(("event", worker, permit[2] - 1), read)
            self.causal.add(self.payload_points[execution["index"]][1], read)
            self.causal.add(read, permit)
            if tail:
                self.causal.add(self.admission_points[self.raw["admission_log"][tail - 1]], read)
            if tail < len(self.orders):
                self.causal.add(read, self.admission_points[self.raw["admission_log"][tail]])
            for member in self.members.values():
                if member["worker_index"] != worker:
                    continue
                if member["order_key"] < tail:
                    self.before(
                        point(member, "closed"), point(execution, "permit"), "ordinary execution bypassed an obligation"
                    )
                else:
                    require(
                        member["observed"][1] > execution["permit"][1],
                        "ordinary permit predates an already observed prefix",
                    )
        require(execution["flags"] == flags, "execution flags disagree with task/permit")
        require(
            (0 < execution["group_seq"] < MAX) if mixed else execution["group_seq"] == 0,
            "invalid execution group identity",
        )

    def slot_lifetimes(self, slots, cores):
        for executions in slots.values():
            executions.sort(key=lambda row: row["permit"][1])
            for previous, current in zip(executions, executions[1:], strict=False):
                require(previous["generation"] < current["generation"], "slot generation reused or reversed")
                require(
                    previous["completion_consumed"][0] <= current["publication_before_cycles"],
                    "payload overwrote an unconsumed slot",
                )
                self.causal.add(point(previous, "completion_consumed"), self.payload_points[current["index"]][0])
                self.before(
                    point(previous, "completion_consumed"), point(current, "permit"), "slot reused before retirement"
                )
                if current["flags"] & COOPERATIVE and not current["flags"] & GM_GROUP:
                    owner = self.blocks[current["index"]]["owner"]
                    require(owner is not None, "missing irreversible claim")
                    self.before(
                        point(previous, "completion_consumed"),
                        point(owner, "claim_before"),
                        "pure slot claimed again before prior work retired",
                    )
        for executions in cores.values():
            executions.sort(key=lambda row: row["entry"][1])
            for previous, current in zip(executions, executions[1:], strict=False):
                self.before(
                    point(previous, "completion_consumed"),
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
                task_id=execution["task_ref"],
                block_idx=execution["block_idx"],
                block_num=execution["block_num"],
                subtask_slot=execution["subtask_slot"],
                pending_slot=execution["pending_slot"],
                dispatch_generation=execution["generation"],
                order_key=execution["order_key"] if execution["flags"] & COOPERATIVE else None,
                group_seq=execution["group_seq"] if execution["flags"] & GM_GROUP else None,
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
            member = self.member(ref, worker)
            require(
                owner["block_idx"] == block_idx
                and owner["block_num"] == task["block_num"]
                and owner["active_mask"] == task["active_mask"]
                and owner["cluster_index"] == self.positions[worker][0],
                "block identity/owner shape mismatch",
            )
            self.chain(owner, "claim_before", "claim_after", "completion_before", "completion_after")
            self.before(point(member, "observed"), point(owner, "claim_before"), "block claimed before obligation")
            self.before(
                point(owner, "completion_after"), point(member, "closed"), "owner closed with an unretired block"
            )
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
            require(execution["worker_index"] == worker, "pure block migrated away from its claimant")
            self.before(point(owner, "claim_after"), point(execution, "permit"), "unclaimed pure execution")
            require(
                owner["claim_after"][0] <= execution["publication_before_cycles"], "pure payload published before claim"
            )
            self.before(
                point(execution, "visible"), point(owner, "completion_before"), "block completed before outputs visible"
            )
            self.before(
                point(owner, "completion_after"),
                point(execution, "completion_consumed"),
                "pure completion ticket after retirement",
            )
        for ref, blocks in self.task_blocks.items():
            by_claim = sorted(blocks, key=lambda block: block["owner"]["block_idx"])
            self.ordered_intervals([b["owner"] for b in by_claim], "claim_before", "claim_after")
            by_ticket = sorted(blocks, key=lambda block: block["owner"]["completion_ticket"])
            require(
                [b["owner"]["completion_ticket"] for b in by_ticket] == list(range(self.tasks[ref]["block_num"])),
                "duplicate or missing block completion ticket",
            )
            self.ordered_intervals([b["owner"] for b in by_ticket], "completion_before", "completion_after")
            owned = defaultdict(list)
            for block in blocks:
                owned[block["owner"]["worker_index"]].append(block["owner"]["block_idx"])
            if self.tasks[ref]["flags"] & SYNC:
                require(all(len(indices) == 1 for indices in owned.values()), "sync_start reused an execution position")
            for worker in self.task_members[ref]:
                if self.tasks[ref]["flags"] & MIX and self.positions[worker][1]:
                    continue
                self.close_cursor(ref, worker, owned[worker])

    def close_cursor(self, ref, worker, owned):
        task, member = self.tasks[ref], self.member(ref, worker)
        cursor, blocks = member["closed_cursor"], task["block_num"]
        require(
            0 < cursor <= blocks
            and all(block < cursor for block in owned)
            and (cursor == blocks or (task["flags"] & SYNC and owned)),
            "invalid close cursor: unclaimed work without a retired sync_start position",
        )
        closed = point(member, "closed")
        read = ("close_cursor_read", ref, worker)
        self.close_reads[(ref, worker)] = read
        self.causal.add(("event", worker, closed[2] - 1), read)
        self.causal.add(read, closed)
        self.causal.add(self.claim_points[(ref, cursor - 1)], read)
        if cursor < blocks:
            self.causal.add(read, self.claim_points[(ref, cursor)])

    def completions(self):
        for ref, task in self.tasks.items():
            cooperative = ref in self.orders
            control = self.controls[ref]
            require(
                control["state"] == 2 and control["admission_state"] == (3 if cooperative else 0),
                "invalid terminal task state",
            )
            require(
                control["next_block"] == control["completed_blocks"] == (task["block_num"] if cooperative else 0),
                "terminal block counters mismatch",
            )
            if self.raw["schema"] == "hbg-native-protocol-v5":
                require(control["admission_claim_lock"] == 0, "invalid terminal admission claim lock")
            if self.raw["schema"] != "hbg-native-protocol-v1":
                positions = self.raw["cluster_count"] * (2 if task["active_mask"] in (2, 4) else 1)
                minimum_blocks = (
                    2
                    if self.raw["schema"] in ("hbg-native-protocol-v4", "hbg-native-protocol-v5")
                    else positions * 2 + (1 if self.raw["schema"] == "hbg-native-protocol-v2" else 0)
                )
                uses_tickets = cooperative and not task["flags"] & SYNC and task["block_num"] >= minimum_blocks
                tickets = control["claim_tickets"]
                require(
                    task["block_num"] <= tickets <= task["block_num"] + positions if uses_tickets else tickets == 0,
                    "invalid terminal claim ticket counter",
                )
            if not task["flags"] & EXECUTABLE:
                continue
            done = self.actors[ref]["completion"]
            require(done is not None, "missing TaskDone")
            self.actor(done, ref)
            if cooperative:
                last = max(self.task_blocks[ref], key=lambda block: block["owner"]["completion_ticket"])
                owner = last["owner"]
                require(owner["worker_index"] == done["worker_index"], "TaskDone not owned by last contributor")
                self.before(
                    point(owner, "completion_after"), point(done, "before"), "TaskDone before last contribution"
                )
                if not task["flags"] & MIX:
                    self.before(
                        point(self.executions[last["index"]], "trace_published"),
                        point(done, "before"),
                        "last pure trace unpublished at TaskDone",
                    )
            else:
                require(self.actors[ref]["admission"] is None, "ordinary/service task has an admission event")
                if not task["flags"] & INLINE:
                    execution = self.executions[task["trace_base"]]
                    require(execution["worker_index"] == done["worker_index"], "ordinary TaskDone wrong worker")
                    self.before(
                        point(execution, "completion_consumed"),
                        point(done, "before"),
                        "ordinary TaskDone before consumption",
                    )
                    if self.orders:
                        self.before(
                            point(execution, "trace_published"),
                            point(done, "before"),
                            "ordered ordinary TaskDone before trace",
                        )
                    else:
                        self.before(
                            point(done, "after"),
                            point(execution, "trace_published"),
                            "ordinary trace/TaskDone order mismatch",
                        )
            self.dependencies(task, done)

    def dependencies(self, task, done):
        ref = task["task_ref"]
        if ref in self.orders:
            earliest = point(self.actors[ref]["admission"], "before")
        elif task["flags"] & INLINE:
            earliest = point(done, "before")
        else:
            execution = self.executions[task["trace_base"]]
            earliest = self.payload_points[execution["index"]][0]
        for dep in task["dependencies"]:
            if self.tasks[dep]["flags"] & EXECUTABLE:
                self.causal.add(self.done_points[dep], earliest)

    def audit(self):
        self.topology()
        self.graph()
        self.admission()
        self.membership()
        self.execution()
        self.block_owners()
        from .native_mix_audit import audit_mix  # noqa: PLC0415

        audit_mix(self)
        self.completions()
        self.causal.validate()
        return dict(
            self.sequence_result,
            protocol_verified=True,
            device_logical_tasks=len(self.actors),
            executions=len(self.executions),
            cooperative_tasks=len(self.orders),
            member_obligations=len(self.members),
            blocks=len(self.blocks),
        )


def audit_native_protocol(section, identities):
    """Require a complete native protocol and a bijection with explicit kernel identities."""
    return NativeAudit(section, identities).audit()
