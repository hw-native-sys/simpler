# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Synthetic wire fixtures; native captures separately validate the real recorder."""

from typing import Any

from test_hbg_native_protocol import capture

MAX = (1 << 64) - 1


class ProtocolBuilder:
    def __init__(
        self,
        shapes,
        clusters=2,
        epoch=0,
        sync_tasks=(),
        deferred_tasks=(),
        fixed_tasks=(),
    ):
        self.raw: dict[str, Any] = capture(epoch)
        self.clock = 0
        self.groups = [0] * clusters
        self.mix_floors = [0] * clusters
        self.identities: list[dict[str, Any]] = []
        self.deferred_tasks = set(deferred_tasks)
        raw = self.raw
        raw.update(
            task_count=len(shapes),
            expected_task_count=len(shapes),
            worker_count=clusters * 3,
            cluster_count=clusters,
            active_worker_count=clusters * 3,
            task_events=[],
            tasks=[],
            workers=[],
            task_controls=[],
            floor_publications=[],
            members=[],
            blocks=[],
            executions=[],
            resolved_task_count=0,
            inline_completed_count=0,
        )
        prefix = 0
        for ref, (mask, blocks, deps) in enumerate(shapes):
            flags = 0 if mask is None else 1
            if mask == 0:
                flags |= 32
            elif mask is not None:
                flags |= (4 if bin(mask).count("1") > 1 else 0) | (8 if blocks > 1 else 0)
            if ref in sync_tasks:
                flags |= 16
            if ref in fixed_tasks:
                flags |= 128
            raw["tasks"].append(
                dict(
                    task_ref=ref,
                    flags=flags,
                    active_mask=mask or (2 if mask == 0 else 0),
                    block_num=blocks if mask is not None else 0,
                    trace_base=prefix if mask is not None else 0,
                    kernel_ids=[0 if mask is None or mask & (1 << lane) else 65535 for lane in range(3)],
                    dependencies=list(deps),
                )
            )
            if mask:
                prefix += blocks * bin(mask).count("1")
            raw["task_controls"].append(
                dict(
                    task_ref=ref,
                    state=2 if mask is None else 0,
                    canonical_total=blocks * bin(mask).count("1") if mask else 1,
                    pending_compute=(blocks * bin(mask).count("1") if mask else 1) if mask is not None else 0,
                    completion_debt=(blocks * bin(mask).count("1") if mask else 1) if mask is not None else 0,
                    next_block=0,
                    completed_subtasks=0,
                    claim_seal=0,
                    completion_backend=0,
                    async_mask=0,
                    counter_index=MAX,
                    counter_threshold=0,
                    counter_limit=0,
                )
            )
            raw["inline_completed_count" if mask is None else "resolved_task_count"] += 1
        raw["trace_count"] = prefix or len(shapes)
        raw["cooperative_task_count"] = sum(bool(task["flags"] & (4 | 16)) for task in raw["tasks"])
        raw["mix_task_count"] = sum(bool(task["flags"] & 4) for task in raw["tasks"])
        raw["sync_mix_task_count"] = sum(bool(task["flags"] & 4 and task["flags"] & 16) for task in raw["tasks"])
        raw["member_count"] = raw["cooperative_task_count"] * raw["worker_count"]
        for worker in range(clusters * 3):
            raw["workers"].append(
                dict(
                    index=worker,
                    worker_index=worker,
                    context_worker_index=worker,
                    run_epoch=epoch,
                    next_sequence=0,
                    error_flags=0,
                    finished=1,
                    core_id=worker + 20,
                    core_type=int(worker % 3 != 0),
                    cluster_index=worker // 3,
                    cluster_workers=[worker // 3 * 3 + lane for lane in range(3)],
                )
            )
        mix, sync, compact = [0] * 3, [0] * 3, 0
        self.mix_refs = [t["task_ref"] for t in raw["tasks"] if t["flags"] & 4]
        self.mix_rows = {ref: i + 1 for i, ref in enumerate(self.mix_refs)}
        self.published_blocks = dict.fromkeys(self.mix_refs, 0)
        raw["floor_count"] = (len(self.mix_refs) + 1) * clusters if self.mix_refs else 0
        for t in raw["tasks"]:
            ref, flags, mask = t["task_ref"], t["flags"], t["active_mask"]
            order = ref
            t.update(
                schedule_rank=order,
                topological_level=0,
                bottom_level=0,
                cooperative_index=compact if flags & (4 | 16) else (1 << 32) - 1,
                required_floor=[0] * 3,
            )
            compact += int(bool(flags & (4 | 16)))
            physical = mask | 2 if flags & 4 else (6 if mask in (2, 4) else mask)
            if flags and not flags & 32:
                for lane in range(3):
                    if physical & (1 << lane):
                        t["required_floor"][lane] = sync[lane] if flags & 4 else mix[lane]
                        if flags & 4:
                            mix[lane] = order + 1
                        elif flags & 16:
                            sync[lane] = order + 1
        if self.mix_refs and self.mix_refs[0]:
            self.publish_floors(None)

    def publish_floors(self, prior):
        row = 0 if prior is None else self.mix_rows[prior]
        floor = self.mix_refs[row] if row < len(self.mix_refs) else len(self.raw["tasks"])
        clusters = self.raw["cluster_count"]
        for cluster in range(clusters):
            rec = dict(
                index=row * clusters + cluster,
                run_epoch=self.raw["run_epoch"],
                cluster_index=cluster,
                worker_index=cluster * 3 + 1,
                prior_mix_ref=(1 << 32) - 1 if prior is None else prior,
                floor=floor,
            )
            self.stamp(rec, "before")
            self.stamp(rec, "after")
            self.raw["floor_publications"].append(rec)
            self.mix_floors[cluster] = floor

    def proof(self, task, worker):
        spec = self.raw["tasks"][task]
        needed = spec["required_floor"][worker % 3]
        return dict(
            observed_mix_floor=(self.mix_floors[worker // 3] if needed and not spec["flags"] & 4 else 0),
            observed_self_floor=(task if needed and spec["flags"] & 4 else 0),
        )

    def commit_member(self, task, worker):
        row = dict(
            index=self.raw["tasks"][task]["cooperative_index"] * self.raw["worker_count"] + worker,
            run_epoch=self.raw["run_epoch"],
            task_ref=task,
            order_key=self.raw["tasks"][task]["schedule_rank"],
            worker_index=worker,
            retired=[0, 0],
            **self.proof(task, worker),
        )
        self.stamp(row, "committed")
        self.raw["members"].append(row)

    def retire_member(self, task, worker):
        row = next(m for m in self.raw["members"] if m["task_ref"] == task and m["worker_index"] == worker)
        self.stamp(row, "retired")

    def tick(self):
        self.clock += 10
        return self.clock

    def stamp(self, record, field):
        worker_id = (
            record["completion_worker"]
            if "completion_worker" in record and field in ("completion_before", "completion_after", "releases_observed")
            else record["worker_index"]
        )
        worker = self.raw["workers"][worker_id]
        worker["next_sequence"] += 1
        record[field] = [self.tick(), worker["next_sequence"]]

    def common(self, task, worker) -> dict[str, Any]:
        spec = self.raw["tasks"][task]
        return dict(
            run_epoch=self.raw["run_epoch"],
            task_ref=task,
            worker_index=worker,
            flags=spec["flags"],
            order_key=spec["schedule_rank"],
        )

    def actor(self, task, worker, name, *, contribution=None) -> dict[str, Any]:
        row: dict[str, Any] = next((r for r in self.raw["task_events"] if r["index"] == task), {})
        if not row:
            row = dict(index=task, completion=None)
            self.raw["task_events"].append(row)
        record = self.common(task, worker)
        row[name] = record
        if contribution is None:
            self.stamp(record, "before")
            self.stamp(record, "after")
        else:
            group = "completion_before" in contribution
            record["flags"] |= 1 << (29 if group else 30)
            prefix = "completion" if group else "contribution"
            record["before"] = contribution[f"{prefix}_before"].copy()
            record["after"] = contribution[f"{prefix}_after"].copy()
        return record

    def members(self, task):
        spec = self.raw["tasks"][task]
        if spec["flags"] & 4:
            return [w["index"] for w in self.raw["workers"] if spec["active_mask"] & (1 << (w["index"] % 3))]
        eligible = [
            w["index"]
            for w in self.raw["workers"]
            if spec["flags"] & 4 or (w["core_type"] == 0) == bool(spec["active_mask"] & 1)
        ]
        return eligible

    def complete_task(self, task, worker, *, contribution=None):
        if task in self.deferred_tasks:
            worker = (worker + 1) % len(self.raw["workers"])
            contribution = None
        record = self.actor(task, worker, "completion", contribution=contribution)
        if task in self.deferred_tasks:
            record["flags"] |= 1 << 31
        self.raw["task_controls"][task].update(state=2, pending_compute=0, completion_debt=0)
        return record

    def claim(self, task, block, worker) -> dict[str, Any]:
        spec = self.raw["tasks"][task]
        row: dict[str, Any] = dict(
            self.common(task, worker),
            block_idx=block,
            cluster_index=worker // 3,
            block_num=spec["block_num"],
            active_mask=spec["active_mask"],
            group_seq=0,
            completion_ticket=MAX,
            completion_credits=0,
            released_seq=[0, 0, 0],
            completion_worker=worker,
        )
        for name in (
            "claim_before",
            "claim_after",
            "publish_before",
            "publish_after",
            "completion_before",
            "completion_after",
            "releases_observed",
        ):
            row[name] = [0, 0]
        result: dict[str, Any] = dict(
            index=spec["trace_base"] + block * bin(spec["active_mask"]).count("1"), owner=row, lanes=[]
        )
        self.raw["blocks"].append(result)
        self.stamp(row, "claim_before")
        self.stamp(row, "claim_after")
        if spec["flags"] & 16 and not spec["flags"] & 4:
            self.commit_member(task, worker)
        control = self.raw["task_controls"][task]
        if not spec["flags"] & 128:
            control["next_block"] += 1
        certificate = dict(
            run_epoch=self.raw["run_epoch"],
            worker_index=worker,
            block_idx=block,
            confirmed_count=control["next_block"] if control["next_block"] == spec["block_num"] else 0,
            seal_value=0,
            published_before=[0, 0],
            published_after=[0, 0],
        )
        result["claim"] = certificate
        if control["next_block"] == spec["block_num"]:
            certificate["seal_value"] = spec["block_num"]
            self.stamp(certificate, "published_before")
            self.stamp(certificate, "published_after")
        if control["next_block"] == spec["block_num"]:
            control["claim_seal"] = spec["block_num"]
        return result

    def complete_block(self, block):
        owner = block["owner"]
        task = owner["task_ref"]
        control = self.raw["task_controls"][task]
        width = bin(owner["active_mask"]).count("1")
        self.stamp(owner, "completion_before")
        owner["completion_ticket"] = control["completed_subtasks"]
        control["completed_subtasks"] += width
        control["pending_compute"] -= width
        control["completion_debt"] -= width
        self.stamp(owner, "completion_after")
        if control["pending_compute"] == 0:
            self.complete_task(task, owner["completion_worker"], contribution=owner)

    def execute(self, task, block, slot, worker, *, owner=None, group=0, skip=False) -> dict[str, Any]:
        spec = self.raw["tasks"][task]
        mixed, cooperative = bool(spec["flags"] & 4), bool(spec["flags"] & 16)
        if not mixed and not cooperative and not spec["flags"] & 128:
            control = self.raw["task_controls"][task]
            control["next_block"] += 1
            if control["next_block"] == spec["block_num"]:
                control["claim_seal"] = spec["block_num"]
        pending = 4 + group // 2 % 2 if mixed else block % 2
        index = (
            spec["trace_base"]
            + block * bin(spec["active_mask"]).count("1")
            + bin(spec["active_mask"] & ((1 << slot) - 1)).count("1")
        )
        row: dict[str, Any] = dict(
            self.common(task, worker),
            index=index,
            block_idx=block,
            pending_slot=pending,
            subtask_slot=slot,
            ready_source=3 if mixed else 0,
            group_seq=group,
            block_num=spec["block_num"],
            cluster_index=worker // 3,
            cluster_lane=worker % 3,
            publication_count=1,
            publication_before_cycles=self.tick(),
            publication_after_cycles=self.tick(),
            **self.proof(task, worker),
        )
        row["flags"] = 1 | (0 if skip else 2) | 64 | (4 if cooperative else 0) | (128 if mixed else 0)
        for name in ("permit", "entry", "returned", "visible", "completion_consumed", "trace_published"):
            row[name] = [0, 0]
        row["contribution_credits"] = 0
        self.raw["executions"].append(row)
        self.stamp(row, "permit")
        self.stamp(row, "entry")
        if not skip:
            self.stamp(row, "returned")
            self.stamp(row, "visible")
        self.stamp(row, "completion_consumed")
        if mixed:
            row.update(
                contribution_before=[0, 0], contribution_after=[0, 0], contribution_ticket=0, contribution_worker=0
            )
        else:
            self.contribute(row, owner)
        self.stamp(row, "trace_published")
        if cooperative and not mixed:
            self.retire_member(task, worker)
        self.identities.append(
            dict(
                run_epoch=self.raw["run_epoch"],
                core_id=worker,
                reg_task_id=index,
                task_id=task,
                block_idx=block,
                block_num=spec["block_num"],
                subtask_slot=slot,
                pending_slot=pending,
                order_key=row["order_key"],
                group_seq=group or None,
                group_kind=2 if mixed else 0,
                cluster_index=worker // 3,
                cluster_lane=worker % 3,
                executed=not skip,
            )
        )
        return row

    def contribute(self, execution, block=None):
        task = execution["task_ref"]
        spec = self.raw["tasks"][task]
        control = self.raw["task_controls"][task]
        self.stamp(execution, "contribution_before")
        if block is not None:
            self.stamp(block["owner"], "completion_before")
        ticket = control["completed_subtasks"]
        control["completed_subtasks"] += 1
        control["pending_compute"] -= 1
        control["completion_debt"] -= 1
        execution.update(contribution_ticket=ticket, contribution_worker=execution["worker_index"])
        self.stamp(execution, "contribution_after")
        if block is not None:
            block["owner"]["completion_ticket"] = ticket
            self.stamp(block["owner"], "completion_after")
        if control["completed_subtasks"] == spec["block_num"] * bin(spec["active_mask"]).count("1"):
            self.complete_task(task, execution["worker_index"], contribution=execution)

    def mix_block(self, task, block, cluster, *, reserved=None) -> dict[str, Any]:
        workers = [cluster * 3 + lane for lane in range(3)]
        result = self.claim(task, block, workers[1]) if reserved is None else reserved
        owner = result["owner"]
        ordered = bool(owner["flags"] & 16)
        group = 2 * self.groups[cluster] + 1
        self.groups[cluster] += 1
        owner["completion_worker"] = workers[1] if owner["active_mask"] & 2 else workers[0]
        owner["group_seq"] = group
        self.stamp(owner, "publish_before")
        self.stamp(owner, "publish_after")
        self.published_blocks[task] += 1
        resources = owner["active_mask"]
        floors = self.raw["tasks"][task]["required_floor"]
        all_arrivals = any(floors[i] for i in range(3) if resources & (1 << i))
        for lane, worker in enumerate(workers):
            if not resources & (1 << lane):
                continue
            row = dict(
                self.common(task, worker),
                index=lane,
                block_idx=block,
                cluster_index=cluster,
                lane=lane,
                group_seq=group,
                block_num=owner["block_num"],
                active_mask=owner["active_mask"],
                resource_mask=resources,
                observed_self_floor=self.proof(task, worker)["observed_self_floor"],
                arrived_seq=[
                    group if resources & (1 << peer) and (peer == lane or all_arrivals) else 0 for peer in range(3)
                ],
                lane_subtasks=[i if owner["active_mask"] & (1 << i) else 255 for i in range(3)],
            )
            for name in (
                "descriptor_ready",
                "arrive_before",
                "arrive_after",
                "arrivals_observed",
                "release_before",
                "release_after",
            ):
                row[name] = [0, 0]
            result["lanes"].append(row)
            self.stamp(row, "descriptor_ready")
            if ordered:
                self.commit_member(task, worker)
            self.stamp(row, "arrive_before")
            self.stamp(row, "arrive_after")
        for row in result["lanes"]:
            self.stamp(row, "arrivals_observed")
            if owner["active_mask"] & (1 << row["lane"]):
                self.execute(task, block, row["lane"], row["worker_index"], group=group)
            self.stamp(row, "release_before")
            self.stamp(row, "release_after")
            if ordered:
                self.retire_member(task, row["worker_index"])
        owner["released_seq"] = [group if resources & (1 << peer) else 0 for peer in range(3)]
        self.stamp(owner, "releases_observed")
        self.complete_block(result)
        if self.published_blocks[task] == owner["block_num"]:
            self.publish_floors(task)
        return result

    def run(self, order=None):
        for task in range(len(self.raw["tasks"])) if order is None else order:
            spec = self.raw["tasks"][task]
            if spec["flags"] == 0:
                continue
            if spec["flags"] & 32:
                self.complete_task(task, 1)
                continue
            if spec["flags"] & (4 | 16):
                last = None
                for block in range(spec["block_num"]):
                    if spec["flags"] & 4:
                        last = self.mix_block(task, block, block % len(self.groups))
                    else:
                        worker = self.members(task)[block % len(self.members(task))]
                        last = self.claim(task, block, worker)
                        self.execute(
                            task,
                            block,
                            0 if spec["active_mask"] == 1 else spec["active_mask"].bit_length() - 1,
                            worker,
                            owner=last,
                        )
                assert last is not None
            else:
                slot = spec["active_mask"].bit_length() - 1
                eligible = self.members(task)
                for block in range(spec["block_num"]):
                    self.execute(task, block, slot, eligible[block % len(eligible)])
        return self.raw, self.identities
