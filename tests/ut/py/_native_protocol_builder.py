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
    def __init__(self, shapes, clusters=2, epoch=0, sync_tasks=()):
        self.raw: dict[str, Any] = capture(epoch)
        self.clock = 0
        self.groups = [0] * clusters
        self.generations = {}
        self.identities: list[dict[str, Any]] = []
        raw = self.raw
        raw.update(
            task_count=len(shapes),
            expected_task_count=len(shapes),
            worker_count=clusters * 3,
            cluster_count=clusters,
            active_worker_count=clusters * 3,
            scheduler_count=clusters * 3,
            task_events=[],
            tasks=[],
            workers=[],
            task_controls=[],
            admissions=[],
            admission_log=[],
            members=[],
            blocks=[],
            executions=[],
            resolved_task_count=0,
            inline_completed_count=0,
        )
        prefix = 0
        for ref, (mask, blocks, deps) in enumerate(shapes):
            flags = 0 if mask is None else 1 | (2 if deps else 0)
            if mask == 0:
                flags |= 32
            elif mask is not None:
                flags |= (4 if mask.bit_count() > 1 else 0) | (8 if blocks > 1 else 0)
            if ref in sync_tasks:
                flags |= 16
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
                prefix += blocks * mask.bit_count()
            raw["task_controls"].append(
                dict(task_ref=ref, state=2, next_block=0, completed_blocks=0, admission_state=0)
            )
            raw["inline_completed_count" if mask is None else "resolved_task_count"] += 1
        raw["trace_count"] = prefix or len(shapes)
        raw["cooperative_task_count"] = sum(bool(task["flags"] & 12) for task in raw["tasks"])
        raw["mix_task_count"] = sum(bool(task["flags"] & 4) for task in raw["tasks"])
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
        if raw["cooperative_task_count"]:
            mix_index = 0
            stride = ((64 + 8 * ((raw["worker_count"] + 63) // 64) + 127) // 128) * 128
            for task in raw["tasks"]:
                mixed = bool(task["flags"] & 4)
                raw["admissions"].append(
                    dict(
                        index=task["task_ref"],
                        run_epoch=0,
                        task_ref=-1,
                        order_key=MAX,
                        block_num=0,
                        member_count=0,
                        active_mask=0,
                        flags=0,
                        member_words_offset=4096 + task["task_ref"] * stride + 64,
                        close_seals_offset=8192 + mix_index * clusters * 8 if mixed else 0,
                        member_words=[0] * ((raw["worker_count"] + 63) // 64),
                        close_seals=[0] * clusters if mixed else [],
                    )
                )
                mix_index += mixed

    def tick(self):
        self.clock += 10
        return self.clock

    def stamp(self, record, field):
        worker = self.raw["workers"][record["worker_index"]]
        worker["next_sequence"] += 1
        record[field] = [self.tick(), worker["next_sequence"]]

    def common(self, task, worker) -> dict[str, Any]:
        spec = self.raw["tasks"][task]
        return dict(
            run_epoch=self.raw["run_epoch"],
            task_ref=task,
            worker_index=worker,
            flags=spec["flags"],
            order_key=self.raw["admission_log"].index(task) if spec["flags"] & 12 else MAX,
        )

    def actor(self, task, worker, name) -> dict[str, Any]:
        row: dict[str, Any] = next((r for r in self.raw["task_events"] if r["index"] == task), {})
        if not row:
            row = dict(index=task, admission=None, completion=None)
            self.raw["task_events"].append(row)
        record = self.common(task, worker)
        row[name] = record
        self.stamp(record, "before")
        self.stamp(record, "after")
        return record

    def members(self, task):
        spec = self.raw["tasks"][task]
        return [
            w["index"]
            for w in self.raw["workers"]
            if spec["flags"] & 4 or (w["core_type"] == 0) == bool(spec["active_mask"] & 1)
        ]

    def admit(self, task, worker=0, observe_members=True):
        spec = self.raw["tasks"][task]
        self.raw["admission_log"].append(task)
        self.raw["commit_tail"] += 1
        record = self.raw["admissions"][task]
        record.update(self.common(task, worker))
        record.pop("worker_index")
        record.update(
            block_num=spec["block_num"], active_mask=spec["active_mask"], member_count=len(self.members(task))
        )
        for member in self.members(task):
            record["member_words"][member // 64] |= 1 << (member % 64)
        self.raw["task_controls"][task]["admission_state"] = 3
        self.actor(task, worker, "admission")
        for member in self.members(task):
            row = dict(
                self.common(task, member),
                index=record["order_key"] * self.raw["worker_count"] + member,
                observed=[0, 0],
                closed=[0, 0],
                observed_tail=self.raw["commit_tail"],
                closed_cursor=0,
                close_seal=0,
                last_group_seq=0,
                pending_mask=0,
            )
            self.raw["members"].append(row)
            if observe_members:
                self.stamp(row, "observed")

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
            released_seq=[0, 0, 0],
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
        result = dict(index=spec["trace_base"] + block * spec["active_mask"].bit_count(), owner=row, lanes=[])
        self.raw["blocks"].append(result)
        self.stamp(row, "claim_before")
        self.stamp(row, "claim_after")
        self.raw["task_controls"][task]["next_block"] += 1
        return result

    def complete_block(self, block):
        owner = block["owner"]
        control = self.raw["task_controls"][owner["task_ref"]]
        self.stamp(owner, "completion_before")
        owner["completion_ticket"] = control["completed_blocks"]
        control["completed_blocks"] += 1
        self.stamp(owner, "completion_after")

    def execute(self, task, block, slot, worker, *, owner=None, group=0, skip=False) -> dict[str, Any]:
        spec = self.raw["tasks"][task]
        mixed, cooperative = bool(spec["flags"] & 4), bool(spec["flags"] & 12)
        pending = block % 2
        key = worker, pending
        self.generations[key] = self.generations.get(key, 0) + 1
        index = (
            spec["trace_base"]
            + block * spec["active_mask"].bit_count()
            + (spec["active_mask"] & ((1 << slot) - 1)).bit_count()
        )
        row: dict[str, Any] = dict(
            self.common(task, worker),
            index=index,
            block_idx=block,
            generation=self.generations[key],
            pending_slot=pending,
            subtask_slot=slot,
            ready_source=3 if cooperative else 0,
            group_seq=group,
            block_num=spec["block_num"],
            cluster_index=worker // 3,
            cluster_lane=worker % 3,
            publication_mode=0,
            publication_count=1,
            publication_before_cycles=self.tick(),
            publication_after_cycles=self.tick(),
            permit_tail=MAX if cooperative else self.raw["commit_tail"],
        )
        row["flags"] = (
            1
            | (0 if skip else 2)
            | 64
            | (4 if cooperative else 16 if self.raw["cooperative_task_count"] else 32)
            | (8 if mixed else 0)
        )
        for name in ("permit", "entry", "returned", "visible", "completion_consumed", "trace_published"):
            row[name] = [0, 0]
        self.raw["executions"].append(row)
        self.stamp(row, "permit")
        self.stamp(row, "entry")
        if not skip:
            self.stamp(row, "returned")
            self.stamp(row, "visible")
        if cooperative and not mixed:
            self.complete_block(owner)
        self.stamp(row, "completion_consumed")
        if not cooperative and not self.raw["cooperative_task_count"]:
            self.actor(task, worker, "completion")
        self.stamp(row, "trace_published")
        if not cooperative and self.raw["cooperative_task_count"]:
            self.actor(task, worker, "completion")
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
                dispatch_generation=self.generations[key],
                order_key=row["order_key"] if cooperative else None,
                group_seq=group or None,
                cluster_index=worker // 3,
                cluster_lane=worker % 3,
                executed=not skip,
            )
        )
        return row

    def mix_block(self, task, block, cluster) -> dict[str, Any]:
        workers = [cluster * 3 + lane for lane in range(3)]
        result = self.claim(task, block, workers[0])
        owner = result["owner"]
        self.groups[cluster] += 1
        group = self.groups[cluster]
        owner["group_seq"] = group
        self.stamp(owner, "publish_before")
        self.stamp(owner, "publish_after")
        for lane, worker in enumerate(workers):
            row = dict(
                self.common(task, worker),
                index=lane,
                block_idx=block,
                cluster_index=cluster,
                lane=lane,
                group_seq=group,
                block_num=owner["block_num"],
                active_mask=owner["active_mask"],
                resource_mask=7,
                arrived_seq=[group] * 3,
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
            for name in ("descriptor_ready", "arrive_before", "arrive_after"):
                self.stamp(row, name)
        for row in result["lanes"]:
            self.stamp(row, "arrivals_observed")
            if owner["active_mask"] & (1 << row["lane"]):
                self.execute(task, block, row["lane"], row["worker_index"], group=group)
            self.stamp(row, "release_before")
            self.stamp(row, "release_after")
        owner["released_seq"] = [group] * 3
        self.stamp(owner, "releases_observed")
        self.complete_block(result)
        return result

    def close(self, task, only_worker=None):
        spec = self.raw["tasks"][task]
        for row in self.raw["members"]:
            if row["task_ref"] != task:
                continue
            worker = row["worker_index"]
            if only_worker is not None and worker != only_worker:
                continue
            row["closed_cursor"] = self.raw["task_controls"][task]["next_block"]
            if spec["flags"] & 4:
                groups = [
                    b["owner"]["group_seq"]
                    for b in self.raw["blocks"]
                    if b["owner"]["task_ref"] == task and b["owner"]["cluster_index"] == worker // 3
                ]
                row["last_group_seq"] = max(groups, default=0)
                row["close_seal"] = row["last_group_seq"] + 1
                self.raw["admissions"][task]["close_seals"][worker // 3] = row["close_seal"]
                if worker % 3:
                    row["closed_cursor"] = MAX
            self.stamp(row, "closed")

    def run(self, order=None):
        for task in range(len(self.raw["tasks"])) if order is None else order:
            spec = self.raw["tasks"][task]
            if spec["flags"] == 0:
                continue
            if spec["flags"] & 32:
                self.actor(task, 1, "completion")
                continue
            if spec["flags"] & 12:
                self.admit(task)
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
                self.actor(task, last["owner"]["worker_index"], "completion")
                self.close(task)
            else:
                slot = spec["active_mask"].bit_length() - 1
                self.execute(task, 0, slot, 0 if slot == 0 else 1)
        return self.raw, self.identities
