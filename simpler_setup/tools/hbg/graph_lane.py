# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The Graph Execution lane: one envelope per modular task, from prepare to last sub-task.

host_build_graph alone has this to draw. A modular task's body is expanded by the
Scheduler (a ``graph_prepare`` phase) and then runs as sub-tasks carrying their
parent in the id; tensormap_and_ringbuffer has neither concept, and its
``contribute_events`` returns nothing.
"""

from __future__ import annotations

from collections import defaultdict

from simpler_setup.tools._runtime_dispatch import normalize_task_id_int

from .task_id import Space, TaskId


def _decode_sub_task_id(task_id):
    """``(parent_id, local_id)`` for a materialized sub-task, else None.

    This filters a stream that also carries ids of other spaces, and each of those
    answers None rather than raising: only a space field no mint can produce raises,
    and that is a corrupt record wherever it is read.
    """
    raw = normalize_task_id_int(task_id)
    if raw is None:
        return None
    parsed = TaskId(raw)
    if parsed.space() is not Space.SUB_TASK:
        return None
    return parsed.parent_id(), parsed.local_id()


def _collect_instances(tasks, scheduler_phases, task_slice_start_us):  # noqa: PLR0912
    """Join sub-task rows to their outer GraphPrepare records.

    Grouping is keyed by ``(run_epoch, outer_task_id)``, not by the outer id
    alone. A graph re-executed in a later run reuses its task ids, so keying on
    the id alone would fold two runs' executions into one instance whose span
    covers both and whose row list is twice as long.
    """
    prepare_by_outer = defaultdict(list)
    dummy_rows = []
    for thread_idx, records in enumerate(scheduler_phases or []):
        for record in records:
            phase = record.get("phase")
            if phase == "graph_prepare":
                outer_task_id = normalize_task_id_int(record.get("task_id"))
                # A graph_prepare record names the outer modular task, which is always
                # GLOBAL. Testing the whole high word rather than the space alone is
                # the stricter check and the one wanted here: a GLOBAL id has a zero
                # parent and zero reserved bits too, so anything else in those bits is
                # a corrupt record rather than a task of another space.
                if outer_task_id is not None and (outer_task_id >> TaskId.PARENT_SHIFT) == 0:
                    prepare_by_outer[(record["run_epoch"], outer_task_id)].append(record)
            elif phase == "dummy_task":
                dummy_rows.append((record, thread_idx))

    rows_by_outer = defaultdict(list)
    for task in tasks:
        decoded = _decode_sub_task_id(task.get("task_id"))
        if decoded is not None:
            outer_task_id, task_index = decoded
            rows_by_outer[(task["run_epoch"], outer_task_id)].append((task, task_index))

    dummy_by_outer = defaultdict(list)
    for record, thread_idx in dummy_rows:
        decoded = _decode_sub_task_id(record.get("task_id"))
        if decoded is not None:
            outer_task_id, task_index = decoded
            dummy_by_outer[(record["run_epoch"], outer_task_id)].append((record, task_index, thread_idx))

    instances = []
    for group_key, prepare_records in prepare_by_outer.items():
        run_epoch, outer_task_id = group_key
        rows = rows_by_outer.get(group_key, [])
        aicpu_rows = dummy_by_outer.get(group_key, [])
        if not rows and not aicpu_rows:
            continue
        task_indices = {task_index for _, task_index in rows}
        task_indices.update(task_index for _, task_index, _ in aicpu_rows)
        starts = [
            task.get("dispatch_time_us", task_slice_start_us(task))
            if task.get("dispatch_time_us", -1) >= 0
            else task_slice_start_us(task)
            for task, _ in rows
        ]
        starts.extend(record["start_time_us"] for record, _, _ in aicpu_rows)
        ends = [
            task.get("finish_time_us", 0) if task.get("finish_time_us", 0) > 0 else task["end_time_us"]
            for task, _ in rows
        ]
        ends.extend(record["end_time_us"] for record, _, _ in aicpu_rows)
        prepare_start_us = min(record["start_time_us"] for record in prepare_records)
        instances.append(
            {
                "outer_task_id": outer_task_id,
                "run_epoch": run_epoch,
                "rows": rows,
                "aicpu_rows": aicpu_rows,
                "visible_task_indices": sorted(task_indices),
                "execution_start_us": min(starts),
                "execution_end_us": max(ends),
                "prepare_start_us": prepare_start_us,
                "prepare_end_us": max(record["end_time_us"] for record in prepare_records),
                "prepare_duration_us": sum(
                    record["end_time_us"] - record["start_time_us"] for record in prepare_records
                ),
                "prepare_slice_count": len(prepare_records),
            }
        )

    instances.sort(key=lambda instance: instance["prepare_start_us"])
    lane_finish_us = []
    for instance_idx, instance in enumerate(instances):
        start_us = instance["prepare_start_us"]
        lane_idx = next((idx for idx, finish_us in enumerate(lane_finish_us) if finish_us <= start_us), -1)
        if lane_idx < 0:
            lane_idx = len(lane_finish_us)
            lane_finish_us.append(0.0)
        lane_finish_us[lane_idx] = instance["execution_end_us"]
        instance["instance_idx"] = instance_idx
        instance["lane_idx"] = lane_idx
    return instances


def contribute_events(tasks, scheduler_phases, ctx):
    """Trace events for this runtime's own lane, empty when the capture has none.

    Every pid and tid here comes from ``ctx``; see LaneContext for why a
    contributor must not number its own lanes.
    """
    instances = _collect_instances(tasks, scheduler_phases, ctx.task_slice_start_us)
    if not instances:
        return []

    events = [
        {"args": {"name": "Graph Execution"}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": ctx.pid},
        {"args": {"sort_index": 1}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": ctx.pid},
    ]
    for lane_idx in sorted({instance["lane_idx"] for instance in instances}):
        events.append(
            {
                "args": {"name": f"Graph_{lane_idx}"},
                "cat": "__metadata",
                "name": "thread_name",
                "ph": "M",
                "pid": ctx.pid,
                "tid": ctx.tid_base + lane_idx,
            }
        )
    for instance in instances:
        outer_display = ctx.display(instance["outer_task_id"])
        task_indices = instance["visible_task_indices"]
        events.append(
            {
                "args": ctx.with_run_epoch(
                    {
                        "outer_task_id": instance["outer_task_id"],
                        "visible_sub_task_count": len(task_indices),
                        "visible_sub_task_local_id_min": min(task_indices),
                        "visible_sub_task_local_id_max": max(task_indices),
                        "prepare_slice_count": instance["prepare_slice_count"],
                        "prepare_duration_us": instance["prepare_duration_us"],
                        "execution_start_us": instance["execution_start_us"],
                        "execution_duration_us": instance["execution_end_us"] - instance["execution_start_us"],
                        "synthetic_id_layout": (
                            f"space{Space.SUB_TASK.value}:(outer_task_id << {TaskId.PARENT_SHIFT}) | sub_task_local_id"
                        ),
                    },
                    instance,
                ),
                "cat": "graph_execution",
                "cname": "rail_animation",
                "name": f"GraphExecution({outer_display}, {len(task_indices)} visible sub-tasks)",
                "ph": "X",
                "pid": ctx.pid,
                "tid": ctx.tid_base + instance["lane_idx"],
                "ts": instance["prepare_start_us"],
                "dur": instance["execution_end_us"] - instance["prepare_start_us"],
            }
        )
    return events
