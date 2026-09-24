# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dependency-only task markers this runtime drains on its AICPU.

``dummy_task`` and ``predicated_skip`` records are tasks that never reach an
AICore: a dependency-only node, and one whose predicate resolved false. They are
drawn on the Worker View rather than the scheduler track, because what they
represent is a DAG node briefly inhabiting the AICPU as a virtual worker.

Only this runtime emits either phase (see phases.py). hbg's counterpart
contributes nothing.
"""

from __future__ import annotations

from simpler_setup.tools._runtime_dispatch import normalize_task_id_int

from .phases import MARKER_PHASES

# A marker's recorded span can be zero-width; Perfetto drops a zero-duration
# slice, so the bar is floored to something visible.
_MARKER_MIN_DUR_US = 0.02


def contribute_worker_lane_events(records, thread_idx, ctx):
    """``(events, anchors)`` for the markers in one scheduler thread's records.

    An anchor is ``(run_epoch, task_id, row)``: the caller keys them by run,
    because a dependency arrow must land on the marker drawn for its own run.
    """
    events = []
    anchors = []
    tid = ctx.tid(thread_idx)
    for record in records:
        phase = record.get("phase")
        if phase not in MARKER_PHASES:
            continue
        start_us = record["start_time_us"]
        dur = max(record["end_time_us"] - start_us, _MARKER_MIN_DUR_US)
        task_id = normalize_task_id_int(record.get("task_id"))
        task_label = ctx.display(task_id) if task_id is not None else "unknown"
        if phase == "dummy_task":
            event_name = f"dummy({task_label})"
        else:
            event_name = ctx.event_name(task_id, task_label)
        event_args = {
            "loop_iter": record.get("loop_iter", 0),
            "task_id": task_id,
            "event-hint": event_name,
        }
        if phase == "dummy_task":
            event_args["phase"] = phase
        else:
            # A rendered predicated_skip is by definition the failing branch:
            # a passing predicate runs the task and records it as one.
            event_args["predicated_pass"] = False
        ctx.with_run_epoch(event_args, record)
        event_id = ctx.next_event_id()
        events.append(
            {
                "args": event_args,
                "cat": "event",
                "id": event_id,
                "name": event_name,
                "ph": "X",
                "pid": 4,
                "tid": tid,
                "ts": start_us,
                "dur": dur,
            }
        )
        if task_id is not None:
            anchors.append(
                (
                    record.get("run_epoch"),
                    task_id,
                    {
                        "task_id": task_id,
                        "run_epoch": record.get("run_epoch"),
                        "start_time_us": start_us,
                        "end_time_us": start_us + dur,
                        "receive_time_us": start_us,
                        "trace_tid": tid,
                        "event_id": event_id,
                    },
                )
            )
    return events, anchors


def marker_task_ids(scheduler_phases):
    """Which task ids this runtime recorded as each marker kind.

    Keyed by phase name -- ``dummy_task`` and ``predicated_skip`` -- because the
    orchestrator lane reads these to tell a dependency-only task from a
    submitted one when a task has no AICore row to speak for it.
    """
    by_kind = {phase: set() for phase in sorted(MARKER_PHASES)}
    for thread_records in scheduler_phases or []:
        for record in thread_records:
            phase = record.get("phase")
            if phase in by_kind:
                task_id = normalize_task_id_int(record.get("task_id"))
                if task_id is not None:
                    by_kind[phase].add(task_id)
    return by_kind
