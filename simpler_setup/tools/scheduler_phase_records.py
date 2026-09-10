# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared scheduler-phase classification for profiling tools."""

import bisect

SCHED_OUTER_PHASES = (
    "complete",
    "async_poll",
    "state_probe",
    "dispatch",
    "worksteal",
    "refill",
    "release",
    "dummy",
    "early_dispatch",
    "drain",
    "graph_prepare",
)

_SCHEDULER_WORK_PHASES = frozenset(
    {
        "complete",
        "state_probe",
        "dispatch",
        "worksteal",
        "refill",
        "release",
        "early_dispatch",
        "drain",
        "graph_prepare",
    }
)
_RESOLUTION_WORK_PHASES = frozenset({"resolve", "resolve_standalone", "async_poll", "dummy"})


def canonical_sched_phase(phase):
    """Map a wire-level phase discriminator to its report label."""
    return "resolve" if phase == "resolve_standalone" else phase


def nested_resolve_record_ids(records):
    """Return Resolve records contained by a Complete or Dummy parent.

    Containment is strict at the end and inclusive at the start. Only a trace
    recorded before the ``resolve_standalone`` discriminator can carry a
    standalone Resolve under this label, and there the P thread's Resolve abuts
    a neighbouring Dummy bar rather than sitting inside it, so a Resolve ending
    exactly at its candidate parent's end is P-thread work.

    The start stays inclusive because TMR opens a Dummy bar and times the first
    dummy's Resolve two ``get_sys_cnt_aicpu()`` reads apart, which share one
    a2a3 sys-cnt tick (20 ns at 50 MHz) often enough that a strict start would
    report genuinely nested Resolve work as standalone and double-count it.
    """
    parents = sorted(
        (
            (record.get("start_time_us", 0), record.get("end_time_us", 0))
            for record in records
            if record.get("phase") in ("complete", "dummy")
        ),
        key=lambda interval: interval[0],
    )
    parent_starts = [interval[0] for interval in parents]
    nested = set()
    for record in records:
        if record.get("phase") != "resolve":
            continue
        start_us = record.get("start_time_us", 0)
        end_us = record.get("end_time_us", 0)
        parent_idx = bisect.bisect_right(parent_starts, start_us) - 1
        if parent_idx < 0:
            continue
        if end_us < parents[parent_idx][1]:
            nested.add(id(record))
    return nested


def scheduler_thread_role(records, assigned_thread_indices, thread_idx, nested_resolve_ids):
    """Classify a scheduler-phase thread as scheduler or resolution."""
    raw_phases = {record.get("phase") for record in records}
    phases = {canonical_sched_phase(phase) for phase in raw_phases}
    has_scheduler_work = bool(phases & _SCHEDULER_WORK_PHASES)
    has_resolution_work = bool(raw_phases & _RESOLUTION_WORK_PHASES)
    is_unassigned_thread = bool(assigned_thread_indices) and thread_idx not in assigned_thread_indices
    has_standalone_resolve = any(
        record.get("phase") == "resolve_standalone"
        or (record.get("phase") == "resolve" and id(record) not in nested_resolve_ids)
        for record in records
    )
    is_resolution_thread = (
        has_resolution_work and not has_scheduler_work and (has_standalone_resolve or is_unassigned_thread)
    )
    return "resolution" if is_resolution_thread else "scheduler"
