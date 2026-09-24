# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Every scheduler phase tensormap_and_ringbuffer emits, how the trace draws them, and how a report reads them.

Stated in full, not layered over a table shared with hbg. A name both runtimes use
is not a shared phase: it is two phases that happen to carry the same label, and
each runtime's own entry is what describes it. ``resolve`` is the clearest case --
here it is a bar nested inside whichever phase observed the FIN; under hbg it is an
AICore completion walk and a dedicated thread's standalone work. Because the two
vocabularies are independent, so is the classification built on them: this module
carries its own ``canonical_sched_phase`` / ``nested_resolve_record_ids`` /
``scheduler_thread_role`` rather than passing a runtime name into a shared one.

The Python mirror of ``simpler::tmr::SchedPhaseKind``
(src/common/tensormap_and_ringbuffer/sched_phase_kind.h). One producer emits all of
these -- this runtime's AICPU scheduler. It has no AICore scheduler, so there is no
second producer to attribute, and none of hbg's AICore phases (``bootstrap``,
``state_probe``, ``worksteal``, ``refill``, ``idle``) can appear in a capture of
this runtime.
"""

from __future__ import annotations

import bisect

# Every phase name a capture of this runtime may carry -- the wire vocabulary, which
# is what a fixture is checked against. ``idle`` is absent: this runtime has no
# AICore scheduler, so nothing publishes an idle record, and the idle a report shows
# is reconstructed from the gaps between work records (sched_overhead_analysis
# Part 2) rather than measured.
PHASES = frozenset(
    {
        "complete",
        "dispatch",
        "dummy",
        "early_dispatch",
        "resolve",
        "drain",
        "drain_prepare",
        "drain_publish",
        "async_poll",
        "release",
        "dummy_task",
        "predicated_skip",
    }
)

# The colour each phase is drawn in. The two markers are included because a colour
# is what makes them drawable at all; MARKER_PHASES below is what keeps them off the
# scheduler track.
PHASE_COLORS = {
    # Outer phases — mutually time-exclusive within an iter
    "complete": "good",  # green
    "dispatch": "terrible",  # red
    "async_poll": "yellow",  # async-wait completion polling (split from complete)
    "release": "olive",  # deferred-release drain (on_task_release work)
    "dummy": "grey",  # dummy_drain pass; this runtime nests Resolve inside
    "early_dispatch": "rail_animation",  # speculative early-dispatch staging
    # sync_start stop-the-world drain: outer bar time-contains the two
    # inner staging passes, so Perfetto nests them by depth on the track.
    "drain": "cq_build_running",  # handle_drain_mode outer
    "drain_prepare": "cq_build_attempt_runnable",  # inner: cluster scan + build_payload
    "drain_publish": "cq_build_attempt_passed",  # inner: MMIO write_reg per subtask (the cohort launch)
    # Nested inside a parent bar here.
    "resolve": "vsync_highlight_color",  # on_task_complete: walk consumer list
    # Separate-lane (Worker View AICPU_N). Both markers are drawn, so both have a
    # colour; MARKER_PHASES below is what keeps them off the scheduler track, so
    # neither value is read by the scheduler lane.
    "dummy_task": "grey",
    "predicated_skip": "grey",
}

# This runtime records every phase under the name a report uses, so nothing is
# aliased. hbg's resolution-thread discriminator is the case this exists for.
PHASE_ALIASES: dict[str, str] = {}

# Phases whose records this runtime draws somewhere other than the scheduler
# track -- a dependency-only node and a predicate that resolved false both
# inhabit the AICPU as a virtual worker, so worker_lane.py puts them on the
# Worker View instead. Naming them here is what keeps the scheduler lane's bar
# set derivable from PHASE_COLORS.
MARKER_PHASES = frozenset({"dummy_task", "predicated_skip"})

# What a scheduler-overhead report calls each of this runtime's phases.
#
# ``idle`` is absent on purpose: the idle row a report shows for this runtime is a
# value the tool synthesized from gaps, not anything this runtime reported, so its
# wording belongs to the tool.
PHASE_REPORT_LABELS = {
    "complete": "Complete (poll handshake, completion handling)",
    "async_poll": "AsyncPoll (async-wait completion: SDMA/RoCE/URMA/CCU)",
    "dispatch": "Dispatch (pop queue, build payload, flush)",
    "dummy": "Dummy (dependency-only task resolution)",
    "early_dispatch": "EarlyDispatch (speculative staging)",
    "drain": "Drain (sync-start staging)",
    "release": "Release (deferred producer release)",
    "resolve": "Resolve (completion/dependency resolution)",
}

# This runtime's phases are shown under the names they are recorded with.
PHASE_DISPLAY_NAMES: dict[str, str] = {}

# Every outer phase this runtime can emit, in the order a report lists them.
OUTER_PHASES = (
    "complete",
    "async_poll",
    "dispatch",
    "release",
    "dummy",
    "early_dispatch",
    "drain",
)

# Phases that make a thread a scheduler, and phases that make it a resolution
# thread -- the two halves scheduler_thread_role() weighs.
WORK_PHASES = frozenset({"complete", "dispatch", "release", "early_dispatch", "drain"})
RESOLUTION_PHASES = frozenset({"resolve", "async_poll", "dummy"})


def canonical_sched_phase(phase: str) -> str:
    """Map a wire-level phase discriminator to its report label.

    Nothing to fold for this runtime: it records every phase under the name a
    report uses. Present so a caller reaches the same entry point on both sides.
    """
    return PHASE_ALIASES.get(phase, phase)


def nested_resolve_record_ids(records):
    """Return Resolve records contained by a Complete or Dummy parent.

    Containment is the only thing that can answer here: this runtime times Resolve
    inside the phase that observed the FIN and records no discriminator to consult,
    unlike hbg's dedicated resolution thread.

    Containment is strict at the end and inclusive at the start. The start stays
    inclusive because this runtime opens a Dummy bar and times the first dummy's
    Resolve two ``get_sys_cnt_aicpu()`` reads apart, which share one a2a3 sys-cnt
    tick (20 ns at 50 MHz) often enough that a strict start would report genuinely
    nested Resolve work as standalone and double-count it.
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


def has_standalone_resolve(records, nested_resolve_ids):
    """Whether these records show Resolve work outside a Complete or Dummy bar.

    Derived from containment, because this runtime records no discriminator that
    would say so directly.
    """
    return any(record.get("phase") == "resolve" and id(record) not in nested_resolve_ids for record in records)


def scheduler_thread_role(records, assigned_thread_indices, thread_idx, nested_resolve_ids):
    """Classify a scheduler-phase thread as scheduler or resolution.

    Which phases count as scheduler work and which as resolution work is this
    runtime's vocabulary -- ``release`` is its own, and ``state_probe`` /
    ``graph_prepare`` are not phases it has at all.
    """
    phases = {canonical_sched_phase(record.get("phase", "")) for record in records}
    has_scheduler_work = bool(phases & WORK_PHASES)
    has_resolution_work = bool(phases & RESOLUTION_PHASES)
    is_unassigned_thread = bool(assigned_thread_indices) and thread_idx not in assigned_thread_indices
    is_resolution_thread = (
        has_resolution_work
        and not has_scheduler_work
        and (has_standalone_resolve(records, nested_resolve_ids) or is_unassigned_thread)
    )
    return "resolution" if is_resolution_thread else "scheduler"
