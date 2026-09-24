# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Every scheduler phase host_build_graph emits, how the trace draws them, and how a report reads them.

Stated in full, not layered over a table shared with tmr. A name both runtimes use
is not a shared phase: it is two phases that happen to carry the same label, and
each runtime's own entry is what describes it. ``resolve`` is the clearest case --
here it is this runtime's AICore completion walk and its dedicated resolution
thread's work; under tmr it is a bar nested inside whichever phase observed the FIN.
Because the two vocabularies are independent, so is the classification built on
them: this module carries its own ``canonical_sched_phase`` /
``nested_resolve_record_ids`` / ``scheduler_thread_role`` rather than passing a
runtime name into a shared one.

The Python mirror of ``simpler::hbg::SchedPhaseKind``
(src/common/host_build_graph/sched_phase_kind.h). Both of this runtime's producers
are covered, because both are this runtime:

  AICPU scheduler -- ``graph_prepare`` (bounded Definition expansion) and
      ``resolve_standalone``, which canonicalizes to ``resolve``.
  AICore scheduler -- ``bootstrap``, ``state_probe``, ``worksteal``, ``refill`` and
      ``idle``, plus ``dispatch`` / ``complete`` / ``resolve``. Written by
      append_scheduler_record() in
      a5/runtime/host_build_graph/host/runtime_maker.cpp. tmr has no AICore
      scheduler at all.
"""

from __future__ import annotations

# Every phase name a capture of this runtime may carry -- the wire vocabulary, which
# is what a fixture is checked against. ``idle`` is in it because the AICore
# scheduler publishes a SchedulerIdleRecord per spin, so a capture really can hold
# one; that is a measured record, unlike the gap reconstruction a report derives for
# an AICPU capture (sched_overhead_analysis Part 2), which no producer writes.
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
        "graph_prepare",
        "resolve_standalone",
        "bootstrap",
        "state_probe",
        "worksteal",
        "refill",
        "idle",
    }
)

# The colour each phase is drawn in on the scheduler track. Keyed by report label, so
# ``resolve_standalone`` is absent -- PHASE_ALIASES folds it into ``resolve`` before a
# bar is drawn.
PHASE_COLORS = {
    # Outer phases — mutually time-exclusive within an iter
    "complete": "good",  # green
    "dispatch": "terrible",  # red
    "async_poll": "yellow",  # async-wait completion polling (split from complete)
    "dummy": "grey",  # dummy_drain pass; this runtime's P bar is standalone
    "early_dispatch": "rail_animation",  # speculative early-dispatch staging
    # sync_start stop-the-world drain: outer bar time-contains the two
    # inner staging passes, so Perfetto nests them by depth on the track.
    "drain": "cq_build_running",  # handle_drain_mode outer
    "drain_prepare": "cq_build_attempt_runnable",  # inner: cluster scan + build_payload
    "drain_publish": "cq_build_attempt_passed",  # inner: MMIO write_reg per subtask (the cohort launch)
    "graph_prepare": "rail_animation",  # bounded Scheduler-side Definition expansion
    # Standalone on this runtime's dedicated P thread.
    "resolve": "vsync_highlight_color",  # on_task_complete: walk consumer list
    # AICore scheduler's own loop.
    "bootstrap": "rail_animation",
    "state_probe": "cq_build_running",
    "worksteal": "cq_build_attempt_failed",
    "refill": "cq_build_attempt_passed",
    "idle": "grey",  # AICore scheduler spin with no progress
}

# Wire-level discriminators this runtime records under a name a report does not
# use. ``resolve_standalone`` is this runtime's Resolve, recorded distinctly because
# it stands alone on a dedicated thread rather than nesting inside a Complete bar.
PHASE_ALIASES = {"resolve_standalone": "resolve"}

# Every phase this runtime records is drawn on the scheduler track, so none is
# held back from the bar set derived from PHASE_COLORS. tmr's marker phases are
# the case this exists for.
MARKER_PHASES = frozenset()

# What a scheduler-overhead report calls each of this runtime's phases. Longer
# than PHASE_DISPLAY_NAMES because a report row has to say what the phase did,
# where a trace bar sits next to the lane that already gives it context.
#
# ``idle`` is absent on purpose. A report reaches this table only on the AICPU
# path, where the idle row is a value the tool synthesized from gaps rather than
# anything this runtime reported -- so its wording belongs to the tool. The AICore
# path prints its measured idle under the raw phase name and consults no table
# (sched_overhead_analysis.print_aicore_scheduler_phase_breakdown).
PHASE_REPORT_LABELS = {
    "complete": "Complete (poll handshake, completion handling)",
    "async_poll": "AsyncPoll (async-wait completion: SDMA/RoCE/URMA/CCU)",
    "dispatch": "Dispatch (pop queue, build payload, flush)",
    "dummy": "Dummy (dependency-only task resolution)",
    "early_dispatch": "EarlyDispatch (speculative staging)",
    "drain": "Drain (sync-start staging)",
    "state_probe": "StateProbe (Scheduler-local Dispatch Slot / Ready state)",
    "worksteal": "Worksteal (remote Inbox claim and dispatch)",
    "refill": "Refill (completed Slot reuse)",
    "graph_prepare": "GraphPrepare (Definition expansion)",
    "resolve": "Resolve (completion/dependency resolution)",
}

# The AICore scheduler names its phases for the reader; the AICPU stream shows
# its phase names as recorded.
PHASE_DISPLAY_NAMES = {
    "complete": "Completion",
    "resolve": "Resolve",
    "state_probe": "StateProbe",
    "dispatch": "Dispatch",
    "worksteal": "Worksteal",
    "refill": "Refill",
}

# Every outer phase this runtime can emit, in the order a report lists them. The
# order carries meaning -- state_probe / dispatch / worksteal / refill is the
# AICore scheduler's own loop -- which is one reason this is each runtime's own to
# state rather than an interleaving of both.
OUTER_PHASES = (
    "complete",
    "async_poll",
    "state_probe",
    "dispatch",
    "worksteal",
    "refill",
    "dummy",
    "early_dispatch",
    "drain",
    "graph_prepare",
)

# Phases that make a thread a scheduler, and phases that make it a resolution
# thread -- the two halves scheduler_thread_role() weighs. Both are matched
# against canonicalized names, so ``resolve`` here covers the
# ``resolve_standalone`` records PHASE_ALIASES folds into it.
WORK_PHASES = frozenset(
    {
        "complete",
        "state_probe",
        "dispatch",
        "worksteal",
        "refill",
        "early_dispatch",
        "drain",
        "graph_prepare",
    }
)
RESOLUTION_PHASES = frozenset({"resolve", "async_poll", "dummy"})


def canonical_sched_phase(phase: str) -> str:
    """Map a wire-level phase discriminator to its report label.

    A record with no phase field reaches here as the empty string, which this
    runtime does not alias and no phase set contains.
    """
    return PHASE_ALIASES.get(phase, phase)


def nested_resolve_record_ids(records):
    """Resolve records this runtime nested inside a parent bar: none, and it can say so.

    Two producers emit a Resolve here and neither needs a containment verdict. The
    AICPU scheduler records its dedicated resolution thread's work as
    ``resolve_standalone``, so the record itself says it stands alone; weighing time
    containment could only contradict it. The AICore scheduler does emit a plain
    ``resolve`` (its completion walk, from publish_aicore_scheduler_profiling in
    a5/runtime/host_build_graph/host/runtime_maker.cpp), but every caller of this
    result guards on ``not is_aicore_scheduler`` before consulting it, so that
    stream never reaches a nesting decision.

    tmr's version is where containment is the only available answer.

    Takes ``records`` and ignores it so both runtimes present one signature to a
    caller that has not yet resolved which runtime it holds.
    """
    del records
    return frozenset()


def has_standalone_resolve(records, nested_resolve_ids):
    """Whether these records show Resolve work this runtime ran on its own thread.

    Answered by the explicit discriminator: a ``resolve_standalone`` record is
    standalone by construction. ``nested_resolve_ids`` is accepted and ignored for
    signature parity with tmr, whose answer depends on it.
    """
    del nested_resolve_ids
    return any(record.get("phase") == "resolve_standalone" for record in records)


def scheduler_thread_role(records, assigned_thread_indices, thread_idx, nested_resolve_ids):
    """Classify a scheduler-phase thread as scheduler or resolution.

    Which phases count as scheduler work and which as resolution work is this
    runtime's vocabulary -- ``state_probe`` and ``graph_prepare`` are its own, and
    ``release`` is not a phase it has at all.
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
