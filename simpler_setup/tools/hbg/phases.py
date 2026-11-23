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
the AICPU producer's dedicated resolution thread emits ``resolve_standalone``;
under tmr, ``resolve`` is nested inside whichever phase observed the FIN.
Because the two vocabularies are independent, so is the classification built on
them: this module carries its own ``canonical_sched_phase`` /
``nested_resolve_record_ids`` / ``scheduler_thread_role`` rather than passing a
runtime name into a shared one.

The Python mirror of ``simpler::hbg::SchedPhaseKind``
(src/common/host_build_graph/sched_phase_kind.h). Both of this runtime's producers
are covered, because both are this runtime:

  AICPU scheduler (a2a3/wse) -- ``graph_prepare`` (bounded Definition expansion) and
      ``resolve_standalone``, which canonicalizes to ``resolve``.
  AICore scheduler (a5) -- ``bootstrap``, ``state_probe`` and
      ``idle``, plus ``dispatch`` / ``complete`` and detailed Scalar phases. Written by
      append_scheduler_record() in
      a5/runtime/host_build_graph/host/runtime_maker.cpp. tmr has no AICore
      scheduler at all.

StateProbe acquisition envelopes can overlap within one discovery batch;
Scalar profiles account for exclusive work.
"""

from __future__ import annotations

# Which of this runtime's two producers writes each phase. This is the primary
# statement, because a stream names its producer: "may this stream carry this phase"
# is the question a reader and a fixture both ask, and PHASES below is a consequence
# of it rather than a second list to keep in step.
#
# The AICPU resolution thread writes ``resolve_standalone``, which PHASE_ALIASES
# folds into ``resolve`` for a report.
PHASES_BY_PRODUCER = {
    "aicpu": frozenset(
        {
            "complete",
            "dispatch",
            "dummy",
            "early_dispatch",
            "drain",
            "drain_prepare",
            "drain_publish",
            "async_poll",
            "graph_prepare",
            "resolve_standalone",
        }
    ),
    "aicore": frozenset(
        {
            "complete",
            "dispatch",
            "bootstrap",
            "state_probe",
            "idle",
        }
    ),
}

# Detailed Scalar phases share the L3 stream. Their hierarchy and exclusive
# statistics are separate from the coarse phase totals.
SCALAR_PHASES = frozenset(
    {
        "scalar_window",
        "scalar_completions",
        "scalar_graph_control",
        "scalar_executor_scan",
        "scalar_prepare",
        "scalar_seal",
        "scalar_async_service",
        "scalar_execution_token_progress",
        "scalar_execution_token_refill",
        "scalar_execution_token_receive",
        "scalar_advance_mix_floor",
        "scalar_self_order_ready",
        "scalar_mix_lane_order_ready",
        "scalar_cluster_owes_before",
        "scalar_cluster_mix_account",
        "scalar_cluster_mix_dependencies",
        "scalar_cluster_mix_prepare_member",
        "scalar_cluster_mix_recycle",
        "scalar_execution_token_select",
        "scalar_fanin_observe",
        "scalar_fill_own_slots",
        "scalar_cluster_mix_service",
        "scalar_cluster_mix_discover_task",
        "scalar_cluster_mix_discover_next",
        "scalar_cluster_mix_pending",
        "scalar_cluster_mix_observe",
        "scalar_cluster_mix_enter",
        "scalar_cluster_mix_release",
        "scalar_cluster_mix_drain",
        "scalar_cluster_mix_all",
        "scalar_prepared_receive",
        "scalar_prepared_claim",
        "scalar_counter_accounting",
        "scalar_prepared_return",
        "scalar_prepared_observe",
        "scalar_prepared_index",
        "scalar_prepared_validate",
        "scalar_group_acquire",
        "scalar_execution_continuation",
        "scalar_execution_token_fanin",
        "scalar_task_issue_head",
        "scalar_task_issue_fanin",
        "scalar_task_supply_closed",
        "scalar_service_cooperative_slots",
        "scalar_ordinary_block_claim",
        "scalar_task_completion_observe",
        "scalar_cached_completion_observe",
        "scalar_cached_completion_lookup",
        "scalar_subtask_contribute",
        "scalar_mix_group_contribute",
        "scalar_returns_contribute",
        "scalar_counter_condition_publish",
        "scalar_execution_identity_capture",
        "scalar_async_begin_prebound",
        "scalar_async_seal_execution",
        "scalar_async_close_computation",
        "scalar_cluster_mix_service_active",
        "scalar_graph_fanin_view",
        "scalar_graph_payload_arguments",
        "scalar_publish_idle",
        "scalar_trace_before_execution",
        "scalar_trace_before_completion",
        "scalar_can_enter_execution",
        "scalar_trace_append",
        "scalar_end_iteration",
        "scalar_build_payload",
        "scalar_publish_built",
        "scalar_select_owned_execution",
        "scalar_retire_execution",
        "scalar_async_check",
        "scalar_async_finish",
        "scalar_bootstrap",
        "scalar_bootstrap_init_wait",
        "scalar_bootstrap_peer_wait",
        "scalar_bootstrap_topology",
        "scalar_launch_wait",
        "scalar_exit_drain",
        "scalar_cache_observe",
        "scalar_cache_publish",
        "scalar_flush_completions",
        "scalar_loop_control",
        "scalar_dispatch_metadata",
        "scalar_dispatch_publish",
        "scalar_trace_finish",
        "scalar_trace_finish_clean",
        "scalar_trace_finish_barrier",
        "scalar_atomic_cas",
        "scalar_atomic_store_issue",
        "scalar_atomic_fetch_add",
        "scalar_atomic_fetch_add_issue",
        "scalar_dcci_out",
        "scalar_dcci_single",
        "scalar_dcci_entire",
        "scalar_dsb_all",
        "scalar_dsb_ddr",
        "scalar_bypass_load",
        "scalar_gm_load",
        "scalar_cnt_notify_load",
        "scalar_bypass_store",
    }
)
PHASES_BY_PRODUCER["aicore"] |= SCALAR_PHASES

# Every phase name a capture of this runtime may carry -- the wire vocabulary, which
# is what a fixture is checked against when it names no producer. ``idle`` is in it
# because the AICore scheduler publishes a SchedulerIdleRecord per spin, so a capture
# really can hold one; that is a measured record, unlike the gap reconstruction a
# report derives for an AICPU capture (sched_overhead_analysis Part 2), which no
# producer writes.
PHASES = frozenset().union(*PHASES_BY_PRODUCER.values())

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
}

# Every outer phase this runtime can emit, in the order a report lists them. The
# order carries meaning -- state_probe / dispatch is the
# AICore scheduler's own loop -- which is one reason this is each runtime's own to
# state rather than an interleaving of both.
OUTER_PHASES = (
    "complete",
    "async_poll",
    "state_probe",
    "dispatch",
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
