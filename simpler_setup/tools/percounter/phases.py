# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Percounter AICore phase vocabulary."""

PHASES_BY_PRODUCER = {
    "aicore": frozenset(
        [
            "scalar_window",
            "scalar_percounter_resolve",
            "scalar_percounter_fanin_ready",
            "scalar_percounter_prepare_subtask",
            "scalar_percounter_run_prepared",
            "scalar_percounter_publish_completion",
            "scalar_percounter_checkpoint_serve",
            "scalar_percounter_mix_prepare",
            "scalar_percounter_mix_arm",
            "scalar_percounter_mix_wait",
            "scalar_percounter_serve_mix",
            "scalar_percounter_mix_drain",
            "scalar_percounter_sync_enqueue",
            "scalar_percounter_sync_promote",
            "scalar_percounter_sync_take",
            "scalar_percounter_sync_serve_block",
            "scalar_percounter_flush_done",
            "scalar_run_percounter_lane",
            "scalar_backoff",
            "scalar_atomic_fetch_add",
            "scalar_atomic_fetch_add_issue",
            "scalar_atomic_cas",
            "scalar_atomic_store_issue",
            "scalar_bypass_load",
            "scalar_dcci_out",
            "scalar_dcci_single",
            "scalar_dcci_entire",
            "scalar_dsb_all",
            "scalar_ssbuf_load",
            "scalar_ssbuf_store",
            "scalar_trace_finish",
            "scalar_trace_finish_clean_entire",
            "scalar_trace_finish_barrier",
            "scalar_bypass_store_issue",
            "scalar_dsb_ddr",
        ]
    ),
    "aicpu": frozenset(),
}
PHASES = PHASES_BY_PRODUCER["aicore"]
MARKER_PHASES = frozenset()
OUTER_PHASES = ()
PHASE_ALIASES = {}
PHASE_COLORS = {}
PHASE_DISPLAY_NAMES = {}
PHASE_REPORT_LABELS = {}
RESOLUTION_PHASES = frozenset()
WORK_PHASES = PHASES


def canonical_sched_phase(phase: str) -> str:
    return phase


def has_standalone_resolve(records, nested_resolve_ids):
    return False


def nested_resolve_record_ids(records):
    return set()


def scheduler_thread_role(records, assigned_thread_indices, thread_idx, nested_resolve_ids):
    return "Scheduler"
