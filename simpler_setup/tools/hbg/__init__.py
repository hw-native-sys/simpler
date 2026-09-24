# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The host_build_graph half of the per-runtime DFX contract (see ``tools._runtime_dispatch``)."""

from .formatter import display, make_formatter, scope_key, task_id_fields
from .graph_lane import contribute_events
from .phases import (
    MARKER_PHASES,
    OUTER_PHASES,
    PHASE_ALIASES,
    PHASE_COLORS,
    PHASE_DISPLAY_NAMES,
    PHASE_REPORT_LABELS,
    PHASES,
    RESOLUTION_PHASES,
    WORK_PHASES,
    canonical_sched_phase,
    has_standalone_resolve,
    nested_resolve_record_ids,
    scheduler_thread_role,
)
from .task_id import Space, TaskId
from .worker_lane import contribute_worker_lane_events, marker_task_ids

__all__ = [
    "MARKER_PHASES",
    "OUTER_PHASES",
    "PHASES",
    "PHASE_ALIASES",
    "PHASE_COLORS",
    "PHASE_DISPLAY_NAMES",
    "PHASE_REPORT_LABELS",
    "RESOLUTION_PHASES",
    "WORK_PHASES",
    "Space",
    "TaskId",
    "canonical_sched_phase",
    "contribute_events",
    "contribute_worker_lane_events",
    "display",
    "has_standalone_resolve",
    "make_formatter",
    "marker_task_ids",
    "nested_resolve_record_ids",
    "scheduler_thread_role",
    "scope_key",
    "task_id_fields",
]
