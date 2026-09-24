# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""This runtime records no per-task marker for the Worker View to draw.

Both runtimes route dependency-only tasks -- an empty active_mask, or a
predicate that resolved false -- to the AICPU scheduler's dummy queue, and both
record the aggregate ``Dummy`` phase for the drain pass. Only tmr additionally
records each task's identity, under ``dummy_task`` / ``predicated_skip``, so
only tmr's markers can be placed on a lane. The difference is in what the two
schedulers instrument, not in what their execution models contain: compare the
dummy drain loop in each runtime's ``scheduler_dispatch.cpp``.

An empty answer here is therefore what the capture supports, not what this
runtime's model implies.
"""

from __future__ import annotations


def contribute_worker_lane_events(records, thread_idx, ctx):
    """Nothing: see the module docstring."""
    del records, thread_idx, ctx
    return [], []


def marker_task_ids(scheduler_phases):
    """No marker kinds: this runtime records neither phase."""
    del scheduler_phases
    return {"dummy_task": set(), "predicated_skip": set()}
