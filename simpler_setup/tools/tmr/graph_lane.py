# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""This runtime contributes no lane of its own to the trace.

The Graph Execution lane hbg draws has no counterpart here: a modular task's body,
its Scheduler-side expansion, and sub-task ids naming a parent are all concepts
tensormap_and_ringbuffer does not have. This is a real answer to the contract, not
a placeholder -- everything tmr puts on the timeline is already drawn by the common
converter from records both runtimes emit.
"""

from __future__ import annotations


def contribute_events(tasks, scheduler_phases, ctx):
    """Nothing: see the module docstring for why tmr has no lane to add."""
    del tasks, scheduler_phases, ctx
    return []
