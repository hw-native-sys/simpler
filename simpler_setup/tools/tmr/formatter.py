# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Whole-graph label style for tmr task ids -- the caller's presentation decision, not TaskId's."""

from __future__ import annotations

from .task_id import TaskId


def make_formatter(task_ids):
    """No compression branch: every tmr id keeps its ring, even when every ring is 0.

    Ring 0 is a ring, not an absence of one -- unlike hbg's GLOBAL space, it carries
    no claim of "unnested", so a graph where every task happens to sit on ring 0
    still prints ``r0t...`` rather than dropping the prefix.
    """
    del task_ids
    return display


def display(task_id):
    """``r{ring}t{local}``, or ``str(task_id)`` if it is not an integer (or numeric string)."""
    t = TaskId.parse(task_id)
    return str(task_id) if t is None else t.display()


def task_id_fields(task_id):
    """The id-layout-dependent fields of a tmr task row: ``ring_id``."""
    return {"ring_id": TaskId(int(task_id)).ring()}


def scope_key(task_id):
    """Which scope boundary this id's task belongs to: its ring.

    One ring per scope depth, so a differing ring is a differing scope -- see
    wait_reduction_sim's reachability-bitmap scope model.
    """
    return TaskId(int(task_id)).ring()
