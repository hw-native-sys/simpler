# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Whole-graph label style for hbg task ids -- the caller's presentation decision, not TaskId's."""

from __future__ import annotations

from .task_id import Space, TaskId


def make_formatter(task_ids):
    """A label style sized to the whole graph: all-GLOBAL drops the ``t`` prefix.

    A GLOBAL id is a task of the run itself -- no parent, no nesting -- so a graph
    made up of nothing else names its tasks by counter alone. Any SUB_TASK or PARAM
    id in the graph means some ids need the prefix to stay unambiguous, so every id
    gets one.
    """
    parsed = [t for t in map(TaskId.parse, task_ids) if t is not None]
    if not all(t.space() is Space.GLOBAL for t in parsed):
        return display

    def bare(task_id):
        t = TaskId.parse(task_id)
        return str(task_id) if t is None else str(t.local_id())

    return bare


def display(task_id):
    """``t{local}`` for a GLOBAL task, ``g{parent}t{local}`` for a sub-task, ``p{local}`` for a parameter.

    Falls back to ``str(task_id)`` for a value that is not even an integer --
    matches str/int JSON encodings alike. A value that parses but decodes to a
    corrupt id space raises (see ``TaskId.space()``): unlike a merely unparsable
    field, that is real capture data that should not render a plausible-looking
    label as if it were valid.
    """
    t = TaskId.parse(task_id)
    return str(task_id) if t is None else t.display()


def task_id_fields(task_id):
    """The id-space-dependent fields of an hbg task row: ``id_space``, plus ``parent_task_id`` for a sub-task."""
    t = TaskId(int(task_id))
    space = t.space()
    fields = {"id_space": space.value}
    if space is Space.SUB_TASK:
        fields["parent_task_id"] = t.parent_id()
    return fields


def scope_key(task_id):
    """Which scope boundary this id's task belongs to: ``(id_space, parent)``.

    Tasks of the run itself share one scope (space GLOBAL, parent 0); each modular
    task's body is its own, keyed by its own parent id -- see wait_reduction_sim's
    reachability-bitmap scope model.
    """
    t = TaskId(int(task_id))
    return t.space().value, t.parent_id()
