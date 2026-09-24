# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The one place a DFX tool picks which runtime's TaskId layout it is looking at.

A task_id carries whichever TaskId layout its runtime uses, and nothing in the value
itself says which -- the layout is chosen from the runtime name a document carries
(``metadata.runtime`` / a top-level ``runtime`` key), never guessed. Common tooling
(swimlane_converter, deps_viewer, critical_path, wait_reduction_sim,
sched_overhead_analysis) resolves that name once and calls ``get(name)`` for the
per-runtime implementation, rather than importing ``tools.hbg`` / ``tools.tmr``
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

HBG_RUNTIME = "host_build_graph"
TMR_RUNTIME = "tensormap_and_ringbuffer"


@dataclass(frozen=True)
class LaneContext:
    """What a runtime's lane contributor is given, and what it must not invent.

    ``pid`` and ``tid_base`` are assigned by the common converter, which owns the
    whole trace's numbering: a contributor that picked its own could collide with
    a lane the converter -- or another runtime's contributor -- already placed.
    Every tid a contributor emits must be ``tid_base`` plus an offset it chooses
    within its own lanes.

    ``display`` is this runtime's own task-id formatter; ``with_run_epoch`` and
    ``task_slice_start_us`` are common rules -- how a record's run is stamped onto
    an event, and where a task row's slice begins -- which are not a runtime's to
    decide and must not be reimplemented per contributor.
    """

    pid: int
    tid_base: int
    display: Callable[[object], str]
    with_run_epoch: Callable[[dict, dict], dict]
    task_slice_start_us: Callable[[dict], float]


@dataclass(frozen=True)
class WorkerLaneContext:
    """What a contributor needs to mark its own task events on the Worker View.

    Everything here is a capability rather than a raw input: ``event_name``
    already resolves a task's func_id and its SPMD status, and ``tid`` already
    knows where a scheduler thread's lane sits. A contributor handed
    ``deps_kernel_map`` and a tid base instead would have to re-derive both, and
    would then own rules that are not its to own.

    ``next_event_id`` hands out the trace's flow-event ids, which are global and
    must stay in issue order -- a contributor emitting its own would collide with
    the task bars already numbered.
    """

    display: Callable[[object], str]
    with_run_epoch: Callable[[dict, dict], dict]
    event_name: Callable[[object, str], str]
    tid: Callable[[int], int]
    next_event_id: Callable[[], int]


def resolve_runtime(runtime_name, *, source="metadata.runtime"):
    """Validate the runtime a document names, refusing anything else.

    Guessing when the name is missing or unrecognised produces labels and id fields
    that read as valid and are wrong -- an hbg sub-task decoded as tmr becomes a
    plausible ``r3t5`` with a billion-scale ring -- so a name this module does not
    know is an error rather than a default.

    Raises:
        ValueError: the name is absent, blank, or not a runtime this module handles.
    """
    if runtime_name in (HBG_RUNTIME, TMR_RUNTIME):
        return runtime_name
    if runtime_name is None or (isinstance(runtime_name, str) and not runtime_name.strip()):
        raise ValueError(
            f"{source} is missing; this capture predates the runtime name and its TaskId "
            f"layout cannot be determined. Re-capture with a current build, which writes "
            f"{HBG_RUNTIME!r} or {TMR_RUNTIME!r}."
        )
    raise ValueError(
        f"{source} is {runtime_name!r}, which this module does not decode; expected "
        f"{HBG_RUNTIME!r} or {TMR_RUNTIME!r}. A task id has no self-describing layout, so "
        f"an unrecognised runtime cannot be decoded by guessing."
    )


def get(runtime_name):
    """The ``hbg`` or ``tmr`` package implementing the per-runtime DFX contract for ``runtime_name``.

    Imported lazily so this module carries no import-time dependency on either
    runtime package, letting both runtime packages import back from here
    (``normalize_task_id_int``) without a cycle.
    """
    if resolve_runtime(runtime_name) == HBG_RUNTIME:
        from simpler_setup.tools import hbg  # noqa: PLC0415

        return hbg
    from simpler_setup.tools import tmr  # noqa: PLC0415

    return tmr


def normalize_task_id_int(v):
    """Unsigned 64-bit task id (matches host JSON / device ``task_id.raw``).

    Runtime-agnostic: normalizes a signed value to unsigned so the high field
    decodes correctly, regardless of which runtime's layout that field holds.
    Returns None if ``v`` is not convertible to int.
    """
    try:
        t = int(v)
    except (TypeError, ValueError):
        return None
    if t < 0:
        t &= (1 << 64) - 1
    return t


def raw_halves(raw):
    """A 64-bit value split at bit 32, high half first.

    Runtime-agnostic: this is a DOT-safe identifier split, not a decode. Splitting
    at 32 happens to be where both layouts place their low field, but that is
    incidental here -- unlike ``local_id()``, this does not claim the low half is a
    layout's local id, so it stays correct even for the invalid sentinel or a
    corrupt record. See ``deps_viewer._node_id``.
    """
    return raw >> 32, raw & 0xFFFFFFFF
