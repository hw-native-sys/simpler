#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Swimlane JSON to Perfetto JSON Converter

Converts performance data JSON (.json) to Chrome Trace Event Format JSON
for visualization in Perfetto (https://ui.perfetto.dev/).

Usage:
    python -m simpler_setup.tools.swimlane_converter  # latest chip_swimlane_records_*.json under ./outputs/
    python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/chip_swimlane_records.json
    python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/chip_swimlane_records.json -o out.json
    python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/chip_swimlane_records.json -k kernel_config.py
    python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/chip_swimlane_records.json -v

SPMD (block_num>1): dependency flows use the earliest visible task slice
per (func_id, task_id) independently in each view; see
docs/dfx/chip-swimlane-profiling.md §3.5.
"""

import argparse
import bisect
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simpler_setup.tools import containment
from simpler_setup.tools.scheduler_phase_records import (
    canonical_sched_phase,
    nested_resolve_record_ids,
    scheduler_thread_role,
)
from simpler_setup.tools.strace_timing import host_process_lanes, parse_spans, span_family

_AICORE_SCHEDULER_PHASE_DISPLAY_NAMES = {
    "complete": "Completion",
    "resolve": "Resolve",
    "state_probe": "StateProbe",
    "dispatch": "Dispatch",
    "worksteal": "Worksteal",
    "refill": "Refill",
}


def _func_id_to_letter(func_id):
    """Map a non-negative integer func_id to a numeric+letter label.

    0 → '0_a', 1 → '1_b', …, 25 → '25_z', 26 → '26_aa', 27 → '27_ab', …
    """
    try:
        n = int(func_id)
    except (TypeError, ValueError):
        return str(func_id)
    letters = []
    m = n + 1  # shift so that 0 maps to 'a' (1-based bijective base-26)
    while m > 0:
        m, rem = divmod(m - 1, 26)
        letters.append(chr(ord("a") + rem))
    return str(n) + "_" + "".join(reversed(letters))


def _task_display_name(func_id, func_id_to_name, tdisp, *, spmd=False):
    """Build the swimlane event label for a task.

    Naming, in priority order:
      - ``func_id < 0`` (unresolved): ``task(<task-id>)``. This is the no-deps.json
        case — without a dep_gen capture the host never carries func_id, so
        every lane is an anonymous ``task(...)`` distinguished only by id.
      - a name mapping exists for the func_id: ``<name>(<task-id>)``.
      - otherwise: ``func_<letter>(<task-id>)`` (resolved id, but no name map entry).

    The caller supplies the runtime-specific task-id label. SPMD logical tasks
    append ``_spmd`` before that suffix unless the name already contains ``spmd``
    (case-insensitive), e.g. ``fa_fused_aic_spmd(r2t18)``.
    """
    try:
        resolved = int(func_id) >= 0
    except (TypeError, ValueError):
        resolved = False
    if not resolved:
        label = f"task({tdisp})"
    elif func_id_to_name and str(func_id) in func_id_to_name:
        label = f"{func_id_to_name[str(func_id)]}({tdisp})"
    else:
        label = f"func_{_func_id_to_letter(func_id)}({tdisp})"
    if spmd and "(" in label:
        base, rest = label.split("(", 1)
        if "spmd" not in base.lower():
            return f"{base}_spmd({rest}"
    return label


def _scheduler_task_key(row):
    """Join key for a ``scheduler_tasks.records`` row: ``(run_epoch, core_id, reg_task_id)``.

    A five-column row carries ``run_epoch`` last. A four-column row is a capture
    written before run identity existed; its epoch is reported as ``None`` rather
    than 0, because 0 is an epoch a device can really be given and coercing to it
    would let such a capture collide with a real run.
    """
    return (int(row[4]) if len(row) > 4 else None, int(row[0]), int(row[1]))


def normalize_task_id_int(v):
    """Unsigned 64-bit task id (matches host JSON / device ``task_id.raw``).

    Normalizes signed values to unsigned so the high field decodes correctly.
    Returns None if ``v`` is not convertible to int.
    """
    try:
        t = int(v)
    except (TypeError, ValueError):
        return None
    if t < 0:
        t &= (1 << 64) - 1
    return t


def _tmr_task_display(task_id):
    """Format a ``tensormap_and_ringbuffer`` task_id for human-readable labels.

    That runtime puts a ring index in bits 39:32 and a local id in the low 32.

    Returns:
        ``r{ring}t{local}`` for every ring (for example ``r0t5`` and ``r2t100``).

    For invalid or non-numeric values, returns str(task_id).
    """
    tid = normalize_task_id_int(task_id)
    if tid is None:
        return str(task_id)
    local = tid & 0xFFFFFFFF
    ring = (tid >> 32) & 0xFF
    return f"r{ring}t{local}"


def _hbg_task_display(task_id):
    """Format a ``host_build_graph`` task_id for human-readable labels.

    That runtime puts an id space in bits 63:62 (0 = GLOBAL, 1 = SUB_TASK, 2 = PARAM),
    a sub-task's parent modular task in bits 51:32, and a local id in the low 32.
    See src/common/host_build_graph/task_id.h.

    Returns:
        ``g{parent}t{local}`` for a sub-task: its parent modular task and own index
        ``p{index}`` for a boundary parameter
        ``t{local}`` for a task of the run itself

    For invalid or non-numeric values, returns str(task_id).
    """
    tid = normalize_task_id_int(task_id)
    if tid is None:
        return str(task_id)
    local = tid & 0xFFFFFFFF
    space = (tid >> 62) & 0x3
    if space == 1:
        return f"g{(tid >> 32) & 0xFFFFF}t{local}"
    if space == 2:
        return f"p{local}"
    return f"t{local}"


HBG_RUNTIME = "host_build_graph"
TMR_RUNTIME = "tensormap_and_ringbuffer"


def resolve_runtime(runtime_name, *, source="metadata.runtime"):
    """Validate the runtime a document names, refusing anything else.

    A task_id carries whichever TaskId layout its runtime uses and nothing in the value
    says which, so the layout is chosen from this name alone. Guessing when the name is
    missing or unrecognised produces labels and id fields that read as valid and are
    wrong -- an hbg sub-task decoded as tmr becomes a plausible `r3t5` with a
    billion-scale ring -- so a name this tool does not know is an error rather than a
    default.

    Every document the repo writes names its runtime unconditionally: the swimlane
    collector emits `metadata.runtime` (and fails to compile without
    SIMPLER_RUNTIME_NAME), and both dep_gen writers emit a top-level `runtime`. A
    missing name therefore means a capture from before those writers, which this tool
    does not decode.

    Public alongside task_display_for because critical_path and deps_viewer resolve
    their own documents and must reach the same verdict this module does.

    Raises:
        ValueError: the name is absent, blank, or not a runtime this tool decodes.
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
        f"{source} is {runtime_name!r}, which this tool does not decode; expected "
        f"{HBG_RUNTIME!r} or {TMR_RUNTIME!r}. A task id has no self-describing layout, so "
        f"an unrecognised runtime cannot be decoded by guessing."
    )


def task_display_for(runtime_name):
    """Pick the task-id formatter for the runtime that minted the records.

    See resolve_runtime for why an unknown name raises rather than defaulting.
    """
    return _hbg_task_display if resolve_runtime(runtime_name) == HBG_RUNTIME else _tmr_task_display


def _task_id_fields_for(runtime_name):
    """Pick the task-row id fields for the runtime that minted the records.

    See resolve_runtime for why an unknown name raises rather than defaulting.
    """
    return _hbg_task_id_fields if resolve_runtime(runtime_name) == HBG_RUNTIME else _tmr_task_id_fields


def _tmr_task_id_fields(task_id):
    """The id-layout-dependent fields of a ``tensormap_and_ringbuffer`` task row."""
    return {"ring_id": (task_id >> 32) & 0xFFFFFFFF}


def _hbg_task_id_fields(task_id):
    """The id-layout-dependent fields of a ``host_build_graph`` task row."""
    space = (task_id >> 62) & 0x3
    fields = {"id_space": space}
    if space == 1:
        fields["parent_task_id"] = (task_id >> 32) & 0xFFFFF
    return fields


def _decode_sub_task_id(task_id):
    """Decode Scheduler-owned sub-task ids.

    ``host_build_graph`` puts a materialized sub-task in id space 1 (SUB_TASK), held in
    the top two bits, with its parent modular task in bits 51:32 and its own index in
    the low 32; the stream-visible outer modular task stays in space 0 (GLOBAL). See
    src/common/host_build_graph/task_id.h.

    The space test reads the top two bits, so a ``tensormap_and_ringbuffer`` id -- whose
    ring sits in bits 39:32 -- never matches, whatever its ring.
    """
    tid = normalize_task_id_int(task_id)
    if tid is None or ((tid >> 62) & 0x3) != 1:
        return None
    return (tid >> 32) & 0xFFFFF, tid & 0xFFFFFFFF


def _collect_graph_execution_instances(tasks, scheduler_phases):  # noqa: PLR0912
    """Join sub-task rows to their outer GraphPrepare records.

    Grouping is keyed by ``(run_epoch, outer_task_id)``, not by the outer id
    alone. A graph re-executed in a later run reuses its task ids, so keying on
    the id alone would fold two runs' executions into one instance whose span
    covers both and whose row list is twice as long.
    """
    prepare_by_outer = defaultdict(list)
    dummy_rows = []
    for thread_idx, records in enumerate(scheduler_phases or []):
        for record in records:
            phase = record.get("phase")
            if phase == "graph_prepare":
                outer_task_id = normalize_task_id_int(record.get("task_id"))
                # A graph_prepare record names the outer modular task, which is always
                # GLOBAL. Testing the whole high word rather than the space alone is
                # the stricter check and the one wanted here: a GLOBAL id has a zero
                # parent and zero reserved bits too, so anything else in those bits is
                # a corrupt record rather than a task of another space.
                if outer_task_id is not None and (outer_task_id >> 32) == 0:
                    prepare_by_outer[(record.get("run_epoch"), outer_task_id)].append(record)
            elif phase == "dummy_task":
                dummy_rows.append((record, thread_idx))

    rows_by_outer = defaultdict(list)
    for task in tasks:
        decoded = _decode_sub_task_id(task.get("task_id"))
        if decoded is not None:
            outer_task_id, task_index = decoded
            rows_by_outer[(task.get("run_epoch"), outer_task_id)].append((task, task_index))

    dummy_by_outer = defaultdict(list)
    for record, thread_idx in dummy_rows:
        decoded = _decode_sub_task_id(record.get("task_id"))
        if decoded is not None:
            outer_task_id, task_index = decoded
            dummy_by_outer[(record.get("run_epoch"), outer_task_id)].append((record, task_index, thread_idx))

    instances = []
    for group_key, prepare_records in prepare_by_outer.items():
        run_epoch, outer_task_id = group_key
        rows = rows_by_outer.get(group_key, [])
        aicpu_rows = dummy_by_outer.get(group_key, [])
        if not rows and not aicpu_rows:
            continue
        task_indices = {task_index for _, task_index in rows}
        task_indices.update(task_index for _, task_index, _ in aicpu_rows)
        starts = [
            task.get("dispatch_time_us", _task_slice_start_us(task))
            if task.get("dispatch_time_us", -1) >= 0
            else _task_slice_start_us(task)
            for task, _ in rows
        ]
        starts.extend(record["start_time_us"] for record, _, _ in aicpu_rows)
        ends = [
            task.get("finish_time_us", 0) if task.get("finish_time_us", 0) > 0 else task["end_time_us"]
            for task, _ in rows
        ]
        ends.extend(record["end_time_us"] for record, _, _ in aicpu_rows)
        prepare_start_us = min(record["start_time_us"] for record in prepare_records)
        instances.append(
            {
                "outer_task_id": outer_task_id,
                "run_epoch": run_epoch,
                "rows": rows,
                "aicpu_rows": aicpu_rows,
                "visible_task_indices": sorted(task_indices),
                "execution_start_us": min(starts),
                "execution_end_us": max(ends),
                "prepare_start_us": prepare_start_us,
                "prepare_end_us": max(record["end_time_us"] for record in prepare_records),
                "prepare_duration_us": sum(
                    record["end_time_us"] - record["start_time_us"] for record in prepare_records
                ),
                "prepare_slice_count": len(prepare_records),
            }
        )

    instances.sort(key=lambda instance: instance["prepare_start_us"])
    lane_finish_us = []
    for instance_idx, instance in enumerate(instances):
        start_us = instance["prepare_start_us"]
        lane_idx = next((idx for idx, finish_us in enumerate(lane_finish_us) if finish_us <= start_us), -1)
        if lane_idx < 0:
            lane_idx = len(lane_finish_us)
            lane_finish_us.append(0.0)
        lane_finish_us[lane_idx] = instance["execution_end_us"]
        instance["instance_idx"] = instance_idx
        instance["lane_idx"] = lane_idx
    return instances


# A scheduler thread index has to fit the runtime's `int8_t core_to_thread[]`,
# and an AICore scheduler index is bounded by the cluster capacity, so no real
# producer emits an id anywhere near this. It exists only so a corrupt artifact
# cannot turn one large id into a multi-gigabyte list.
_MAX_SCHEDULER_ID = 256


def _place_streams_by_scheduler_id(streams_records, streams_metadata):
    """Return the stream lists re-indexed so each stream sits at its own
    ``scheduler_id``, with the omitted ids left as empty slots.

    The position in these lists *is* the scheduler thread index to every
    consumer: `sched_overhead_analysis.compute_dag_stats_from_deps` keys its
    per-thread accumulators on the values in ``core_to_thread`` (which the AICPU
    fills with its own thread indices), and the scene tests compare list
    positions against the same table. The writer omits a stream that recorded
    nothing, so appending the survivors in encounter order would renumber every
    stream above the gap and charge their work to a thread that does not exist.

    Falls back to the encounter order when any id is missing, negative, out of
    range, or repeated — the invariant cannot be restored from those, and
    keeping the previous shape beats raising on an artifact that is merely odd.
    """
    ids = [metadata.get("scheduler_id") for metadata in streams_metadata]
    if not all(isinstance(sid, int) and not isinstance(sid, bool) and 0 <= sid < _MAX_SCHEDULER_ID for sid in ids):
        return streams_records, streams_metadata
    if len(set(ids)) != len(ids):
        return streams_records, streams_metadata

    # Every real producer shares one producer per artifact, so a gap inherits it
    # and keeps lane naming consistent with the streams around it.
    producers = {metadata.get("producer") for metadata in streams_metadata if metadata.get("producer")}
    gap_producer = producers.pop() if len(producers) == 1 else None

    placed_records = [[] for _ in range(max(ids) + 1)] if ids else []
    placed_metadata = [
        {
            "platform": None,
            "producer": gap_producer,
            "scheduler_id": index,
            "worker_id": index,
            "core_type": None,
            "physical_core_id": None,
            "capture": None,
        }
        for index in range(len(placed_records))
    ]
    for sid, records, metadata in zip(ids, streams_records, streams_metadata):
        placed_records[sid] = records
        placed_metadata[sid] = metadata
    return placed_records, placed_metadata


_CLOCK_ALIGNMENT_FIELDS = (
    "status",
    "device_anchor_cycles",
    "host_anchor_ns",
    "host_anchor_min_ns",
    "host_anchor_max_ns",
)


def _capture_without_alignment(raw):
    """Compare capture data independently of the saved mapping."""
    document = dict(raw)
    metadata = dict(document.get("metadata") or {})
    metadata.pop("clock_alignment", None)
    document["metadata"] = metadata
    return document


@dataclass(frozen=True)
class _StoredClockAlignment:
    """A saved device-to-Host mapping, using the capture's counter frequency."""

    record: dict
    frequency_hz: int

    @property
    def place_lo_ns(self):
        return self.record["host_anchor_min_ns"]

    def map_cycles_to_host_ns(self, cycles):
        return self.record["host_anchor_ns"] + (
            (cycles - self.record["device_anchor_cycles"]) * 1_000_000_000 / self.frequency_hz
        )


def _read_clock_alignment(raw):
    """Validate saved anchors, bounds, and the capture counter frequency."""
    metadata = raw.get("metadata") or {}
    record = metadata.get("clock_alignment")
    if record is None:
        return None
    if not isinstance(record, dict):
        raise ValueError("clock_alignment must be an object")
    if record.get("status") == "unavailable":
        return None
    if record.get("status") != "bounded":
        raise ValueError("unsupported clock_alignment status")
    if any(type(record.get(key)) is not int for key in _CLOCK_ALIGNMENT_FIELDS[1:]):
        raise ValueError("clock_alignment anchors and bounds must be integers")
    frequency_hz = metadata.get("clock_freq_hz")
    if type(frequency_hz) is not int or frequency_hz <= 0:
        raise ValueError("capture metadata has no usable clock_freq_hz")
    lo, hi = record["host_anchor_min_ns"], record["host_anchor_max_ns"]
    if record["device_anchor_cycles"] <= 0 or not 0 < lo <= record["host_anchor_ns"] <= hi:
        raise ValueError("invalid clock_alignment bounds")
    return _StoredClockAlignment(record, frequency_hz)


def _clock_alignment_record(placement):
    """Serialize one mapping, bounding both window slack and phase-join freedom."""
    if placement.capture is None or placement.join is None:
        raise ValueError("clock alignment requires device records")
    anchor = placement.capture.extent[0]
    candidates = [
        containment.place(
            placement.host,
            placement.capture,
            containment.Join(origin, placement.join.interval_cycles, placement.join.sources),
        )
        for origin in placement.join.interval_cycles
    ]
    lo = math.floor(min(item.map_cycles_to_host_ns(anchor) for item in candidates))
    hi = math.ceil(max(item.map_cycles_to_host_ns(anchor) + item.slack_ns for item in candidates))
    return {
        "status": "bounded",
        "device_anchor_cycles": anchor,
        "host_anchor_ns": int(round(placement.map_cycles_to_host_ns(anchor))),
        "host_anchor_min_ns": lo,
        "host_anchor_max_ns": hi,
    }


def _json_object_members(text, start):
    """Locate value spans without reformatting the surrounding JSON."""
    decoder = json.JSONDecoder()

    def skip_space(position):
        while text[position] in " \t\r\n":
            position += 1
        return position

    cursor = skip_space(start + 1)
    while text[cursor] != "}":
        key, cursor = decoder.raw_decode(text, cursor)
        cursor = skip_space(cursor)
        if text[cursor] != ":":
            raise ValueError("invalid JSON object member")
        value_start = skip_space(cursor + 1)
        _value, value_end = decoder.raw_decode(text, value_start)
        yield key, value_start, value_end
        cursor = skip_space(value_end)
        if text[cursor] == ",":
            cursor = skip_space(cursor + 1)
        elif text[cursor] != "}":
            raise ValueError("invalid JSON object separator")


def _replace_alignment_text(text, record):
    """Only the alignment value is rewritten; compact task rows stay compact."""
    root_start = len(text) - len(text.lstrip())
    metadata_start = next(
        (start for key, start, _end in _json_object_members(text, root_start) if key == "metadata"),
        None,
    )
    if metadata_start is None or text[metadata_start] != "{":
        raise ValueError("capture metadata must be an object")
    encoded = json.dumps(record, separators=(",", ":"), ensure_ascii=True)
    members = list(_json_object_members(text, metadata_start))
    for key, start, end in members:
        if key == "clock_alignment":
            return text[:start] + encoded + text[end:]
    newline = "\r\n" if "\r\n" in text else "\n"
    insertion = members[-1][2] if members else metadata_start + 1
    addition = ("," if members else "") + newline + '    "clock_alignment": ' + encoded
    return text[:insertion] + addition + text[insertion:]


def _write_clock_alignment_record(path, raw, record):
    """Replace the complete JSON atomically, preserving its data and permissions."""
    metadata = raw.setdefault("metadata", {})
    if metadata.get("clock_alignment") == record:
        return
    # Conversion must use the mapping calculated for this run even when the
    # source is read-only or changes before the best-effort writeback.
    metadata["clock_alignment"] = record
    path = Path(path).resolve()
    # The writeback source must still contain the same raw capture.
    source_text = path.read_bytes().decode("utf-8")
    current = json.loads(source_text)
    if _capture_without_alignment(current) != _capture_without_alignment(raw):
        raise ValueError("capture changed during clock alignment")
    updated_text = _replace_alignment_text(source_text, record)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            os.fchmod(output.fileno(), path.stat().st_mode & 0o777)
            output.write(updated_text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _host_record_bounds(raw):
    values = []
    # These are Host-clock records, and only they bound the invocation.
    for name in ("host_orchestrator_phases", "host_device_uploads"):
        for record in containment._phase_records(raw.get(name)):
            values.extend(int(record[field]) for field in ("start_host_ns", "end_host_ns") if record.get(field, 0) > 0)
    return (min(values), max(values)) if values else None


def _matching_capture_host_windows(raw, spans, sidecar):
    windows = containment.host_windows(spans)
    by_name = {(span.pid, span.inv, span.name): span for span in spans}
    windows = [
        window
        for window in windows
        if not by_name[(window.pid, window.inv, containment.RUNNER_SPAN)].is_device
        and by_name[(window.pid, window.inv, containment.DEVICE_WALL_SPAN)].is_device
    ]
    roots = {(span.pid, span.inv): span for span in spans if span.name == containment.RUN_SPAN and not span.is_device}
    identity = containment.capture_identity(sidecar)
    pid = (sidecar or {}).get("host_pid")
    if pid is not None:
        windows = [window for window in windows if window.pid == int(pid)]
    if identity is not None:
        windows = [window for window in windows if window.identity is None or window.identity == identity]
    host_bounds = _host_record_bounds(raw)
    if host_bounds is not None:
        windows = [
            window
            for window in windows
            if (root := roots.get((window.pid, window.inv))) is not None
            and root.ts <= host_bounds[0] <= host_bounds[1] <= root.ts + root.dur
        ]
    windows = [
        window for window in windows if window.start_ns > 0 and window.duration_ns > 0 and window.device_wall_ns > 0
    ]
    if not windows:
        raise ValueError("no matching Host runner_run/device_wall windows")
    return windows


def _is_hbg_host_capture(raw):
    """HBG level-3/4 captures containing Host data require clock alignment."""
    if raw.get("chip_swimlane_level") not in (3, 4):
        return False
    metadata = raw.get("metadata") or {}
    has_host_capture = (
        metadata.get("orchestrator_source") == "host"
        or bool(raw.get("host_orchestrator_phases"))
        or bool(raw.get("host_device_uploads"))
        or isinstance(metadata.get("host_capture"), dict)
    )
    if not has_host_capture:
        return False
    return metadata.get("runtime") == HBG_RUNTIME


def _prepare_capture_clock_alignment(path, host_logs=None):
    """Return an enriched HBG capture and any available placement diagnostics.

    Explicit logs request a fresh calculation. With no override, a valid saved
    mapping is sufficient and no log or sidecar is read.
    """
    path = Path(path)
    raw = json.loads(path.read_text())
    if not _is_hbg_host_capture(raw):
        return raw, None
    try:
        saved = _read_clock_alignment(raw) if host_logs is None else None
        if saved is not None:
            return raw, saved
    except ValueError as error:
        print(f"Warning: ignoring saved clock alignment: {error}", file=sys.stderr)

    placement = None
    try:
        logs = (
            [Path(log) for log in host_logs]
            if host_logs is not None
            else (sorted(path.parent.glob("host_clock_alignment.*.log")) or sorted(path.parent.glob("host.*.log")))
        )
        if not logs:
            raise ValueError("no Host log available")
        spans = []
        for log in logs:
            with log.open() as input_file:
                spans.extend(parse_spans(input_file))
        sidecar_path = path.parent / "dispatch_identity.json"
        sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else None
        windows = _matching_capture_host_windows(raw, spans, sidecar)
        if len(windows) != 1:
            raise ValueError("capture does not uniquely identify a Host invocation")
        capture = containment.capture_windows(raw)
        placement = containment.place(windows[0], capture)
        record = _clock_alignment_record(placement)
    except (OSError, ValueError, KeyError, TypeError) as error:
        placement = None
        record = {"status": "unavailable", "reason": str(error)}
        print(f"Warning: clock alignment skipped: {error}", file=sys.stderr)
    try:
        _write_clock_alignment_record(path, raw, record)
    except (OSError, ValueError) as error:
        # A read-only source can still be converted with the computed mapping.
        raw.setdefault("metadata", {})["clock_alignment"] = record
        print(f"Warning: could not save clock alignment: {error}", file=sys.stderr)
    return raw, placement


def read_perf_data(filepath, *, timeline_origin_ns=None, placement=None):
    """Read and decode performance data from a swimlane JSON file."""
    with open(filepath) as file:
        data = json.load(file)
    return _decode_perf_data(data, timeline_origin_ns=timeline_origin_ns, placement=placement)


def _decode_perf_data(data, *, timeline_origin_ns=None, placement=None):  # noqa: PLR0912, PLR0915
    """Decode performance data from an already-loaded swimlane document.

    Host dumps raw cycle-domain per-stream records plus metadata; this
    function joins AICore execution records with Scheduler task timing. Schema:

        {
          "chip_swimlane_level": <1..4>,
          "metadata": {
            "clock_freq_hz": <int>,
            "num_cores": <int>,
            "core_types": ["aic"|"aiv", ...],   # indexed by core_id
            "core_to_thread": [<int>, ...]      # optional (level >= 3)
          },
          "aicore_tasks": [[core_id, task_token_raw, reg_task_id, start_cycles,
                            end_cycles, receive_to_start_cycles, run_epoch], ...],
          "scheduler_tasks": {
            "producer": "<aicpu|aicore>",
            "records": [[core_id, reg_task_id, dispatch_cycles, finish_cycles, run_epoch], ...]
          },
          "scheduler_records": {"streams": [...]},
          "aicpu_lifecycle_records": [{aicpu_thread_id, ..._cycles}, ...],
          "aicpu_orchestrator_phases":  [ [ {submit_idx, task_id, start_cycles, end_cycles, run_epoch}, ... ], ... ],
          "host_orchestrator_phases":   [ [ {submit_idx, task_id, start_host_ns, end_host_ns}, ... ], ... ]
        }

    Run identity. Per-task tokens are per-run: reg_task_id restarts at 0 every
    run, and a graph's task ids repeat whenever it is re-executed. So the join
    key is (run_epoch, core_id, reg_task_id) — the epoch is not decoration, it
    is what makes the key unique once one file holds more than one run. Task
    rows and graph execution instances carry it for the same reason.

    Older captures have no identity: aicore_tasks rows of 5 (v2) or 6 (v3)
    columns, and four-column scheduler_tasks rows whose phase records lack
    ``run_epoch``. Those parse with run_epoch = None, which means "this capture
    did not record a run identity". It is deliberately not 0 — 0 is an epoch a
    device can be given, so defaulting to it would make a legacy file claim to be
    run 0 and let it collide with a real one.

    Shape is read from the data, not from a version field: these artifacts are
    written by platform C++ in this repo and read by this module from the same
    checkout and the same build, so a declared version can never disagree with
    the rows it describes, and a producer wrong about its own rows would be
    wrong about the number too. Consistency *within* a stream is what is
    enforced instead — mixed row widths, or phase records that disagree on
    whether they carry an epoch, are rejected.

    aicore_tasks columns (v3 schema): the trailing receive_to_start_cycles
    is a uint32 delta = AICore-side `start_time - receive_time`, where
    receive_time is captured immediately after AICore's
    `read_reg(DATA_MAIN_BASE)` returns the new task_id (before the per-task
    dcci + ack pair). Lets DFX split per-task head_OH into the
    AICPU→AICore NoC propagation (dispatch_ts → receive_time, hardware-
    bound) and the AICore-local dcci + ack cost (receive_time → start_time,
    software-tunable). Archived v2 JSON without this column still parses;
    the field is exposed as 0 for those.

    ``placement`` optionally supplies a ``containment.Placement`` — where this
    capture's device clock sits on the Host CLOCK_MONOTONIC axis, derived from
    the Host span that brackets it (see ``containment``). With one, device
    records are emitted on the Host timeline. Placement diagnostics report
    block-start slack and separate phase-join freedom; saved anchor bounds
    include both. With no explicit placement, a valid saved
    ``metadata.clock_alignment`` is reused. If neither is available, device
    records remain relative and the HBG composite only preserves causal order.

    ``timeline_origin_ns`` optionally supplies a Host CLOCK_MONOTONIC origin
    shared by several same-host Rank files. It requires an explicit or saved
    placement: a Host origin is meaningless on a relative device timeline.

    Returns a dict shaped for `generate_chrome_trace_json`,
    `print_task_statistics`, and `sched_overhead_analysis`: `tasks`,
    `scheduler_records` (plus the legacy internal alias), `aicpu_orchestrator_phases`,
    `core_to_thread`.

    The join logic that used to live in `export_swimlane_json` (host C++):

      - per-core `reg_task_id → (task_token_raw, start_cycles, end_cycles)` map
        from `aicore_tasks` (the AICore is the canonical identity producer)
      - `base_time_cycles` = min non-zero timestamp across all streams (task,
        phase, orch)
      - cycles → µs via `clock_freq_hz` from metadata (a2a3=50 MHz, a5=1 GHz —
        the freq MUST come from the host, never be hardcoded here)
      - join `scheduler_tasks.records` by `(core_id, reg_task_id)`; unmatched rows are
        dropped and counted
      - archived JSON with `aicpu_tasks` is accepted as an AICPU-produced stream
      - level 1 accepts AICore-only task records; higher levels require Scheduler
        dispatch/finish timing for every emitted task
      - sort joined `tasks` by `task_id` (= task_token_raw)
      - convert phase records from `*_cycles` → `*_time_us`

    Raises:
        ValueError: If the JSON is malformed.
    """
    level = int(data.get("chip_swimlane_level"))
    if level not in [1, 2, 3, 4]:
        raise ValueError(f"Unsupported chip_swimlane_level: {level} (expected 1, 2, 3, or 4)")

    metadata = data.get("metadata") or {}
    saved_alignment = metadata.get("clock_alignment")
    if placement is None:
        try:
            placement = _read_clock_alignment(data)
        except ValueError as error:
            print(f"Warning: ignoring saved clock alignment: {error}", file=sys.stderr)
            saved_alignment = {"status": "unavailable", "reason": str(error)}
    clock_freq_hz = int(metadata.get("clock_freq_hz") or 0)
    if clock_freq_hz <= 0:
        raise ValueError(f"metadata missing/zero clock_freq_hz: {clock_freq_hz}")
    core_types = list(metadata.get("core_types") or [])
    core_to_thread = list(metadata.get("core_to_thread") or [])

    aicore_rows = data.get("aicore_tasks") or []
    scheduler_task_section = data.get("scheduler_tasks")
    legacy_aicpu_rows = data.get("aicpu_tasks")
    if scheduler_task_section is not None:
        if legacy_aicpu_rows is not None:
            raise ValueError("both scheduler_tasks and legacy aicpu_tasks are present")
        if not isinstance(scheduler_task_section, dict):
            raise ValueError("scheduler_tasks must be an object")
        scheduler_task_producer = scheduler_task_section.get("producer")
        if scheduler_task_producer not in ("aicpu", "aicore"):
            raise ValueError("scheduler_tasks.producer must be 'aicpu' or 'aicore'")
        scheduler_task_rows = scheduler_task_section.get("records")
        if not isinstance(scheduler_task_rows, list) or any(
            not isinstance(row, list) or len(row) not in (4, 5) for row in scheduler_task_rows
        ):
            raise ValueError("scheduler_tasks.records must contain four- or five-column arrays")
        # Shape is read from the rows, not from a version field. Mixed widths in
        # one stream mean the producer is inconsistent with itself, which is the
        # drift a version number was supposed to catch and cannot: the writer
        # stamps the number, so a writer that is wrong about its own rows is
        # wrong about the number too.
        row_widths = {len(row) for row in scheduler_task_rows}
        if len(row_widths) > 1:
            raise ValueError(f"scheduler_tasks.records mixes row widths {sorted(row_widths)}")
    else:
        scheduler_task_rows = legacy_aicpu_rows or []
        scheduler_task_producer = "aicpu" if legacy_aicpu_rows is not None else None
    lifecycle_raw = data.get("aicpu_lifecycle_records") or []
    if not isinstance(lifecycle_raw, list) or any(not isinstance(record, dict) for record in lifecycle_raw):
        raise ValueError("aicpu_lifecycle_records must be an array of objects")
    scheduler_stream_metadata = []
    scheduler_section = data.get("scheduler_records")
    if scheduler_section is not None:
        if not isinstance(scheduler_section, dict):
            raise ValueError("scheduler_records must be an object")
        scheduler_streams = scheduler_section.get("streams")
        if not isinstance(scheduler_streams, list):
            raise ValueError("scheduler_records.streams must be an array")
        sched_phases_raw = []
        for stream_index, stream in enumerate(scheduler_streams):
            if not isinstance(stream, dict):
                raise ValueError(f"scheduler_records.streams[{stream_index}] must be an object")
            records = stream.get("records")
            metrics = stream.get("metrics") or []
            if not isinstance(records, list) or not isinstance(metrics, list):
                raise ValueError(f"scheduler stream {stream_index} records/metrics must be arrays")
            # Exact-match on purpose: an unexpected or missing key is producer
            # drift, and this is the only place that would catch it. `run_epoch`
            # is optional only in the sense that a capture written before run
            # identity existed lacks it; within one stream every record either
            # has it or none does, which is checked below.
            record_fields = {
                "start_cycles",
                "end_cycles",
                "loop_iter",
                "kind",
                "tasks_processed",
                "task_id",
            }
            if records and isinstance(records[0], dict) and "run_epoch" in records[0]:
                record_fields = record_fields | {"run_epoch"}
            merged_records = []
            for record_index, record in enumerate(records):
                if not isinstance(record, dict) or set(record) != record_fields:
                    raise ValueError(
                        f"scheduler stream {stream_index} record {record_index} must contain exactly "
                        f"{sorted(record_fields)}"
                    )
                if int(record["end_cycles"]) < int(record["start_cycles"]):
                    raise ValueError(f"scheduler stream {stream_index} record {record_index} has a negative interval")
                merged_records.append(dict(record))
            for metric in metrics:
                if not isinstance(metric, dict) or "record_index" not in metric:
                    raise ValueError(f"scheduler stream {stream_index} has malformed metrics")
                record_index = int(metric["record_index"])
                if record_index < 0 or record_index >= len(merged_records):
                    raise ValueError(
                        f"scheduler stream {stream_index} metric record_index {record_index} is out of range"
                    )
                metric_values = {key: value for key, value in metric.items() if key != "record_index"}
                overwritten_fields = set(metric_values) & record_fields
                if overwritten_fields:
                    raise ValueError(
                        f"scheduler stream {stream_index} metric overwrites fixed record fields: "
                        f"{sorted(overwritten_fields)}"
                    )
                merged_records[record_index].update(metric_values)
            sched_phases_raw.append(merged_records)
            scheduler_stream_metadata.append(
                {
                    key: stream.get(key)
                    for key in (
                        "platform",
                        "producer",
                        "scheduler_id",
                        "worker_id",
                        "core_type",
                        "physical_core_id",
                        "capture",
                    )
                }
            )
        sched_phases_raw, scheduler_stream_metadata = _place_streams_by_scheduler_id(
            sched_phases_raw, scheduler_stream_metadata
        )
    else:
        sched_phases_raw = data.get("aicpu_scheduler_phases") or []
        scheduler_stream_metadata = [
            {
                "platform": None,
                "producer": "aicpu",
                "scheduler_id": index,
                "worker_id": index,
                "core_type": "aicpu",
                "physical_core_id": None,
                "capture": None,
            }
            for index in range(len(sched_phases_raw))
        ]
    orch_phases_raw = data.get("aicpu_orchestrator_phases") or []
    host_orch_phases_raw = data.get("host_orchestrator_phases") or []
    raw_host_capture = metadata.get("host_capture")
    host_mode = (
        metadata.get("orchestrator_source") == "host"
        or bool(host_orch_phases_raw)
        or isinstance(raw_host_capture, dict)
    )
    if orch_phases_raw and host_mode:
        raise ValueError("both AICPU and host orchestrator phases are present; clock-domain source is ambiguous")

    # Which TaskId layout the records in this document carry. Resolved once here, and
    # strictly: nothing in a task_id value says which runtime minted it, so an absent or
    # unrecognised name is refused rather than guessed at.
    runtime_name = resolve_runtime(metadata.get("runtime"))
    task_id_fields = _task_id_fields_for(runtime_name)

    actual_host_record_count = sum(len(records) for records in host_orch_phases_raw)
    if isinstance(raw_host_capture, dict):
        host_capture = dict(raw_host_capture)
        capture_status = str(host_capture.get("status") or "unknown")
        raw_dropped_records = host_capture.get("dropped_records")
        dropped_records = int(raw_dropped_records) if raw_dropped_records is not None else None
        reported_records = host_capture.get("recorded_records")
        count_matches = reported_records is None or int(reported_records) == actual_host_record_count
        # Completeness is per kind: the producer records every timed host
        # operation, of which this file carries the ones that submit a task, so
        # `expected_records` is the pass's task count and not its record count.
        # `pool_records`, when present, is the whole population and is carried
        # through for context rather than checked here.
        expected_records = host_capture.get("expected_records")
        expected_count_matches = expected_records is not None and int(expected_records) == actual_host_record_count
        host_capture_complete = (
            capture_status == "complete" and dropped_records == 0 and count_matches and expected_count_matches
        )
        validation_errors = []
        if not count_matches:
            validation_errors.append("recorded_record_count_mismatch")
        if expected_records is None:
            validation_errors.append("expected_record_count_missing")
        elif not expected_count_matches:
            validation_errors.append("expected_record_count_mismatch")
        if validation_errors:
            host_capture["converter_validation_errors"] = validation_errors
    elif host_mode:
        host_capture = {
            "status": "unknown",
            "recorded_records": actual_host_record_count,
            "dropped_records": None,
            "error": "legacy_capture_status_missing",
        }
        host_capture_complete = False
    else:
        host_capture = None
        host_capture_complete = False

    host_timestamps = [
        int(pr[field])
        for thread_records in host_orch_phases_raw
        for pr in thread_records
        for field in ("start_host_ns", "end_host_ns")
    ]
    source_host_origin_ns = int(
        metadata.get("host_orchestration_origin_ns") or metadata.get("host_timeline_origin_ns") or 0
    )
    if host_timestamps and source_host_origin_ns == 0:
        source_host_origin_ns = min(host_timestamps)
    host_origin_ns = source_host_origin_ns
    composite_timestamps = host_timestamps + [
        int(record[field])
        for record in data.get("host_device_uploads") or []
        for field in ("start_host_ns", "end_host_ns")
    ]
    # The unaligned composite preserves upload-before-dispatch causality only;
    # its seam is not a measurement of cross-domain latency.
    host_composite_end_us = (max(composite_timestamps) - host_origin_ns) / 1000.0 if composite_timestamps else 0.0

    # AICore lookup keyed by (run_epoch, core_id, reg_task_id). Two dispatches of
    # the same task_token_raw to the same core (SPMD over-subscription, MIX
    # cluster spread) each get their own reg_task_id, so core+reg_task_id is
    # unique per dispatch *within one run* even when task_token_raw collides.
    # It is not unique across runs: reg_task_id restarts at 0 every run, so a
    # file holding two runs has the same core+reg_task_id twice. run_epoch is
    # what separates them, which is why it leads the key.
    #
    # `*rest` makes three generations of row parse: v2 (5 cols, no
    # receive_to_start_cycles), v3 (6 cols) and v4 (7 cols, + run_epoch).
    # A pre-v4 row has no identity, and `None` records exactly that — it is
    # never coerced to 0, which is a real epoch a device can be given.
    # Key is (run_epoch, core_id, reg_task_id); run_epoch is None for a
    # pre-identity capture. Spelled loosely because this module has no
    # `from __future__ import annotations` and pyright targets 3.9.
    aicore_lookup: dict[tuple, tuple[int, int, int, int]] = {}
    for row_index, row in enumerate(aicore_rows):
        if not isinstance(row, list) or len(row) not in (5, 6, 7):
            raise ValueError(f"aicore_tasks[{row_index}] must contain five, six or seven columns")
        core_id, task_token_raw, reg_task_id, start_cycles, end_cycles, *rest = row
        start_cycles = int(start_cycles)
        end_cycles = int(end_cycles)
        r2s_cycles = int(rest[0]) if rest else 0
        run_epoch = int(rest[1]) if len(rest) > 1 else None
        if not (0 < start_cycles <= end_cycles):
            raise ValueError(f"aicore_tasks[{row_index}] has invalid timing: expected 0 < start_cycles <= end_cycles")
        if not (0 <= r2s_cycles < start_cycles):
            raise ValueError(
                f"aicore_tasks[{row_index}] has invalid receive_to_start_cycles: "
                "expected 0 <= receive_to_start_cycles < start_cycles"
            )
        key = (run_epoch, int(core_id), int(reg_task_id))
        if key in aicore_lookup:
            raise ValueError(f"duplicate aicore_tasks join key: {key}")
        aicore_lookup[key] = (
            int(task_token_raw),
            start_cycles,
            end_cycles,
            r2s_cycles,
        )

    scheduler_task_keys = [_scheduler_task_key(row) for row in scheduler_task_rows]
    if len(scheduler_task_keys) != len(set(scheduler_task_keys)):
        raise ValueError("scheduler_tasks contains duplicate (run_epoch, core_id, reg_task_id) join keys")
    for row_index, row in enumerate(scheduler_task_rows):
        dispatch_cycles = int(row[2])
        finish_cycles = int(row[3])
        if not (0 < dispatch_cycles <= finish_cycles):
            raise ValueError(
                f"scheduler_tasks.records[{row_index}] has invalid timing: "
                "expected 0 < dispatch_cycles <= finish_cycles"
            )
    if level >= 2:
        missing_scheduler_keys = sorted(set(aicore_lookup) - set(scheduler_task_keys))
        if missing_scheduler_keys:
            preview = ", ".join(str(key) for key in missing_scheduler_keys[:3])
            raise ValueError(
                f"level {level} requires Scheduler task timing for every AICore task; "
                f"missing {len(missing_scheduler_keys)} join key(s): {preview}"
            )

    # base_time = min non-zero timestamp across every stream that will be
    # emitted. Used as the cycle-domain zero for all µs conversions.
    base_time_cycles = None

    def _track(v):
        nonlocal base_time_cycles
        if v > 0 and (base_time_cycles is None or v < base_time_cycles):
            base_time_cycles = v

    for row in aicore_rows:
        # Column count varies (v2: 5, v3: 6); only the timing columns matter
        # for base_time tracking. For v3, the per-task receive_time =
        # start_cycles - receive_to_start_cycles is earlier than start_cycles
        # itself; track it so Worker View task bars that start at receive_time
        # don't land at a negative offset relative to the kernel start.
        start_c = int(row[3])
        end_c = int(row[4])
        r2s_c = int(row[5]) if len(row) > 5 else 0
        _track(start_c - r2s_c)
        _track(end_c)
    for _, _, d, f, *_epoch in scheduler_task_rows:
        _track(int(d))
        _track(int(f))
    for thread_records in sched_phases_raw:
        for pr in thread_records:
            _track(int(pr.get("start_cycles", 0)))
            _track(int(pr.get("end_cycles", 0)))
    for thread_records in orch_phases_raw:
        for pr in thread_records:
            _track(int(pr.get("start_cycles", 0)))
            _track(int(pr.get("end_cycles", 0)))
    lifecycle_cycle_fields = (
        "handshake_start_cycles",
        "handshake_complete_cycles",
        "config_start_cycles",
        "topology_complete_cycles",
        "context_publish_start_cycles",
        "context_publish_complete_cycles",
        "bootstrap_wait_start_cycles",
        "bootstrap_complete_cycles",
        "register_release_start_cycles",
        "register_release_end_cycles",
        "exit_signal_start_cycles",
        "exit_signal_end_cycles",
        "exit_wait_start_cycles",
        "exit_wait_end_cycles",
    )
    for record in lifecycle_raw:
        for field in lifecycle_cycle_fields:
            _track(int(record.get(field, 0)))

    if base_time_cycles is None:
        base_time_cycles = 0

    source_host_origin_ns = host_origin_ns
    if placement is not None and host_origin_ns == 0:
        # A capture with no Host records of its own still gets a Host origin
        # from the window it is placed in, so its lane starts where the run did
        # rather than at an arbitrary first record.
        host_origin_ns = int(placement.place_lo_ns)
    if timeline_origin_ns is not None:
        if placement is None:
            raise ValueError("a shared timeline origin requires a containment placement for this capture")
        host_origin_ns = int(timeline_origin_ns)
        if host_origin_ns <= 0:
            raise ValueError(f"invalid shared timeline origin: {host_origin_ns}")

    cycles_to_us_factor = 1_000_000.0 / float(clock_freq_hz)

    def _to_us(cycles):
        if cycles <= 0:
            return 0.0
        if placement is not None:
            return (placement.map_cycles_to_host_ns(cycles) - host_origin_ns) / 1000.0
        relative_device_us = (cycles - base_time_cycles) * cycles_to_us_factor
        if host_mode:
            # Fail-soft diagnostic layout: preserve both clock domains and only
            # encode the known happens-before relation. The physical gap at
            # this seam is unknown and must never feed cross-domain latency.
            return host_composite_end_us + relative_device_us
        return relative_device_us

    def _core_type(core_id):
        if 0 <= core_id < len(core_types):
            return core_types[core_id]
        return "aiv"

    tasks = []
    unmatched_per_core: dict[int, int] = defaultdict(int)

    if scheduler_task_rows:
        for row in scheduler_task_rows:
            core_id, reg_task_id, dispatch_cycles, finish_cycles, *epoch_rest = row
            core_id = int(core_id)
            reg_task_id = int(reg_task_id)
            run_epoch = int(epoch_rest[0]) if epoch_rest else None
            ac = aicore_lookup.get((run_epoch, core_id, reg_task_id))
            if ac is None:
                unmatched_per_core[core_id] += 1
                continue
            task_token_raw, start_cycles, end_cycles, r2s_cycles = ac
            dispatch_cycles = int(dispatch_cycles)
            finish_cycles = int(finish_cycles)
            start_us = _to_us(start_cycles)
            end_us = _to_us(end_cycles)
            dispatch_us = _to_us(dispatch_cycles)
            receive_us = _to_us(start_cycles - r2s_cycles)
            local_setup_us = start_us - receive_us
            tasks.append(
                {
                    "task_id": task_token_raw,
                    "func_id": -1,
                    "core_id": core_id,
                    "core_type": _core_type(core_id),
                    **task_id_fields(task_token_raw),
                    "start_time_us": start_us,
                    "end_time_us": end_us,
                    "duration_us": end_us - start_us,
                    "dispatch_time_us": dispatch_us,
                    "finish_time_us": _to_us(finish_cycles),
                    "receive_time_us": receive_us,
                    "local_setup_us": local_setup_us,
                    "propagation_us": receive_us - dispatch_us,
                    "run_epoch": run_epoch,
                }
            )
    elif aicore_rows and level == 1:
        for row in aicore_rows:
            core_id, task_token_raw, _reg_task_id, start_cycles, end_cycles, *rest = row
            r2s_cycles = int(rest[0]) if rest else 0
            run_epoch = int(rest[1]) if len(rest) > 1 else None
            core_id = int(core_id)
            task_token_raw = int(task_token_raw)
            start_us = _to_us(int(start_cycles))
            end_us = _to_us(int(end_cycles))
            receive_us = _to_us(int(start_cycles) - r2s_cycles)
            local_setup_us = start_us - receive_us
            tasks.append(
                {
                    "task_id": task_token_raw,
                    "func_id": -1,
                    "core_id": core_id,
                    "core_type": _core_type(core_id),
                    **task_id_fields(task_token_raw),
                    "start_time_us": start_us,
                    "end_time_us": end_us,
                    "duration_us": end_us - start_us,
                    "receive_time_us": receive_us,
                    "local_setup_us": local_setup_us,
                    "run_epoch": run_epoch,
                    # propagation_us requires a Scheduler dispatch timestamp.
                }
            )
    elif aicore_rows:
        raise ValueError(f"level {level} requires Scheduler task timing records")

    # Sorting by task_id alone would interleave two runs' executions of the same
    # task. Epoch leads so each run's tasks stay contiguous; None (a pre-identity
    # capture) sorts before any real epoch and cannot compare against one.
    tasks.sort(key=lambda t: (t.get("run_epoch") is not None, t.get("run_epoch") or 0, int(t["task_id"])))

    total_unmatched = sum(unmatched_per_core.values())
    if total_unmatched > 0:
        worst = sorted(unmatched_per_core.items(), key=lambda kv: -kv[1])[:3]
        worst_str = ", ".join(f"core {c}: {n}" for c, n in worst)
        print(
            f"Warning: {total_unmatched} Scheduler task timing record(s) had no matching AICore record "
            f"(producer={scheduler_task_producer}, top offenders: {worst_str}); "
            "the missing AICore buffer(s) were dropped on rotation. Bump PLATFORM_AICORE_BUFFERS_PER_CORE if you "
            "see this regularly.",
            file=sys.stderr,
        )

    def _phase_us(pr):
        # Host already omits pop_hit / pop_miss for Complete records (terse
        # emit), so we don't need to re-strip zero deltas here.
        out = dict(pr)
        out["start_time_us"] = _to_us(int(pr.get("start_cycles", 0)))
        out["end_time_us"] = _to_us(int(pr.get("end_cycles", 0)))
        out.pop("start_cycles", None)
        out.pop("end_cycles", None)
        return out

    aicpu_scheduler_phases = []
    for thread_records in sched_phases_raw:
        converted = []
        for pr in thread_records:
            kind = pr.get("kind", "unknown")
            out = _phase_us(pr)
            # Downstream code branches on "phase" as the sched-record
            # discriminator; surface "kind" under that name.
            out["phase"] = kind
            out.pop("kind", None)
            converted.append(out)
        aicpu_scheduler_phases.append(converted)

    aicpu_orchestrator_phases = []
    for thread_records in orch_phases_raw:
        converted = []
        for pr in thread_records:
            out = _phase_us(pr)
            out["phase"] = "orch_submit"
            converted.append(out)
        aicpu_orchestrator_phases.append(converted)

    aicpu_lifecycle_records = []
    for record_index, record in enumerate(lifecycle_raw):
        converted = dict(record)
        for field in lifecycle_cycle_fields:
            if field not in converted:
                continue
            cycles = int(converted.pop(field))
            if cycles > 0:
                converted[field.removesuffix("_cycles") + "_time_us"] = _to_us(cycles)
        converted["record_index"] = record_index
        aicpu_lifecycle_records.append(converted)

    host_device_uploads = []
    for pr in data.get("host_device_uploads") or []:
        start_ns = int(pr.get("start_host_ns", 0))
        end_ns = int(pr.get("end_host_ns", 0))
        if start_ns < host_origin_ns or end_ns < start_ns:
            raise ValueError(f"invalid host device upload: origin={host_origin_ns}, start={start_ns}, end={end_ns}")
        out = dict(pr)
        out["start_time_us"] = (start_ns - host_origin_ns) / 1000.0
        out["end_time_us"] = (end_ns - host_origin_ns) / 1000.0
        out.pop("start_host_ns", None)
        out.pop("end_host_ns", None)
        host_device_uploads.append(out)

    host_orchestrator_phases = []
    for thread_records in host_orch_phases_raw:
        converted = []
        for pr in thread_records:
            start_ns = int(pr.get("start_host_ns", 0))
            end_ns = int(pr.get("end_host_ns", 0))
            if start_ns < host_origin_ns or end_ns < start_ns:
                raise ValueError(
                    f"invalid host orchestrator phase: origin={host_origin_ns}, start={start_ns}, end={end_ns}"
                )
            out = dict(pr)
            out["start_time_us"] = (start_ns - host_origin_ns) / 1000.0
            out["end_time_us"] = (end_ns - host_origin_ns) / 1000.0
            out["phase"] = "orch_submit"
            out.pop("start_host_ns", None)
            out.pop("end_host_ns", None)
            converted.append(out)
        if converted:
            host_orchestrator_phases.append(converted)

    out = {
        "chip_swimlane_level": level,
        "tasks": tasks,
    }
    # Carried through so every downstream stage picks the same TaskId layout this
    # decode did, rather than re-deriving it from something that only correlates.
    # Unconditional: resolve_runtime above already refused a document without it.
    out["runtime"] = runtime_name
    if scheduler_task_producer is not None:
        out["scheduler_task_producer"] = scheduler_task_producer
    if aicpu_scheduler_phases:
        out["aicpu_scheduler_phases"] = aicpu_scheduler_phases
        out["scheduler_records"] = aicpu_scheduler_phases
        out["scheduler_streams"] = scheduler_stream_metadata
    if aicpu_orchestrator_phases:
        out["aicpu_orchestrator_phases"] = aicpu_orchestrator_phases
        out["orchestrator_source"] = "aicpu"
    if aicpu_lifecycle_records:
        out["aicpu_lifecycle_records"] = aicpu_lifecycle_records
    if host_device_uploads:
        out["host_device_uploads"] = host_device_uploads
    if host_mode:
        out["orchestrator_source"] = "host"
        trace_status = "complete" if host_capture_complete else "partial"
        out["timeline_metadata"] = {
            "layout": "containment_spliced" if placement is not None else "causal_composite",
            "trace_status": trace_status,
            "relation": metadata.get("timeline_relation", "host_orchestration_precedes_device"),
            "host_capture": host_capture,
            "host_records_complete": host_capture_complete,
            # Cross-domain gaps retain placement and phase-join uncertainty,
            # represented together by saved anchor bounds when available.
            "cross_domain_latency_available": placement is not None and host_capture_complete,
            "source_timeline_origin_ns": source_host_origin_ns,
            "timeline_origin_ns": host_origin_ns,
        }
        if placement is None:
            out["timeline_metadata"].update(
                {
                    "cross_domain_gap_unknown": True,
                    "logical_seam_us": host_composite_end_us,
                }
            )
        if host_orchestrator_phases:
            out["aicpu_orchestrator_phases"] = host_orchestrator_phases
        else:
            out["timeline_metadata"]["host_records_missing"] = True
    elif placement is not None:
        out["timeline_metadata"] = {
            "layout": "containment_spliced",
            # The placement is what makes the capture readable on the Host axis;
            # whether the capture itself is whole is `host_capture`'s business,
            # and a capture with no Host records of its own has none to lose.
            "trace_status": "complete",
            "host_records_complete": False,
            "cross_domain_latency_available": True,
            "source_timeline_origin_ns": source_host_origin_ns,
            "timeline_origin_ns": host_origin_ns,
        }
    if saved_alignment is not None and "timeline_metadata" not in out:
        out["timeline_metadata"] = {"layout": "device_relative", "cross_domain_latency_available": False}
    if "timeline_metadata" in out:
        if saved_alignment is not None:
            out["timeline_metadata"]["clock_alignment"] = saved_alignment
        if isinstance(placement, containment.Placement):
            out["timeline_metadata"]["placement"] = placement.metadata()
        host_clock_domain_id = metadata.get("host_clock_domain_id")
        if host_clock_domain_id:
            out["timeline_metadata"]["host_clock_domain_id"] = str(host_clock_domain_id)
    if core_to_thread:
        out["core_to_thread"] = core_to_thread
    return out


def load_deps_json(deps_path):
    """Load a dep_gen replay output (``deps.json``).

    deps.json is the sole source of truth for the task graph in this tool:
    the device hot path no longer records per-task fanout (see PR #863). The
    typical workflow is a dep_gen run once per topology (``--enable-dep-gen``)
    to produce ``deps.json``, then any number of ``--enable-chip-swimlane`` runs
    that join their per-task timing against that captured graph.

    Returns:
        dict[int, list[int]] mapping ``pred_raw → [succ_raw, ...]``, or
        ``None`` if the file is missing, unreadable, or not v2-shaped. Tasks
        with no successors are absent from the dict (``defaultdict``-like
        lookup-miss semantics).
    """
    deps_path = Path(deps_path)
    if not deps_path.exists():
        return None
    try:
        with deps_path.open() as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print(f"Warning: failed to read {deps_path}: {e}", file=sys.stderr)
        return None
    edges = data.get("edges")
    if not isinstance(edges, list):
        print(f"Warning: {deps_path} has no 'edges' array", file=sys.stderr)
        return None
    # The converter only needs flow-event endpoints (not the per-edge tensor
    # annotations). Project annotated edges down to a (pred, succ) set and
    # dedup so multiple annotated edges sharing the same pair (distinct arg
    # / source / overlap) collapse to a single flow event.
    by_pred: dict[int, list[int]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        pred = normalize_task_id_int(edge.get("pred"))
        succ = normalize_task_id_int(edge.get("succ"))
        if pred is None or succ is None:
            continue
        key = (pred, succ)
        if key in seen:
            continue
        seen.add(key)
        by_pred[pred].append(succ)
    return dict(by_pred)


def load_deps_kernel_map(deps_path):
    """Build a ``task_id → kernel_ids[3]`` map from deps.json's ``tasks[]``.

    a2a3 dep_gen captures per-task ``kernel_ids = [aic, aiv0, aiv1]`` so the
    swimlane post-processor can resolve ``func_id`` at TASK_TIMING (level=1)
    where the AICore record alone is on disk and carries ``func_id == -1``.
    The trace generator uses the per-record ``core_type`` to pick the right
    subslot: ``aic → kernel_ids[0]``, ``aiv → kernel_ids[1]`` (falling back
    to ``[2]`` if AIV0 is inactive). Same pattern fanout edges already use
    (deps.json is the offline-joined identity source).

    Returns:
        dict[int, list[int]] mapping ``task_id_raw → [aic, aiv0, aiv1]``,
        or ``None`` if the file is missing / unreadable / lacks the field.
        Entries without ``kernel_ids`` (pre-schema deps.json from older
        runs) are silently skipped — the caller treats a missing map as
        "no override available" and emits the ``func_-1_(...)`` fallback.
    """
    deps_path = Path(deps_path)
    if not deps_path.exists():
        return None
    try:
        with deps_path.open() as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return None
    kmap: dict[int, list[int]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        tid = normalize_task_id_int(task.get("task_id"))
        kids = task.get("kernel_ids")
        if tid is None or not isinstance(kids, list) or len(kids) != 3:
            continue
        kmap[tid] = [int(k) for k in kids]
    return kmap if kmap else None


def load_deps_block_map(deps_path):
    """Build a ``task_id → block_num`` map from deps.json's ``tasks[]``.

    Returns:
        dict[int, int] mapping ``task_id_raw → block_num``, or ``None`` if
        the file is missing / unreadable / lacks the field. Entries without
        ``block_num`` default to 1 (non-SPMD).
    """
    deps_path = Path(deps_path)
    if not deps_path.exists():
        return None
    try:
        with deps_path.open() as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return None
    bmap: dict[int, int] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        tid = normalize_task_id_int(task.get("task_id"))
        if tid is None:
            continue
        try:
            block_num = int(task.get("block_num", 1))
        except (TypeError, ValueError):
            block_num = 1
        block_num = max(block_num, 1)
        bmap[tid] = block_num
    return bmap if bmap else None


def _identify_spmd_task_ids(task_maps_by_run, deps_block_map=None):
    """Return task_ids whose dependency flow endpoints collapse to one subtask row.

    Multiplicity is counted **within a single run**. A task that executes once
    per run is not SPMD, however many runs a capture holds: counting rows in a
    map merged across runs turns every repeated single-core task into a
    spurious SPMD task, which both mislabels it and makes its dependency fan
    count the number of runs.

    ``deps_block_map`` stays authoritative where present, in both directions —
    ``block_num > 1`` marks SPMD outright, and ``block_num == 1`` is not
    second-guessed by observation.
    """
    spmd_ids: set[int] = set()
    if deps_block_map:
        for tid, block_num in deps_block_map.items():
            if block_num > 1:
                spmd_ids.add(tid)
    for task_map in task_maps_by_run.values():
        for tid, recs in task_map.items():
            if tid in spmd_ids or len(recs) <= 1:
                continue
            if deps_block_map and tid in deps_block_map:
                continue  # authoritative — don't second-guess block_num==1
            core_types = {r.get("core_type") for r in recs}
            if len(core_types) == 1:
                spmd_ids.add(tid)
    return spmd_ids


def _task_slice_start_us(task):
    """Start of the task slice emitted in Worker View."""
    receive_time_us = task.get("receive_time_us")
    return receive_time_us if receive_time_us is not None else task["start_time_us"]


def _scheduler_slice_start_us(task):
    """Start of the task slice emitted in Scheduler View."""
    dispatch_time_us = task.get("dispatch_time_us")
    finish_time_us = task.get("finish_time_us")
    if dispatch_time_us is None or dispatch_time_us < 0 or finish_time_us is None or finish_time_us <= 0:
        return float("inf")
    return dispatch_time_us


def _with_run_epoch(args, row):
    """Add ``run_epoch`` to an event's args when the row carries one.

    Omitted rather than defaulted when the row has no identity: a capture
    written before run identity existed genuinely does not know its run, and 0
    is an epoch a device can really be given, so filling it in would make the
    unknown indistinguishable from run 0.
    """
    run_epoch = row.get("run_epoch")
    if run_epoch is not None:
        args["run_epoch"] = run_epoch
    return args


# An absent (thread, run) bucket: no phases, and therefore no starts.
_NO_COMPLETES: tuple = ((), ())


def _build_complete_index(scheduler_phases):
    """Per-thread complete-phase lookup, built once for the whole conversion.

    Each entry carries the thread's completes in start order (``all``) and the
    same records bucketed by run, each bucket paired with its start times
    (``by_run``). The starts are cached next to the phases because every finish
    bisects them: rebuilding that list per query turns an indexed lookup into a
    full scan of the run's phases, and both the finish counter and the
    completion arrows query once per task.
    """
    index = []
    for thread_records in scheduler_phases:
        sorted_completes = sorted(
            (r for r in thread_records if r.get("phase") == "complete"),
            key=lambda r: r["start_time_us"],
        )
        by_run: dict = defaultdict(list)
        for record in sorted_completes:
            by_run[record.get("run_epoch")].append(record)
        index.append(
            {
                "all": sorted_completes,
                "by_run": {epoch: (phases, [c["start_time_us"] for c in phases]) for epoch, phases in by_run.items()},
            }
        )
    return index


def _select_complete_for_finish(indexed_phases, finish_us):
    """The complete phase containing ``finish_us``, else the next one to start.

    ``indexed_phases`` is one run's ``(phases, starts)`` pair from
    :func:`_build_complete_index`. Complete phases on one thread do not overlap,
    so the only candidate that can contain the timestamp is the last one
    starting at or before it. The next-start fallback covers a finish the
    scheduler had not yet drained — and is exactly why the pair has to be one
    run's: across runs it would step forward into a phase that executed in a
    different execution of the graph.
    """
    phases, starts = indexed_phases
    if not phases:
        return None
    idx = bisect.bisect_right(starts, finish_us)
    if idx > 0:
        prev_c = phases[idx - 1]
        if prev_c["start_time_us"] <= finish_us <= prev_c["end_time_us"]:
            return prev_c
    if idx < len(phases):
        return phases[idx]
    return None


def _execution_key(row):
    """Identity of one execution instance: ``(run_epoch, task_id, core_id)``.

    Task ids and core ids both repeat across runs, so an index keyed on that
    pair alone is overwritten by whichever run is emitted last, and every
    earlier run's flow then binds to a slice belonging to the later one. The
    static graph in deps.json is run-independent and stays keyed by bare
    task_id; this key is only for per-execution indexes.

    ``run_epoch`` is ``None`` for a capture that recorded no identity, which
    keeps legacy rows in their own domain rather than sharing one with run 0.
    """
    return (row.get("run_epoch"), row["task_id"], row["core_id"])


def _flow_anchor_rows(task_id, task_map, spmd_task_ids, slice_start, aicpu_worker_anchor_map=None):
    """Flow anchor rows selected by one view's visible slice start.

    Non-SPMD keeps every subtask row. SPMD collapses rows by
    ``(func_id, task_id)`` and keeps the earliest visible slice in each group,
    so MIX tasks with shared task_id but distinct AIC/AIV func_id values remain
    visible as separate dependency endpoints.

    Both ``task_map`` and ``aicpu_worker_anchor_map`` must already be scoped to
    a single run. Task ids repeat across runs, so a map spanning two of them
    returns anchors from both: either mixed endpoints, or — when the merged rows
    make one task's two executions look like two SPMD subtasks — the earliest
    run's row standing in for every run.
    """
    recs = task_map.get(task_id, [])
    if not recs and aicpu_worker_anchor_map:
        return aicpu_worker_anchor_map.get(task_id, [])
    if not recs:
        return []
    if task_id not in spmd_task_ids:
        return recs
    by_func: dict[int, dict] = {}
    for row in recs:
        func_id = row.get("func_id", -1)
        prev = by_func.get(func_id)
        if prev is None or (slice_start(row), row.get("core_id", 0)) < (
            slice_start(prev),
            prev.get("core_id", 0),
        ):
            by_func[func_id] = row
    return list(by_func.values())


def _worker_flow_anchor_rows(task_id, task_map, spmd_task_ids, aicpu_worker_anchor_map=None):
    """Worker View flow anchors."""
    return _flow_anchor_rows(task_id, task_map, spmd_task_ids, _task_slice_start_us, aicpu_worker_anchor_map)


def _scheduler_flow_anchor_rows(task_id, task_map, spmd_task_ids, aicpu_worker_anchor_map=None):
    """Scheduler View flow anchors."""
    return _flow_anchor_rows(task_id, task_map, spmd_task_ids, _scheduler_slice_start_us, aicpu_worker_anchor_map)


def _flow_row_pairs(pred_id, succ_id, task_map, spmd_task_ids, anchor_rows, aicpu_worker_anchor_map=None):
    """(pred_row, succ_row) pairs for one logical dependency edge in one view."""
    pred_rows = anchor_rows(pred_id, task_map, spmd_task_ids, aicpu_worker_anchor_map)
    succ_rows = anchor_rows(succ_id, task_map, spmd_task_ids, aicpu_worker_anchor_map)
    if not pred_rows or not succ_rows:
        return []
    return [(pred_row, succ_row) for pred_row in pred_rows for succ_row in succ_rows]


def _dependency_task_fan_count(task_id, spmd_task_ids, task_map, deps_block_map=None):
    """Logical subtask count for dependency metadata (SPMD block_num, else 1).

    ``task_map`` is this run's. A task the run never executed on AICore — a
    dummy, predicated-skip or alloc node anchored on an AICPU slice — has no row
    here and contributes one logical endpoint, so the count is defined without
    dereferencing a row that does not exist.
    """
    if task_id in spmd_task_ids:
        if deps_block_map and task_id in deps_block_map:
            return deps_block_map[task_id]
        return len(task_map.get(task_id) or ()) or 1
    return 1


def _append_dependency_flow_pair(  # noqa: PLR0913
    events,
    flow_id,
    flow_name,
    src_pid,
    src_tid,
    src_ts,
    src_event_id,
    dst_pid,
    dst_tid,
    dst_ts,
    dst_event_id,
    *,
    input_task_count=1,
    output_task_count=1,
):
    flow_s = {
        "cat": "flow",
        "id": flow_id,
        "name": flow_name,
        "ph": "s",
        "pid": src_pid,
        "tid": src_tid,
        "ts": src_ts,
        "input_task_count": input_task_count,
        "output_task_count": output_task_count,
    }
    if src_event_id is not None:
        flow_s["bind_id"] = src_event_id
    events.append(flow_s)

    flow_f = {
        "cat": "flow",
        "id": flow_id,
        "name": flow_name,
        "ph": "f",
        "pid": dst_pid,
        "tid": dst_tid,
        "ts": dst_ts,
        "bp": "e",
        "input_task_count": input_task_count,
        "output_task_count": output_task_count,
    }
    if dst_event_id is not None:
        flow_f["bind_id"] = dst_event_id
    events.append(flow_f)


def resolve_func_id_from_kernel_map(task_id, core_type, kernel_map):
    """Look up the active ``func_id`` for a TASK_TIMING record via dep_gen.

    Picks the kernel_ids[3] subslot by record ``core_type``. Returns the
    resolved func_id (>= 0) on a hit, or -1 if no usable subslot was found
    (caller keeps the original -1 and emits the ``func_-1_(...)`` fallback
    name). The choice for ``aiv`` prefers AIV0 ([1]) and falls back to AIV1
    ([2]) — works for pure-AIV and MIX-with-single-AIV records; for MIX
    records that span both AIVs the host swimlane record only tells us the
    lane is "aiv", so the resolver may name an AIV1 lane after AIV0's
    kernel. Acceptable trade-off until the host emits a lane-disambiguated
    core_type ("aiv0" / "aiv1").
    """
    if kernel_map is None or task_id is None:
        return -1
    kids = kernel_map.get(int(task_id))
    if not kids:
        return -1
    if core_type == "aic":
        return kids[0] if kids[0] >= 0 else -1
    # "aiv": prefer AIV0, fall back to AIV1.
    for idx in (1, 2):
        if kids[idx] >= 0:
            return kids[idx]
    return -1


def resolve_task_func_id_from_kernel_map(task_id, kernel_map):
    """Look up the first active ``func_id`` for a task without an AICore row."""
    if kernel_map is None or task_id is None:
        return -1
    kids = kernel_map.get(int(task_id))
    if not kids:
        return -1
    return next((func_id for func_id in kids if func_id >= 0), -1)


def load_kernel_config(config_path):
    """Load kernel configuration from kernel_config.py file.

    Args:
        config_path: Path to kernel_config.py file

    Returns:
        dict: Mapping from func_id (as string) to function name
              Example: {"0": "QK", "1": "SF", "2": "PV", "3": "UP"}
              Entries without 'func_id' or 'name' are skipped with a warning

    Raises:
        ValueError: If file cannot be loaded or KERNELS definition is missing
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ValueError(f"Kernel config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("kernel_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract func_id to name mapping from KERNELS list
    if not hasattr(module, "KERNELS"):
        raise ValueError("kernel_config.py missing KERNELS definition")

    func_id_to_name = {}
    for kernel in module.KERNELS:
        # Skip entries without func_id
        if "func_id" not in kernel:
            print(f"Warning: Kernel entry missing 'func_id', skipping: {kernel}", file=sys.stderr)
            continue

        func_id = kernel["func_id"]

        # If name is missing, we'll fall back to default naming (Func_{func_id})
        if "name" not in kernel:
            print(
                f"Warning: Kernel entry for func_id={func_id} missing 'name', will use default naming",
                file=sys.stderr,
            )
            continue

        # Store as string to match JSON format
        func_id_to_name[str(func_id)] = kernel["name"]

    return func_id_to_name


def load_func_names_json(json_path):
    """Load name mapping from a SceneTest JSON file.

    Each level's mapping carries ``callable_id_to_name`` for its
    next-level-down callables and a ``level`` tag.  The tool uses
    ``callable_id_to_name`` directly — no cross-level merging.

    Returns:
        tuple: (callable_id_to_name dict, orchestrator_name str or None)
    """
    path = Path(json_path)
    if not path.exists():
        raise ValueError(f"Func names JSON not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return data.get("callable_id_to_name", {}), data.get("orchestrator_name")


def print_task_statistics(tasks, func_id_to_name=None, chip_swimlane_level=None):
    """Print task statistics grouped by func_id.

    Exec = kernel execution time (end_time_us - start_time_us) on AICore.
    Latency = Scheduler view: finish_time_us - dispatch_time_us (includes head OH + Exec + tail OH).
    High Latency with low Exec means scheduler/polling overhead (tail OH = finish_ts recorded
    when the scheduler loop next sees the completed handshake; reordering the loop to process
    completed tasks first reduces this).

    Args:
        tasks: List of task dicts
        func_id_to_name: Optional dict mapping func_id to function name
        chip_swimlane_level: Source collection level. Level 2 and above include
            Scheduler per-task dispatch/finish timestamps.
    """
    has_scheduler_timing = any(
        task.get("dispatch_time_us", -1) >= 0 and task.get("finish_time_us", 0) > 0 for task in tasks
    )

    # Group tasks by func_id with extended metrics
    func_stats: defaultdict[Any, dict[str, Any]] = defaultdict(
        lambda: {
            "durations": [],
            "head_overheads": [],
            "tail_overheads": [],
            "propagations": [],  # dispatch_ts → AICore receive_time (NoC + FFTS)
            "local_setups": [],  # receive_time → start_time (dcci + ack on AICore)
            "latencies": [],
            "total_exec_time": 0.0,
            "total_latency": 0.0,
        }
    )

    # Track global min dispatch and max finish times
    min_dispatch_time = float("inf")
    max_finish_time = float("-inf")
    min_aicore_time = float("inf")
    max_aicore_time = float("-inf")

    for task in tasks:
        func_id = task["func_id"]
        duration = task["duration_us"]
        func_stats[func_id]["durations"].append(duration)

        start_time = task["start_time_us"]
        end_time = task["end_time_us"]
        receive_time = task.get("receive_time_us")
        min_aicore_time = min(min_aicore_time, receive_time if receive_time is not None else start_time)
        max_aicore_time = max(max_aicore_time, end_time)

        if "local_setup_us" in task:
            func_stats[func_id]["local_setups"].append(task["local_setup_us"])

        # Calculate new metrics if dispatch_time_us and finish_time_us are available
        if has_scheduler_timing and "dispatch_time_us" in task and "finish_time_us" in task:
            dispatch_time = task["dispatch_time_us"]
            finish_time = task["finish_time_us"]

            # Head overhead: start_time_us - dispatch_time_us
            head_overhead = start_time - dispatch_time
            func_stats[func_id]["head_overheads"].append(head_overhead)

            # Tail overhead: finish_time_us - end_time_us
            tail_overhead = finish_time - end_time
            func_stats[func_id]["tail_overheads"].append(tail_overhead)

            # Head OH split (v3 schema only — falls back to absent when the
            # AICore record came from a pre-receive_time build).
            if "propagation_us" in task:
                func_stats[func_id]["propagations"].append(task["propagation_us"])

            # Latency: finish_time_us - dispatch_time_us
            latency = finish_time - dispatch_time
            func_stats[func_id]["latencies"].append(latency)

            # Accumulate execution time and latency for ratio calculation
            func_stats[func_id]["total_exec_time"] += duration
            func_stats[func_id]["total_latency"] += latency

            # Track global times
            min_dispatch_time = min(min_dispatch_time, dispatch_time)
            max_finish_time = max(max_finish_time, finish_time)

    # Print statistics
    print("\n" + "=" * 140)
    print("Task Statistics by Function")
    level_descriptions = {
        1: "AICore timing only",
        2: "AICore + Scheduler task timing",
        3: "AICore + Scheduler task timing + scheduler phases",
        4: "full collection with orchestrator phases",
    }
    if chip_swimlane_level is None:
        level_description = "unknown"
        level_value = "unknown"
    else:
        level_description = level_descriptions.get(chip_swimlane_level, "unknown")
        level_value = chip_swimlane_level
    print(f"  Source chip_swimlane_level: {level_value} ({level_description}; recorded in chip_swimlane_records.json)")
    print("  Exec = kernel time on AICore; Latency = dispatch->finish (incl. head OH + Exec + tail OH)")
    print("  Head OH split (v3): Prop = NoC propagation (dispatch_ts→AICore receive); Local = dcci+ack (receive→start)")
    print("=" * 140)
    print(
        f"{'Func_ID':<8} {'Func_Name':<12} {'Count':>5}   {'Avg Exec(us)':>12}  "
        f"{'Avg Latency(us)':>15}  {'Exec%':>6}   {'Avg Head OH(us)':>15}  {'Avg Tail OH(us)':>15}  "
        f"{'Avg Prop(us)':>12}  {'Avg Local(us)':>13}"
    )
    print("-" * 140)

    # Sort by func_id for consistent output
    total_count = 0
    total_duration = 0.0

    for func_id in sorted(func_stats.keys()):
        stats = func_stats[func_id]
        durations = stats["durations"]
        count = len(durations)
        sum_duration = sum(durations)
        avg_duration = sum_duration / count

        # Accumulate totals
        total_count += count
        total_duration += sum_duration

        # Get function name
        if func_id_to_name and str(func_id) in func_id_to_name:
            func_name = func_id_to_name[str(func_id)]
        else:
            func_name = f"func_{_func_id_to_letter(func_id)}"

        # Calculate averages
        avg_head_overhead = (
            sum(stats["head_overheads"]) / len(stats["head_overheads"]) if stats["head_overheads"] else 0
        )
        avg_tail_overhead = (
            sum(stats["tail_overheads"]) / len(stats["tail_overheads"]) if stats["tail_overheads"] else 0
        )
        avg_latency = stats["total_latency"] / count if count > 0 else 0
        # `None` (not NaN) signals "no v3 receive_time data on this func" so
        # the print line below renders a dash. NaN would force ruff's
        # PLR0124 self-compare idiom.
        avg_propagation = sum(stats["propagations"]) / len(stats["propagations"]) if stats["propagations"] else None
        avg_local_setup = sum(stats["local_setups"]) / len(stats["local_setups"]) if stats["local_setups"] else None

        # Calculate execution ratio: total_exec_time / total_latency
        exec_ratio = (stats["total_exec_time"] / stats["total_latency"] * 100) if stats["total_latency"] > 0 else 0

        latency_str = f"{avg_latency:.2f}" if has_scheduler_timing else "-"
        exec_ratio_str = f"{exec_ratio:.1f}%" if has_scheduler_timing else "-"
        head_str = f"{avg_head_overhead:.2f}" if has_scheduler_timing else "-"
        tail_str = f"{avg_tail_overhead:.2f}" if has_scheduler_timing else "-"
        prop_str = f"{avg_propagation:>12.2f}" if avg_propagation is not None else f"{'-':>12}"
        local_str = f"{avg_local_setup:>13.2f}" if avg_local_setup is not None else f"{'-':>13}"
        print(
            f"{func_id:<8} {func_name:<12} {count:>5}   {avg_duration:>12.2f}  {latency_str:>15}  "
            f"{exec_ratio_str:>6}   {head_str:>15}  {tail_str:>15}  "
            f"{prop_str}  {local_str}"
        )

    # Print total row
    print("-" * 140)

    # Calculate total latency (sum of all latencies)
    total_latency_sum = sum(stats["total_latency"] for stats in func_stats.values())
    total_latency_str = f"{total_latency_sum:.2f}" if has_scheduler_timing else "-"
    print(f"{'TOTAL':<21} {total_count:>5}   {total_duration:>12.2f}  {total_latency_str:>15}")

    # Print total test execution time
    if has_scheduler_timing and min_dispatch_time != float("inf") and max_finish_time != float("-inf"):
        total_test_time = max_finish_time - min_dispatch_time
        print(f"\nTotal Test Time: {total_test_time:.2f} us (from earliest dispatch to latest finish)")
    elif not has_scheduler_timing and min_aicore_time != float("inf") and max_aicore_time != float("-inf"):
        aicore_observed_span = max_aicore_time - min_aicore_time
        print(
            f"\nAICore Observed Span: {aicore_observed_span:.2f} us (from earliest AICore receive to latest AICore end)"
        )

    # Task execution vs Scheduler overhead summary
    if has_scheduler_timing and total_count > 0 and total_latency_sum > 0:
        avg_exec_us = total_duration / total_count
        avg_latency_us = total_latency_sum / total_count
        exec_latency_ratio_pct = total_duration / total_latency_sum * 100
        print("\n--- Task execution vs Scheduler overhead ---")
        print(
            f"  Per-task (all):  Avg Exec = {avg_exec_us:.2f} us,  "
            f"Avg Latency (dispatch->finish) = {avg_latency_us:.2f} us,  "
            f"Exec/Latency = {exec_latency_ratio_pct:.2f}%"
        )
        print("  (Latency = dispatch→finish; Exec = AICore kernel time per task)")

    print("=" * 110)


def build_overhead_counter_events(tasks, deps_edges, pid=2):  # noqa: PLR0912
    """Per-engine + system overhead counter tracks for the Perfetto trace.

    Emits 8 counter (``"ph":"C"``) series under the AICPU Scheduler process
    (``pid=2``) — it is scheduler-overhead analysis, so it lives in the sched
    group — overlaid alongside the scheduler's own tracks and the AICore task
    bars to see, at every instant, why time is or is not wasted (see
    docs/dfx/sched-overhead-model.md):

      {aic,aiv}_idle     core of that type NOT executing (k - running)
      {aic,aiv}_ready    that type's tasks whose producers have ended but that
                         are not yet dispatched. A MIX task (records on BOTH
                         engines) counts for BOTH.
      {aic,aiv}_overhead 1 when idle>0 AND ready>0 (free core + ready work the
                         scheduler hasn't placed)
      all_overhead       1 when EVERY present engine is overhead (whole chip
                         blocked — e.g. a MIX waiting to launch)
      has_overhead       1 when every engine that HAS ready work is overhead
                         (engines with no work are ignored)

    Readiness keys off producer end_time; a task whose predecessors are all
    absent from the perf set falls back to its own dispatch (no unverifiable
    early readiness). Needs ``deps_edges`` (pred -> [succ]); returns [] without it.
    """
    if not deps_edges or not tasks or any(task.get("dispatch_time_us") is None for task in tasks):
        return []

    def _u64(x):
        try:
            x = int(x)
        except (TypeError, ValueError):
            return None
        return x & ((1 << 64) - 1) if x < 0 else x

    cores_by_type = defaultdict(set)
    for t in tasks:
        cores_by_type[t.get("core_type")].add(t.get("core_id"))
    types = [ty for ty in ("aic", "aiv") if cores_by_type.get(ty)]
    if not types:
        return []
    k = {ty: len(cores_by_type[ty]) for ty in types}

    types_of = defaultdict(set)
    disp, end = {}, {}
    for t in tasks:
        tid = _u64(t.get("task_id"))
        if tid is None:
            continue
        # Keyed by (run_epoch, task_id): the same task id in two runs is two
        # executions. Keying on the id alone would take min(dispatch) from one
        # run and max(end) from the other, producing a span covering both.
        key = (t.get("run_epoch"), tid)
        types_of[key].add(t.get("core_type"))
        disp[key] = min(disp.get(key, t["dispatch_time_us"]), t["dispatch_time_us"])
        end[key] = max(end.get(key, t["end_time_us"]), t["end_time_us"])

    preds = defaultdict(set)
    for pred, succs in deps_edges.items():
        p = _u64(pred)
        for s in succs:
            ss = _u64(s)
            if p is not None and ss is not None and p != ss:
                preds[ss].add(p)
    ready = {}
    for (epoch, tid), dp in disp.items():
        # deps_edges is the run-independent static graph, so it is looked up by
        # bare task id — but the predecessor *execution* must be the one from
        # this same run, or a run-2 successor would wait on a run-1 producer.
        in_perf = [p for p in preds.get(tid, ()) if (epoch, p) in end]
        ready[(epoch, tid)] = max(end[(epoch, p)] for p in in_perf) if in_perf else dp

    w0 = min(t["start_time_us"] for t in tasks)
    w1 = max(t["end_time_us"] for t in tasks)
    run = {ty: defaultdict(int) for ty in types}
    rw = {ty: defaultdict(int) for ty in types}
    times = {w0, w1}
    for t in tasks:
        ty = t.get("core_type")
        if ty not in run:
            continue
        s = max(w0, min(t["start_time_us"], w1))
        e = max(w0, min(t["end_time_us"], w1))
        if e > s:
            run[ty][s] += 1
            run[ty][e] -= 1
            times.update((s, e))
    for key, dp in disp.items():
        r = max(w0, min(ready[key], w1))
        dd = max(w0, min(dp, w1))
        if dd > r:
            for ty in types_of[key]:  # MIX -> credit both engines
                if ty in rw:
                    rw[ty][r] += 1
                    rw[ty][dd] -= 1
            times.update((r, dd))

    tids = {
        "oh_aic_idle": 9101,
        "oh_aic_ready": 9102,
        "oh_aic_overhead": 9103,
        "oh_aiv_idle": 9111,
        "oh_aiv_ready": 9112,
        "oh_aiv_overhead": 9113,
        "oh_all_overhead": 9121,
        "oh_has_overhead": 9122,
    }
    # No process metadata: pid=2 is the AICPU Scheduler process (named/sorted
    # elsewhere). Emitting it here would override that — these counters just
    # join the scheduler group as extra tracks (names prefixed "oh_").
    events = []
    run_c = {ty: 0 for ty in types}
    rw_c = {ty: 0 for ty in types}
    prev = {}
    order = sorted(times)
    for i in range(len(order) - 1):
        a = order[i]
        for ty in types:
            run_c[ty] += run[ty][a]
            rw_c[ty] += rw[ty][a]
        ov = {}
        vals = {}
        for ty in types:
            idle = k[ty] - run_c[ty]
            vals[f"oh_{ty}_idle"] = idle
            vals[f"oh_{ty}_ready"] = rw_c[ty]
            ov[ty] = 1 if (idle > 0 and rw_c[ty] > 0) else 0
            vals[f"oh_{ty}_overhead"] = ov[ty]
        vals["oh_all_overhead"] = 1 if all(ov[ty] for ty in types) else 0
        work = [ty for ty in types if rw_c[ty] > 0]
        vals["oh_has_overhead"] = 1 if (work and all(k[ty] - run_c[ty] > 0 for ty in work)) else 0
        for name, v in vals.items():
            if name in tids and prev.get(name) != v:
                events.append(
                    {
                        "ph": "C",
                        "cat": "overhead",
                        "name": name,
                        "pid": pid,
                        "tid": tids[name],
                        "ts": round(a, 3),
                        "args": {name: v},
                    }
                )
                prev[name] = v
    return events


def generate_chrome_trace_json(  # noqa: PLR0912, PLR0913, PLR0915
    tasks,
    output_path,
    func_id_to_name=None,
    verbose=False,
    scheduler_phases=None,
    scheduler_streams=None,
    orchestrator_phases=None,
    core_to_thread=None,
    orchestrator_name=None,
    orchestrator_source=None,
    runtime_name=None,
    timeline_metadata=None,
    deps_edges=None,
    deps_kernel_map=None,
    deps_block_map=None,
    emit_overhead=False,
    host_device_uploads=None,
    aicpu_lifecycle_records=None,
):
    """Generate Chrome Trace Event Format JSON from task data.

    Args:
        tasks: List of task dicts with fields:
            - task_id, func_id, core_id, core_type
            - start_time_us, end_time_us, duration_us
            - dispatch_time_us (optional, Scheduler dispatch timestamp)
            - finish_time_us (optional, Scheduler finish timestamp)
        output_path: Path to output JSON file
        func_id_to_name: Optional dict mapping func_id to function name
        verbose: Print progress information
        scheduler_phases: Optional list of per-scheduler record lists (chip_swimlane_level >= 3)
        scheduler_streams: Optional metadata for each scheduler record list
        orchestrator_phases: Optional list of per-task orchestrator phase records (chip_swimlane_level >= 4)
        core_to_thread: Optional list mapping core_id (index) to scheduler thread index (-1 = unassigned)
        aicpu_lifecycle_records: Optional A5 HBG AICPU control-plane lifecycle records

    Generates processes in the trace:
        - pid=5 "Graph Execution": one end-to-end envelope per Graph task
        - pid=1 "Host/AICPU Orchestrator": orchestrator phase bars (chip_swimlane_level >= 4)
        - pid=2 "AICPU/AICore Scheduler": scheduler record bars (chip_swimlane_level >= 3)
        - pid=3 "Scheduler View": dispatch_time_us to finish_time_us
        - pid=4 "Worker View": per-subtask kernel execution on physical cores
    """
    if verbose:
        print("Generating Chrome Trace JSON...")
        print(f"  Tasks: {len(tasks)}")
        if func_id_to_name:
            print(f"  Function names: {len(func_id_to_name)} entries")

    # Step 1: Build core_to_tid mapping (using only core_id, not core_type)
    unique_cores = set()
    for task in tasks:
        unique_cores.add(task["core_id"])

    core_to_tid = {}
    for core_id in sorted(unique_cores):
        core_to_tid[core_id] = 10000 + core_id * 10

    if verbose:
        print(f"  Unique cores: {len(unique_cores)}")

    # The TaskId layout every label in this trace is formatted with. Chosen once, from
    # the runtime the document names, because nothing in a task_id value says which
    # runtime minted it.
    task_display = task_display_for(runtime_name)

    # Recover func_id for TASK_TIMING (level=1) records, which the host
    # emits as func_id=-1. Resolve once here against dep_gen's per-task
    # kernel_ids[3] (picking the subslot by core_type) and write it back onto
    # the task, so every downstream consumer — Worker View, Scheduler View, and
    # event-hints — sees the same real func_id. See
    # resolve_func_id_from_kernel_map() for the AIV0-vs-AIV1 tie-break and the
    # host-side contract.
    if deps_kernel_map is not None:
        for task in tasks:
            if int(task["func_id"]) < 0:
                resolved = resolve_func_id_from_kernel_map(task["task_id"], task.get("core_type"), deps_kernel_map)
                if resolved >= 0:
                    task["func_id"] = resolved

    graph_instances = _collect_graph_execution_instances(tasks, scheduler_phases)

    # Step 2: Generate JSON events
    events = []

    if aicpu_lifecycle_records:
        events.append(
            {"args": {"name": "AICPU Lifecycle"}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 6}
        )
        events.append(
            {"args": {"sort_index": 2}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": 6}
        )
        lifecycle_intervals = (
            ("handshake_partition", "handshake_start_time_us", "handshake_complete_time_us"),
            ("topology_config", "config_start_time_us", "topology_complete_time_us"),
            ("context_publish", "context_publish_start_time_us", "context_publish_complete_time_us"),
            ("bootstrap_wait", "bootstrap_wait_start_time_us", "bootstrap_complete_time_us"),
            ("register_release", "register_release_start_time_us", "register_release_end_time_us"),
            ("exit_signal", "exit_signal_start_time_us", "exit_signal_end_time_us"),
            ("exit_wait", "exit_wait_start_time_us", "exit_wait_end_time_us"),
        )
        for record in aicpu_lifecycle_records:
            thread_id = int(record.get("aicpu_thread_id", record.get("record_index", 0)))
            tid = 60000 + thread_id
            events.append(
                {
                    "args": {"name": f"AICPU Thread {thread_id}"},
                    "cat": "__metadata",
                    "name": "thread_name",
                    "ph": "M",
                    "pid": 6,
                    "tid": tid,
                }
            )
            identity = {"aicpu_thread_id": thread_id}
            for name, start_field, end_field in lifecycle_intervals:
                start = float(record.get(start_field, 0.0))
                end = float(record.get(end_field, 0.0))
                if end <= 0 or end < start:
                    continue
                events.append(
                    {
                        "args": identity,
                        "cat": "aicpu_lifecycle",
                        "name": name,
                        "ph": "X",
                        "pid": 6,
                        "tid": tid,
                        "ts": start,
                        "dur": end - start,
                    }
                )

    # Metadata event: Process names and sort order.
    # pid is renumbered in pipeline order (top → bottom in Perfetto):
    #   pid=1  AICPU Orchestrator  (submits tasks — earliest)
    #   pid=5  Graph Execution     (Scheduler-local expansion + execution)
    #   pid=2  AICPU Scheduler     (pops ready, dispatches, completes)
    #   pid=3  Scheduler View      (AICPU-eye view of each worker's dispatch→finish)
    #   pid=4  Worker View         (physical AIC/AIV execution rows)
    # sort_index intentionally equals pid so JSON ordering is self-evident.
    task_map: dict[int, list] = defaultdict(list)
    for t in tasks:
        task_map[t["task_id"]].append(t)
    # Per-run view of the same grouping. Dependency rendering must use this one:
    # deps.json is the run-independent static graph, so two runs of the same
    # graph share every edge, and pairing predecessor rows with successor rows
    # out of the merged map would draw arrows from one run's producer to the
    # other's consumer.
    task_maps_by_run: dict[object, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for t in tasks:
        task_maps_by_run[t.get("run_epoch")][t["task_id"]].append(t)
    if not task_maps_by_run:
        # A capture can legitimately have no AICore task rows and still have
        # dependency arrows, anchored on AICPU dummy/alloc slices. Keep one
        # (empty) run so the edge loops below still execute for that path.
        task_maps_by_run[None] = defaultdict(list)
    # Whether a task is SPMD is a property of the task, so the result is one set
    # for the whole capture — but it is *observed* per run, because row count in
    # a merged map is the number of runs rather than a block count. Every
    # consumer that then looks rows up does so in that run's own map.
    spmd_task_ids = _identify_spmd_task_ids(task_maps_by_run, deps_block_map)

    if graph_instances:
        events.append(
            {"args": {"name": "Graph Execution"}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 5}
        )
        events.append(
            {"args": {"sort_index": 1}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": 5}
        )
        for lane_idx in sorted({instance["lane_idx"] for instance in graph_instances}):
            events.append(
                {
                    "args": {"name": f"Graph_{lane_idx}"},
                    "cat": "__metadata",
                    "name": "thread_name",
                    "ph": "M",
                    "pid": 5,
                    "tid": 5000 + lane_idx,
                }
            )
        for instance in graph_instances:
            outer_display = task_display(instance["outer_task_id"])
            task_indices = instance["visible_task_indices"]
            events.append(
                {
                    "args": _with_run_epoch(
                        {
                            "outer_task_id": instance["outer_task_id"],
                            "visible_sub_task_count": len(task_indices),
                            "visible_sub_task_local_id_min": min(task_indices),
                            "visible_sub_task_local_id_max": max(task_indices),
                            "prepare_slice_count": instance["prepare_slice_count"],
                            "prepare_duration_us": instance["prepare_duration_us"],
                            "execution_start_us": instance["execution_start_us"],
                            "execution_duration_us": instance["execution_end_us"] - instance["execution_start_us"],
                            "synthetic_id_layout": "space1:(outer_task_id << 32) | sub_task_local_id",
                        },
                        instance,
                    ),
                    "cat": "graph_execution",
                    "cname": "rail_animation",
                    "name": f"GraphExecution({outer_display}, {len(task_indices)} visible sub-tasks)",
                    "ph": "X",
                    "pid": 5,
                    "tid": 5000 + instance["lane_idx"],
                    "ts": instance["prepare_start_us"],
                    "dur": instance["execution_end_us"] - instance["prepare_start_us"],
                }
            )

    events.append({"args": {"name": "Worker View"}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 4})
    events.append({"args": {"sort_index": 4}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": 4})

    # Check if any task has Scheduler timestamps.
    has_scheduler_task_data = any(
        task.get("dispatch_time_us", 0) >= 0 and task.get("finish_time_us", 0) > 0 for task in tasks
    )

    if has_scheduler_task_data:
        events.append(
            {"args": {"name": "Scheduler View"}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 3}
        )
        events.append(
            {"args": {"sort_index": 3}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": 3}
        )

    # Metadata events: Thread names (one per core)
    for core_id, tid in core_to_tid.items():
        # Find first task with this core_id to get core_type
        core_type = None
        for task in tasks:
            if task["core_id"] == core_id:
                core_type = task["core_type"]
                break

        # core_type is now a string ("aic" or "aiv")
        core_type_str = (core_type or "unknown").upper()
        thread_name = f"{core_type_str}_{core_id}"
        events.append(
            {"args": {"name": thread_name}, "cat": "__metadata", "name": "thread_name", "ph": "M", "pid": 4, "tid": tid}
        )

    # Duration events (Complete events "X")
    # Build task_id -> event_id mapping for flow events
    task_to_event_id: dict[tuple, int] = {}
    task_to_scheduler_event_id: dict[tuple, int] = {}
    task_to_scheduler_tid: dict[tuple, int] = {}
    # Dummy / predicated-skip / alloc DAG nodes have no AICore kernel row, so
    # their dependency arrows anchor on the AICPU worker slice instead. Kept per
    # run for the same reason the task map is: these task ids repeat every run,
    # and a map spanning runs hands a producer from one run to a consumer in
    # another.
    aicpu_anchor_maps_by_run: dict[object, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    event_id = 0

    AICPU_TID_BASE = 19000  # noqa: N806
    scheduler_thread_indices = {int(thread_idx) for thread_idx in (core_to_thread or []) if int(thread_idx) >= 0}
    scheduler_thread_indices.update(
        thread_idx for thread_idx, thread_records in enumerate(scheduler_phases or []) if thread_records
    )
    scheduler_thread_count = (
        max(scheduler_thread_indices) + 1 if scheduler_thread_indices else len(scheduler_phases or [])
    )
    orchestrator_thread_base = scheduler_thread_count
    aicpu_worker_thread_count = scheduler_thread_count + len(orchestrator_phases or [])

    # core_to_thread and non-empty phase pools carry runtime scheduler indices;
    # phase arrays may also contain trailing empty capacity slots. Orchestrator
    # pools are ordinal-only and occupy the dedicated threads after schedulers.
    for thread_idx in range(aicpu_worker_thread_count):
        events.append(
            {
                "args": {"name": f"AICPU_{thread_idx}"},
                "cat": "__metadata",
                "name": "thread_name",
                "ph": "M",
                "pid": 4,
                "tid": AICPU_TID_BASE + thread_idx,
            }
        )

    def worker_flow_endpoint(row):
        event_id_key = row.get("event_id")
        if "trace_tid" in row:
            return row["trace_tid"], event_id_key
        return core_to_tid[row["core_id"]], task_to_event_id.get(_execution_key(row))

    # Invert deps (pred -> [succ]) into a fanin map (succ -> [pred]) so each task
    # bar can show both its consumers (fanout) and producers (fanin) with counts.
    fanin_map: dict[int, list] = defaultdict(list)
    if deps_edges:
        for pred, succs in deps_edges.items():
            for succ in succs:
                fanin_map[succ].append(pred)

    for task in tasks:
        tid = core_to_tid[task["core_id"]]
        local_setup_us = task.get("local_setup_us", 0.0) or 0.0
        ts = _task_slice_start_us(task)
        dur = task["end_time_us"] - ts

        # func_id is already resolved (level=1 records recovered from
        # dep_gen's kernel_ids up front; see the pre-pass above). Without a
        # deps.json the id stays -1 and the lane is named task(<task-id>).
        func_id = task["func_id"]
        tdisp = task_display(task["task_id"])
        task_name = _task_display_name(func_id, func_id_to_name, tdisp, spmd=task["task_id"] in spmd_task_ids)

        # fanout (consumers) / fanin (producers) hints from deps.json — the device
        # hot path no longer carries them. Each leads with the degree (count) so
        # broadcast / reduction nodes are obvious without expanding the list.
        fanout_ids = deps_edges.get(task["task_id"], []) if deps_edges else []
        fanin_ids = fanin_map.get(task["task_id"], [])
        fanout_str = f"{len(fanout_ids)}: [" + ", ".join(task_display(x) for x in fanout_ids) + "]"
        fanin_str = f"{len(fanin_ids)}: [" + ", ".join(task_display(x) for x in fanin_ids) + "]"

        events.append(
            {
                "args": _with_run_epoch(
                    {
                        "event-hint": f"Task:{tdisp}, FuncId:{func_id}, CoreId:{task['core_id']}",
                        "fanout-hint": fanout_str,
                        "fanin-hint": fanin_str,
                        "duration-us": dur,
                        "kernel-duration-us": task["duration_us"],
                        "local_setup_us": local_setup_us,
                        "taskId": task["task_id"],
                    },
                    task,
                ),
                "cat": "event",
                "id": event_id,
                "name": task_name,
                "ph": "X",
                "pid": 4,
                "tid": tid,
                "ts": ts,
                "dur": dur,
            }
        )

        # Record mapping for flow events
        task_to_event_id[_execution_key(task)] = event_id
        event_id += 1

    # Scheduler View duration events (dispatch_time to finish_time)
    # Assign overlapping tasks on the same core to different tids so Perfetto
    # renders each bar on its own row (Perfetto requires strict nesting on a tid).
    if has_scheduler_task_data:
        # Build per-core sorted task lists and assign sub-lanes.
        # Each core gets a base tid from core_to_tid; overlapping tasks get base+1.
        _core_scheduler_tasks: dict[int, list] = defaultdict(list)
        for task in tasks:
            d = task.get("dispatch_time_us", 0)
            f = task.get("finish_time_us", 0)
            if d < 0 or f <= 0:
                continue
            _core_scheduler_tasks[task["core_id"]].append(task)
        for ct_list in _core_scheduler_tasks.values():
            ct_list.sort(key=lambda t: t["dispatch_time_us"])

        scheduler_tid_set: set[int] = set()
        for core_id, ct_list in _core_scheduler_tasks.items():
            base_tid = core_to_tid[core_id]
            # Greedy lane assignment: track finish time per sub-lane
            lane_finish = [0.0]  # lane 0 = base_tid
            for task in ct_list:
                d = task["dispatch_time_us"]
                assigned = -1
                for lane_idx, lf in enumerate(lane_finish):
                    if lf <= d:
                        assigned = lane_idx
                        break
                if assigned < 0:
                    assigned = len(lane_finish)
                    lane_finish.append(0.0)
                lane_finish[assigned] = task["finish_time_us"]
                tid = base_tid if assigned == 0 else base_tid + assigned
                task_to_scheduler_tid[_execution_key(task)] = tid
                scheduler_tid_set.add(tid)

        # Thread name metadata for Scheduler View (one entry per unique tid used)
        for core_id, base_tid in core_to_tid.items():
            ct_list = _core_scheduler_tasks.get(core_id)
            core_type_str = ct_list[0]["core_type"].upper() if ct_list else "unknown"
            base_name = f"{core_type_str}_{core_id}"
            # Base lane always gets metadata (even if no tasks, for consistency)
            if base_tid in scheduler_tid_set or not scheduler_tid_set:
                events.append(
                    {
                        "args": {"name": base_name},
                        "cat": "__metadata",
                        "name": "thread_name",
                        "ph": "M",
                        "pid": 3,
                        "tid": base_tid,
                    }
                )
            # Overflow lane (at most one: dual-slot dispatch means max 2 concurrent tasks per core)
            overflow_tid = base_tid + 1
            if overflow_tid in scheduler_tid_set:
                events.append(
                    {
                        "args": {"name": base_name},
                        "cat": "__metadata",
                        "name": "thread_name",
                        "ph": "M",
                        "pid": 3,
                        "tid": overflow_tid,
                    }
                )

        for task in tasks:
            dispatch_us = task.get("dispatch_time_us", 0)
            finish_us = task.get("finish_time_us", 0)
            # 0us is a valid timestamp (base-time aligned); only reject negative/invalid values.
            if dispatch_us < 0 or finish_us <= 0:
                continue

            tid = task_to_scheduler_tid.get(_execution_key(task), core_to_tid[task["core_id"]])
            scheduler_duration_us = finish_us - dispatch_us

            # Get function name if available (task(<task-id>) when no deps.json
            # resolved the func_id; see _task_display_name).
            func_id = task["func_id"]
            tdisp = task_display(task["task_id"])
            task_name = _task_display_name(func_id, func_id_to_name, tdisp, spmd=task["task_id"] in spmd_task_ids)

            events.append(
                {
                    "args": _with_run_epoch(
                        {
                            "event-hint": f"Task:{tdisp}, FuncId:{func_id}, CoreId:{task['core_id']}",
                            "dispatch-time-us": dispatch_us,
                            "finish-time-us": finish_us,
                            "scheduler-duration-us": scheduler_duration_us,
                            "taskId": task["task_id"],
                        },
                        task,
                    ),
                    "cat": "event",
                    "id": event_id,
                    "name": task_name,
                    "ph": "X",
                    "pid": 3,
                    "tid": tid,
                    "ts": dispatch_us,
                    "dur": scheduler_duration_us,
                }
            )
            task_to_scheduler_event_id[_execution_key(task)] = event_id
            event_id += 1

    flow_id = 0
    hb_violation_count = 0
    deps_flow_count = 0
    edges_by_pred = deps_edges or {}

    # AICPU Scheduler phase events (chip_swimlane_level >= 3)
    if scheduler_phases:

        def sched_lane_tid(thread_idx, lane=0):
            return 30000 + thread_idx * 10 + lane

        # Process metadata
        producers = {stream.get("producer") for stream in (scheduler_streams or []) if stream.get("producer")}
        scheduler_process_name = "AICore Scheduler" if producers == {"aicore"} else "AICPU Scheduler"
        events.append(
            {"args": {"name": scheduler_process_name}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 2}
        )
        events.append(
            {"args": {"sort_index": 2}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": 2}
        )

        # Phase color mapping. The Perfetto sched lane only renders the
        # two work phases (complete, dispatch). Idle is the wall-clock gap
        # between consecutive work bars — Perfetto's empty-track regions
        # already convey that visually, so we don't paint a synthetic bar
        # for it. (Idle is still tallied numerically by
        # sched_overhead_analysis.py Part 2 via gap reconstruction.)
        phase_colors = {
            # Outer phases — mutually time-exclusive within an iter
            "complete": "good",  # green
            "dispatch": "terrible",  # red
            "async_poll": "yellow",  # async-wait completion polling (split from complete)
            "release": "olive",  # deferred-release drain (on_task_release work)
            "dummy": "grey",  # dummy_drain pass (TMR Resolve nests inside; HBG P bar is standalone)
            "early_dispatch": "rail_animation",  # speculative early-dispatch staging
            # sync_start stop-the-world drain: outer bar time-contains the two
            # inner staging passes, so Perfetto nests them by depth on the track.
            "drain": "cq_build_running",  # handle_drain_mode outer
            "drain_prepare": "cq_build_attempt_runnable",  # inner: cluster scan + build_payload
            "drain_publish": "cq_build_attempt_passed",  # inner: MMIO write_reg per subtask (the cohort launch)
            "graph_prepare": "rail_animation",  # bounded Scheduler-side Definition expansion
            "bootstrap": "rail_animation",
            "fanin": "cq_build_running",
            "state_probe": "cq_build_running",
            "ready_claim": "cq_build_attempt_runnable",
            "ready_steal": "cq_build_attempt_failed",
            "direct_refill": "cq_build_attempt_passed",
            "worksteal": "cq_build_attempt_failed",
            "refill": "cq_build_attempt_passed",
            "idle": "grey",
            # Inner in TMR; standalone on HBG's dedicated P thread.
            "resolve": "vsync_highlight_color",  # on_task_complete: walk consumer list
            # Separate-lane (Worker View AICPU_N) — fallback color if it ever lands on Sched
            "dummy_task": "grey",
        }

        # Per-complete task-finish rows are retained as an attribution check.
        # The runtime's Complete `tasks_processed` field is the authoritative
        # number of AICore FIN/retire events in the phase (including non-final
        # SPMD sub-block retires), not necessarily a logical-task count.
        # Bucketed per (thread, run): a finish belongs to a complete phase of
        # its own run. Attributing across runs inflates the receiving phase's
        # count with finishes it never drained, and that count is displayed —
        # and, for a record carrying no raw `tasks_processed`, becomes the
        # displayed count outright.
        complete_index = _build_complete_index(scheduler_phases)

        finishes_per_complete: dict[int, int] = defaultdict(int)
        if core_to_thread:
            for t in tasks:
                f_us = t.get("finish_time_us")
                if f_us is None or f_us < 0:
                    continue
                t_cid = t["core_id"]
                if t_cid >= len(core_to_thread):
                    continue
                t_thr = core_to_thread[t_cid]
                if t_thr < 0 or t_thr >= len(complete_index):
                    continue
                t_comp = _select_complete_for_finish(
                    complete_index[t_thr]["by_run"].get(t.get("run_epoch"), _NO_COMPLETES), f_us
                )
                if t_comp is None:
                    continue
                finishes_per_complete[id(t_comp)] += 1

        # AICPU worker-marker width — start/end on the device coincide; render
        # as a 0.02 us sliver so Perfetto does not collapse it to a hairline.
        AICPU_WORKER_MARKER_MIN_DUR_US = 0.02  # noqa: N806

        assigned_thread_indices = {
            assigned for assigned in (core_to_thread or []) if isinstance(assigned, int) and assigned >= 0
        }
        for thread_idx, thread_records in enumerate(scheduler_phases):
            tid = sched_lane_tid(thread_idx, 0)
            resolve_tid = sched_lane_tid(thread_idx, 1)
            nested_resolve_ids = nested_resolve_record_ids(thread_records)
            is_resolution_thread = (
                scheduler_thread_role(thread_records, assigned_thread_indices, thread_idx, nested_resolve_ids)
                == "resolution"
            )

            # Thread name metadata
            stream = scheduler_streams[thread_idx] if scheduler_streams and thread_idx < len(scheduler_streams) else {}
            is_aicore_scheduler = stream.get("producer") == "aicore"
            scheduler_id = stream.get("scheduler_id", thread_idx)
            worker_id = stream.get("worker_id")
            lane_name = f"Sched_{scheduler_id}"
            if is_aicore_scheduler:
                display_id = worker_id if worker_id is not None else scheduler_id
                lane_name = f"Scheduler_{display_id}"
            events.append(
                {
                    "args": {"name": lane_name},
                    "cat": "__metadata",
                    "name": "thread_name",
                    "ph": "M",
                    "pid": 2,
                    "tid": tid,
                }
            )
            if nested_resolve_ids and not is_aicore_scheduler:
                events.append(
                    {
                        "args": {"name": f"Sched_{thread_idx}"},
                        "cat": "__metadata",
                        "name": "thread_name",
                        "ph": "M",
                        "pid": 2,
                        "tid": resolve_tid,
                    }
                )

            # Render work phases (complete / dispatch) plus the real operations
            # that otherwise hide inside an idle stretch (poll = completion-scan
            # that retired nothing; release = on_task_release drain). Genuine
            # spin emits no record and shows as a blank gap.
            #
            # Dependency-only task markers do NOT live on the sched track —
            # they represent DAG nodes briefly inhabiting the AICPU as a
            # virtual worker, so we route them to Worker View (pid=4) AICPU_N
            # (where N = the AICPU id of the sched thread that drained them).
            for record in thread_records:
                raw_phase = record.get("phase", "unknown")
                phase = canonical_sched_phase(raw_phase)
                if phase in ("dummy_task", "predicated_skip"):
                    start_us = record["start_time_us"]
                    end_us = record["end_time_us"]
                    dur = max(end_us - start_us, AICPU_WORKER_MARKER_MIN_DUR_US)
                    task_id = normalize_task_id_int(record.get("task_id"))
                    task_label = task_display(task_id) if task_id is not None else "unknown"
                    if phase == "dummy_task":
                        event_name = f"dummy({task_label})"
                    else:
                        func_id = resolve_task_func_id_from_kernel_map(task_id, deps_kernel_map)
                        event_name = _task_display_name(
                            func_id,
                            func_id_to_name,
                            task_label,
                            spmd=task_id in spmd_task_ids,
                        )
                    event_args = {
                        "loop_iter": record.get("loop_iter", 0),
                        "task_id": task_id,
                        "event-hint": event_name,
                    }
                    if phase == "dummy_task":
                        event_args["phase"] = phase
                    else:
                        event_args["predicated_pass"] = False
                    _with_run_epoch(event_args, record)
                    events.append(
                        {
                            "args": event_args,
                            "cat": "event",
                            "id": event_id,
                            "name": event_name,
                            "ph": "X",
                            "pid": 4,
                            "tid": AICPU_TID_BASE + thread_idx,
                            "ts": start_us,
                            "dur": dur,
                        }
                    )
                    if task_id is not None:
                        aicpu_anchor_maps_by_run[record.get("run_epoch")][task_id].append(
                            {
                                "task_id": task_id,
                                "run_epoch": record.get("run_epoch"),
                                "start_time_us": start_us,
                                "end_time_us": start_us + dur,
                                "receive_time_us": start_us,
                                "trace_tid": AICPU_TID_BASE + thread_idx,
                                "event_id": event_id,
                            }
                        )
                    event_id += 1
                    continue
                if phase not in (
                    "complete",
                    "async_poll",
                    "dispatch",
                    "release",
                    "resolve",
                    "early_dispatch",
                    "dummy",
                    "drain",
                    "drain_prepare",
                    "drain_publish",
                    "graph_prepare",
                    "bootstrap",
                    "fanin",
                    "state_probe",
                    "ready_claim",
                    "ready_steal",
                    "direct_refill",
                    "worksteal",
                    "refill",
                    "idle",
                ):
                    continue
                start_us = record["start_time_us"]
                end_us = record["end_time_us"]
                dur = end_us - start_us
                tasks_processed = record.get("tasks_processed", 0)

                # Queue-depth snapshot fields. Layout per
                # ChipSwimlaneAicpuSchedPhaseRecord docstring: [AIC, AIV, MIX].
                shared_at_start = record.get("shared_at_start")
                shared_at_end = record.get("shared_at_end")
                depths_valid = (
                    isinstance(shared_at_start, list)
                    and isinstance(shared_at_end, list)
                    and len(shared_at_start) == 3
                    and len(shared_at_end) == 3
                )

                # Phase block. When queue depths are present, fold them into
                # args so hover on a complete/dispatch bar surfaces the
                # before/after queue state alongside the phase metadata.
                phase_args = _with_run_epoch(
                    {
                        "phase": phase,
                        "loop_iter": record.get("loop_iter", 0),
                        "tasks_processed": tasks_processed,
                    },
                    record,
                )
                task_id = normalize_task_id_int(record.get("task_id"))
                if is_aicore_scheduler and task_id is not None:
                    phase_args["task_id"] = task_id
                if depths_valid:
                    # Perfetto's args SQL parses key names; `[...]` looks like
                    # an array-index op and crashes the details-panel query.
                    # Encode the AIC/AIV/MIX layout inline so the key stays
                    # parser-safe while still self-documenting.
                    phase_args.update(
                        {
                            "shared_at_start (aic,aiv,mix)": list(shared_at_start),
                            "shared_at_end (aic,aiv,mix)": list(shared_at_end),
                        }
                    )
                if phase == "complete":
                    matched_finish_rows = finishes_per_complete.get(id(record), 0)
                    if "tasks_processed" in record:
                        # Preserve the raw runtime count for trace consumers;
                        # it includes both final and non-final SPMD FINs.
                        phase_args["finishes_processed"] = tasks_processed
                        phase_args["finish_rows_attributed"] = matched_finish_rows
                    else:
                        # Older/synthetic records have no raw count. Keep the
                        # converter useful by falling back to row attribution.
                        phase_args["finishes_processed"] = matched_finish_rows
                        tasks_processed = matched_finish_rows
                        phase_args["tasks_processed"] = tasks_processed
                display_phase = (
                    _AICORE_SCHEDULER_PHASE_DISPLAY_NAMES.get(phase, phase) if is_aicore_scheduler else phase
                )
                if not is_aicore_scheduler:
                    display_name = f"{display_phase}({tasks_processed})"
                elif task_id is not None:
                    display_name = f"{display_phase}({task_display(task_id)})"
                else:
                    display_name = display_phase
                event_tid = (
                    resolve_tid
                    if not is_aicore_scheduler and raw_phase == "resolve" and id(record) in nested_resolve_ids
                    else tid
                )
                events.append(
                    {
                        "args": phase_args,
                        "cat": "scheduler",
                        "cname": phase_colors.get(phase, "generic_work"),
                        "name": display_name,
                        "ph": "X",
                        "pid": 2,
                        "tid": event_tid,
                        "ts": start_us,
                        "dur": dur,
                    }
                )

                # Queue-depth counter tracks (Perfetto "ph": "C"). Emit ONE
                # sample per phase at its end_us — phase N's end is phase N+1's
                # start, so emitting both is redundant. Two samples at the
                # SAME ts (e.g. final-drain emit where start_time==end_time)
                # also breaks Perfetto's rate calc (divide-by-zero → NULL).
                # Complete/Dispatch and every HBG P-thread phase carry live
                # shared-queue snapshots. Other runtime phases zero-fill them.
                phase_has_live_depths = phase in ("complete", "dispatch") or (
                    is_resolution_thread and phase in ("resolve", "async_poll", "dummy")
                )
                if not phase_has_live_depths:
                    continue
                if not depths_valid:
                    continue
                # Shared queue: dedicated tid 3999 so all 3 schedulers'
                # snapshots compose onto one timeline (it's the same global
                # queue regardless of who sampled it). Samples from different
                # threads at slightly different ts are fine — Perfetto plots
                # them in time order to render the step function.
                events.append(
                    {
                        "args": {"AIC": shared_at_end[0], "AIV": shared_at_end[1], "MIX": shared_at_end[2]},
                        "cat": "queue",
                        "name": "shared_ready_queue",
                        "ph": "C",
                        "pid": 2,
                        "tid": 3999,
                        "ts": end_us,
                    }
                )

        # Name the shared-queue pseudo-thread + give it a sort index that
        # places it after the 3 scheduler threads but still inside the
        # AICPU Scheduler process row, so the user reads top-to-bottom:
        # Sched_0 / Sched_1 / Sched_2 / Shared queue (global).
        events.append(
            {
                "args": {"name": "shared_ready_queue (global)"},
                "cat": "__metadata",
                "name": "thread_name",
                "ph": "M",
                "pid": 2,
                "tid": 3999,
            }
        )
        events.append(
            {
                "args": {"sort_index": 100},
                "cat": "__metadata",
                "name": "thread_sort_index",
                "ph": "M",
                "pid": 2,
                "tid": 3999,
            }
        )

    # Host-to-device transfer lane. These are bind segments, not orchestrator
    # operations, and they are the ones a device timeline can say something about:
    # drawn beside the device lanes they show the handover the device waits on,
    # where the rest of the bind stage is host-only setup with no counterpart here.
    if host_device_uploads:
        events.append(
            {"args": {"name": "Host Prepare"}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 1}
        )
        events.append(
            {"args": {"name": "H2D"}, "cat": "__metadata", "name": "thread_name", "ph": "M", "pid": 1, "tid": 4100}
        )
        for upload in host_device_uploads:
            start_us = float(upload.get("start_time_us", 0.0))
            end_us = float(upload.get("end_time_us", start_us))
            events.append(
                {
                    "name": str(upload.get("phase", "upload")),
                    "cat": "host_h2d",
                    "ph": "X",
                    "pid": 1,
                    "tid": 4100,
                    "ts": start_us,
                    "dur": max(0.0, end_us - start_us),
                    "cname": "thread_state_iowait",
                    "args": {"bytes": upload.get("detail", 0)},
                }
            )

    # AICPU Orchestrator lane (chip_swimlane_level >= 4)
    #
    # Per-event AicpuPhaseRecord[] is the single source of truth for
    # orchestrator timing. There is no separate aggregate summary — the
    # device-side LOG_INFO "orch_start=… orch_end=… orch_cost=…" log
    # line covers the run-window envelope for debugging without swimlane.
    if orchestrator_phases:
        # Process metadata
        if orchestrator_source == "host":
            orch_process_label = "Host Orchestrator"
        else:
            orch_process_label = f"AICPU {orchestrator_name}" if orchestrator_name else "AICPU Orchestrator"
        events.append(
            {"args": {"name": orch_process_label}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 1}
        )
        events.append(
            {"args": {"sort_index": 1}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": 1}
        )

        # Thread name metadata for each orchestrator thread
        for orch_idx in range(len(orchestrator_phases)):
            tid = 4000 + orch_idx
            name = f"Host_{orch_idx}" if orchestrator_source == "host" else f"Orch_{orch_idx}"
            events.append(
                {"args": {"name": name}, "cat": "__metadata", "name": "thread_name", "ph": "M", "pid": 1, "tid": tid}
            )

        # Per-task orchestrator phase bars. As of PR-X the device folds
        # all 6 sub-step phases into one ORCH_SUBMIT record covering the
        # submit's entire [start, end] window. Legacy per-sub-step phase
        # strings remain in the color map so old captures still render.
        orch_phase_colors = {
            "orch_submit": "rail_animation",  # purple — primary
            # Legacy per-sub-step phases (old captures only):
            "orch_sync": "thread_state_iowait",
            "orch_alloc": "terrible",
            "orch_params": "good",
            "orch_lookup": "thread_state_running",
            "orch_insert": "olive",
            "orch_fanin": "rail_animation",
        }

        # Build the runtime task-id sets so the orch-phase loop can distinguish
        # alloc_tensors() calls from submitted tasks. deps.json remains the
        # identity source when a runtime timing record is missing.
        regular_task_ids = {int(t.get("task_id", -1)) for t in tasks}
        dummy_task_ids = set()
        predicated_skip_task_ids = set()
        if scheduler_phases:
            for thread_records in scheduler_phases:
                for rec in thread_records:
                    phase = rec.get("phase")
                    if phase in ("dummy_task", "predicated_skip"):
                        task_id = normalize_task_id_int(rec.get("task_id"))
                        if task_id is not None:
                            if phase == "dummy_task":
                                dummy_task_ids.add(task_id)
                            else:
                                predicated_skip_task_ids.add(task_id)
        deps_dummy_task_ids = {
            task_id
            for task_id, kernel_ids in (deps_kernel_map or {}).items()
            if all(kernel_id < 0 for kernel_id in kernel_ids)
        }
        missing_dummy_record_warnings = set()

        for orch_idx, thread_records in enumerate(orchestrator_phases):
            tid = 4000 + orch_idx
            orch_worker_thread_idx = orchestrator_thread_base + orch_idx
            for record in thread_records:
                phase = record.get("phase", "unknown")
                start_us = record["start_time_us"]
                end_us = record["end_time_us"]
                dur = end_us - start_us
                submit_idx = record.get("submit_idx", 0)
                task_id = record.get("task_id", -1)

                # Strip "orch_" prefix for display name
                display_name = phase.replace("orch_", "") if phase.startswith("orch_") else phase

                # The document runtime selects how the full task_id.raw value is displayed.
                if task_id >= 0:
                    label = f"{display_name}({task_display(task_id)})"
                else:
                    label = f"{display_name}({submit_idx})"

                event = {
                    "args": _with_run_epoch({"phase": phase, "submit_idx": submit_idx, "task_id": task_id}, record),
                    "cat": "orchestrator",
                    "cname": orch_phase_colors.get(phase, "generic_work"),
                    "name": label,
                    "ph": "X",
                    "pid": 1,
                    "tid": tid,
                    "ts": start_us,
                    "dur": dur,
                }
                events.append(event)

                # deps.json identifies submitted tasks independently of the
                # timing buffers. Runtime records provide the fallback for
                # captures without task metadata.
                if phase == "orch_submit" and task_id >= 0:
                    if deps_kernel_map is not None and task_id in deps_kernel_map:
                        is_dummy = task_id in deps_dummy_task_ids
                        is_regular = not is_dummy
                        if is_dummy and task_id not in dummy_task_ids and task_id not in missing_dummy_record_warnings:
                            print(
                                f"Warning: dummy({task_display(task_id)}) has no dummy_task scheduler record; "
                                "its Worker View bar cannot be rendered.",
                                file=sys.stderr,
                            )
                            missing_dummy_record_warnings.add(task_id)
                    else:
                        is_regular = task_id in regular_task_ids
                        is_dummy = task_id in dummy_task_ids
                    is_predicated_skip = task_id in predicated_skip_task_ids
                    if not is_regular and not is_dummy and not is_predicated_skip:
                        events.append(
                            {
                                "args": _with_run_epoch(
                                    {
                                        "phase": "alloc",
                                        "task_id": task_id,
                                        "event-hint": f"alloc({task_display(task_id)})",
                                    },
                                    record,
                                ),
                                "cat": "event",
                                "cname": "olive",
                                "id": event_id,
                                "name": f"alloc({task_display(task_id)})",
                                "ph": "X",
                                "pid": 4,
                                "tid": AICPU_TID_BASE + orch_worker_thread_idx,
                                "ts": start_us,
                                "dur": max(dur, 0.02),
                            }
                        )
                        aicpu_anchor_maps_by_run[record.get("run_epoch")][task_id].append(
                            {
                                "task_id": task_id,
                                "run_epoch": record.get("run_epoch"),
                                "start_time_us": start_us,
                                "end_time_us": start_us + max(dur, 0.02),
                                "receive_time_us": start_us,
                                "trace_tid": AICPU_TID_BASE + orch_worker_thread_idx,
                                "event_id": event_id,
                            }
                        )
                        event_id += 1

    # Flow events (Flow events "s" and "f" for dependencies). Edges come from
    # deps.json (dep_gen replay); without one we emit no flow events at all,
    # since the device hot path no longer carries fanout (PR #863).
    # SPMD logical tasks anchor Worker View dependency arrows on the earliest
    # visible kernel slice per (func_id, task_id). Dummy/alloc DAG nodes have no
    # kernel row, so their arrows anchor on the AICPU worker slice emitted above.
    # The runs to walk come from the union of both identity sources: a run can
    # contribute only AICPU phases (dummy / alloc nodes and no AICore kernel
    # row at all), and iterating the task map alone would skip it entirely.
    dependency_run_keys = sorted(
        set(task_maps_by_run) | set(aicpu_anchor_maps_by_run),
        key=lambda epoch: (epoch is not None, epoch or 0),
    ) or [None]
    for run_epoch, pred_id, succ_ids in (
        (_e, _p, _s) for _e in dependency_run_keys for _p, _s in edges_by_pred.items()
    ):
        run_task_map = task_maps_by_run.get(run_epoch) or {}
        run_anchor_map = aicpu_anchor_maps_by_run.get(run_epoch) or {}
        if pred_id not in run_task_map and pred_id not in run_anchor_map:
            continue

        for succ_id in succ_ids:
            if succ_id not in run_task_map and succ_id not in run_anchor_map:
                if verbose:
                    print(
                        f"Warning: Task {task_display(pred_id)} (raw {pred_id}) "
                        f"references non-existent successor {task_display(succ_id)} (raw {succ_id})"
                    )
                continue

            row_pairs = _flow_row_pairs(
                pred_id,
                succ_id,
                run_task_map,
                spmd_task_ids,
                _worker_flow_anchor_rows,
                run_anchor_map,
            )
            if not row_pairs:
                continue

            output_task_count = _dependency_task_fan_count(pred_id, spmd_task_ids, run_task_map, deps_block_map)
            input_task_count = _dependency_task_fan_count(succ_id, spmd_task_ids, run_task_map, deps_block_map)
            for pred_row, succ_row in row_pairs:
                src_bar_start_us = _task_slice_start_us(pred_row)
                src_end_us = pred_row["end_time_us"]
                dst_ts_start = _task_slice_start_us(succ_row)
                hb_violated = src_end_us > dst_ts_start
                flow_name = "hb_violation" if hb_violated else "dependency"
                if hb_violated:
                    hb_violation_count += 1
                src_tid, src_event_id = worker_flow_endpoint(pred_row)
                dst_tid, dst_event_id = worker_flow_endpoint(succ_row)
                _append_dependency_flow_pair(
                    events,
                    flow_id,
                    flow_name,
                    4,
                    src_tid,
                    src_bar_start_us,
                    src_event_id,
                    4,
                    dst_tid,
                    dst_ts_start,
                    dst_event_id,
                    input_task_count=input_task_count,
                    output_task_count=output_task_count,
                )
                flow_id += 1
                deps_flow_count += 1

    if verbose:
        if deps_edges is not None:
            print(f"  Dependency flow events: {deps_flow_count} edges (source: deps.json)")
            if spmd_task_ids:
                print(
                    f"  SPMD tasks: {len(spmd_task_ids)} logical task(s); "
                    "dependency arrows independently anchor on each view's earliest visible slice per function"
                )
        else:
            print("  Flow events: 0 (no deps.json — re-run dep_gen and pass --deps-json to add arrows)")
        if hb_violation_count > 0:
            print(f"  Happens-before violations: {hb_violation_count} edge(s) flagged as 'hb_violation'")

    # Scheduler View dependency mirror (Scheduler timestamps).
    if has_scheduler_task_data:
        for run_task_map, pred_id, succ_ids in (
            (_m, _p, _s) for _m in task_maps_by_run.values() for _p, _s in edges_by_pred.items()
        ):
            if pred_id not in run_task_map:
                continue

            for succ_id in succ_ids:
                if succ_id not in run_task_map:
                    continue

                row_pairs = _flow_row_pairs(
                    pred_id,
                    succ_id,
                    run_task_map,
                    spmd_task_ids,
                    _scheduler_flow_anchor_rows,
                )
                if not row_pairs:
                    continue

                output_task_count = _dependency_task_fan_count(pred_id, spmd_task_ids, run_task_map, deps_block_map)
                input_task_count = _dependency_task_fan_count(succ_id, spmd_task_ids, run_task_map, deps_block_map)
                for pred_row, succ_row in row_pairs:
                    src_dispatch_us = pred_row.get("dispatch_time_us", 0)
                    src_finish_us = pred_row.get("finish_time_us", 0)
                    dst_dispatch_us = succ_row.get("dispatch_time_us", 0)
                    dst_finish_us = succ_row.get("finish_time_us", 0)
                    # Skip when Scheduler timestamps are missing or zero (matches Scheduler
                    # View bar emission, which rejects finish_us <= 0).
                    if src_dispatch_us < 0 or src_finish_us <= 0 or dst_dispatch_us < 0 or dst_finish_us <= 0:
                        continue
                    scheduler_hb_violated = src_finish_us > dst_dispatch_us
                    scheduler_flow_name = "hb_violation" if scheduler_hb_violated else "dependency"
                    _append_dependency_flow_pair(
                        events,
                        flow_id,
                        scheduler_flow_name,
                        3,
                        task_to_scheduler_tid.get(_execution_key(pred_row), core_to_tid[pred_row["core_id"]]),
                        src_dispatch_us,
                        task_to_scheduler_event_id.get(_execution_key(pred_row)),
                        3,
                        task_to_scheduler_tid.get(_execution_key(succ_row), core_to_tid[succ_row["core_id"]]),
                        dst_dispatch_us,
                        task_to_scheduler_event_id.get(_execution_key(succ_row)),
                        input_task_count=input_task_count,
                        output_task_count=output_task_count,
                    )
                    flow_id += 1

    # Complete-phase flow arrows. The complete phase wraps the Scheduler's
    # completion-polling loop: it observes AICore subtask FINs, increments
    # the slot's per-task subtask counter, and on the LAST subtask of a
    # logical task it walks the fanout list and releases each consumer's
    # fanin refcount. The arrows are built in two stages.
    #
    # Inbound: per-task, NOT per-subtask. A task is logically "completed"
    # only when its LAST subtask is observed (the one that triggers
    # logical completion count in firmware). For SPMD with N subtasks across
    # N cores, the earlier N-1 subtasks just bump the slot's
    # completed_subtasks counter inside whatever complete phase happened to
    # poll them; only the LAST subtask's finish actually completes the
    # task. So per task: take max(finish_time_us) across its subtasks and
    # find the complete phase that CONTAINS that time. Each task view starts
    # its visual arrow on the same independently selected anchor it uses for
    # SPMD dependency arrows, then lands at the last subtask's Scheduler finish
    # timestamp inside the complete phase. This keeps related SPMD flows
    # consistent within each view without changing completion attribution.
    #
    # Outbound: per-consumer, gated on full fanin. Each consumer in
    # deps.json has multiple producer fanin edges; refcount += 1 fires
    # whenever ANY producer's complete walks its fanout, but the consumer
    # only becomes ready when ALL producers have completed. The complete
    # that triggers the LAST refcount bump is the one that "released" the
    # consumer — that's the causal edge. Compute by walking complete
    # phases in temporal order and tracking each consumer's satisfied
    # fanin count.
    if scheduler_phases and core_to_thread:
        # `complete_index` was built once in the pre-pass above, which runs
        # under the weaker `if scheduler_phases:` guard and therefore always
        # precedes this block. Rebuilding it here would sort and bucket the same
        # records a second time.

        # Group subtask records by task_id; SPMD tasks have multiple rows.
        # Keyed by (run_epoch, task_id): completion attribution and fanin
        # release describe one execution, so two runs of the same task are two
        # entries. Merging them would let run 2's complete satisfy run 1's
        # consumer and push its fanin past its total.
        tasks_by_id: dict[tuple, list[dict]] = defaultdict(list)
        for t in tasks:
            tasks_by_id[(t.get("run_epoch"), t["task_id"])].append(t)

        # For each task: completion = LAST subtask's finish observation.
        # The owning thread is determined by core_to_thread of that last
        # subtask's core — typical case is the same thread observed
        # earlier subtasks too, but we don't assume. Each task view selects its
        # own earliest visible subtask slice; both flows end at the LAST
        # subtask's Scheduler finish timestamp, preserving completion attribution.
        task_to_complete: dict[tuple, dict] = {}
        # (run_epoch, task_id) -> (last_end_us, last_finish_us, core_id)
        task_last_subtask: dict[tuple, tuple[float, float, int]] = {}
        task_worker_anchors: dict[tuple, list[dict]] = {}
        task_scheduler_anchors: dict[tuple, list[dict]] = {}
        for task_key, recs in tasks_by_id.items():
            run_epoch, tid = task_key
            # This run's rows only: the merged map either returns both runs'
            # rows, or — when they look like two SPMD subtasks of one task —
            # collapses to the earliest run's row for every run.
            run_task_map = task_maps_by_run.get(run_epoch) or {}
            valid_finishes = [
                (r.get("finish_time_us"), r.get("end_time_us"), r["core_id"])
                for r in recs
                if r.get("finish_time_us") is not None and r["finish_time_us"] >= 0 and r.get("end_time_us") is not None
            ]
            worker_anchors = _worker_flow_anchor_rows(tid, run_task_map, spmd_task_ids)
            scheduler_anchors = _scheduler_flow_anchor_rows(tid, run_task_map, spmd_task_ids)
            if not valid_finishes or not worker_anchors:
                continue
            last_finish_us, last_end_us, last_cid = max(valid_finishes, key=lambda x: x[0])
            if last_cid >= len(core_to_thread):
                continue
            owning_thread = core_to_thread[last_cid]
            if owning_thread < 0 or owning_thread >= len(complete_index):
                continue
            task_last_subtask[task_key] = (last_end_us, last_finish_us, last_cid)
            task_worker_anchors[task_key] = worker_anchors
            task_scheduler_anchors[task_key] = scheduler_anchors
            # Find the complete phase that CONTAINS this last_finish_us.
            # Fall back to the next-starting complete if none contains
            # (rare: AICore reported the finish but the scheduler hadn't
            # entered its next complete phase by run end). Bisect for O(log N).
            chosen = _select_complete_for_finish(
                complete_index[owning_thread]["by_run"].get(run_epoch, _NO_COMPLETES), last_finish_us
            )
            if chosen is not None:
                task_to_complete[task_key] = chosen

        # ---- Inbound: one arrow per independently selected view anchor ----
        # Source ts = <bar end> - epsilon so it lands INSIDE the task X event
        # selected for that view. Without this anchoring Perfetto can't bind the
        # flow to a slice and the arrow is invisible when you click the task.
        # The pid=2 endpoint (thread + ts) is identical for both views, so
        # completion attribution is unchanged; only the visual source differs.
        FLOW_EPSILON_US = 0.01
        for task_key, comp in task_to_complete.items():
            run_epoch, tid = task_key
            _last_end_us, last_finish_us, last_cid = task_last_subtask[task_key]
            owning_thread = core_to_thread[last_cid]
            dst_tid = sched_lane_tid(owning_thread, 0)
            dst_ts = comp["start_time_us"]
            if comp["start_time_us"] <= last_finish_us <= comp["end_time_us"]:
                dst_ts = last_finish_us
            for anchor in task_worker_anchors[task_key]:
                # Worker View (pid=4): anchor on the kernel slice (end_time_us).
                src_tid = core_to_tid[anchor["core_id"]]
                src_event_id = task_to_event_id.get((run_epoch, tid, anchor["core_id"]))
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "complete",
                        "ph": "s",
                        "pid": 4,
                        "tid": src_tid,
                        "ts": anchor["end_time_us"] - FLOW_EPSILON_US,
                        **({"bind_id": src_event_id} if src_event_id is not None else {}),
                    }
                )
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "complete",
                        "ph": "f",
                        "pid": 2,
                        "tid": dst_tid,
                        "ts": dst_ts,
                        "bp": "e",
                    }
                )
                flow_id += 1

            for anchor in task_scheduler_anchors[task_key]:
                # Scheduler View (pid=3): anchor on the Scheduler dispatch→finish
                # bar (source ts = finish_time_us). Skip when the anchor has
                # no Scheduler finish — its pid=3 bar doesn't exist to bind to.
                anchor_finish_us = anchor.get("finish_time_us")
                if anchor_finish_us is None or anchor_finish_us <= 0:
                    continue
                sched_src_tid = task_to_scheduler_tid.get(
                    (run_epoch, tid, anchor["core_id"]), core_to_tid[anchor["core_id"]]
                )
                sched_src_event_id = task_to_scheduler_event_id.get((run_epoch, tid, anchor["core_id"]))
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "complete",
                        "ph": "s",
                        "pid": 3,
                        "tid": sched_src_tid,
                        "ts": anchor_finish_us - FLOW_EPSILON_US,
                        **({"bind_id": sched_src_event_id} if sched_src_event_id is not None else {}),
                    }
                )
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "complete",
                        "ph": "f",
                        "pid": 2,
                        "tid": dst_tid,
                        "ts": dst_ts,
                        "bp": "e",
                    }
                )
                flow_id += 1

        # ---- Outbound: per-consumer, gated on full fanin ----
        if deps_edges is not None:
            # Invert deps_edges to consumer → predecessors for fanin counting.
            preds_for_consumer: dict[int, list[int]] = defaultdict(list)
            for pred, succs in deps_edges.items():
                for succ in succs:
                    preds_for_consumer[succ].append(pred)
            fanin_total = {c: len(preds) for c, preds in preds_for_consumer.items()}
            # Satisfied counts are per (run_epoch, consumer): deps.json is the
            # run-independent graph, so without the epoch a second run's
            # completions would keep incrementing the first run's consumers.
            fanin_satisfied: dict[tuple, int] = defaultdict(int)

            # Reverse map: complete_phase id → the executions it completed.
            complete_to_tasks: dict[int, list[tuple]] = defaultdict(list)
            for task_key, comp in task_to_complete.items():
                complete_to_tasks[id(comp)].append(task_key)

            # Walk completes in temporal order (by end_time). Within each,
            # walk the tasks it completed; for each completed task, bump
            # its consumers' satisfied fanin. The complete that pushes a
            # consumer's satisfied count to its total is the one that
            # released that consumer.
            all_completes = []
            for thr_idx, entry in enumerate(complete_index):
                phases = entry["all"]
                for p in phases:
                    all_completes.append((p["end_time_us"], thr_idx, p))
            # Explicit key restricts the comparison to (end_time_us, thr_idx).
            # Without it, ties in both fields fall through to comparing the
            # third element (a dict), which raises TypeError in Python 3.
            all_completes.sort(key=lambda x: (x[0], x[1]))

            # Earliest dispatch per task_id (for arrow target).
            earliest_dispatch_us: dict[tuple, tuple[float, int]] = {}
            for task_key, recs in tasks_by_id.items():
                valid = [
                    (r.get("dispatch_time_us"), r["core_id"])
                    for r in recs
                    if r.get("dispatch_time_us") is not None and r["dispatch_time_us"] >= 0
                ]
                if not valid:
                    continue
                d_us, d_cid = min(valid, key=lambda x: x[0])
                if d_cid >= len(core_to_thread):
                    continue
                d_thr = core_to_thread[d_cid]
                if d_thr < 0:
                    continue
                earliest_dispatch_us[task_key] = (d_us, d_thr)

            for end_us, comp_thr, comp in all_completes:
                completed_keys = complete_to_tasks.get(id(comp), ())
                if not completed_keys:
                    continue
                triggered: list[tuple] = []
                for completed_epoch, completed_tid in completed_keys:
                    # Static graph looked up by bare id; the consumer it
                    # releases is the one in the same run.
                    for consumer in deps_edges.get(completed_tid, ()):
                        consumer_key = (completed_epoch, consumer)
                        fanin_satisfied[consumer_key] += 1
                        if fanin_satisfied[consumer_key] == fanin_total.get(consumer, 0):
                            triggered.append(consumer_key)
                if not triggered:
                    continue
                src_tid = sched_lane_tid(comp_thr, 0)
                for consumer_key in triggered:
                    if consumer_key not in earliest_dispatch_us:
                        continue
                    d_us, d_thr = earliest_dispatch_us[consumer_key]
                    # Skip degenerate "dispatched before complete ended" —
                    # the consumer was popped/dispatched off a still-in-flight
                    # release path while the complete was still running;
                    # the arrow would point backwards.
                    if d_us < end_us:
                        continue
                    events.append(
                        {
                            "cat": "flow",
                            "id": flow_id,
                            "name": "complete→ready",
                            "ph": "s",
                            "pid": 2,
                            "tid": src_tid,
                            # Anchor inside the complete phase X event so
                            # clicking the complete block surfaces this arrow.
                            "ts": end_us - FLOW_EPSILON_US,
                        }
                    )
                    events.append(
                        {
                            "cat": "flow",
                            "id": flow_id,
                            "name": "complete→ready",
                            "ph": "f",
                            "pid": 2,
                            "tid": sched_lane_tid(d_thr, 0),
                            "ts": d_us,
                            "bp": "e",
                        }
                    )
                    flow_id += 1

    # Scheduler DISPATCH → task execution arrows
    if scheduler_phases and has_scheduler_task_data:
        # Build core_id → scheduler thread mapping.
        # Prefer explicit core_to_thread from perf JSON (written by AICPU after orchestration).
        # Fall back to voting heuristic for older data without the mapping.
        core_to_sched_thread = {}

        if core_to_thread:
            for core_id, thread_idx in enumerate(core_to_thread):
                if thread_idx >= 0:
                    core_to_sched_thread[core_id] = thread_idx
            if verbose:
                print(f"  Core-to-thread mapping: {len(core_to_sched_thread)} cores (from perf JSON)")
        else:
            # Fallback: infer via voting (for perf JSON without core_to_thread field)
            dispatch_phases_by_thread = {}
            for thread_idx, thread_records in enumerate(scheduler_phases):
                dispatch_records = [r for r in thread_records if r.get("phase") == "dispatch"]
                if dispatch_records:
                    dispatch_phases_by_thread[thread_idx] = dispatch_records

            core_thread_votes = defaultdict(lambda: defaultdict(int))
            for task in tasks:
                dispatch_us = task.get("dispatch_time_us", 0)
                if dispatch_us < 0:
                    continue
                core_id = task["core_id"]
                for thread_idx, dispatch_records in dispatch_phases_by_thread.items():
                    for dr in dispatch_records:
                        if dr["start_time_us"] <= dispatch_us <= dr["end_time_us"]:
                            core_thread_votes[core_id][thread_idx] += 1
                            break

            for core_id, votes in core_thread_votes.items():
                core_to_sched_thread[core_id] = max(votes.items(), key=lambda kv: kv[1])[0]
            if verbose:
                print(f"  Core-to-thread mapping: {len(core_to_sched_thread)} cores (inferred via voting)")

        for task in tasks:
            dispatch_us = task.get("dispatch_time_us", 0)
            if dispatch_us < 0:
                continue

            matched_thread = core_to_sched_thread.get(task["core_id"])

            if matched_thread is not None:
                sched_tid = sched_lane_tid(matched_thread, 0)
                core_tid = core_to_tid[task["core_id"]]
                scheduler_view_tid = task_to_scheduler_tid.get(_execution_key(task), core_tid)

                # Flow: scheduler DISPATCH → Worker View task start
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "dispatch",
                        "ph": "s",
                        "pid": 2,
                        "tid": sched_tid,
                        "ts": dispatch_us,
                    }
                )
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "dispatch",
                        "ph": "f",
                        "pid": 4,
                        "tid": core_tid,
                        "ts": task["start_time_us"],
                        "bp": "e",
                    }
                )
                flow_id += 1

                # Flow: scheduler DISPATCH → Scheduler View task start
                scheduler_event_id = task_to_scheduler_event_id.get(_execution_key(task))
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "dispatch",
                        "ph": "s",
                        "pid": 2,
                        "tid": sched_tid,
                        "ts": dispatch_us,
                    }
                )
                flow_f = {
                    "cat": "flow",
                    "id": flow_id,
                    "name": "dispatch",
                    "ph": "f",
                    "pid": 3,
                    "tid": scheduler_view_tid,
                    "ts": dispatch_us,
                    "bp": "e",
                }
                if scheduler_event_id is not None:
                    flow_f["bind_id"] = scheduler_event_id
                events.append(flow_f)
                flow_id += 1

    # Orchestrator → scheduler dispatch:
    # Anchor each task's dispatch arrow on the end of its orch_submit record
    # (covers the entire submit_task() span). Legacy captures with the older
    # per-sub-step phases (orch_fanin / orch_params) are accepted as fallbacks.
    cross_domain_aligned = not (
        orchestrator_source == "host"
        and timeline_metadata
        and not timeline_metadata.get("cross_domain_latency_available", False)
    )
    if orchestrator_phases and scheduler_phases and cross_domain_aligned:
        orch_anchor_by_task = {}
        for orch_idx, thread_records in enumerate(orchestrator_phases):
            for record in thread_records:
                phase = record.get("phase")
                task_id = record.get("task_id", -1)
                if task_id < 0:
                    continue
                tid_k = normalize_task_id_int(task_id)
                if tid_k is None:
                    continue
                # First-seen orch_submit wins; legacy orch_fanin / orch_params
                # only fill in when no orch_submit exists for that task. The
                # explicit "not already submit→dispatch" guard preserves first-
                # seen semantics even if a (defensive) duplicate orch_submit
                # ever appears for the same task.
                existing = orch_anchor_by_task.get(tid_k)
                if phase == "orch_submit" and (existing is None or existing[2] != "submit→dispatch"):
                    orch_anchor_by_task[tid_k] = (record, orch_idx, "submit→dispatch")
                elif existing is None and phase == "orch_fanin":
                    orch_anchor_by_task[tid_k] = (record, orch_idx, "fanin→dispatch")
                elif existing is None and phase == "orch_params":
                    orch_anchor_by_task[tid_k] = (record, orch_idx, "params→dispatch")

        if has_scheduler_task_data and orch_anchor_by_task:
            for task in tasks:
                tid = normalize_task_id_int(task.get("task_id"))
                if tid is None:
                    continue

                dispatch_us = task.get("dispatch_time_us", 0)
                if dispatch_us < 0:
                    continue

                matched_thread = core_to_sched_thread.get(task["core_id"])
                if matched_thread is None:
                    continue

                sched_tid = sched_lane_tid(matched_thread, 0)

                anchor = orch_anchor_by_task.get(tid)
                if anchor is None:
                    continue

                anchor_rec, orch_idx, flow_name = anchor
                anchor_us = anchor_rec["end_time_us"]

                orch_tid = 4000 + orch_idx

                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": flow_name,
                        "ph": "s",
                        "pid": 1,
                        "tid": orch_tid,
                        "ts": anchor_us,
                    }
                )
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": flow_name,
                        "ph": "f",
                        "pid": 2,
                        "tid": sched_tid,
                        "ts": dispatch_us,
                        "bp": "e",
                    }
                )
                flow_id += 1

    if verbose:
        print(f"  Total events: {len(events)}")
        print(f"  Flow events: {flow_id}")

    # Step 3: Write JSON file (with traceEvents wrapper to match C++ output)
    if emit_overhead:
        oh = build_overhead_counter_events(tasks, deps_edges)
        events.extend(oh)
        if verbose:
            print(f"  Overhead Analysis: {sum(1 for e in oh if e.get('ph') == 'C')} counter points (8 tracks)")

    trace: dict[str, object] = {"traceEvents": events}
    metadata = dict(timeline_metadata) if timeline_metadata else {}
    # Downstream tools (critical_path) re-format task ids from this trace alone, so it
    # has to name the runtime whose TaskId layout its labels and ids follow. Always set:
    # task_display_for above already refused a caller that named none.
    metadata["runtime"] = resolve_runtime(runtime_name)
    if metadata:
        trace["metadata"] = metadata
    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(trace, f, indent=2)

    if verbose and output_path is not None:
        print(f"JSON written to: {output_path}")
    return trace


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Convert swimlane performance JSON to Chrome Trace Event JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                            # Use latest .json in outputs/, output to outputs/
  %(prog)s outputs/<case>_<ts>/chip_swimlane_records.json   # Output: outputs/merged_swimlane_20260210_143526.json
  %(prog)s outputs/<case>_<ts>/chip_swimlane_records.json -o custom_output.json
  %(prog)s outputs/<case>_<ts>/chip_swimlane_records.json \
      -k examples/host_build_graph/paged_attention/kernels/kernel_config.py
  %(prog)s outputs/<case>_<ts>/chip_swimlane_records.json -v
  %(prog)s build_output/<case>/dfx_outputs --dispatch d0
        """,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help=(
            "Input JSON file, or a dfx_outputs directory containing rank*/dN/. "
            "If omitted, uses the latest chip_swimlane_records_*.json in outputs/."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file (default: merged_swimlane.json for a file, l3_swimlane.json for a directory)",
    )
    parser.add_argument(
        "-k",
        "--kernel-config",
        help="Path to kernel_config.py file for func_id to function name mapping",
    )
    parser.add_argument(
        "--func-names",
        help=(
            "Path to name_map*.json for func_id to function name mapping. "
            "Defaults to a unique sibling name_map*.json next to the input."
        ),
    )
    parser.add_argument(
        "--deps-json",
        help=(
            "Path to a dep_gen replay deps.json (defaults to sibling of the perf JSON). "
            "Without one the trace has no dependency arrows — re-run with --enable-dep-gen first."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--dispatch",
        help="Local capture directory to merge for directory input, for example d0",
    )
    parser.add_argument(
        "--dispatch-id",
        help="Parent dispatch identity to merge for directory input, formatted as RUN_ID:TASK_SLOT",
    )
    parser.add_argument(
        "--host-log",
        action="append",
        help="Host [STRACE] log holding the chip.run.runner_run windows the captures are placed in "
        "(repeatable). Single-file input prefers sibling host_clock_alignment.*.log over host.*.log. "
        "Directory input reads both at the root and rank*/d*/host_clock_alignment.*.log. "
        "A single capture with no matching logs remains unaligned; valid saved alignment needs no logs.",
    )
    parser.add_argument(
        "--rank-pid",
        action="append",
        metavar="RANK=PID[:INV]",
        help="Pin one Rank's capture to the Host invocation that ran it (repeatable). Only needed when the "
        "captures carry no dispatch_identity.json and Ranks running the same shape cannot be told apart by "
        "their device windows. Give :INV when the process ran more than once.",
    )
    parser.add_argument(
        "--overhead",
        action="store_true",
        help="Add an 'Overhead Analysis' track (8 counter lines: per-engine "
        "idle/ready/overhead + system all_overhead/has_overhead). Needs deps.json.",
    )
    return parser


def _resolve_input_path(args):
    """Resolve input path, auto-selecting newest outputs/<case>/chip_swimlane_records.json if unspecified."""
    if args.input is not None:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            return None
        return input_path

    outputs_dir = Path.cwd() / "outputs"
    json_files = list(outputs_dir.glob("*/chip_swimlane_records.json"))
    if not json_files:
        print(f"Error: No outputs/*/chip_swimlane_records.json found under {outputs_dir}", file=sys.stderr)
        print("Run a test with --enable-chip-swimlane first, or specify an explicit input.", file=sys.stderr)
        return None

    input_path = max(json_files, key=lambda p: p.stat().st_mtime)
    if args.verbose:
        print(f"Auto-selected latest file: {input_path}")
    return input_path


def _resolve_output_path(args, input_path):
    """Determine output path from args or derive from input directory name."""
    if args.output:
        return Path(args.output)

    if input_path.is_dir():
        return input_path / "l3_swimlane.json"

    # Default: write merged_swimlane.json next to the input. The parent
    # directory name (e.g. outputs/<case>_<ts>/) already disambiguates runs.
    return input_path.parent / "merged_swimlane.json"


def _print_verbose_data_info(data, verbose):
    """Print verbose summary of loaded performance data, including phase counts
    when present (chip_swimlane_level >= SCHED_PHASES)."""
    if not verbose:
        return
    print("\n=== Performance Data ===")
    print(f"  L2 perf level: {data['chip_swimlane_level']}")
    print(f"  Task Count: {len(data['tasks'])}")
    if data["tasks"]:
        start_times = [t["start_time_us"] for t in data["tasks"]]
        end_times = [t["end_time_us"] for t in data["tasks"]]
        min_time = min(start_times)
        max_time = max(end_times)
        print(f"  Time Range: {min_time:.3f} us - {max_time:.3f} us (span: {max_time - min_time:.3f} us)")
    print()
    scheduler_phases = data.get("aicpu_scheduler_phases")
    orchestrator_phases = data.get("aicpu_orchestrator_phases")
    core_to_thread = data.get("core_to_thread")
    if scheduler_phases:
        print(f"  Scheduler threads: {len(scheduler_phases)}")
        print(f"  Total phase records: {sum(len(t) for t in scheduler_phases)}")
    if orchestrator_phases:
        print(f"  Orchestrator threads: {len(orchestrator_phases)}")
        print(f"  Total orchestrator phase records: {sum(len(t) for t in orchestrator_phases)}")
        # submit_count is derivable as the number of orch_submit records (one per submit).
        # Legacy captures fall back to orch_fanin (was last phase of submit pre-fold).
        submit_count = sum(1 for thread in orchestrator_phases for r in thread if r.get("phase") == "orch_submit")
        if submit_count == 0:
            submit_count = sum(1 for thread in orchestrator_phases for r in thread if r.get("phase") == "orch_fanin")
        if submit_count:
            print(f"  Orchestrator: {submit_count} tasks submitted")
    if core_to_thread:
        print(f"  Core-to-thread mapping: {len(core_to_thread)} cores")


def _find_sibling_name_map(input_path):
    """Return the unique sibling ``name_map*.json``, if one exists."""
    candidates = sorted(path for path in Path(input_path).parent.glob("name_map*.json") if path.is_file())
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        candidate_names = ", ".join(path.name for path in candidates)
        print(
            f"Warning: multiple sibling name maps found next to {input_path}: {candidate_names}; "
            "pass --func-names explicitly.",
            file=sys.stderr,
        )
    return None


def _load_func_names(args, input_path):
    """Load func_id→name mapping from an explicit or sibling source.

    Returns:
        tuple: (func_id_to_name dict, orchestrator_name str or None)
    """
    func_names_path = Path(args.func_names) if args.func_names else None
    if func_names_path is None and not args.kernel_config:
        func_names_path = _find_sibling_name_map(input_path)

    if func_names_path is not None:
        if args.verbose:
            source = "Auto-discovered" if not args.func_names else "Loading"
            print(f"{source} func names from: {func_names_path}")
        func_names, orchestrator_name = load_func_names_json(func_names_path)
        if args.verbose:
            print(f"  Loaded {len(func_names)} function name mappings:")
            for func_id, name in sorted(func_names.items(), key=lambda x: int(x[0])):
                print(f"    func_id={func_id}: {name}")
            if orchestrator_name:
                print(f"  Orchestrator: {orchestrator_name}")
            print()
        return func_names, orchestrator_name

    if args.kernel_config:
        if args.verbose:
            print(f"Loading kernel config from: {args.kernel_config}")
        func_names = load_kernel_config(args.kernel_config)
        if args.verbose:
            print(f"  Loaded {len(func_names)} function name mappings from kernel_config.py:")
            for func_id, name in sorted(func_names.items(), key=lambda x: int(x[0])):
                print(f"    func_id={func_id}: {name}")
            print()
        return func_names, None

    return {}, None


_RANK_DIR_PATTERN = re.compile(r"rank([0-9]+)")
_DISPATCH_DIR_PATTERN = re.compile(r"d[0-9]+")
_DISPATCH_ID_PATTERN = re.compile(r"([0-9]+):([0-9]+)")
_RANK_PID_STRIDE = 100
# Perfetto orders process groups by pid, so the pid *is* the layout: the two
# clock domains read as two blocks only if every Host-domain pid sorts below
# every Chip one. The Host block takes the first stride, and the Chip views
# start at the stride above it.
#
# Inside the Host block the dispatching processes come first, one pid each —
# the L3 scheduler that sent the work, and whatever level sent to it — because
# they are what the Ranks below them are a consequence of.
_DISPATCHER_PID_BASE = 1
_DISPATCHER_PID_LIMIT = 10
# Then one run of pids per Rank, which keeps its three lanes adjacent.
_HOST_BLOCK_PID_BASE = _DISPATCHER_PID_BASE + _DISPATCHER_PID_LIMIT
# Lanes one Rank contributes to the Host block: its own call tree, the device
# phases the Host log carries, and the window they are placed in.
_HOST_LANES_PER_RANK = 3


def _l3_rank_dirs(root):
    root = Path(root)
    rank_dirs = sorted(path for path in root.glob("rank*") if path.is_dir())
    if not rank_dirs:
        raise ValueError(f"no rankN directories found under {root}")

    discovered = []
    seen_ranks = set()
    for rank_dir in rank_dirs:
        match = _RANK_DIR_PATTERN.fullmatch(rank_dir.name)
        if match is None:
            raise ValueError(f"invalid Rank directory name: {rank_dir.name} (expected rankN)")
        rank = int(match.group(1))
        if rank in seen_ranks:
            raise ValueError(f"duplicate Rank number {rank} under {root}")
        seen_ranks.add(rank)
        discovered.append((rank, rank_dir))
    return sorted(discovered)


def _load_dispatch_identity(capture_dir):
    path = Path(capture_dir) / "dispatch_identity.json"
    if not path.is_file():
        return None
    with path.open() as file:
        identity = json.load(file)
    if (
        not isinstance(identity, dict)
        or isinstance(identity.get("schema_version"), bool)
        or identity.get("schema_version") != 1
    ):
        raise ValueError(f"unsupported dispatch identity schema: {path}")

    integer_fields = (
        "run_id",
        "task_slot",
        "group_index",
        "group_size",
        "chip_rank",
        "local_capture_index",
        "endpoint_dispatch_id",
        "pipeline_slot",
        "pipeline_generation",
    )
    for field in integer_fields:
        value = identity.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"dispatch identity {field} must be an integer: {path}")
    if identity["run_id"] <= 0 or identity["task_slot"] < 0:
        raise ValueError(f"dispatch identity has an invalid parent key: {path}")
    if identity["endpoint_dispatch_id"] <= 0 or identity["pipeline_slot"] < 0 or identity["pipeline_generation"] <= 0:
        raise ValueError(f"dispatch identity has invalid endpoint diagnostics: {path}")
    if identity["group_size"] <= 0 or not 0 <= identity["group_index"] < identity["group_size"]:
        raise ValueError(f"dispatch identity has invalid group membership: {path}")
    # Optional: a sidecar written before the field existed carries no host pid,
    # and its capture is then paired by how well the device windows fit.
    host_pid = identity.get("host_pid")
    if host_pid is not None and (isinstance(host_pid, bool) or not isinstance(host_pid, int) or host_pid <= 0):
        raise ValueError(f"dispatch identity host_pid must be a positive integer: {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", identity.get("callable_digest", "")):
        raise ValueError(f"dispatch identity has an invalid callable digest: {path}")

    rank_match = _RANK_DIR_PATTERN.fullmatch(Path(capture_dir).parent.name)
    dispatch_match = _DISPATCH_DIR_PATTERN.fullmatch(Path(capture_dir).name)
    if rank_match is None or dispatch_match is None:
        raise ValueError(f"dispatch identity is not below rankN/dN: {path}")
    path_rank = int(rank_match.group(1))
    path_capture_index = int(Path(capture_dir).name.removeprefix("d"))
    if identity["chip_rank"] != path_rank or identity["local_capture_index"] != path_capture_index:
        raise ValueError(f"dispatch identity disagrees with its rankN/dN path: {path}")
    return identity


def _validate_parent_dispatch_group(discovered, expected_key):
    if not discovered:
        raise ValueError(f"no Rank captures found for parent dispatch {expected_key[0]}:{expected_key[1]}")
    identities = [identity for _, _, identity in discovered]
    group_sizes = {identity["group_size"] for identity in identities}
    callable_digests = {identity.get("callable_digest") for identity in identities}
    if len(group_sizes) != 1 or len(callable_digests) != 1:
        raise ValueError(f"inconsistent metadata for parent dispatch {expected_key[0]}:{expected_key[1]}")
    group_size = next(iter(group_sizes))
    if group_size <= 1:
        raise ValueError(
            f"parent dispatch {expected_key[0]}:{expected_key[1]} is an individual submission; pair it by dN"
        )
    ranks = [rank for rank, _, _ in discovered]
    if len(set(ranks)) != len(ranks):
        duplicates = sorted({rank for rank in ranks if ranks.count(rank) > 1})
        raise ValueError(
            f"parent dispatch {expected_key[0]}:{expected_key[1]} has several captures on the same "
            f"Rank {duplicates}; one Rank contributes at most one capture to a group"
        )
    group_indexes = [identity["group_index"] for identity in identities]
    if len(discovered) != group_size or sorted(group_indexes) != list(range(group_size)):
        raise ValueError(
            f"incomplete parent dispatch {expected_key[0]}:{expected_key[1]}: "
            f"expected group indexes 0..{group_size - 1}, found {sorted(group_indexes)}"
        )
    return {
        "dispatch_pairing": "parent_dispatch_identity",
        "dispatch_identity": {
            "run_id": expected_key[0],
            "task_slot": expected_key[1],
            "group_size": group_size,
            "callable_digest": next(iter(callable_digests)),
        },
    }


def _discover_l3_parent_dispatch_inputs(rank_dirs, dispatch_identity):
    match = _DISPATCH_ID_PATTERN.fullmatch(dispatch_identity)
    if match is None:
        raise ValueError("--dispatch-id must use RUN_ID:TASK_SLOT (for example, 17:5)")
    expected_key = (int(match.group(1)), int(match.group(2)))
    discovered = []
    for rank, rank_dir in rank_dirs:
        for capture_dir in sorted(
            (path for path in rank_dir.glob("d*") if path.is_dir()),
            key=lambda path: int(path.name.removeprefix("d"))
            if _DISPATCH_DIR_PATTERN.fullmatch(path.name)
            else sys.maxsize,
        ):
            if not (capture_dir / "chip_swimlane_records.json").is_file():
                continue
            identity = _load_dispatch_identity(capture_dir)
            if identity is None:
                continue
            if (identity["run_id"], identity["task_slot"]) == expected_key:
                discovered.append((rank, capture_dir / "chip_swimlane_records.json", identity))
    pairing = _validate_parent_dispatch_group(discovered, expected_key)
    return [(rank, records_path) for rank, records_path, _ in sorted(discovered)], pairing


def _discover_l3_local_capture_inputs(rank_dirs, dispatch):
    if not _DISPATCH_DIR_PATTERN.fullmatch(dispatch or ""):
        raise ValueError("--dispatch must name dN (for example, --dispatch d0)")

    # One list of (rank, records_path, identity) triples rather than two parallel
    # lists: the identity belongs to the capture it was read from, and a pairing
    # that depends on two lists staying index-aligned is the failure this
    # function exists to prevent.
    captures = []
    for rank, rank_dir in rank_dirs:
        records_path = rank_dir / dispatch / "chip_swimlane_records.json"
        if not records_path.is_file():
            raise ValueError(f"rank{rank} is missing {dispatch}/chip_swimlane_records.json")
        captures.append((rank, records_path, _load_dispatch_identity(records_path.parent)))

    present_count = sum(identity is not None for _, _, identity in captures)
    if present_count not in (0, len(captures)):
        raise ValueError(f"{dispatch} has dispatch identity metadata for only {present_count}/{len(captures)} Ranks")
    pairing = {"dispatch_pairing": "local_capture_index"}
    if present_count:
        group_keys = {(identity["run_id"], identity["task_slot"]) for _, _, identity in captures}
        has_group = any(identity["group_size"] > 1 for _, _, identity in captures)
        if has_group:
            if len(group_keys) != 1:
                raise ValueError(
                    f"{dispatch} refers to different parent dispatches across Ranks; use --dispatch-id RUN_ID:TASK_SLOT"
                )
            pairing = _validate_parent_dispatch_group(captures, next(iter(group_keys)))
        else:
            pairing["dispatch_identity_status"] = "individual_submissions"
    return sorted((rank, records_path) for rank, records_path, _ in captures), pairing


def _discover_l3_rank_inputs(root, dispatch, dispatch_identity=None):
    if bool(dispatch) == bool(dispatch_identity):
        raise ValueError("directory input requires exactly one of --dispatch dN or --dispatch-id RUN_ID:TASK_SLOT")
    rank_dirs = _l3_rank_dirs(root)
    if dispatch_identity:
        return _discover_l3_parent_dispatch_inputs(rank_dirs, dispatch_identity)
    return _discover_l3_local_capture_inputs(rank_dirs, dispatch)


def discover_l3_conversion_targets(root):
    """Discover safe automatic conversion units below one L3 output root.

    Public because the SceneTest postprocessor needs the same units this
    module's directory mode would pick, one CLI invocation per unit.

    Raises ValueError only when nothing below ``root`` can be paired safely. A
    remainder that cannot be paired by dN downgrades to a stderr warning while
    the parent-identity targets are still returned.
    """
    captures_by_rank = {}
    parent_groups = defaultdict(list)
    for rank, rank_dir in _l3_rank_dirs(root):
        captures = []
        for capture_dir in sorted(
            (path for path in rank_dir.glob("d*") if path.is_dir()),
            key=lambda path: int(path.name.removeprefix("d"))
            if _DISPATCH_DIR_PATTERN.fullmatch(path.name)
            else sys.maxsize,
        ):
            if not _DISPATCH_DIR_PATTERN.fullmatch(capture_dir.name):
                raise ValueError(f"invalid capture directory name: {capture_dir} (expected dN)")
            records_path = capture_dir / "chip_swimlane_records.json"
            if not records_path.is_file():
                continue
            identity = _load_dispatch_identity(capture_dir)
            capture = (rank, records_path, identity)
            captures.append(capture)
            if identity is not None and identity["group_size"] > 1:
                parent_groups[(identity["run_id"], identity["task_slot"])].append(capture)
        captures_by_rank[rank] = captures

    targets = []
    semantically_paired_paths = set()
    for parent_key, captures in sorted(parent_groups.items()):
        pairing = _validate_parent_dispatch_group(captures, parent_key)
        paths = [records_path for _, records_path, _ in sorted(captures)]
        semantically_paired_paths.update(paths)
        targets.append(
            {
                "dispatch": None,
                "dispatch_id": f"{parent_key[0]}:{parent_key[1]}",
                "output_stem": f"l3_swimlane_run{parent_key[0]}_task{parent_key[1]}",
                "capture_dirs": [path.parent for path in paths],
                "pairing": pairing,
            }
        )

    fallback_sets = []
    for rank, captures in sorted(captures_by_rank.items()):
        fallback_sets.append(
            (
                rank,
                {
                    records_path.parent.name: records_path.parent
                    for _, records_path, _ in captures
                    if records_path not in semantically_paired_paths
                },
            )
        )
    if fallback_sets:
        expected = set(fallback_sets[0][1])
        if any(set(captures) != expected for _, captures in fallback_sets[1:]):
            # Only the dN-paired remainder is unsafe here. A parent-identity
            # target is paired by (run_id, task_slot) and is unaffected by what
            # the leftover dN sets look like, so dropping those too would
            # discard exactly the pairings this identity exists to make.
            detail = ", ".join(f"rank{rank}={sorted(captures)}" for rank, captures in fallback_sets)
            message = f"refusing to pair asymmetric local capture indexes under {root}: {detail}"
            if not targets:
                raise ValueError(message)
            print(f"Warning: {message}", file=sys.stderr)
            return targets
        for dispatch in sorted(expected, key=lambda name: int(name.removeprefix("d"))):
            targets.append(
                {
                    "dispatch": dispatch,
                    "dispatch_id": None,
                    "output_stem": f"l3_swimlane_{dispatch}",
                    "capture_dirs": [captures[dispatch] for _, captures in fallback_sets],
                    "pairing": {"dispatch_pairing": "local_capture_index"},
                }
            )
    return targets


def _rank_clock_domain(rank, records_path, raw):
    """The Host clock the Rank's own process timestamps on.

    Containment places a Rank inside a window read from that Host's
    CLOCK_MONOTONIC, so two Ranks are comparable exactly when they name the same
    clock. A capture from before the field existed is accepted with a warning:
    the merge it is asked for is same-host by construction, and refusing an old
    capture buys no correctness a reader could act on.
    """
    clock_domain = (raw.get("metadata") or {}).get("host_clock_domain_id")
    if not clock_domain:
        print(
            f"Warning: rank{rank} predates metadata.host_clock_domain_id ({records_path}); "
            "the merge assumes every Rank ran against one Host clock",
            file=sys.stderr,
        )
        return None
    return str(clock_domain)


def _discover_host_logs(root, explicit):
    """The `[STRACE]` logs that hold the outer windows for these captures.

    Persistent ``host.<pid>.log`` files live at the case/level output root;
    ``host_clock_alignment.<pid>.log`` holds capture-local timing spans.
    See ``docs/dfx/host-trace.md``.
    """
    if explicit:
        paths = [Path(item) for item in explicit]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ValueError(f"--host-log names a file that does not exist: {', '.join(missing)}")
        return paths
    paths = sorted(
        {
            *Path(root).glob("host_clock_alignment.*.log"),
            *Path(root).glob("host.*.log"),
            *Path(root).glob("rank*/d*/host_clock_alignment.*.log"),
        }
    )
    if not paths:
        raise ValueError(
            f"no host_clock_alignment.*.log or host.*.log under {root}: cross-Rank placement reads "
            "each Rank's device work out of the "
            f"{containment.RUNNER_SPAN} window that contains it. Pass --host-log, or re-run with a Host log "
            f"threshold of TIMING or finer."
        )
    return paths


def _parse_rank_pid_pins(values):
    """Read ``--rank-pid RANK=PID`` / ``RANK=PID:INV`` into what containment takes.

    The invocation form exists because a process runs many invocations and the
    bare pid only identifies one of them when the run had exactly one.
    """
    pins = {}
    for item in values or []:
        rank, separator, target = item.partition("=")
        pid, _, inv = target.partition(":")
        if not separator or not rank.strip().isdigit() or not pid.strip().isdigit():
            raise ValueError(f"--rank-pid must be RANK=PID or RANK=PID:INV (for example, 0=4242), not {item!r}")
        if inv and not inv.strip().isdigit():
            raise ValueError(f"--rank-pid invocation must be a number, as RANK=PID:INV, not {item!r}")
        pins[int(rank)] = (int(pid), int(inv)) if inv else int(pid)
    return pins


def _place_rank_captures(host_log_paths, raw_inputs, identities, host_pids, pins):
    """Bound where each Rank's device records sit on the Host timeline.

    Returns the spans as well: the same Host log that supplies the outer window
    also holds that process's own call tree, which the merged trace draws beside
    the Rank it belongs to.
    """
    spans = []
    for path in host_log_paths:
        with path.open(errors="replace") as log:
            spans.extend(parse_spans(log))
    windows = containment.host_windows(spans)
    if not windows:
        raise ValueError(
            f"the Host logs hold no {containment.RUNNER_SPAN} / {containment.DEVICE_WALL_SPAN} pair; "
            "nothing brackets the device work"
        )
    captures = {rank: containment.capture_windows(raw) for rank, raw in raw_inputs.items()}
    pins = dict(pins)
    for rank, raw in raw_inputs.items():
        if rank in pins or host_pids.get(rank) is None or not _is_hbg_host_capture(raw):
            continue
        # HBG Host records fall within their own chip.run invocation.
        # They disambiguate synchronous launches, whose dispatch IDs are zero.
        try:
            matches = _matching_capture_host_windows(raw, spans, {"host_pid": host_pids[rank]})
        except ValueError:
            # Logs without a chip.run root cannot use the Host-record bounds.
            continue
        identity = identities.get(rank)
        matches = [window for window in matches if identity is None or window.identity in (None, identity)]
        if len(matches) == 1:
            pins[rank] = (matches[0].pid, matches[0].inv)
    pairs, pairing = containment.pair_captures(
        windows, captures, forced=pins, identities=identities, host_pids=host_pids
    )
    placements = {rank: containment.place(pairs[rank], captures[rank]) for rank in captures}
    return placements, pairing, spans


def _host_block_lane(rank, lane_index, label):
    """Metadata for one lane of a Rank's Host block.

    Perfetto orders process groups by pid, so the Host block has to keep every
    one of its pids below the first Chip view. Each Rank owns a fixed run of
    them, which keeps its three lanes adjacent.
    """
    pid = _HOST_BLOCK_PID_BASE + rank * _HOST_LANES_PER_RANK + lane_index
    # rank0's Chip views start at one stride, so the whole Host block has to fit
    # below it: past that the two blocks interleave and lanes silently merge.
    if pid >= _RANK_PID_STRIDE:
        raise ValueError(
            f"rank{rank} needs Host-block pid {pid}, which does not fit below the per-Rank stride "
            f"{_RANK_PID_STRIDE}; the merge supports at most "
            f"{(_RANK_PID_STRIDE - _HOST_BLOCK_PID_BASE) // _HOST_LANES_PER_RANK} Ranks"
        )
    return pid, [
        {"args": {"name": f"rank{rank} / {label}"}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": pid},
        {"args": {"sort_index": pid}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": pid},
    ]


def _dispatcher_spans(spans, chip_pids, window_ns):
    """The dispatching processes' spans that belong to the merged dispatch.

    Their logs cover the whole run while the merge covers one dispatch, so the
    Ranks' own span on the Host axis selects what to keep — in two passes,
    because the level above the chip emits spans of two different scopes.

    A span with an invocation is kept or dropped with its whole invocation:
    `node.submit` runs to completion before the Rank it dispatched to starts,
    so an overlap test applied span by span would draw the dispatch and drop
    what led to it.

    ``inv=0`` says the span belongs to a thread rather than to one run —
    `node.scheduler_loop` is emitted that way (`Scheduler::…`, run id 0), so
    its "invocation" is every loop the process ever ran and cannot be the unit.
    Those are taken span by span, against the story the first pass selected: a
    loop is drawn when it overlaps what the dispatch did, which is wider than
    the Ranks' own windows at both ends.
    """
    window_lo, window_hi = window_ns
    # By family, not only by pid: a chip child that no placement paired with is
    # still a chip child, and labelling it a dispatcher would be a lie.
    host_spans = [
        span for span in spans if span.pid not in chip_pids and not span.is_device and span_family(span.name) != "chip"
    ]

    def overlaps(span, lo, hi):
        return span.ts < hi and span.ts + span.dur > lo

    concurrent = {(span.pid, span.inv) for span in host_spans if span.inv and overlaps(span, window_lo, window_hi)}
    selected = [span for span in host_spans if span.inv and (span.pid, span.inv) in concurrent]

    story_lo = min([window_lo] + [span.ts for span in selected])
    story_hi = max([window_hi] + [span.ts + span.dur for span in selected])
    selected += [span for span in host_spans if not span.inv and overlaps(span, story_lo, story_hi)]
    return selected


def _dispatcher_block_events(spans, global_origin_ns):
    """The processes that dispatched to these Ranks, drawn above them.

    Each process writes its `host.<pid>.log` at the root of the level
    namespace it owns, so a direct L3 run supplies the scheduler's own `node.*`
    spans — and an L4's `network1.*` above them — from the persistent logs.
    Under a parent-assigned `nodeN` namespace the levels above it write one
    directory up, outside the root this merge reads, so only that namespace's
    own levels reach here. They are Host CLOCK_MONOTONIC and same-host
    cross-process comparable, so they go straight onto the axis: no
    containment, no slack. Containment is only ever needed for the device
    clock, which is why nothing in this block carries a bound.
    """
    processes = host_process_lanes(spans)
    if len(processes) > _DISPATCHER_PID_LIMIT:
        raise ValueError(
            f"{len(processes)} dispatching processes is more than the {_DISPATCHER_PID_LIMIT} the Host block "
            "reserves pids for"
        )

    events = []
    for index, (pid, process) in enumerate(sorted(processes.items())):
        block_pid = _DISPATCHER_PID_BASE + index
        events += [
            {
                "args": {"name": process["label"]},
                "cat": "__metadata",
                "name": "process_name",
                "ph": "M",
                "pid": block_pid,
            },
            {
                "args": {"sort_index": block_pid},
                "cat": "__metadata",
                "name": "process_sort_index",
                "ph": "M",
                "pid": block_pid,
            },
        ]
        events += [
            {
                "args": {"name": name},
                "cat": "__metadata",
                "name": "thread_name",
                "ph": "M",
                "pid": block_pid,
                "tid": tid,
            }
            for tid, name in process["lanes"].items()
        ]
        for span, attrs, tid in process["spans"]:
            events.append(
                {
                    "name": span.name,
                    "cat": "host",
                    "ph": "X",
                    "pid": block_pid,
                    "tid": tid,
                    "ts": (span.ts - global_origin_ns) / 1000.0,
                    "dur": span.dur / 1000.0,
                    "args": {"inv": span.inv, "os_pid": span.pid, "os_tid": span.tid, "depth": span.depth, **attrs},
                }
            )
    return events


def _host_call_tree_events(rank, placement, spans, global_origin_ns):
    """The Rank's own Host call tree — `chip.run` and everything under it.

    Drawn on the Host axis directly, since that is the clock it was recorded
    on: this lane carries no placement error at all, unlike the two below it.
    They nest by timestamp on one track because they are one thread's call
    tree, which is what `depth` already says.
    """
    pid, events = _host_block_lane(rank, 0, "Host")
    events.append(
        {
            "args": {"name": f"pid {placement.host.pid}"},
            "cat": "__metadata",
            "name": "thread_name",
            "ph": "M",
            "pid": pid,
            "tid": 0,
        }
    )
    for span in sorted(spans, key=lambda item: (item.ts, -item.dur)):
        events.append(
            {
                "name": span.name,
                "cat": "host",
                "ph": "X",
                "pid": pid,
                "tid": 0,
                "ts": (span.ts - global_origin_ns) / 1000.0,
                "dur": span.dur / 1000.0,
                "args": {"rank": rank, "inv": span.inv, "os_pid": span.pid, "depth": span.depth},
            }
        )
    return events


def _host_device_phase_events(rank, placement, spans, global_origin_ns):
    """The `clk=dev` spans the Host log carries, placed by containment.

    These cover the head of the run that the capture does not record at all —
    preamble, SO load and graph build produce no swimlane record — so they are
    drawn beside the capture rather than instead of it. One lane per phase
    name: the phases are reduced across AICPU threads, so two of them can
    overlap without either containing the other, which one track cannot show.
    """
    pid, events = _host_block_lane(rank, 1, "Device phases (placed)")
    slack_ns = int(round(placement.slack_ns))
    lane_of = {}
    for span in sorted(spans, key=lambda item: (item.ts, -item.dur)):
        if span.name not in lane_of:
            lane_of[span.name] = len(lane_of)
            events.append(
                {
                    "args": {"name": span.name.rpartition(".")[2]},
                    "cat": "__metadata",
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": lane_of[span.name],
                }
            )
        events.append(
            {
                "name": span.name,
                "cat": "host.device",
                "ph": "X",
                "pid": pid,
                "tid": lane_of[span.name],
                "ts": (placement.phase_ns_to_host_ns(span.ts) - global_origin_ns) / 1000.0,
                "dur": span.dur / 1000.0,
                "args": {"rank": rank, "device_ts_ns": span.ts, "slack_ns": slack_ns},
            }
        )
    return events


def _placement_bound_events(rank, placement, global_origin_ns):
    """Draw each Rank's error bound as slices, not only as metadata.

    The lane says two things a reader needs before comparing Ranks: the outer
    window the device work provably sits in, and the interval its start can
    fall in for the selected phase join. The lane shows placement slack;
    phase-join freedom is reported separately in the join metadata.
    """
    pid, events = _host_block_lane(rank, 2, "Placement Bound")
    outer_start_us = (placement.host.start_ns - global_origin_ns) / 1000.0
    args = placement.metadata()
    events += [
        {"args": {"name": "outer window"}, "cat": "__metadata", "name": "thread_name", "ph": "M", "pid": pid, "tid": 0},
        {
            "args": {"name": "placement slack"},
            "cat": "__metadata",
            "name": "thread_name",
            "ph": "M",
            "pid": pid,
            "tid": 1,
        },
        {
            "name": f"{containment.RUNNER_SPAN} (rank{rank})",
            "cat": "containment",
            "ph": "X",
            "pid": pid,
            "tid": 0,
            "ts": outer_start_us,
            "dur": placement.host.duration_ns / 1000.0,
            "args": args,
        },
        {
            "name": f"slack {placement.slack_ns / 1000.0:.1f} us",
            "cat": "containment",
            "ph": "X",
            "pid": pid,
            "tid": 1,
            "ts": outer_start_us,
            "dur": placement.slack_ns / 1000.0,
            "args": args,
        },
    ]
    return events


def _load_rank_local_artifacts(records_path):
    name_map_path = _find_sibling_name_map(records_path)
    if name_map_path is None:
        func_names, orchestrator_name = {}, None
    else:
        func_names, orchestrator_name = load_func_names_json(name_map_path)

    deps_path = records_path.parent / "deps.json"
    return {
        "dispatch_identity": _load_dispatch_identity(records_path.parent),
        "func_names": func_names,
        "orchestrator_name": orchestrator_name,
        "deps_path": deps_path,
        "deps_edges": load_deps_json(deps_path),
        "deps_kernel_map": load_deps_kernel_map(deps_path),
        "deps_block_map": load_deps_block_map(deps_path),
    }


def _namespace_rank_trace(trace, rank, slack_ns):
    # One stride above the Host block, so `rank0 / Worker View` still sorts
    # below every Rank's placement window.
    pid_base = (rank + 1) * _RANK_PID_STRIDE
    for event in trace.get("traceEvents", []):
        if "pid" in event:
            # Every single-Rank view pid must fit inside one stride, or two Ranks
            # land on the same namespaced pid and their lanes silently merge.
            base_pid = int(event["pid"])
            if not 0 <= base_pid < _RANK_PID_STRIDE:
                raise ValueError(f"single-Rank view pid {base_pid} does not fit the per-Rank stride {_RANK_PID_STRIDE}")
            event["pid"] = pid_base + base_pid
        if event.get("ph") == "M" and event.get("name") == "process_name":
            name = event.get("args", {}).get("name")
            if name:
                event["args"]["name"] = f"rank{rank} / {name}"
        elif event.get("ph") == "M" and event.get("name") == "process_sort_index":
            # Mirrors the pid, which is what actually orders the groups.
            sort_index = int(event.get("args", {}).get("sort_index", 0))
            event["args"]["sort_index"] = pid_base + sort_index
        for id_field in ("id", "bind_id"):
            if id_field in event:
                event[id_field] = f"r{rank}:{event[id_field]}"
        # Perfetto treats every counter arg as a separate numeric series. Rank
        # identity is already encoded in the PID, so adding it to ``ph: C``
        # would create a bogus constant counter alongside the real values.
        if event.get("ph") not in ("M", "C"):
            args = event.setdefault("args", {})
            args["rank"] = rank
            # The bound belongs on the slice a reader clicks, not only in the
            # document's metadata: it is what separates a real displacement
            # between two Ranks from the width of their placement. The full
            # record stays in `metadata.ranks[]`, which this keys into by rank.
            args["slack_ns"] = slack_ns
    return trace


def _generate_l3_trace(args, root):  # noqa: PLR0912
    if args.func_names or args.kernel_config or args.deps_json:
        raise ValueError("directory input auto-loads per-Rank name/dependency files; global overrides are not allowed")

    rank_inputs, pairing_metadata = _discover_l3_rank_inputs(root, args.dispatch, args.dispatch_id)
    raw_inputs = {}
    rank_identities = {}
    rank_host_pids = {}
    clock_domains = set()
    for rank, records_path in rank_inputs:
        with records_path.open() as f:
            raw_inputs[rank] = json.load(f)
        # The sidecar names both halves of the pairing: `host_pid` is the
        # ChipWorker child that served this capture, and the dispatch key is
        # which of that child's invocations it was. The Host log's root
        # `chip.run` span names the second, and its filename the first, so the
        # two artifacts pair exactly instead of by how well their device
        # windows happen to fit.
        sidecar = _load_dispatch_identity(records_path.parent)
        identity = containment.capture_identity(sidecar)
        if identity is not None:
            rank_identities[rank] = identity
        host_pid = (sidecar or {}).get("host_pid")
        if host_pid is not None:
            rank_host_pids[rank] = int(host_pid)
        clock_domain = _rank_clock_domain(rank, records_path, raw_inputs[rank])
        if clock_domain is not None:
            clock_domains.add(clock_domain)
    if len(clock_domains) > 1:
        raise ValueError(
            f"Rank inputs come from different Host clocks ({sorted(clock_domains)}), so their windows are not "
            "comparable. Placing them on one axis needs the window of the level that dispatched to both — the "
            "cross-host splice is not implemented."
        )

    host_logs = _discover_host_logs(root, args.host_log)
    placements, host_pairing, host_spans = _place_rank_captures(
        host_logs, raw_inputs, rank_identities, rank_host_pids, _parse_rank_pid_pins(args.rank_pid)
    )
    # The Host log holds several processes; each Rank draws only the spans of
    # the invocation its capture was paired with.
    spans_by_invocation = defaultdict(list)
    for span in host_spans:
        spans_by_invocation[(span.pid, span.inv)].append(span)

    # What the Ranks occupy on the Host axis. It bounds the origin below and
    # selects which of the dispatching processes' spans belong to this merge.
    rank_spans = [
        span
        for placement in placements.values()
        for span in spans_by_invocation[(placement.host.pid, placement.host.inv)]
        if not span.is_device
    ]
    # The axis starts at the earliest thing drawn on it. That is not always a
    # placement: a Rank's `chip.run` opens before the `runner_run` window
    # inside it, and the scheduler's `node.dispatch` opens before that again,
    # so an origin taken from the windows alone would put both at a negative
    # timestamp.
    window_lo = min([int(placement.place_lo_ns) for placement in placements.values()] + [s.ts for s in rank_spans])
    window_hi = max(
        [int(placement.host.end_ns) for placement in placements.values()] + [s.ts + s.dur for s in rank_spans]
    )
    chip_pids = {placement.host.pid for placement in placements.values()}
    dispatcher_spans = _dispatcher_spans(host_spans, chip_pids, (window_lo, window_hi))
    global_origin_ns = min([window_lo] + [span.ts for span in dispatcher_spans])
    all_events = _dispatcher_block_events(dispatcher_spans, global_origin_ns)
    rank_metadata = []
    # Every Rank of one L3 run is the same runtime, so the merged trace names it for
    # downstream tools. A set rather than a scalar so a mixed input is refused below
    # instead of silently taking whichever Rank came last.
    rank_runtimes = set()
    for rank, records_path in rank_inputs:
        placement = placements[rank]
        raw = raw_inputs[rank]
        if _is_hbg_host_capture(raw):
            record = _clock_alignment_record(placement)
            try:
                _write_clock_alignment_record(records_path, raw, record)
            except (OSError, ValueError) as error:
                print(f"Warning: could not save clock alignment: {error}", file=sys.stderr)
        data = _decode_perf_data(raw, timeline_origin_ns=global_origin_ns, placement=placement)
        artifacts = _load_rank_local_artifacts(records_path)
        dispatch_identity = artifacts["dispatch_identity"]
        trace = generate_chrome_trace_json(
            data["tasks"],
            None,
            artifacts["func_names"],
            args.verbose,
            orchestrator_name=artifacts["orchestrator_name"],
            scheduler_phases=data.get("aicpu_scheduler_phases"),
            scheduler_streams=data.get("scheduler_streams"),
            orchestrator_phases=data.get("aicpu_orchestrator_phases"),
            orchestrator_source=data.get("orchestrator_source"),
            runtime_name=data.get("runtime"),
            timeline_metadata=data.get("timeline_metadata"),
            core_to_thread=data.get("core_to_thread"),
            host_device_uploads=data.get("host_device_uploads"),
            aicpu_lifecycle_records=data.get("aicpu_lifecycle_records"),
            deps_edges=artifacts["deps_edges"],
            deps_kernel_map=artifacts["deps_kernel_map"],
            deps_block_map=artifacts["deps_block_map"],
            emit_overhead=args.overhead,
        )
        _namespace_rank_trace(trace, rank, int(round(placement.slack_ns)))
        all_events.extend(trace["traceEvents"])
        invocation = spans_by_invocation[(placement.host.pid, placement.host.inv)]
        all_events.extend(
            _host_call_tree_events(
                rank, placement, [span for span in invocation if not span.is_device], global_origin_ns
            )
        )
        all_events.extend(
            _host_device_phase_events(
                rank, placement, [span for span in invocation if span.is_device], global_origin_ns
            )
        )
        all_events.extend(_placement_bound_events(rank, placement, global_origin_ns))

        timeline = data["timeline_metadata"]
        rank_runtimes.add(data.get("runtime"))
        rank_metadata.append(
            {
                "rank": rank,
                "input": str(records_path),
                "trace_status": timeline["trace_status"],
                "source_timeline_origin_ns": timeline["source_timeline_origin_ns"],
                "placement": placement.metadata(),
                "host_pairing": host_pairing[rank],
                "host_capture": timeline.get("host_capture"),
                "dispatch_identity": dispatch_identity,
            }
        )

    # One merged trace carries one TaskId layout, so Ranks naming different runtimes
    # have no single answer. Refused rather than left unnamed: an unnamed trace would
    # push the same guess onto every downstream label instead of stopping here.
    if len(rank_runtimes) > 1:
        raise ValueError(
            "Ranks of this run name different runtimes "
            f"({', '.join(sorted(str(name) for name in rank_runtimes))}); one merged trace "
            "carries one TaskId layout, so these captures cannot be spliced together."
        )

    metadata = {
        "layout": "containment_spliced_multi_rank",
        "runtime": resolve_runtime(next(iter(rank_runtimes)) if rank_runtimes else None),
        "dispatch": args.dispatch,
        "dispatch_id": args.dispatch_id,
        "host_clock_domain_id": next(iter(clock_domains)) if clock_domains else None,
        "host_logs": [str(path) for path in host_logs],
        "global_origin_ns": global_origin_ns,
        # The processes that dispatched to these Ranks, drawn on the Host clock
        # directly — no containment, so no slack applies to their lanes.
        "dispatcher_pids": sorted({span.pid for span in dispatcher_spans}),
        "rank_count": len(rank_metadata),
        **pairing_metadata,
        "trace_status": "partial" if any(rank["trace_status"] != "complete" for rank in rank_metadata) else "complete",
        "ranks": rank_metadata,
        # Worst case for an interval read between two Ranks: each end carries
        # its own Rank's slack, so the two largest bound any pair. null means
        # fewer than two Ranks were placed, never that the comparison is exact.
        "cross_rank_uncertainty_ns": containment.cross_uncertainty_ns(placements.values()),
    }
    output_path = _resolve_output_path(args, Path(root))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"traceEvents": all_events, "metadata": metadata}, f, indent=2)
    return output_path, rank_metadata


def main():
    args = _build_parser().parse_args()

    input_path = _resolve_input_path(args)
    if input_path is None:
        return 1

    try:
        if input_path.is_dir():
            output_path, rank_metadata = _generate_l3_trace(args, input_path)
            print("\n✓ Multi-Rank conversion complete")
            print(f"  Input:  {input_path}")
            print(f"  Ranks:  {', '.join('rank' + str(item['rank']) for item in rank_metadata)}")
            # The bound belongs next to the output, not only inside it: a
            # reader who compares two Ranks needs to know it before they look.
            slacks = [item["placement"]["slack_ns"] for item in rank_metadata]
            width = (
                f"{min(slacks) / 1000.0:.1f}"
                if min(slacks) == max(slacks)
                else (f"{min(slacks) / 1000.0:.1f}–{max(slacks) / 1000.0:.1f}")
            )
            bound = f"  Bound:  each Rank placed within {width} us of its {containment.RUNNER_SPAN} window"
            # Same rule the document publishes as `cross_rank_uncertainty_ns`,
            # so a one-Rank merge cannot print a threshold the document calls
            # unknown — one capture's own slack bounds nothing across Ranks.
            cross_ns = containment.sum_two_widest_slacks(slacks)
            if cross_ns is not None:
                bound += f"; a cross-Rank gap under {cross_ns / 1000.0:.1f} us is undecided"
            print(bound)
            print(f"  Output: {output_path}")
            print(f"\nTo visualize: Open https://ui.perfetto.dev/ and drag in {output_path}")
            return 0
        if args.dispatch or args.dispatch_id:
            raise ValueError("--dispatch and --dispatch-id are only valid when input is a dfx_outputs directory")
        if args.verbose:
            print(f"Reading performance data from: {input_path}")
        raw, placement = _prepare_capture_clock_alignment(input_path, args.host_log)
        data = _decode_perf_data(raw, placement=placement)
        _print_verbose_data_info(data, args.verbose)

        func_names, orchestrator_name = _load_func_names(args, input_path)

        output_path = _resolve_output_path(args, input_path)

        deps_path = Path(args.deps_json) if args.deps_json else Path(input_path).parent / "deps.json"
        deps_edges = load_deps_json(deps_path)
        # Load the per-task kernel_ids map separately so the trace generator
        # can resolve func_id=-1 records (TASK_TIMING / level=1) back to
        # the real kernel name. Optional — pre-schema deps.json without
        # kernel_ids and SCHEDULE_TIMING+ runs both leave this at None.
        deps_kernel_map = load_deps_kernel_map(deps_path)
        deps_block_map = load_deps_block_map(deps_path)
        if deps_edges is not None:
            if args.verbose:
                print(f"  Using deps.json edges ({sum(len(v) for v in deps_edges.values())} total) from {deps_path}")
                if deps_kernel_map is not None:
                    print(f"  Using deps.json kernel_ids for {len(deps_kernel_map)} tasks (level=1 name recovery)")
        else:
            print(
                f"Warning: no usable deps.json at {deps_path}; Perfetto trace will have no dependency arrows. "
                f"Run a dep_gen capture (--enable-dep-gen) and pass --deps-json <path> to add them.",
                file=sys.stderr,
            )

        generate_chrome_trace_json(
            data["tasks"],
            str(output_path),
            func_names,
            args.verbose,
            orchestrator_name=orchestrator_name,
            scheduler_phases=data.get("aicpu_scheduler_phases"),
            scheduler_streams=data.get("scheduler_streams"),
            orchestrator_phases=data.get("aicpu_orchestrator_phases"),
            orchestrator_source=data.get("orchestrator_source"),
            runtime_name=data.get("runtime"),
            timeline_metadata=data.get("timeline_metadata"),
            core_to_thread=data.get("core_to_thread"),
            host_device_uploads=data.get("host_device_uploads"),
            aicpu_lifecycle_records=data.get("aicpu_lifecycle_records"),
            deps_edges=deps_edges,
            deps_kernel_map=deps_kernel_map,
            deps_block_map=deps_block_map,
            emit_overhead=args.overhead,
        )
        if args.overhead and deps_edges is None:
            print(
                "Warning: --overhead needs deps.json for task readiness; no deps found, "
                "Overhead Analysis track skipped.",
                file=sys.stderr,
            )

        print("\n✓ Conversion complete")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"\nTo visualize: Open https://ui.perfetto.dev/ and drag in {output_path}")

        print_task_statistics(data["tasks"], func_names, chip_swimlane_level=data["chip_swimlane_level"])

        # Scheduler-overhead deep-dive is a SEPARATE manual tool now: it needs
        # the task DAG (deps.json) captured in its own --enable-dep-gen run
        # (co-running dep_gen with swimlane perturbs the timing), so it can't be
        # produced accurately inline here. Run it explicitly:
        #   python -m simpler_setup.tools.sched_overhead_analysis \
        #       --chip-swimlane-records-json <this> --deps-json <deps from dep_gen run>
        print(
            "\nScheduler-overhead deep-dive: run sched_overhead_analysis manually with a "
            "separately-captured deps.json (--enable-dep-gen)."
        )

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
