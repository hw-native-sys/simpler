#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Onboard validation for early enqueue on a5 ``tensormap_and_ringbuffer``, over ``Worker.submit``.

This runtime orchestrates on the device, so one AICPU op resets the shared arena, dispatches
until every task completes, retires the cores and destroys the runtime context across its own
span. Its three pooled regions are one instance each, which is why the runtime answered "no" to
joined launch until now: the single instance is only safe if a successor's op cannot begin before
its predecessor's has returned. a5's merged stream ordering is exactly that guarantee, so what
this suite establishes is that the guarantee holds on the public route and that nothing the two
runs share is read or written out of turn.

What is checked, for exactly the runs each case submitted:

  admitted   two mailbox frames hold two *different* runs, both TASK_LAUNCHED with both sticky
             acceptance words set. At depth one the endpoint negotiates a single task frame, so
             that state is unreachable by construction rather than merely unobserved.
  ordered    for a named (predecessor, successor) pair, the successor's native submission
             *completed* while the identified predecessor's whole-operator boundary was still
             unfired.
  serial     the device's own passive timestamps show that pair ran one whole operator at a time,
             in that order — the successor's `aicore_start` against the predecessor's
             `whole_operator_end`. This is the property the shared arena depends on, so it is
             measured rather than inferred from host overlap.
  results    every run's own output from its own inputs, including the first, with no warm-up, no
             dropped run and no widened tolerance.
  refill     sixteen consecutive submissions keep both resource sets turning over.
  endpoints  a Worker with two a5 devices reaches that state on *each* endpoint independently,
             which is the actual scope of the capability: it is answered per device context, and
             no gate on the path checks how many endpoints a Worker has.
  attributed a run that fails beside a live predecessor reports its own status, and the
             predecessor still produces its own correct result — with the join for *that exact
             pair* asserted, so the case cannot pass under serial submission.
  refused    a worker holding a live comm resource establishes no joined launch at all.
  default    with ``launch_depth`` unset, none of it happens.

**No device stream waits on the markers.** Recording an event is not a wait, so they add no
ordering to the streams they sit in and cannot make a missing production wait look satisfied.

Every counted edge belongs to the case that counted it: records are read only from this worker's
own chip children's log files through a per-file byte cursor opened at each file's current size,
and both ends of a pair must be runs this case submitted, which :class:`_CaseRuns` decides from a
dispatch-id floor taken before the case submits anything.

The evidence reader below is a third copy of the one the a2a3 and a5 ``host_build_graph`` classes
carry, for the reason their own note gives: importing a module that defines ``SceneTestCase``
classes would collect those classes here as well, and the three trees have no shared importable
package. Consolidating them onto one arch-neutral helper needs package files in both runtime trees
and is deliberately left to its own change rather than bundled here.

Its evidence wait is for the readings the ordering assertion consumes, not for the joined-launch
record alone: a child writes that record when a successor's native submission completes and a run's
boundary markers when that run retires, so the two arrive separately and the drain loop is the only
thing that advances the cursor. Every pair is also scoped to the chip child that wrote it, because a
run key is ``(dispatch id, pipeline slot)`` and each child allocates both for itself — two endpoints
of one Worker name the same ``(1, 0)``. A reading that never arrives is reported as missing, never
passed.
"""

import contextlib
import ctypes
import tempfile
import threading
import time
from pathlib import Path

import pytest
import torch
from simpler.task_interface import (
    MAILBOX_RUN_ENTRY_STRIDE,
    MAILBOX_RUN_ENTRY_TABLE_OFFSET,
    DataType,
    TaskArgs,
    TensorArgType,
    mailbox_run_entry_count,
)
from simpler.task_interface import ArgDirection as D
from simpler.worker import (
    _OFF_ENTRY_ACCEPTED,
    _OFF_ENTRY_GENERATION,
    _OFF_ENTRY_STATE,
    _TASK_ACCEPTED,
    _TASK_LAUNCHED,
    _mailbox_load_i32,
    _mailbox_load_u64,
    _read_run_entry_identity,
)

from simpler_setup import SceneTestCase, scene_test
from simpler_setup.tools.strace_timing import parse_spans

#: The point-in-time host span one joined launch publishes, carrying both run identities and
#: carrying `observed` separately from `unfired` so a failed query reads as a failed query.
_JOIN_SPAN = "chip.run.joined_launch"
#: The run's own host span, whose attributes name the run its invocation belongs to.
_RUN_SPAN = "chip.run"
#: That run's two passive device-boundary marker times. `aic_start` is the instant its AICore
#: stream was released to begin its kernel; `wo_end` is the instant its own AICore kernel had
#: returned. Both are `aclrtEventGetTimestamp` readings on one chip-wide counter.
_BOUNDARY_SPAN = "chip.run.runner_run.device_boundaries"

_KERNELS = "kernels"

#: Tasks the chain orchestration submits. Every task names the same three caller tensors and
#: declares the output INOUT, so the tensor map orders the chain through that one tensor and no
#: intermediate is created — nothing here occupies the ring heap. The length only has to make the
#: graph non-trivial; the run's duration comes from the first task's spin.
_CHAIN_LENGTH = 64
# Long enough that the host can observe the overlap window, bounded so a regression fails on an
# assertion rather than on the op-execute timeout.
_DEVICE_SPIN_ITERS = 200_000_000
# Shorter per run: the refill case runs sixteen back to back and each only needs to outlast one
# sampling pass rather than a whole host observation window.
_REFILL_SPIN_ITERS = 40_000_000
_SIZE = 128 * 128
#: The task frames a5 TMR's endpoint negotiates with the opt-in set: one per run-resource set its
#: published depth of two grants. Each case asserts the count it actually got rather than assuming
#: this one, because the count is what decides whether a successor has anywhere to go.
_FRAME_COUNT = 2
#: Consecutive submissions the refill case makes.
_REFILL_RUNS = 16
#: How long to keep draining the children's log files after their runs have finished. The writers
#: are asynchronous, so this covers the lag between a record being accepted and written, not the
#: run itself. Exhausting it is an absence, which the caller's assertion reports as one.
_EVIDENCE_BUDGET_S = 10.0
#: Lines of each child's own log a failing numeric check carries, from the end of the sequence.
_DIAGNOSTIC_LOG_LINES = 900

_DIR_TAGS = {
    D.IN: TensorArgType.INPUT,
    D.OUT: TensorArgType.OUTPUT_EXISTING,
    D.INOUT: TensorArgType.INOUT,
}

_CALLABLES = {
    "callables": [
        {
            "name": "vector",
            "orchestration": {
                "source": f"{_KERNELS}/orchestration/pipelined_vector_orch.cpp",
                "function_name": "aicpu_orchestration_entry",
                "signature": [D.IN, D.IN, D.OUT],
            },
            "incores": [
                {
                    "func_id": 0,
                    "source": f"{_KERNELS}/aiv/spin_add.cpp",
                    "core_type": "aiv",
                    "signature": [D.IN, D.IN, D.OUT],
                },
            ],
        },
        {
            "name": "reported_fatal",
            "orchestration": {
                "source": f"{_KERNELS}/orchestration/reported_fatal_orch.cpp",
                "function_name": "aicpu_orchestration_entry",
                "signature": [D.IN, D.IN, D.OUT],
            },
            "incores": [
                {
                    "func_id": 0,
                    "source": f"{_KERNELS}/aiv/spin_add.cpp",
                    "core_type": "aiv",
                    "signature": [D.IN, D.IN, D.OUT],
                },
            ],
        },
    ],
}


def _chip_args(handles, orch_signature, *scalars):
    """A ``TaskArgs`` naming each ``Buffer`` as a whole-buffer float32 view, tagged by signature."""
    args = TaskArgs()
    for handle, direction in zip(handles, orch_signature):
        args.add_tensor(handle.tensor((_SIZE,), DataType.FLOAT32), _DIR_TAGS[direction])
    for value in scalars:
        args.add_scalar(value)
    return args


def _negotiated_frame_counts(worker):
    """The task frames this Worker registered each of its chip endpoints with.

    The child decides the count after init — only an initialized ``ChipWorker`` can answer whether
    its runtime joins native launches — and publishes it; the parent registers each endpoint with
    exactly that number. Reading the parent's record is therefore reading the route a case actually
    runs on, not the frames the mailbox happens to be laid out for.

    Returned per endpoint rather than as one number, because the capability is answered per device
    context: a Worker with two devices negotiates two counts, and a case that drives both has to
    hold each of them.
    """
    return [int(count) for count in worker._chip_run_resource_set_counts]  # noqa: SLF001 -- white-box negotiation


@contextlib.contextmanager
def _frame_views(worker, chip_index=0):
    """One endpoint's negotiated task frames as ``(address, buffer)``, all released before exit.

    A view sliced out of the chip mailbox's shared memory is an *export* of it, and an export is
    exactly what makes ``SharedMemory.close()`` raise ``BufferError``. A failing assertion keeps
    the frame locals of everything on the stack alive into the fixture's ``Worker.close()``, so a
    view this module does not release itself becomes a teardown error that replaces the case's own
    finding. Releasing them here is what keeps a failure readable.

    ``_CaseRuns`` and ``_RunTrace`` deliberately keep none of these: the first stores an integer
    floor and identity tuples, the second file offsets, so a view's lifetime is this scope's alone.
    """
    shm_buf = worker._chip_shms[chip_index].buf  # noqa: SLF001 -- white-box mailbox observation
    assert shm_buf is not None
    # Left unnamed on purpose: a ctypes object built from a buffer holds an export for as long as
    # it lives, so binding it would reintroduce exactly the leak this function exists to close.
    mailbox_addr = ctypes.addressof(ctypes.c_char.from_buffer(shm_buf))
    # One view per run-state entry, which is where a run's whole life is
    # published: the 64 KiB request frames carry only the payload and go back to
    # the parent as soon as the child has copied it out.
    entry_count = mailbox_run_entry_count(_negotiated_frame_counts(worker)[chip_index])
    views = [
        shm_buf[
            MAILBOX_RUN_ENTRY_TABLE_OFFSET + index * MAILBOX_RUN_ENTRY_STRIDE : MAILBOX_RUN_ENTRY_TABLE_OFFSET
            + (index + 1) * MAILBOX_RUN_ENTRY_STRIDE
        ]
        for index in range(entry_count)
    ]
    try:
        yield [
            (mailbox_addr + MAILBOX_RUN_ENTRY_TABLE_OFFSET + index * MAILBOX_RUN_ENTRY_STRIDE, view)
            for index, view in enumerate(views)
        ]
    finally:
        for view in views:
            view.release()


def _coherent_snapshot(entries):
    """Every entry's ``(identity, state, accepted)`` at one instant, or None.

    Two passes over the **states**, not only over the identities. A single pass is not an
    instant: it can read entry A launched, and then — after A has retired and its successor E
    has launched — read E launched too, and report both as launched when neither overlapped the
    other. Re-reading every state and requiring each to be unchanged is what makes the returned
    set a claim about one point in time rather than about a window.

    The identities are re-read for the same reason they always were: an entry whose identity
    moved between the passes was reused mid-read, so nothing sampled from it names a run.
    """
    before_identity = [_read_run_entry_identity(buf) for _, buf in entries]

    def _states():
        return [
            (_mailbox_load_i32(addr + _OFF_ENTRY_STATE), _mailbox_load_i32(addr + _OFF_ENTRY_ACCEPTED))
            for addr, _ in entries
        ]

    before_state = _states()
    after_state = _states()
    after_identity = [_read_run_entry_identity(buf) for _, buf in entries]
    if before_identity != after_identity or before_state != after_state:
        return None
    return [(identity, state, accepted) for identity, (state, accepted) in zip(after_identity, after_state)]


def _run_key(identity):
    """The ``(dispatch id, pipeline slot)`` pair a chip child gives its native run.

    Both halves reach the native descriptor unchanged from this frame, so a record's identities are
    comparable with what the parent can see. A frame that has never carried a run reads as dispatch
    zero and is not a key.
    """
    _protocol, _run_id, slot_id, _generation, dispatch_id, *_rest = identity
    return None if dispatch_id == 0 else (dispatch_id, slot_id)


def _launched_run_keys(snapshot):
    """The keys of every run a snapshot shows launched and accepted, or None when it is unusable."""
    if snapshot is None:
        return None
    keys = set()
    for identity, state, accepted in snapshot:
        if state != _TASK_LAUNCHED or accepted != _TASK_ACCEPTED:
            continue
        key = _run_key(identity)
        if key is None:
            return None
        keys.add(key)
    return keys


def _two_distinct_runs_launched(snapshot):
    """Whether the snapshot holds two different runs, both launched and both accepted.

    Counted over the entries that carry a launched run, never over the whole table. The table is
    `M = granted + K` entries so that a hand-off can be taken before a set frees, so entries
    beyond the runs in flight are idle by construction — a predicate requiring every entry
    launched could never be satisfied at any grant.
    """
    keys = _launched_run_keys(snapshot)
    if keys is None:
        return False
    return len(keys) >= _FRAME_COUNT


def _dispatch_floor(frames):
    """The highest dispatch id this endpoint carried before the caller submitted anything.

    Read from a coherent snapshot so the frames are not mixed across an instant. A frame that has
    never carried a run reads as zero, so a fresh Worker's floor is zero.
    """
    deadline = time.monotonic() + 1.0
    while True:
        snapshot = _coherent_snapshot(frames)
        if snapshot is not None:
            return max((identity[4] for identity, _, _ in snapshot), default=0)
        if time.monotonic() >= deadline:
            raise AssertionError("the mailbox frames never read coherently, so no dispatch floor could be taken")
        time.sleep(0.001)


class _CaseRuns:
    """The runs *this* case submitted, and nothing else.

    A `WorkerThread` allocates dispatch ids monotonically and the parent writes them into the
    frames in that order, so the highest id in any frame before the case submits anything is the
    last id issued to this Worker. Every run the case goes on to submit therefore has a strictly
    higher id, and that floor is what keeps a **previous** case's record out of the evidence — a
    byte cursor alone cannot, because the earlier case's writer is asynchronous, and a frame
    snapshot alone cannot, because the frames still hold the previous case's terminal identities.

    Holds no mailbox view: the floor is an integer and the keys are tuples, so an instance
    outliving its :func:`_frame_views` scope keeps nothing exported from the shared memory.
    """

    def __init__(self, frames):
        self.floor = _dispatch_floor(frames)
        self._keys: set[tuple] = set()

    def note(self, snapshot):
        """Record every run a coherent snapshot names that this case could have submitted."""
        if snapshot is None:
            return
        for identity, _, _ in snapshot:
            key = _run_key(identity)
            if key is not None and key[0] > self.floor:
                self._keys.add(key)

    def __contains__(self, key):
        return key in self._keys

    def __len__(self):
        return len(self._keys)

    def sorted_keys(self):
        return sorted(self._keys)


def _child_log_directory():
    """The directory this worker's chip children append their host logs to."""
    from _task_interface import _host_log_directory  # noqa: PLC0415
    from simpler.task_interface import _bind_host_log_session_directory  # noqa: PLC0415

    return _host_log_directory() or _bind_host_log_session_directory()


def _attributes(span):
    """One span's ``k=v`` attribute string as a dict."""
    return dict(pair.split("=", 1) for pair in span.attrs.split() if "=" in pair)


def _join_record(span):
    """One joined-launch span as a record, or a marker for one that did not survive its write."""
    attributes = _attributes(span)
    try:
        return {
            "pid": span.pid,
            "inv": span.inv,
            "successor": (int(attributes["s_disp"]), int(attributes["s_slot"])),
            "predecessor": (int(attributes["p_disp"]), int(attributes["p_slot"])),
            "observed": attributes["observed"] == "1",
            "unfired": attributes["unfired"] == "1",
            "query_rc": int(attributes["rc"]),
        }
    except (KeyError, ValueError):
        # The logger marks a field it truncated with `~`. Keep the raw text rather than dropping
        # it: a record that cannot be read is a different failure from no record at all.
        return {"pid": span.pid, "inv": span.inv, "malformed": span.attrs}


class _RunTrace:
    """A byte cursor over each of this worker's chip children's host-log files.

    The host log is one append-only file per process (``host.<pid>.log``), so a window into it is
    a byte offset per file. A count over the concatenation of several files is not a cursor: a
    file that sorts earlier appending between two reads shifts every later record.

    Each cursor opens at its files' current size, so its window begins where it was created: an
    earlier case on the same worker, or an earlier session that reused a pid, cannot supply a
    record to a later one.

    The directory is the process tree's own log spool, not the case's ``output_prefix``: it
    outlives every capture and is removed only at owner-process exit. So a failing case can still
    read what its children wrote, which :meth:`child_log_window` is for.
    """

    def __init__(self, worker):
        directory = Path(_child_log_directory())
        self._pids = list(worker._chip_pids)  # noqa: SLF001 -- white-box child attribution
        self._paths = {pid: directory / f"host.{pid}.log" for pid in self._pids}
        self._offsets = {pid: self._size(pid) for pid in self._pids}
        #: Where each file stood when this cursor opened, so a diagnostic window can start there
        #: rather than at whatever the record cursor has since consumed.
        self._origins = dict(self._offsets)
        #: ``(pid, inv)`` -> the ``(dispatch id, pipeline slot)`` that invocation ran.
        self.run_of_invocation: dict[tuple, tuple] = {}
        #: ``(pid, inv)`` -> that run's two device-boundary marker times, or their reasons.
        self.boundary_of_invocation: dict[tuple, dict] = {}

    def _size(self, pid):
        try:
            return self._paths[pid].stat().st_size
        except OSError:
            return 0

    @property
    def pids(self):
        """The chip children whose records this cursor will accept."""
        return list(self._pids)

    def take(self):
        """Every joined-launch record appended since the previous call.

        The other two families are folded into this cursor's own state rather than returned:
        they are looked up by invocation, not iterated.
        """
        fresh = []
        for pid in self._pids:
            blob = b""
            with contextlib.suppress(OSError):
                blob = self._paths[pid].read_bytes()
            window = blob[self._offsets[pid] :]
            consumed = window.rfind(b"\n") + 1
            if consumed <= 0:
                continue
            self._offsets[pid] += consumed
            lines = window[:consumed].decode("utf-8", errors="replace").splitlines()
            for span in parse_spans(lines):
                if span.pid != pid:
                    continue
                if span.name == _JOIN_SPAN:
                    fresh.append(_join_record(span))
                elif span.name == _RUN_SPAN:
                    self._note_identity(pid, span)
                elif span.name == _BOUNDARY_SPAN:
                    self._note_boundaries(pid, span)
        return fresh

    def _note_identity(self, pid, span):
        attributes = _attributes(span)
        with contextlib.suppress(KeyError, ValueError):
            self.run_of_invocation[pid, span.inv] = (int(attributes["dispatch_id"]), int(attributes["slot_id"]))

    def _note_boundaries(self, pid, span):
        """One run's marker times. A position with a non-zero rc carries no time, by construction.

        The emitter writes 0 for an unavailable position and the rc that says why, so a reader must
        check the rc rather than the value — a zero could not otherwise be told from an absence.
        """
        attributes = _attributes(span)
        with contextlib.suppress(KeyError, ValueError):
            aicore_rc = int(attributes["aic_rc"])
            whole_operator_rc = int(attributes["wo_rc"])
            self.boundary_of_invocation[pid, span.inv] = {
                "device": int(attributes["dev_id"]),
                "aicore_start": int(attributes["aic_start"]) if aicore_rc == 0 else None,
                "aicore_rc": aicore_rc,
                "whole_operator_end": int(attributes["wo_end"]) if whole_operator_rc == 0 else None,
                "whole_operator_rc": whole_operator_rc,
                "hz": int(attributes["ts_hz"]),
            }

    def boundaries_of_run(self, pid, run_key):
        """That run's marker times on this child, or None while they have not been published."""
        for (row_pid, inv), row in self.boundary_of_invocation.items():
            if row_pid == pid and self.run_of_invocation.get((row_pid, inv)) == run_key:
                return row
        return None

    def child_log_window(self, keep_lines=_DIAGNOSTIC_LOG_LINES):
        """The child's own account of this case, for a failure message to carry.

        Every line each chip child wrote since this cursor opened, unfiltered. A child's file holds
        only that child's own records — the parent's spans go to its stderr — so the `[STRACE]`
        lines here are `chip.run.bind`, `chip.run.stage_inputs` and the rest of the per-run phases,
        which is exactly what a failure needs and what nothing else in a CI job log carries.

        Tail-limited per child because a sixteen-run case writes a great deal; the tail is the end
        of the sequence, which is where a run that produced nothing shows it.
        """
        window = {}
        for pid in self._pids:
            blob = b""
            with contextlib.suppress(OSError):
                blob = self._paths[pid].read_bytes()
            lines = blob[self._origins[pid] :].decode("utf-8", errors="replace").splitlines()
            window[pid] = lines[-keep_lines:]
        return window


def _established_pairs(records, case_runs, pid=None):
    """The pairs whose ordering the records actually establish.

    A record counts only when the query answered *and* the predecessor's whole-operator boundary
    had not fired: that pair's successor completed its native submission before its predecessor
    completed. A failed query counts for neither side. Both ends must be runs this case submitted,
    and must differ.

    ``pid`` scopes the records to one chip child, and a caller that is making a claim about one
    endpoint must pass it. A run key is ``(dispatch id, pipeline slot)`` and a child allocates both
    for itself, so two endpoints of one Worker routinely name the same ``(1, 0)`` — and the byte
    cursor cannot separate them either, because a child that writes late writes into a window
    another endpoint's check has already opened.
    """
    return {
        (record["predecessor"], record["successor"])
        for record in records
        if record.get("observed")
        and record.get("unfired")
        and (pid is None or record["pid"] == pid)
        and record["successor"] != record["predecessor"]
        and record["successor"] in case_runs
        and record["predecessor"] in case_runs
    }


def _comparable_boundaries(trace, pid, pair):
    """One pair's two boundary rows as read on one child, when they can be compared at all.

    Returns ``(predecessor row, successor row)`` or None. Comparability is a property of a single
    child: the two rows must name the same device and the same tick rate, the predecessor must
    carry its whole-operator end, and the successor its AICore start. A missing position is not a
    slower reading of the same fact — the emitter writes 0 for one it could not take — so None here
    covers both "not written yet" and "never available", and only the caller's deadline tells the
    two apart.
    """
    ahead = trace.boundaries_of_run(pid, pair[0])
    behind = trace.boundaries_of_run(pid, pair[1])
    if ahead is None or behind is None:
        return None
    if ahead["device"] != behind["device"] or ahead["hz"] != behind["hz"]:
        return None
    if ahead["whole_operator_end"] is None or behind["aicore_start"] is None:
        return None
    return ahead, behind


def _whole_operator_order(trace, records, case_runs, pid=None):
    """Split the established pairs by the *whole-operator* relation on the device's own clock.

    Returns ``(ordered, overlapping, within_tick, unmeasured)``:

    * **ordered** — the successor's AICore stream was released strictly after the predecessor's own
      AICore kernel had returned. This is the whole-operator serialization the queued wait
      constructs, and on this runtime it is also what makes the single shared arena safe, so it is
      measured rather than inferred.
    * **overlapping** — the successor was released before that instant. A real failure of the
      ordering edge.
    * **within_tick** — the two readings are the same tick. The predecessor's marker is queued
      ahead of the boundary event the successor's wait consumes, but the device clock can assign
      those distinct operations one value. This shows no measurable overlap, not strict ordering
      below one tick, so the claim is held against ``ordered | within_tick``.
    * **unmeasured** — a position was unavailable, the runs report different devices, or the
      readings are on different tick rates. Never a pass.

    ``pid`` scopes both the records and the boundary lookup to one child. Without it a pair's rows
    are hunted across every child that wrote a record, which can place a pair on the other
    endpoint's clock readings — see :func:`_established_pairs`.
    """
    pids = [pid] if pid is not None else sorted({record["pid"] for record in records if not record.get("malformed")})
    ordered, overlapping, within_tick, unmeasured = set(), set(), set(), set()
    for pair in _established_pairs(records, case_runs, pid):
        decided = False
        for candidate in pids:
            rows = _comparable_boundaries(trace, candidate, pair)
            if rows is None:
                continue
            decided = True
            end = rows[0]["whole_operator_end"]
            start = rows[1]["aicore_start"]
            if start > end:
                ordered.add(pair)
            elif start == end:
                within_tick.add(pair)
            else:
                overlapping.add(pair)
        if not decided:
            unmeasured.add(pair)
    return ordered, overlapping, within_tick, unmeasured


def _boundaries_settled(trace, records, case_runs, pid=None):
    """Whether every pair these records establish can already be placed on the device clock.

    This is the predicate an evidence wait must use, because it is the question the ordering
    assertion asks. Waiting only for a pair to be *established* returns as soon as the join span
    lands, and a run publishes its boundary markers when it retires — a later event, written by an
    asynchronous child writer. They are separate records, so the arrival of the first is no evidence
    about the second, and a reader satisfied by the join alone goes on to read a successor row that
    is simply not there yet.

    False while anything is still missing, so the caller's own deadline decides when an absence has
    been given its full window. Nothing here weakens what is asserted afterwards: the same
    classification runs again, and a pair that never becomes comparable stays ``unmeasured``.
    """
    if not _established_pairs(records, case_runs, pid):
        return False
    return not _whole_operator_order(trace, records, case_runs, pid)[3]


def _boundary_detail(trace, pairs, pid=None):
    """Per-pair marker readings, per child, for a failure message that names why rather than that.

    Keyed by child as well as by pair. A pair's two halves are per-child run keys, so one entry per
    pair would report whichever child happened to be read last and silently drop the rest — which
    is the difference between "this endpoint never published the successor's row" and "some other
    endpoint has rows for the same key".
    """
    children = [pid] if pid is not None else trace.pids
    return {
        (child, pair): {
            "predecessor": trace.boundaries_of_run(child, pair[0]),
            "successor": trace.boundaries_of_run(child, pair[1]),
        }
        for child in children
        for pair in pairs
    }


def _log_window_text(trace):
    """Every chip child's own log window, labelled by child, for a failure message to carry."""
    return "\n".join(f"--- child {pid} ---\n" + "\n".join(lines) for pid, lines in trace.child_log_window().items())


def _await_records(cursor, records, budget_s, satisfied):
    """Drain into ``records`` until ``satisfied``, or until the budget runs out.

    The child's writer is asynchronous and the parent's flush drains only the parent's own sink, so
    a read taken straight after ``wait()`` is no guarantee the child's record has been written.
    This polls instead, and returns on the budget rather than raising, so the caller's own
    assertion names which half was missing.

    The predicate has to name the evidence the caller then *consumes*, not a record that merely
    precedes it: this loop is the only thing that advances the cursor, so whatever has not been
    read by the time it returns is not in the trace at all, however long the caller waits after.
    """
    deadline = time.monotonic() + budget_s
    while True:
        records.extend(cursor.take())
        if satisfied(records):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def _wait_for_one_launched_frame(frames, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = _coherent_snapshot(frames)
        if snapshot is not None and any(state == _TASK_LAUNCHED for _, state, _ in snapshot):
            return
        time.sleep(0.001)
    raise AssertionError("no run reached its device launch fence")


def _drain_handles(handles, what):
    """Wait for every submitted run before the caller's resources leave scope.

    No failure is raised from here, so a run that failed cannot replace the caller's own finding
    while it is unwinding. The failures are returned instead: a caller that reaches its normal end
    with a non-empty list has no primary error to preserve and reports these itself.
    """
    failures = []
    for handle in handles:
        try:
            handle.wait()
        except Exception as error:  # noqa: BLE001 -- must not replace the caller's failure
            failures.append(str(error))
            print(f"[{what} cleanup] a run did not finish: {error}")
    return failures


def _assert_results(expectations, trace, case_runs, what):
    """Every run's own output, exactly — and with the device's account when one does not match.

    The tolerance is unchanged: each output must still be close to the value computed from that
    run's own inputs. What this adds is attribution. An output that never changed and an output
    computed from the wrong bytes are the same assertion failure, and `assert_close` alone reports
    only the number; the run identities this case saw, their device-boundary markers and the
    children's own log window are what tell the two apart.
    """
    mismatches = []
    for index, (out, expected) in enumerate(expectations):
        try:
            torch.testing.assert_close(out, expected)
        except AssertionError as error:
            mismatches.append((index, str(error), float(out.reshape(-1)[0]), float(expected.reshape(-1)[0])))
    if not mismatches:
        return
    # Only now, and only once: the children's writers are asynchronous, so the rows that describe
    # these runs may not have been written when the outputs were compared. A passing case pays
    # none of this.
    _await_records(trace, [], _EVIDENCE_BUDGET_S, lambda seen: bool(trace.boundary_of_invocation))
    runs = case_runs.sorted_keys()
    boundaries = {pid: {run: trace.boundaries_of_run(pid, run) for run in runs} for pid in trace.pids}
    summary = "; ".join(
        f"run #{index}: first element {actual} against {wanted}" for index, _, actual, wanted in mismatches
    )
    log_window = _log_window_text(trace)
    raise AssertionError(
        f"{what}: {len(mismatches)} of {len(expectations)} runs produced the wrong output. {summary}.\n"
        f"This case's runs (dispatch id, slot) above floor {case_runs.floor}: {runs}\n"
        f"Device boundary markers per child: {boundaries}\n"
        f"First failure in full:\n{mismatches[0][1]}\n"
        f"Children's own log window:\n{log_window}"
    )


class _A5TmrEarlyEnqueueBase(SceneTestCase):
    """The submission sequence and its expectations, shared by every class below."""

    CALLABLE = _CALLABLES

    def test_run(self):
        """Not the standard single-case golden path.

        These classes have no single orchestration callback and no generated argument set: each
        drives its own multi-run sequence and checks the results itself. Requesting no fixtures
        keeps the skip from allocating a device for nothing.
        """
        pytest.skip("early enqueue drives its own multi-run sequences; see the cases below")

    @staticmethod
    def _endpoint_pid(worker, chip_index=0):
        """The chip child whose records are this endpoint's.

        ``_chip_pids`` and ``_chip_shms`` are appended together in device order, so one chip index
        names the same child in the mailbox views, in the negotiated counts and in the log cursor.
        """
        return worker._chip_pids[chip_index]  # noqa: SLF001 -- white-box child attribution

    @staticmethod
    def _tensor_from_host_buffer(worker, value):
        buffer = worker.create_buffer(_SIZE * torch.float32.itemsize)
        tensor = torch.frombuffer(buffer.shm.buf, dtype=torch.float32, count=_SIZE)
        tensor.fill_(value)
        return buffer, tensor

    def _submit_vector(self, worker, arg_buffers, output_prefix, *, spin_iters=0, chip_index=0, callable_name="vector"):
        """One ``Worker.submit`` of the chain onto one endpoint.

        A non-empty output prefix is what makes the chip child bind its host log to a file instead
        of leaving it on the inherited stderr, which is what gives the joined-launch spans a
        destination this process can read by child pid. No diagnostic flag is set, so nothing else
        is written there.
        """
        handle = type(self)._st_chip_handles[callable_name]
        signature = type(self)._st_chip_handles[f"{callable_name}_sig"]
        config = self._build_config(self.CASES[0]["config"], output_prefix=output_prefix)

        def graph(orch, _args, _cfg):
            orch.submit_next_level(handle, _chip_args(arg_buffers, signature, spin_iters), config, worker=chip_index)

        return worker.submit(graph)

    @staticmethod
    def _expected(a, b):
        """The chain's result for one run's inputs, read before the run overwrites its output."""
        return a + _CHAIN_LENGTH * b

    def _three_buffers(self, worker, a_value, b_value):
        """One run's ``(buffers, output tensor, expected tensor)``."""
        buffers = []
        tensors = []
        for value in (a_value, b_value, 0.0):
            buffer, tensor = self._tensor_from_host_buffer(worker, value)
            buffers.append(buffer)
            tensors.append(tensor)
        a, b, out = tensors
        return buffers, out, self._expected(a, b)

    def _observe_two_launched_runs(self, frames, timeout, case_runs):
        """Watch for two different runs launched and accepted at once.

        Returns whether that state was ever observed, and records every run a sampled frame names
        in ``case_runs`` so the ordering records can be held to this case's own runs. The watch
        ends when no frame is launched any more, which is as long as there is anything left to
        observe.
        """
        deadline = time.monotonic() + timeout
        saw_any_launched = False
        while time.monotonic() < deadline:
            snapshot = _coherent_snapshot(frames)
            case_runs.note(snapshot)
            if _two_distinct_runs_launched(snapshot):
                return True
            if snapshot is not None:
                launched = any(state == _TASK_LAUNCHED for _, state, _ in snapshot)
                if launched:
                    saw_any_launched = True
                elif saw_any_launched:
                    return False
            time.sleep(0.001)
        raise AssertionError(f"no run stayed launched long enough to sample within {timeout}s")

    def _assert_two_run_window(self, worker, what, *, chip_index=0):
        """Admission, the ordering it exists for, what the device then did, and both results.

        Shared by the single-endpoint case and each endpoint of the two-device case, because the
        claim is the same one per device context.
        """
        negotiated = _negotiated_frame_counts(worker)[chip_index]
        assert negotiated == _FRAME_COUNT, (
            f"{what}: endpoint {chip_index} negotiated {negotiated} task frame(s) at launch_depth=2, so no "
            f"successor can occupy one while its predecessor runs and the overlap below is unreachable"
        )
        pid = self._endpoint_pid(worker, chip_index)
        trace = _RunTrace(worker)
        with _frame_views(worker, chip_index) as frames:
            case_runs = _CaseRuns(frames)
            with tempfile.TemporaryDirectory(prefix="simpler-a5-tmr-early-enqueue-") as output_prefix:
                handles: list = []
                buffers: list = []
                expectations: list = []
                try:
                    for a_value, b_value in ((2.0, 3.0), (5.0, 7.0)):
                        group, out, expected = self._three_buffers(worker, a_value, b_value)
                        buffers.extend(group)
                        expectations.append((out, expected))
                        handles.append(
                            self._submit_vector(
                                worker,
                                group,
                                output_prefix,
                                spin_iters=_DEVICE_SPIN_ITERS,
                                chip_index=chip_index,
                            )
                        )
                        if len(handles) == 1:
                            _wait_for_one_launched_frame(frames, 20.0)
                    together = self._observe_two_launched_runs(frames, 60.0, case_runs)
                finally:
                    # Before anything of this case's leaves scope, whatever the submission or the
                    # observation did: a run still in flight owns its arguments and its output
                    # buffer, and the temporary output directory is this case's too.
                    drain_failures = _drain_handles(handles, what)
                # Reached only with no primary error, so these have nothing to hide behind.
                assert not drain_failures, f"{what}: a run of this case did not finish: {drain_failures}"
                _assert_results(expectations, trace, case_runs, what)

                assert together, (
                    f"{what}: two different runs were never observed launched and accepted at the same time, so "
                    f"a successor's submission never reached the device while its predecessor was executing"
                )

                records: list[dict] = []
                # The wait is for the readings the ordering assertion below consumes, not for the
                # join span alone: that span is written at submission and a run's boundary markers
                # when it retires, so a join-only predicate returns before the successor's row
                # exists and no later read can recover it.
                settled = _await_records(
                    trace,
                    records,
                    _EVIDENCE_BUDGET_S,
                    lambda seen: _boundaries_settled(trace, seen, case_runs, pid),
                )
                assert records, (
                    f"{what}: no joined-launch record reached this process from child {pid} (children "
                    f"{trace.pids}), so nothing carried the ordering of the two accepted runs"
                )
                pairs = _established_pairs(records, case_runs, pid)
                assert pairs, (
                    f"{what}: no record from child {pid} established a pair of this case's runs: "
                    f"records={records}, case runs={case_runs.sorted_keys()} above floor {case_runs.floor}"
                )

                ordered, overlapping, within_tick, unmeasured = _whole_operator_order(trace, records, case_runs, pid)
                assert not overlapping, (
                    f"{what}: a joined successor's AICore stream was released before its predecessor's whole "
                    f"operator had finished, which is what this runtime's shared arena depends on: "
                    f"{sorted(overlapping)}; {_boundary_detail(trace, overlapping, pid)}"
                )
                assert ordered | within_tick, (
                    f"{what}: no established pair has comparable whole-operator boundary readings on child "
                    f"{pid} after waiting {_EVIDENCE_BUDGET_S}s for them "
                    f"({'the wait was satisfied' if settled else 'the wait ran out'}): "
                    f"unmeasured={sorted(unmeasured)}; {_boundary_detail(trace, pairs, pid)}; "
                    f"this case's runs above floor {case_runs.floor}: {case_runs.sorted_keys()}; "
                    f"records={records}\nChildren's own log window:\n{_log_window_text(trace)}"
                )


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestA5TmrEarlyEnqueueDepthTwo(_A5TmrEarlyEnqueueBase):
    """At ``launch_depth=2`` a successor's work reaches a5 TMR's device while its predecessor runs."""

    CASES = [
        {
            "name": "a5_tmr_early_enqueue",
            "platforms": ["a5"],
            "config": {"device_count": 1, "num_sub_workers": 0, "launch_depth": 2},
            "params": {},
        },
    ]

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        del kwargs
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        assert str(worker._config["platform"]) in case["platforms"]  # noqa: SLF001 -- scene-test validation
        assert worker._launch_depth == 2, (  # noqa: SLF001 -- scene-test validation
            f"this class needs a Worker at launch_depth=2, got {worker._launch_depth}"  # noqa: SLF001
        )
        self.test_two_runs_are_launched_and_accepted_at_once("a5", worker)
        self.test_sixteen_runs_keep_refilling_both_sets("a5", worker)

    @pytest.mark.platforms(["a5"])
    def test_two_runs_are_launched_and_accepted_at_once(self, st_platform, st_worker):
        """Admission, the ordering the admission exists for, and what the device then did."""
        if st_platform != "a5":
            pytest.skip("a5 TMR early enqueue is gated to the a5 onboard tensormap_and_ringbuffer route")
        self._assert_two_run_window(st_worker, "a5 TMR two-run window")

    @pytest.mark.platforms(["a5"])
    def test_sixteen_runs_keep_refilling_both_sets(self, st_platform, st_worker):
        """Sixteen consecutive submissions that keep both sets occupied as they turn over.

        Correct results alone would pass under ordinary serial execution, so they are not the
        evidence. What is:

        two-deep    some instant showed two different runs launched and accepted at once.
        sustained   that happened again later in the sequence, over a strictly higher frontier of
                    dispatch ids — a run admitted after earlier ones had gone.
        ordered     no resource set was ever seen carrying a *lower* dispatch id than one it had
                    already carried, so a set passes from one run to a later-submitted one.
        distinct    every one of the sixteen runs produced its own output.

        What the reuse observation is: the mailbox showing a set under a second run's identity.
        That is a physical observation of the set being handed on, and the production path only
        hands it on after the earlier run's lifetime closed — but this case reads the handover, it
        does not witness each step of that closure.
        """
        if st_platform != "a5":
            pytest.skip("a5 TMR early enqueue is gated to the a5 onboard tensormap_and_ringbuffer route")
        negotiated = _negotiated_frame_counts(st_worker)[0]
        assert negotiated == _FRAME_COUNT, (
            f"a5 TMR negotiated {negotiated} task frame(s) at launch_depth=2, so the refill below has only one "
            f"frame to turn over and nothing it could observe two-deep"
        )
        pairs_seen: list[frozenset] = []
        dispatches_by_slot: dict[int, list[int]] = {}
        widest = 0
        stop = threading.Event()
        trace = _RunTrace(st_worker)

        with _frame_views(st_worker) as frames:
            case_runs = _CaseRuns(frames)

            def sample():
                nonlocal widest
                while not stop.is_set():
                    snapshot = _coherent_snapshot(frames)
                    case_runs.note(snapshot)
                    keys = _launched_run_keys(snapshot)
                    if keys:
                        widest = max(widest, len(keys))
                        for dispatch_id, slot_id in keys:
                            seen = dispatches_by_slot.setdefault(slot_id, [])
                            if not seen or seen[-1] != dispatch_id:
                                seen.append(dispatch_id)
                        if len(keys) == _FRAME_COUNT:
                            pair = frozenset(keys)
                            if pair not in pairs_seen:
                                pairs_seen.append(pair)
                    time.sleep(0.001)

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            handles: list = []
            buffers: list = []
            try:
                with tempfile.TemporaryDirectory(prefix="simpler-a5-tmr-early-enqueue-refill-") as output_prefix:
                    expectations = []
                    try:
                        for index in range(_REFILL_RUNS):
                            group, out, expected = self._three_buffers(st_worker, float(index + 2), float(index + 3))
                            buffers.extend(group)
                            expectations.append((out, expected))
                            handles.append(
                                self._submit_vector(st_worker, group, output_prefix, spin_iters=_REFILL_SPIN_ITERS)
                            )
                    finally:
                        # Every submitted run is drained before this case's buffers or its output
                        # directory leave scope, whether the loop finished or a submission failed
                        # part way.
                        drain_failures = _drain_handles(handles, "a5 TMR sixteen-run refill")
                    assert not drain_failures, f"a run of this case did not finish: {drain_failures}"
                    _assert_results(expectations, trace, case_runs, "a5 TMR sixteen-run refill")
            finally:
                # The sampler reads the frame views, so it stops before they are released — which
                # is this scope's exit, whether the body finished or raised.
                stop.set()
                sampler.join(timeout=5.0)

        assert len(handles) == _REFILL_RUNS, f"only {len(handles)} of {_REFILL_RUNS} runs were submitted"
        assert widest == _FRAME_COUNT, (
            f"the widest instant held {widest} launched run(s), so two runs were never in flight over the "
            f"two sets a5 TMR grants: per-set dispatch ids {dispatches_by_slot}"
        )
        frontiers = [max(dispatch_id for dispatch_id, _ in pair) for pair in pairs_seen]
        assert len(pairs_seen) >= 2 and max(frontiers) > min(frontiers), (
            f"two runs were never seen in flight together a second time over later identities, so the second "
            f"set was filled once rather than refilled: frontiers={frontiers}, "
            f"pairs={[sorted(pair) for pair in pairs_seen]}"
        )
        regressed = {slot: ids for slot, ids in dispatches_by_slot.items() if ids != sorted(ids)}
        assert not regressed, (
            f"a resource set was seen carrying an earlier run after a later one, so the sets are not passing "
            f"from one run to its successor: {regressed}"
        )


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestA5TmrEarlyEnqueueTwoEndpoints(_A5TmrEarlyEnqueueBase):
    """Two a5 endpoints of one Worker each reach the two-deep state, independently.

    This is the capability's actual scope rather than a widening of it: nothing on the admission
    path checks how many endpoints a Worker has, so a level-3 Worker configured at
    ``launch_depth=2`` passes the request to *every* local device child. Each child is a separate
    process owning its own device context, pooled arena, slot storage and retained temporaries, so
    the two endpoints share none of the storage the ordering protects — and this case is what makes
    that a measurement rather than an argument from the process model.
    """

    CASES = [
        {
            "name": "a5_tmr_early_enqueue_two_endpoints",
            "platforms": ["a5"],
            "config": {"device_count": 2, "num_sub_workers": 0, "launch_depth": 2},
            "params": {},
        },
    ]

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        del kwargs
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        assert str(worker._config["platform"]) in case["platforms"]  # noqa: SLF001 -- scene-test validation
        self.test_each_endpoint_enqueues_early("a5", worker)
        self.test_a_live_comm_domain_refuses_the_join("a5", worker)

    @pytest.mark.platforms(["a5"])
    def test_each_endpoint_enqueues_early(self, st_platform, st_worker):
        """Every endpoint negotiates two frames, and each reaches the two-deep ordered state."""
        if st_platform != "a5":
            pytest.skip("a5 TMR early enqueue is gated to the a5 onboard tensormap_and_ringbuffer route")
        counts = _negotiated_frame_counts(st_worker)
        assert len(counts) == 2, (
            f"this case needs two a5 endpoints on one Worker, got {len(counts)}; without a second endpoint the "
            f"per-device-context scope of the capability is not observed at all"
        )
        assert counts == [_FRAME_COUNT, _FRAME_COUNT], (
            f"the two endpoints negotiated {counts} task frames, so the capability did not reach every local "
            f"device child of this Worker"
        )
        # Sequential rather than concurrent on purpose: each endpoint's claim is about its own
        # device, and driving both at once would leave a failure unable to say which one failed.
        for chip_index in range(2):
            self._assert_two_run_window(st_worker, f"a5 TMR endpoint {chip_index}", chip_index=chip_index)

    @pytest.mark.platforms(["a5"])
    def test_a_live_comm_domain_refuses_the_join(self, st_platform, st_worker):
        """A worker holding a live comm resource establishes no joined launch.

        The single cross-endpoint interaction on this path, and the reason a multi-chip collective
        workload does not overlap at all: ``permits_joined_launch`` ends with
        ``!holds_live_comm_resources()``, which covers comm sessions, global domains and live
        exported device regions. A child releases those before its device reset, so a run ordered
        behind another while one is live would be ordered across that release.

        Driven here rather than in its own class because the realistic shape of the gate is a
        domain that actually spans this Worker's endpoints, and the two devices are already booked.
        """
        if st_platform != "a5":
            pytest.skip("a5 TMR early enqueue is gated to the a5 onboard tensormap_and_ringbuffer route")
        handle = type(self)._st_chip_handles["vector"]
        signature = type(self)._st_chip_handles["vector_sig"]
        trace = _RunTrace(st_worker)
        stop = threading.Event()

        # Endpoint 0 drives this case, and only its child's records are its evidence: both
        # endpoints name their runs with their own dispatch ids, so an absence claim that accepted
        # the other child's record could fail on a run this case never submitted.
        pid = self._endpoint_pid(st_worker, 0)
        with _frame_views(st_worker, 0) as frames:
            case_runs = _CaseRuns(frames)

            def sample():
                while not stop.is_set():
                    case_runs.note(_coherent_snapshot(frames))
                    time.sleep(0.001)

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            try:
                with tempfile.TemporaryDirectory(prefix="simpler-a5-tmr-early-enqueue-comm-") as output_prefix:
                    config = self._build_config(self.CASES[0]["config"], output_prefix=output_prefix)
                    buffers: list = []
                    expectations: list = []
                    groups = []
                    for a_value, b_value in ((2.0, 3.0), (5.0, 7.0)):
                        group, out, expected = self._three_buffers(st_worker, a_value, b_value)
                        buffers.extend(group)
                        expectations.append((out, expected))
                        groups.append(group)

                    def graph(orch, _args, _cfg):
                        # Both runs are dispatched from inside the domain's lifetime, so the gate
                        # is live for the whole window in which a join could have been authorized.
                        # Both go to endpoint 0: the claim is about one worker's own two runs while
                        # it holds a comm resource, and the domain is what makes it hold one.
                        with orch.allocate_domain(name="early_enqueue", workers=[0, 1], window_size=4096):
                            for group in groups:
                                orch.submit_next_level(
                                    handle,
                                    _chip_args(group, signature, _DEVICE_SPIN_ITERS),
                                    config,
                                    worker=0,
                                )

                    run_handle = st_worker.submit(graph)
                    drain_failures = _drain_handles([run_handle], "a5 TMR comm refusal")
                    assert not drain_failures, f"the comm-refusal case's own graph failed: {drain_failures}"
                    _assert_results(expectations, trace, case_runs, "a5 TMR comm refusal")
            finally:
                # The sampler reads the frame views, so it stops before they are released.
                stop.set()
                sampler.join(timeout=5.0)
                observed = len(case_runs)

        # The absence below is about runs this case is shown to have run, not about nothing having
        # been sampled.
        assert observed >= 2, (
            f"only {observed} of this case's runs were ever seen in a negotiated frame above dispatch floor "
            f"{case_runs.floor}, so no joined-launch absence can be attributed to this case"
        )
        # Drained on the budget rather than asserted immediately: an absence has to be given the
        # same window a presence would get.
        records: list[dict] = []
        _await_records(trace, records, _EVIDENCE_BUDGET_S, lambda seen: bool(_established_pairs(seen, case_runs, pid)))
        assert not _established_pairs(records, case_runs, pid), (
            f"a joined launch was established while a comm domain was live: {records}"
        )


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestA5TmrEarlyEnqueueErrorAttribution(_A5TmrEarlyEnqueueBase):
    """A run that fails beside a live predecessor reports its own status, and only its own.

    This is the case for the second production change. Only a run's own published snapshot now
    decides the status it is reported with; the shared memory header, which whichever run occupies
    the arena next resets and refills, is read for the diagnostic detail line alone and the line
    says so. Before, its value could become one run's reported failure while describing another's.

    The failing run is the *successor* deliberately. Its orchestration latches its own fatal code
    and returns, exhausting no resource and hanging no core, so the predecessor ahead of it is
    unaffected and must still produce its correct result. A poisoned runner after the failure can
    then affect no later run, because there is none.

    What is asserted is the attribution, not the numeral: onboard, a generic CANN status can mask a
    device-side code, so pinning the reported number here would be pinning the masking race rather
    than this change. Which run reports a failure at all is the property the change protects.

    This case does **not** cover which status channel was read. A device-side fatal publishes a
    terminal snapshot, so it exercises the preferred channel rather than the missing-snapshot one,
    and which of the two a given failure takes is not this case's to decide. That contract is
    pinned deterministically at the host boundary instead, by
    ``TrbRuntimeTempBufferTest.FailedExecutionReadsThisRunsPublishedSnapshot`` and
    ``...WithoutASnapshotKeepsItsExecutionError`` in
    ``tests/ut/cpp/common/tensormap_and_ringbuffer/test_trb_runtime_temp_buffer.cpp``, which drive
    both channels directly over the real maker.
    """

    CASES = [
        {
            "name": "a5_tmr_early_enqueue_error_attribution",
            "platforms": ["a5"],
            "config": {"device_count": 1, "num_sub_workers": 0, "launch_depth": 2},
            "params": {},
        },
    ]

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        del kwargs
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        assert str(worker._config["platform"]) in case["platforms"]  # noqa: SLF001 -- scene-test validation
        self.test_a_failing_successor_reports_its_own_status("a5", worker)

    @pytest.mark.platforms(["a5"])
    def test_a_failing_successor_reports_its_own_status(self, st_platform, st_worker):
        """This exact pair was joined, the predecessor's result survives, and the failure is its own.

        The join evidence is scoped to *these two* runs. Without it the case would pass under
        ordinary serial submission — the fatal callable declining the join, or the predecessor
        finishing before the successor was submitted — and so would establish nothing about a
        failure beside a live predecessor. Another pair's record cannot stand in: the keys below
        are this case's own two runs, taken above a dispatch floor read before it submitted
        anything.
        """
        if st_platform != "a5":
            pytest.skip("a5 TMR early enqueue is gated to the a5 onboard tensormap_and_ringbuffer route")
        assert _negotiated_frame_counts(st_worker)[0] == _FRAME_COUNT, (
            "this case needs two negotiated frames, so that the failing run is admitted while its predecessor "
            "is still executing rather than after it has retired"
        )
        trace = _RunTrace(st_worker)
        stop = threading.Event()
        pid = self._endpoint_pid(st_worker)
        with _frame_views(st_worker) as frames:
            case_runs = _CaseRuns(frames)

            def sample():
                while not stop.is_set():
                    case_runs.note(_coherent_snapshot(frames))
                    time.sleep(0.001)

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            try:
                with tempfile.TemporaryDirectory(prefix="simpler-a5-tmr-early-enqueue-error-") as output_prefix:
                    buffers: list = []
                    good_handle = None
                    failing_handle = None
                    good_failure = None
                    failing_error = None
                    try:
                        good_group, good_out, good_expected = self._three_buffers(st_worker, 2.0, 3.0)
                        failing_group, _failing_out, _unused = self._three_buffers(st_worker, 5.0, 7.0)
                        buffers.extend(good_group)
                        buffers.extend(failing_group)
                        good_handle = self._submit_vector(
                            st_worker, good_group, output_prefix, spin_iters=_DEVICE_SPIN_ITERS
                        )
                        _wait_for_one_launched_frame(frames, 20.0)
                        failing_handle = self._submit_vector(
                            st_worker, failing_group, output_prefix, callable_name="reported_fatal"
                        )
                    finally:
                        # The predecessor is drained first and separately: its result is the half
                        # of this case that must survive, and its wait must not be skipped by the
                        # successor's failure.
                        if good_handle is not None:
                            good_failure = _drain_handles([good_handle], "a5 TMR error attribution predecessor")
                        if failing_handle is not None:
                            try:
                                failing_handle.wait()
                            except Exception as error:  # noqa: BLE001 -- this case's subject
                                failing_error = str(error)
            finally:
                # The sampler reads the frame views, so it stops before they are released — which
                # is this scope's exit, whether the body finished or raised.
                stop.set()
                sampler.join(timeout=5.0)
                observed = case_runs.sorted_keys()

        assert not good_failure, (
            f"the predecessor failed while its successor was the one made to fail: {good_failure}. Its own run "
            f"must be unaffected by a later run's fatal report"
        )
        torch.testing.assert_close(good_out, good_expected)
        assert failing_error is not None, (
            "the run whose orchestration reports a fatal code completed successfully, so this case establishes "
            "nothing about how a failure is attributed"
        )

        # Dispatch ids are allocated in submission order, so of this case's two runs the lower is
        # the predecessor that must survive and the higher is the one made to fail. That is what
        # names the pair the record below has to be about.
        assert len(observed) == 2, (
            f"this case's two runs were not both seen in the negotiated frames above dispatch floor "
            f"{case_runs.floor}: {observed}. Without both identities the record below cannot be attributed "
            f"to this good/fatal pair"
        )
        predecessor, successor = observed[0], observed[1]
        records: list[dict] = []
        _await_records(
            trace,
            records,
            _EVIDENCE_BUDGET_S,
            lambda seen: (predecessor, successor) in _established_pairs(seen, case_runs, pid),
        )
        pairs = _established_pairs(records, case_runs, pid)
        assert (predecessor, successor) in pairs, (
            f"the failing run {successor} was never recorded as having completed its native submission while "
            f"its predecessor {predecessor} still had an unfired whole-operator boundary, so it was not "
            f"submitted early and this case's claim about a failure beside a live predecessor is unsupported. "
            f"Established pairs for this case: {sorted(pairs)}; records={records}"
        )


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestA5TmrEarlyEnqueueDepthOneControl(_A5TmrEarlyEnqueueBase):
    """With ``launch_depth`` unset, a5 TMR keeps the serial path: no successor reaches the device.

    The control reads the route it is actually on. With the opt-in unset the child negotiates one
    task frame and its endpoint runs the single-frame loop, which publishes ``_TASK_READY`` then
    ``_TASK_DONE`` and no ``_TASK_LAUNCHED`` at all — so the two-frames-at-one-instant observation
    the opt-in class makes is not merely absent here, it is not a state this route publishes. The
    same claim is therefore established from what this route does publish: the negotiated frame
    count, and the absence of any joined-launch record for runs this case is shown to have run.
    """

    CASES = [
        {
            "name": "a5_tmr_early_enqueue_depth_one",
            "platforms": ["a5"],
            "config": {"device_count": 1, "num_sub_workers": 0},
            "params": {},
        },
    ]

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        del kwargs
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        assert str(worker._config["platform"]) in case["platforms"]  # noqa: SLF001 -- scene-test validation
        self.test_no_overlap_without_the_opt_in("a5", worker)

    @pytest.mark.platforms(["a5"])
    def test_no_overlap_without_the_opt_in(self, st_platform, st_worker):
        """The default is unchanged: the same two submissions produce no joined launch at all."""
        if st_platform != "a5":
            pytest.skip("a5 TMR early enqueue is gated to the a5 onboard tensormap_and_ringbuffer route")
        assert st_worker._launch_depth == 1, (  # noqa: SLF001 -- scene-test validation
            f"this control needs a Worker at the default launch_depth, got {st_worker._launch_depth}"  # noqa: SLF001
        )
        # One frame is the whole reason a successor cannot reach the device by default: with a
        # single frame the parent has nowhere to publish a second run while the first is live, so
        # the second is dispatched only once the first has left.
        negotiated = _negotiated_frame_counts(st_worker)[0]
        assert negotiated == 1, (
            f"the default negotiated {negotiated} task frames on a5 TMR, so this control is not observing the "
            f"serial route it exists to pin"
        )
        trace = _RunTrace(st_worker)
        stop = threading.Event()
        pid = self._endpoint_pid(st_worker)
        with _frame_views(st_worker) as frames:
            case_runs = _CaseRuns(frames)

            def sample():
                while not stop.is_set():
                    case_runs.note(_coherent_snapshot(frames))
                    time.sleep(0.001)

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            try:
                with tempfile.TemporaryDirectory(prefix="simpler-a5-tmr-early-enqueue-control-") as output_prefix:
                    handles: list = []
                    buffers: list = []
                    expectations: list = []
                    try:
                        for a_value, b_value in ((2.0, 3.0), (5.0, 7.0)):
                            group, out, expected = self._three_buffers(st_worker, a_value, b_value)
                            buffers.extend(group)
                            expectations.append((out, expected))
                            handles.append(
                                self._submit_vector(st_worker, group, output_prefix, spin_iters=_DEVICE_SPIN_ITERS)
                            )
                    finally:
                        drain_failures = _drain_handles(handles, "a5 TMR depth-one control")
                    assert not drain_failures, f"a run of this case did not finish: {drain_failures}"
                    _assert_results(expectations, trace, case_runs, "a5 TMR depth-one control")
            finally:
                # The sampler reads the frame views, so it stops before they are released.
                stop.set()
                sampler.join(timeout=5.0)
                observed = len(case_runs)

        # Both submissions passed through that one frame, so the absence below is an absence about
        # runs this case is shown to have run rather than about nothing having been sampled.
        assert observed >= 2, (
            f"only {observed} of this case's runs were ever seen in the negotiated frame above dispatch floor "
            f"{case_runs.floor}, so no joined-launch absence can be attributed to this case"
        )
        records: list[dict] = []
        _await_records(trace, records, _EVIDENCE_BUDGET_S, lambda seen: bool(_established_pairs(seen, case_runs, pid)))
        assert not _established_pairs(records, case_runs, pid), (
            f"a joined launch was established with launch_depth unset: {records}"
        )


#: Run-resource sets a case asks for when it is proving the capacity has no fixed number in it.
#: Above the three every per-set owner used to be laid out for, so a pass here cannot be a pass on
#: a width that happened to exist.
_WIDE_SETS = 5
#: How long the window is watched for the full width, after every submission is out. Bounded so a
#: regression fails on the assertion rather than on the scene-test deadline.
_WIDE_WINDOW_BUDGET_S = 60.0
#: How long a stopped sampler is given to return. Exceeding it is a failure, not a timeout that is
#: then ignored: the sampler takes no blocking call and checks its stop event every iteration.
_OBSERVER_JOIN_BUDGET_S = 5.0
#: How long the first sample is waited for, so the observer's span is established before any
#: submission rather than assumed from `Thread.start`.
_OBSERVER_READY_BUDGET_S = 10.0


class _EntrySampler:
    """Background sampling of one endpoint's run-state entries, with a proved ownership boundary.

    The sampler reads integer addresses into the chip mailbox's shared memory, so it must be
    *proved* stopped before that mapping can go. A timed `join` whose result is not checked proves
    nothing, and raising after a timeout still unwinds into the release — so this holds its own
    export of the shared buffer for the thread's whole life. An export is what makes
    ``SharedMemory.close()`` raise ``BufferError`` rather than unmap, so a sampler that will not
    stop leaves the segment mapped and its close reports the live reader, instead of this case
    reading freed memory. The export is released only after the thread is observed dead.

    `__enter__` returns once the sampler has completed a sample, so "sampling spans the body" is
    established rather than assumed from `Thread.start`. A sampler that raises carries its
    exception to the caller rather than losing it on a daemon thread.

    Subclasses implement `observe(snapshot)`; the loop owns the cadence, and every path sleeps —
    a tight loop on an unstable table spends the window competing with the thread publishing
    into it.
    """

    def __init__(self, worker, entries, case_runs, chip_index=0):
        self._entries = entries
        self._case_runs = case_runs
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        shm_buf = worker._chip_shms[chip_index].buf  # noqa: SLF001 -- white-box mailbox observation
        assert shm_buf is not None
        self._mapping_hold = shm_buf[0:1]
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.samples = 0
        self.incoherent = 0

    def observe(self, snapshot) -> None:
        """Called once per sample with a coherent snapshot, or None when the read was unstable."""
        raise NotImplementedError

    def _run(self):
        try:
            while not self._stop.is_set():
                snapshot = _coherent_snapshot(self._entries)
                self._case_runs.note(snapshot)
                self.samples += 1
                if snapshot is None:
                    self.incoherent += 1
                else:
                    self.observe(snapshot)
                self._ready.set()
                time.sleep(0.001)
        except BaseException as error:  # noqa: BLE001 -- carried to the caller by __exit__
            self._error = error
            self._ready.set()

    def __enter__(self):
        self._thread.start()
        if not self._ready.wait(timeout=_OBSERVER_READY_BUDGET_S):
            self._shutdown()
            raise AssertionError(
                f"the entry sampler did not complete a sample within {_OBSERVER_READY_BUDGET_S}s, so nothing "
                f"below would have been observed"
            )
        if self._error is not None:
            self._shutdown()
            raise AssertionError(f"the entry sampler failed before sampling: {self._error!r}")
        return self

    def _shutdown(self):
        self._stop.set()
        self._thread.join(timeout=_OBSERVER_JOIN_BUDGET_S)
        if self._thread.is_alive():
            raise AssertionError(
                f"the entry sampler did not exit within {_OBSERVER_JOIN_BUDGET_S}s of being stopped; its export "
                f"of the chip mailbox is deliberately still held, so this endpoint's shared memory stays mapped "
                f"and its close will report the live reader rather than this case reading freed memory"
            )
        self._mapping_hold.release()

    def __exit__(self, exc_type, _exc, _tb):
        self._shutdown()
        if exc_type is None and self._error is not None:
            raise AssertionError(f"the entry sampler failed while sampling: {self._error!r}")
        return False


class _WorkedCapacityObserver(_EntrySampler):
    """The worked example's evidence: sets held, runs launched, and how each set turned over.

    `widest_held` is the capacity actually in use at one instant, taken from the entry generations'
    parity; `widest` is how many of those were launched and accepted. They are different numbers,
    and keeping them apart is what separates the grant from the launch budget.

    `frontiers` records the highest dispatch id at each new widest instant, so a width reached once
    can be told from one reached again over later runs. `dispatches_by_slot` records the order each
    set carried, which is what shows a set passing from one run to its successor.
    """

    def __init__(self, worker, entries, case_runs, chip_index=0):
        super().__init__(worker, entries, case_runs, chip_index)
        self.widest = 0
        self.widest_held = 0
        self.held_sets: set[int] = set()
        self.frontiers: list[int] = []
        self.dispatches_by_slot: dict[int, list[int]] = {}

    def observe(self, snapshot) -> None:
        held = _held_run_sets(self._entries)
        if held is not None:
            self.held_sets.update(held)
            self.widest_held = max(self.widest_held, len(held))
        keys = _launched_run_keys(snapshot)
        if not keys:
            return
        if len(keys) >= self.widest and len(keys) > 1:
            self.widest = len(keys)
            self.frontiers.append(max(dispatch_id for dispatch_id, _ in keys))
        for dispatch_id, slot_id in keys:
            seen = self.dispatches_by_slot.setdefault(slot_id, [])
            if not seen or seen[-1] != dispatch_id:
                seen.append(dispatch_id)


class _WideWindowObserver(_EntrySampler):
    """Watches for the widest instant of launched, accepted runs across a submission window.

    Samples to its deadline rather than stopping at the first instant that shows nothing launched:
    runs retire at different times, so a gap between one retiring and the next launching is an
    ordinary part of the window and says nothing about whether the width was ever reached.

    `history` records the launched count at each transition. It is diagnostic context for a failure
    — it shows whether any width was reached and how the window moved — and on its own it cannot
    tell a product clamp from an observation that missed the instant: both can produce a history
    whose maximum is below the target.
    """

    def __init__(self, worker, entries, case_runs, want, chip_index=0):
        super().__init__(worker, entries, case_runs, chip_index)
        self._want = want
        self.together = None
        self.widest = set()
        self.history: list[int] = []
        self._last = -1

    def observe(self, snapshot) -> None:
        keys = _launched_run_keys(snapshot)
        if keys is None:
            self.incoherent += 1
            return
        if len(keys) != self._last:
            self.history.append(len(keys))
            self._last = len(keys)
        if len(keys) > len(self.widest):
            self.widest = set(keys)
        if len(keys) >= self._want and self.together is None:
            self.together = set(keys)

    def report(self) -> str:
        return (
            f"widest instant {sorted(self.widest)} over {self.samples} samples "
            f"({self.incoherent} incoherent); launched-count transitions {self.history}"
        )


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestA5TmrCapacityBeyondTheOldCeiling(_A5TmrEarlyEnqueueBase):
    """At ``pipeline_depth=5`` five runs are in flight at once on the shared-scratch runtime.

    This is the backend the old declaration bounded at two, on the reasoning that one shared
    device scratch cannot serve more runs. It can: the AICPU stream is a FIFO and that order is
    transitive, so a run ordered behind the run ahead of it is ordered behind everything that run
    was ordered behind. What actually bounds concurrency here is the per-run task-args region, and
    a context gets one of those per set it was granted.

    What each check can see:

    granted    the endpoint negotiated five task frames, so five runs have somewhere to go.
    accepted   five different runs were observed launched and accepted at one coherent instant.
    distinct   every run's own output is correct, so the shared scratch served five runs in order
               rather than letting any two of them write over each other.
    """

    CASES = [
        {
            "name": "a5_tmr_capacity_beyond_the_old_ceiling",
            "platforms": ["a5"],
            "config": {
                "device_count": 1,
                "num_sub_workers": 0,
                "launch_depth": _WIDE_SETS,
                "pipeline_depth": _WIDE_SETS,
            },
            "params": {},
        },
    ]

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        del kwargs
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        assert str(worker._config["platform"]) in case["platforms"]  # noqa: SLF001 -- scene-test validation
        self._require_wide_sets(worker)
        self.test_five_runs_are_launched_and_accepted_at_once("a5", worker)

    @staticmethod
    def _require_wide_sets(worker):
        assert worker._pipeline_depth_request == _WIDE_SETS, (  # noqa: SLF001 -- scene-test validation
            f"this class needs a Worker at pipeline_depth={_WIDE_SETS}, got {worker._pipeline_depth_request}"  # noqa: SLF001
        )
        # The grant, not the request: this runtime used to declare two, so a negotiation that
        # still lands on two is the clamp this case exists to refuse.
        negotiated = _negotiated_frame_counts(worker)[0]
        assert negotiated == _WIDE_SETS, (
            f"the a5 TMR endpoint negotiated {negotiated} task frames for a {_WIDE_SETS}-set request, "
            f"so the capacity was clamped somewhere and nothing below could observe it"
        )

    @pytest.mark.platforms(["a5"])
    def test_five_runs_are_launched_and_accepted_at_once(self, st_platform, st_worker):
        """Five accepted submissions over five sets, each with its own correct result."""
        if st_platform != "a5":
            pytest.skip("this capacity case is gated to the a5 onboard tensormap_and_ringbuffer route")
        self._require_wide_sets(st_worker)
        trace = _RunTrace(st_worker)
        with _frame_views(st_worker) as frames:
            case_runs = _CaseRuns(frames)
            with tempfile.TemporaryDirectory(prefix="simpler-a5-tmr-wide-capacity-") as output_prefix:
                handles: list = []
                buffers: list = []
                expectations: list = []
                # Every buffer before any submission. Allocating between submits puts five device
                # round trips inside the window the five runs have to share, which is the host
                # racing itself rather than the capacity being measured.
                groups = []
                for index in range(_WIDE_SETS):
                    group, out, expected = self._three_buffers(st_worker, float(2 * index + 2), float(2 * index + 3))
                    groups.append(group)
                    buffers.extend(group)
                    expectations.append((out, expected))
                observer = _WideWindowObserver(st_worker, frames, case_runs, _WIDE_SETS)
                try:
                    # Sampling starts before the first submission and ends after the last: the
                    # instant this case is about is somewhere inside that interval, and a sampler
                    # that starts after it can only see what is left.
                    with observer:
                        for group in groups:
                            handles.append(
                                self._submit_vector(st_worker, group, output_prefix, spin_iters=_DEVICE_SPIN_ITERS)
                            )
                        deadline = time.monotonic() + _WIDE_WINDOW_BUDGET_S
                        while observer.together is None and time.monotonic() < deadline:
                            time.sleep(0.001)
                    together = observer.together
                finally:
                    # Before anything of this case's leaves scope, whatever the submission or the
                    # observation did: a run still in flight owns its arguments and its output.
                    drain_failures = _drain_handles(handles, "a5 TMR wide-capacity window")
                assert not drain_failures, f"a run of this case did not finish: {drain_failures}"
                _assert_results(expectations, trace, case_runs, "a5 TMR wide-capacity window")

                assert observer.samples > 0, "the observer never sampled, so this case established nothing"
                assert together is not None, (
                    f"{_WIDE_SETS} different runs were never observed launched and accepted at the same time, "
                    f"so this runtime's shared scratch never carried more than its old declaration allowed. "
                    f"{observer.report()}"
                )
                assert len(together) == _WIDE_SETS, (
                    f"fewer than {_WIDE_SETS} runs were named together: {sorted(together)}. {observer.report()}"
                )


#: The contract's worked example: eight run-resource sets with four of them on the device at
#: once, and sixty-four submissions turning over through them. Eight is above every width this
#: layout ever had, and the launch budget is deliberately *below* the grant — the two are
#: separate budgets and a case that conflates them proves neither.
_WORKED_SETS = 8
_WORKED_LAUNCH_DEPTH = 4
_WORKED_SUBMISSIONS = 64


def _held_run_sets(entries):
    """Which resource sets hold a run at one coherent instant, or None.

    An entry's generation is odd while it is occupied and even once released, so the odd ones are
    the runs in flight — staged, activated and launched alike — and the sets they name are the
    capacity actually in use. That is a different number from how many are *launched*: a run
    holding its set while it waits behind the launch budget is precisely what the sets beyond that
    budget buy, and it is the only observation that separates the two bounds.

    Two passes, for the reason :func:`_coherent_snapshot` takes two: one pass over a table being
    refilled can read a set after it was released and another before it was taken, and so report an
    instant that never existed.
    """

    def _read():
        return [
            (_mailbox_load_u64(addr + _OFF_ENTRY_GENERATION), _read_run_entry_identity(buf)[2]) for addr, buf in entries
        ]

    before = _read()
    after = _read()
    if before != after:
        return None
    return {slot for generation, slot in after if generation % 2 == 1}


def _observe_launch_prefix(entries, want, timeout, case_runs):
    """Watch for `want` different runs launched and accepted at one coherent instant.

    Returns the widest set of keys seen together, which the caller compares against what the
    launch budget allows. Unlike the capacity cases this does not require the *grant*'s worth of
    runs to be launched: `launch_depth` bounds how many reach the device, and that bound is the
    claim.
    """
    deadline = time.monotonic() + timeout
    widest: set = set()
    while time.monotonic() < deadline:
        snapshot = _coherent_snapshot(entries)
        case_runs.note(snapshot)
        keys = _launched_run_keys(snapshot)
        if keys is not None and len(keys) > len(widest):
            widest = set(keys)
            if len(widest) >= want:
                return widest
        time.sleep(0.001)
    return widest


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestA5TmrWideFailureAttribution(_A5TmrEarlyEnqueueBase):
    """One run fails amid a wide prefix of accepted runs; each outcome is its own.

    The narrow sibling pairs one good run with one fatal at the old width. Here four good runs and
    one fatal are admitted together at `_WIDE_SETS`, which is the state rows 29 and 32 are about: a
    poisoned runner after the failure must affect no run that was already accepted, and each of the
    concurrent failures keeps its own attribution rather than being collapsed into one.

    What each check can see:

    attributed  exactly one handle raises, and it is the one whose orchestration latches a fatal
                code. The other four return without error.
    distinct    each surviving run's own output equals its own inputs' arithmetic, so a failure
                beside them did not corrupt the sets they held.
    own text    the failure's message is non-empty and reaches the caller through its own handle.
    settled     every handle is drained before anything of this case's leaves scope, on every path,
                and the failing one is waited exactly once.

    What is asserted is the attribution, not the numeral — onboard, a generic CANN status can mask
    a device-side code, so pinning the number would pin the masking race rather than the ownership
    this case is about.
    """

    CASES = [
        {
            "name": "a5_tmr_wide_failure_attribution",
            "platforms": ["a5"],
            "config": {
                "device_count": 1,
                "num_sub_workers": 0,
                "launch_depth": _WIDE_SETS,
                "pipeline_depth": _WIDE_SETS,
            },
            "params": {},
        },
    ]

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        del kwargs
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        assert str(worker._config["platform"]) in case["platforms"]  # noqa: SLF001 -- scene-test validation
        self.test_one_failure_amid_a_wide_prefix_is_its_own("a5", worker)

    @pytest.mark.platforms(["a5"])
    def test_one_failure_amid_a_wide_prefix_is_its_own(self, st_platform, st_worker):
        """Four good runs and one fatal, all accepted at a wide grant."""
        if st_platform != "a5":
            pytest.skip("a5 TMR early enqueue is gated to the a5 onboard tensormap_and_ringbuffer route")
        negotiated = _negotiated_frame_counts(st_worker)[0]
        assert negotiated == _WIDE_SETS, (
            f"the endpoint negotiated {negotiated} sets for a {_WIDE_SETS}-set request, so the failing run would "
            f"not be one of a wide prefix of accepted runs"
        )

        good_handles: list = []
        expectations: list = []
        buffers: list = []
        failing_handle = None
        failing_error = None
        good_failures: list = []
        with tempfile.TemporaryDirectory(prefix="simpler-a5-tmr-wide-failure-") as output_prefix:
            try:
                # The good runs first and the fatal last, so every one of them is already accepted
                # when the failure is reported: a failure ahead of them would prove nothing about
                # a run that was already holding a set.
                for index in range(_WIDE_SETS - 1):
                    group, out, expected = self._three_buffers(st_worker, float(2 * index + 2), float(2 * index + 3))
                    buffers.extend(group)
                    expectations.append((out, expected))
                    good_handles.append(
                        self._submit_vector(st_worker, group, output_prefix, spin_iters=_DEVICE_SPIN_ITERS)
                    )
                failing_group, _out, _unused = self._three_buffers(st_worker, 11.0, 13.0)
                buffers.extend(failing_group)
                failing_handle = self._submit_vector(
                    st_worker, failing_group, output_prefix, callable_name="reported_fatal"
                )
            finally:
                # The surviving runs are drained first and separately: their results are the half
                # of this case that must survive, and their waits must not be skipped by the
                # failure behind them.
                good_failures = _drain_handles(good_handles, "a5 TMR wide failure attribution")
                if failing_handle is not None:
                    try:
                        failing_handle.wait()
                    except Exception as error:  # noqa: BLE001 -- this case's subject
                        failing_error = str(error)

            assert not good_failures, (
                f"a run accepted before the failure reported one of its own: {good_failures}. A later run's "
                f"fatal report must not become an earlier run's outcome"
            )
            assert len(good_handles) == _WIDE_SETS - 1, (
                f"only {len(good_handles)} of {_WIDE_SETS - 1} surviving runs were submitted, so the failing run "
                f"was not one of a wide prefix"
            )
            assert failing_error is not None, (
                "the run whose orchestration reports a fatal code completed successfully, so this case "
                "establishes nothing about how a failure is attributed"
            )
            assert failing_error.strip(), "the failure reached the caller with no text of its own"
            for out, expected in expectations:
                torch.testing.assert_close(out, expected)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestA5TmrWideTwoEndpoints(_A5TmrEarlyEnqueueBase):
    """Two endpoints at a wide grant: each negotiates it, and each run's result is its own.

    Capacity is answered per device context, and nothing on the path checks how many endpoints a
    Worker has — so a grant above the old width has to hold on each of them independently. The
    narrow sibling establishes the per-endpoint overlap at the old width; this is row 35's wide
    form.

    What each check can see:

    granted     every endpoint negotiated `_WIDE_SETS`, read from each child's own published count,
                so neither was clamped while the other was not.
    admitted    a wide prefix is accepted on each endpoint, with no submission refused.
    aggregated  every run's own output is correct, on both endpoints, so the two contexts' sets
                held their own parameters and results rather than sharing any of them.

    It does not assert a simultaneous launched instant on each endpoint; that is the narrow
    sibling's subject, and widening it would be re-proving the same overlap rather than the
    per-endpoint grant this row is about.
    """

    CASES = [
        {
            "name": "a5_tmr_wide_two_endpoints",
            "platforms": ["a5"],
            "config": {
                "device_count": 2,
                "num_sub_workers": 0,
                "launch_depth": _WIDE_SETS,
                "pipeline_depth": _WIDE_SETS,
            },
            "params": {},
        },
    ]

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        del kwargs
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        assert str(worker._config["platform"]) in case["platforms"]  # noqa: SLF001 -- scene-test validation
        self.test_each_endpoint_carries_a_wide_grant("a5", worker)

    @pytest.mark.platforms(["a5"])
    def test_each_endpoint_carries_a_wide_grant(self, st_platform, st_worker):
        """A wide prefix on each of two endpoints, every result its own."""
        if st_platform != "a5":
            pytest.skip("a5 TMR early enqueue is gated to the a5 onboard tensormap_and_ringbuffer route")
        counts = _negotiated_frame_counts(st_worker)
        assert len(counts) == 2, f"this case needs two chip endpoints, got {len(counts)}"
        for chip_index, negotiated in enumerate(counts):
            assert negotiated == _WIDE_SETS, (
                f"endpoint {chip_index} negotiated {negotiated} sets for a {_WIDE_SETS}-set request, so the grant "
                f"did not hold on every endpoint"
            )

        handles: list = []
        buffers: list = []
        expectations: list = []
        drain_failures: list = []
        with tempfile.TemporaryDirectory(prefix="simpler-a5-tmr-wide-endpoints-") as output_prefix:
            try:
                # Interleaved across the endpoints, so neither is drained before the other is
                # admitted: a sequence that filled one and emptied it would establish the width
                # once rather than on each.
                for index in range(_WIDE_SETS):
                    for chip_index in range(len(counts)):
                        group, out, expected = self._three_buffers(
                            st_worker, float(2 * index + 2 + chip_index), float(2 * index + 3 + chip_index)
                        )
                        buffers.extend(group)
                        expectations.append((out, expected))
                        handles.append(
                            self._submit_vector(
                                st_worker, group, output_prefix, spin_iters=_REFILL_SPIN_ITERS, chip_index=chip_index
                            )
                        )
            finally:
                # Before anything of this case's leaves scope, whatever the submissions did.
                drain_failures = _drain_handles(handles, "a5 TMR wide two endpoints")

            assert not drain_failures, f"a run of this case did not finish: {drain_failures}"
            assert len(handles) == _WIDE_SETS * len(counts), (
                f"only {len(handles)} of {_WIDE_SETS * len(counts)} runs were submitted across the two endpoints"
            )
            for out, expected in expectations:
                torch.testing.assert_close(out, expected)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestA5TmrWorkedCapacityExample(_A5TmrEarlyEnqueueBase):
    """The contract's worked example, end to end: eight sets, four launched, sixty-four runs.

    Three claims this suite's other cases cannot make, because each needs the two budgets to be
    *different* numbers and the run count to exceed the sets:

    separated  `launch_depth=4` under `pipeline_depth=8` must hold the device to four runs while
               more than four are admitted and prepared. A case where the two are equal cannot
               tell a launch budget from a resource budget.
    sustained  sixty-four submissions retire through eight reused sets, with the widest observed
               instant reached again later over a strictly higher frontier of dispatch ids — a set
               carrying a *later* run, not one opening window.
    ordered    no set is ever seen carrying an earlier run after a later one, so a set passes from
               one run to its successor rather than being shared.

    What the reuse observation is: a run-state entry showing a set under a second run's identity.
    That is a physical observation of the set being handed on, and the production path only hands
    it on after the earlier run's lifetime closed — this case reads the handover, it does not
    witness each step of that closure.
    """

    CASES = [
        {
            "name": "a5_tmr_worked_capacity_example",
            "platforms": ["a5"],
            "config": {
                "device_count": 1,
                "num_sub_workers": 0,
                "launch_depth": _WORKED_LAUNCH_DEPTH,
                "pipeline_depth": _WORKED_SETS,
            },
            "params": {},
        },
    ]

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        del kwargs
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        assert str(worker._config["platform"]) in case["platforms"]  # noqa: SLF001 -- scene-test validation
        self._require_worked_budgets(worker)
        self.test_sixty_four_runs_retire_through_eight_sets("a5", worker)

    @staticmethod
    def _require_worked_budgets(worker):
        assert worker._pipeline_depth_request == _WORKED_SETS, (  # noqa: SLF001 -- scene-test validation
            f"this class needs pipeline_depth={_WORKED_SETS}, got {worker._pipeline_depth_request}"  # noqa: SLF001
        )
        assert worker._launch_depth == _WORKED_LAUNCH_DEPTH, (  # noqa: SLF001 -- scene-test validation
            f"this class needs launch_depth={_WORKED_LAUNCH_DEPTH}, got {worker._launch_depth}"  # noqa: SLF001
        )
        negotiated = _negotiated_frame_counts(worker)[0]
        assert negotiated == _WORKED_SETS, (
            f"the endpoint negotiated {negotiated} sets for a {_WORKED_SETS}-set request, so the capacity was "
            f"clamped somewhere and nothing below could observe it"
        )

    @pytest.mark.platforms(["a5"])
    def test_sixty_four_runs_retire_through_eight_sets(self, st_platform, st_worker):
        """Sustained submission at the contracted budgets, with the launch bound held."""
        if st_platform != "a5":
            pytest.skip("the worked capacity example is gated to the a5 onboard tensormap_and_ringbuffer route")
        self._require_worked_budgets(st_worker)
        trace = _RunTrace(st_worker)
        with _frame_views(st_worker) as entries:
            case_runs = _CaseRuns(entries)

            handles: list = []
            buffers: list = []
            expectations: list = []
            # Sampling spans the whole body, and the scope is what proves the sampler stopped
            # before the entry views it reads are released — on every path, including the
            # readiness handshake failing and a submission raising. See `_EntrySampler`.
            with _WorkedCapacityObserver(st_worker, entries, case_runs) as observer:
                with tempfile.TemporaryDirectory(prefix="simpler-a5-tmr-worked-") as output_prefix:
                    try:
                        for index in range(_WORKED_SUBMISSIONS):
                            group, out, expected = self._three_buffers(
                                st_worker, float(index % 7 + 2), float(index % 5 + 3)
                            )
                            buffers.extend(group)
                            expectations.append((out, expected))
                            handles.append(
                                self._submit_vector(st_worker, group, output_prefix, spin_iters=_REFILL_SPIN_ITERS)
                            )
                    finally:
                        # Every submitted run is drained before this case's buffers or its output
                        # directory leave scope, whether the loop finished or a submission failed
                        # part way. More than three are outstanding at this point, which is itself
                        # part of what this case covers.
                        drain_failures = _drain_handles(handles, "a5 worked capacity example")
                    assert not drain_failures, f"a run of this case did not finish: {drain_failures}"
                    _assert_results(expectations, trace, case_runs, "a5 worked capacity example")

        assert len(handles) == _WORKED_SUBMISSIONS, f"only {len(handles)} of {_WORKED_SUBMISSIONS} runs were submitted"
        # Four on the device at once, and never a fifth: the launch budget is a separate bound
        # from the eight sets, and a run beyond it is prepared rather than launched.
        assert observer.widest >= 2, (
            f"no instant held two launched runs, so nothing above the serial path was observed: "
            f"per-set dispatch ids {observer.dispatches_by_slot}"
        )
        # The configured grant, observed rather than inferred: every one of the eight sets holding a
        # run at one coherent instant. That is what separates the grant from the launch budget —
        # four is all the launch budget can ever account for — and it is the claim an implementation
        # advertising eight while recycling fewer cannot satisfy.
        #
        # Occupancy comes from the entry generation's parity, which is what the protocol itself uses
        # to mean occupied, so a set counted here holds a run whether that run is staged, activated
        # or launched.
        assert observer.widest_held == _WORKED_SETS, (
            f"at most {observer.widest_held} of {_WORKED_SETS} resource sets were ever held at one instant, so the "
            f"grant this Worker was given is not the capacity it reached: sets seen carrying a run "
            f"{sorted(observer.held_sets)}"
        )
        assert len(observer.held_sets) == _WORKED_SETS, (
            f"only {len(observer.held_sets)} resource set(s) ever carried a run across {_WORKED_SUBMISSIONS} "
            f"submissions: {sorted(observer.held_sets)}"
        )
        assert observer.widest <= _WORKED_LAUNCH_DEPTH, (
            f"{observer.widest} runs were launched and accepted at one instant, above the launch_depth of "
            f"{_WORKED_LAUNCH_DEPTH} this Worker was configured with"
        )
        # The launch budget reached, not merely respected. With the grant full and the device
        # busy, a peak below it would mean the configured depth is not what the lane actually
        # fills — which a bound-only assertion cannot tell from a serial pipeline.
        assert observer.widest == _WORKED_LAUNCH_DEPTH, (
            f"the widest instant held {observer.widest} launched runs, not the {_WORKED_LAUNCH_DEPTH} this Worker "
            f"was "
            f"given, so the launch budget was never reached: per-set dispatch ids {observer.dispatches_by_slot}"
        )
        assert len(observer.frontiers) >= 2 and max(observer.frontiers) > min(observer.frontiers), (
            f"the widest instant was never reached again over later identities, so the sets were filled once "
            f"rather than refilled: observer.frontiers={observer.frontiers}"
        )
        regressed = {slot: ids for slot, ids in observer.dispatches_by_slot.items() if ids != sorted(ids)}
        assert not regressed, (
            f"a resource set was seen carrying an earlier run after a later one, so the sets are not passing "
            f"from one run to its successor: {regressed}"
        )
        assert len(observer.dispatches_by_slot) >= 2, (
            f"only {len(observer.dispatches_by_slot)} resource set(s) were ever seen carrying a run: "
            f"{observer.dispatches_by_slot}"
        )
