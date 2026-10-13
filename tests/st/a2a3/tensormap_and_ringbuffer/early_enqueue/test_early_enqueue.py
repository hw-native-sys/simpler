#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Onboard validation for early enqueue on a2a3 ``tensormap_and_ringbuffer``, over ``Worker.submit``.

This runtime orchestrates on the device, so one AICPU op resets the shared arena, dispatches
until every task completes, retires the cores and destroys the runtime context across its own
span. Its three pooled regions are one instance each, which is why the runtime answered "no" to
joined launch until now: the single instance is only safe if a successor's op cannot begin before
its predecessor's has returned.

a2a3 constructs that guarantee across its two streams, and the two halves are different in kind:

  - The successor's AICore work waits on an **event** — its stream takes a queued wait on the
    predecessor's whole-operator boundary, and records a proof event immediately behind that wait
    so the predecessor can retire its own boundary without awaiting this run's kernels.
  - The successor's AICPU op is ordered by the **queue** — a stream is a FIFO, and the
    predecessor's AICPU payload, its own-AICore-boundary wait, its whole-operator marker and its
    boundary record all sit ahead of it, so the arena reset cannot be dispatched until the
    predecessor's own AICore kernel has ended.

That second half only holds behind a boundary covering both of the predecessor's kernels. The
own-AICore wait is therefore queued *after* the AICPU payload — the AICPU kernel spins in the
handshake until the AICore workers publish, so holding its launch behind its own AICore kernel's
end would deadlock the pair — and a predecessor that published no whole-operator boundary is
refused as a join target.

What this suite establishes is that the guarantee holds on the public route and that nothing the
two runs share is read or written out of turn.

What is checked, for exactly the runs each case submitted:

  admitted   two mailbox frames hold two *different* runs, both TASK_LAUNCHED with both sticky
             acceptance words set. Two frames exist at any launch depth on this endpoint, so what
             the default lacks is the authorization to launch into the second, not somewhere to put
             it — which is why the control below keeps the ordinary capacity and varies only
             ``launch_depth``.
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
  endpoints  a Worker with two a2a3 devices reaches that state on *each* endpoint independently,
             which is the actual scope of the capability: it is answered per device context, and
             no gate on the path checks how many endpoints a Worker has.
  attributed a run that fails beside a live predecessor reports its own status, and the
             predecessor still produces its own correct result — with the join for *that exact
             pair* asserted, so the case cannot pass under serial submission.
  bounded    a third submission parks in the orchestrator's admission wait while two runs hold the
             two sets, and a retirement is what releases it.
  through    a caller's host output buffer still carries its own D2H bytes when it is read only
             after the successor has joined and both have retired — with the join for those exact
             two runs established, so a serial fallback cannot satisfy it.
  owned      a caller's **device** allocation named by a live run is refused by ``Worker.free``
             and accepted once the chain retires, so the caller is still its owner afterwards;
             and a caller's **host** buffer an in-flight run named is refused by
             ``Worker.release_buffer`` on the same terms.
  closed     a run is left outstanding and the fixture's own ``close()`` is what drains it.
  default    with ``launch_depth`` unset, none of it happens.

Comm resources are deliberately outside this capability: ``permits_joined_launch`` ends with
``!holds_live_comm_resources()``, so a worker holding one establishes no joined launch at all, and
the a5 TMR suite drives that gate.

**No device stream waits on the markers.** Recording an event is not a wait, so they add no
ordering to the streams they sit in and cannot make a missing production wait look satisfied.

Every counted edge belongs to the case that counted it: records are read only from this worker's
own chip children's log files through a per-file byte cursor opened at each file's current size,
and both ends of a pair must be runs this case submitted, which :class:`_CaseRuns` decides from a
dispatch-id floor taken before the case submits anything.

The evidence reader below is a fourth copy of the one the ``host_build_graph`` classes and the a5 TMR
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
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import DataType, TaskArgs, TensorArgType
from simpler.worker import (
    _OFF_ACCEPTED,
    _OFF_STATE,
    _TASK_ACCEPTED,
    _TASK_LAUNCHED,
    MAILBOX_FRAME_SIZE,
    _mailbox_load_i32,
    _read_task_frame_identity,
)

from simpler_setup import SceneTestCase, scene_test
from simpler_setup.scene_test import build_output_prefix
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
#: The task frames a2a3 TMR's endpoint negotiates with the opt-in set: one per run-resource set its
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
#: How long the backpressure case waits for its deferred submission to be *counted* as parked in
#: admission. The park itself happens as soon as the thread reaches `begin_run`; this budget only
#: covers thread start-up, and exhausting it is a reported evidence failure rather than a pass.
_PARK_BUDGET_S = 60.0
#: How long a case waits for a helper thread it started to return before calling it unsettled.
#: Long enough to cover a submission parked in admission that a retirement has just released, and
#: bounded so a thread that never returns is a reported failure rather than a hang.
_THREAD_SETTLE_S = 60.0

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
    return [int(count) for count in worker._chip_task_frame_counts]  # noqa: SLF001 -- white-box negotiation


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
    frame_count = _negotiated_frame_counts(worker)[chip_index]
    views = [
        shm_buf[(1 + index) * MAILBOX_FRAME_SIZE : (2 + index) * MAILBOX_FRAME_SIZE] for index in range(frame_count)
    ]
    try:
        yield [(mailbox_addr + (1 + index) * MAILBOX_FRAME_SIZE, view) for index, view in enumerate(views)]
    finally:
        for view in views:
            view.release()


def _coherent_snapshot(frames):
    """Every frame's ``(identity, state, accepted)``, or None when they moved mid-read.

    The frames are separate words, so reading them one after another can mix two instants.
    Re-reading each identity after the states settles that: a set whose identities are unchanged
    describes named runs at one point in their lives, which is what makes a claim about the *pair*
    attributable at all.
    """
    before = [_read_task_frame_identity(buf) for _, buf in frames]
    samples = [(_mailbox_load_i32(addr + _OFF_STATE), _mailbox_load_i32(addr + _OFF_ACCEPTED)) for addr, _ in frames]
    after = [_read_task_frame_identity(buf) for _, buf in frames]
    if before != after:
        return None
    return [(identity, state, accepted) for identity, (state, accepted) in zip(after, samples)]


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
    """Whether the snapshot holds two different runs, both launched and both accepted."""
    if snapshot is None or len(snapshot) != _FRAME_COUNT:
        return False
    if not all(state == _TASK_LAUNCHED and accepted == _TASK_ACCEPTED for _, state, accepted in snapshot):
        return False
    return len({identity for identity, _, _ in snapshot}) == _FRAME_COUNT


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


class _A2a3TmrEarlyEnqueueBase(SceneTestCase):
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
            with tempfile.TemporaryDirectory(prefix="simpler-a2a3-tmr-early-enqueue-") as output_prefix:
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
class TestA2a3TmrEarlyEnqueueDepthTwo(_A2a3TmrEarlyEnqueueBase):
    """At ``launch_depth=2`` a successor's work reaches a2a3 TMR's device while its predecessor runs."""

    CASES = [
        {
            "name": "a2a3_tmr_early_enqueue",
            "platforms": ["a2a3"],
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
        self.test_two_runs_are_launched_and_accepted_at_once("a2a3", worker)
        self.test_sixteen_runs_keep_refilling_both_sets("a2a3", worker)

    @pytest.mark.platforms(["a2a3"])
    def test_two_runs_are_launched_and_accepted_at_once(self, st_platform, st_worker):
        """Admission, the ordering the admission exists for, and what the device then did."""
        if st_platform != "a2a3":
            pytest.skip("a2a3 TMR early enqueue is gated to the a2a3 onboard tensormap_and_ringbuffer route")
        self._assert_two_run_window(st_worker, "a2a3 TMR two-run window")

    @pytest.mark.platforms(["a2a3"])
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
        if st_platform != "a2a3":
            pytest.skip("a2a3 TMR early enqueue is gated to the a2a3 onboard tensormap_and_ringbuffer route")
        negotiated = _negotiated_frame_counts(st_worker)[0]
        assert negotiated == _FRAME_COUNT, (
            f"a2a3 TMR negotiated {negotiated} task frame(s) at launch_depth=2, so the refill below has only one "
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
                with tempfile.TemporaryDirectory(prefix="simpler-a2a3-tmr-early-enqueue-refill-") as output_prefix:
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
                        drain_failures = _drain_handles(handles, "a2a3 TMR sixteen-run refill")
                    assert not drain_failures, f"a run of this case did not finish: {drain_failures}"
                    _assert_results(expectations, trace, case_runs, "a2a3 TMR sixteen-run refill")
            finally:
                # The sampler reads the frame views, so it stops before they are released — which
                # is this scope's exit, whether the body finished or raised.
                stop.set()
                sampler.join(timeout=5.0)

        assert len(handles) == _REFILL_RUNS, f"only {len(handles)} of {_REFILL_RUNS} runs were submitted"
        assert widest == _FRAME_COUNT, (
            f"the widest instant held {widest} launched run(s), so two runs were never in flight over the "
            f"two sets a2a3 TMR grants: per-set dispatch ids {dispatches_by_slot}"
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
class TestA2a3TmrEarlyEnqueueTwoEndpoints(_A2a3TmrEarlyEnqueueBase):
    """Two a2a3 endpoints of one Worker each reach the two-deep state, independently.

    This is the capability's actual scope rather than a widening of it: nothing on the admission
    path checks how many endpoints a Worker has, so a level-3 Worker configured at
    ``launch_depth=2`` passes the request to *every* local device child. Each child is a separate
    process owning its own device context, pooled arena, slot storage and retained temporaries, so
    the two endpoints share none of the storage the ordering protects — and this case is what makes
    that a measurement rather than an argument from the process model.
    """

    CASES = [
        {
            "name": "a2a3_tmr_early_enqueue_two_endpoints",
            "platforms": ["a2a3"],
            "config": {"device_count": 2, "num_sub_workers": 0, "launch_depth": 2},
            "params": {},
        },
    ]

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        del kwargs
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        assert str(worker._config["platform"]) in case["platforms"]  # noqa: SLF001 -- scene-test validation
        self.test_each_endpoint_enqueues_early("a2a3", worker)

    @pytest.mark.platforms(["a2a3"])
    def test_each_endpoint_enqueues_early(self, st_platform, st_worker):
        """Every endpoint negotiates two frames, and each reaches the two-deep ordered state."""
        if st_platform != "a2a3":
            pytest.skip("a2a3 TMR early enqueue is gated to the a2a3 onboard tensormap_and_ringbuffer route")
        counts = _negotiated_frame_counts(st_worker)
        assert len(counts) == 2, (
            f"this case needs two a2a3 endpoints on one Worker, got {len(counts)}; without a second endpoint the "
            f"per-device-context scope of the capability is not observed at all"
        )
        assert counts == [_FRAME_COUNT, _FRAME_COUNT], (
            f"the two endpoints negotiated {counts} task frames, so the capability did not reach every local "
            f"device child of this Worker"
        )
        # Sequential rather than concurrent on purpose: each endpoint's claim is about its own
        # device, and driving both at once would leave a failure unable to say which one failed.
        for chip_index in range(2):
            self._assert_two_run_window(st_worker, f"a2a3 TMR endpoint {chip_index}", chip_index=chip_index)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestA2a3TmrEarlyEnqueueErrorAttribution(_A2a3TmrEarlyEnqueueBase):
    """A run that fails beside a live predecessor reports its own status, and only its own.

    This is the case for the second production change. Only a run's own published snapshot decides
    the status it is reported with; the shared memory header — which whichever run occupies the
    arena next resets and refills — is **not read at all** on this path any more. Before, its value
    could become one run's reported failure while describing another's.

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
    both channels directly over the real maker. Those cases are updated with this change: they
    previously expected a2a3 to fall back to the shared header and now expect both runtimes to read
    the run-owned record only, with the same planted wrong header code and the same zero-read
    oracle.
    """

    CASES = [
        {
            "name": "a2a3_tmr_early_enqueue_error_attribution",
            "platforms": ["a2a3"],
            "config": {"device_count": 1, "num_sub_workers": 0, "launch_depth": 2},
            "params": {},
        },
    ]

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        del kwargs
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        assert str(worker._config["platform"]) in case["platforms"]  # noqa: SLF001 -- scene-test validation
        self.test_a_failing_successor_reports_its_own_status("a2a3", worker)

    @pytest.mark.platforms(["a2a3"])
    def test_a_failing_successor_reports_its_own_status(self, st_platform, st_worker):
        """This exact pair was joined, the predecessor's result survives, and the failure is its own.

        The join evidence is scoped to *these two* runs. Without it the case would pass under
        ordinary serial submission — the fatal callable declining the join, or the predecessor
        finishing before the successor was submitted — and so would establish nothing about a
        failure beside a live predecessor. Another pair's record cannot stand in: the keys below
        are this case's own two runs, taken above a dispatch floor read before it submitted
        anything.
        """
        if st_platform != "a2a3":
            pytest.skip("a2a3 TMR early enqueue is gated to the a2a3 onboard tensormap_and_ringbuffer route")
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
                with tempfile.TemporaryDirectory(prefix="simpler-a2a3-tmr-early-enqueue-error-") as output_prefix:
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
                            good_failure = _drain_handles([good_handle], "a2a3 TMR error attribution predecessor")
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
class TestA2a3TmrEarlyEnqueueDepthOneControl(_A2a3TmrEarlyEnqueueBase):
    """With ``launch_depth`` unset, a2a3 TMR keeps the serial path: no successor reaches the device.

    The control reads the route it is actually on, and that route has the *ordinary* capacity. The
    endpoint negotiates its task frames from the runtime's published pipeline depth, not from the
    launch budget, so this child negotiates the same two frames as the opt-in class and runs the
    same staged-frame loop — including publishing ``_TASK_LAUNCHED`` for whichever run is on the
    device. What the default withholds is the authorization for a *second* run to be launched while
    the first is still live. So the claim is established from the two things that do distinguish the
    routes: the negotiated frame count is the ordinary one, and no joined launch is ever established
    for runs this case is shown to have run.
    """

    CASES = [
        {
            "name": "a2a3_tmr_early_enqueue_depth_one",
            "platforms": ["a2a3"],
            "config": {"device_count": 1, "num_sub_workers": 0},
            "params": {},
        },
    ]

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        del kwargs
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        assert str(worker._config["platform"]) in case["platforms"]  # noqa: SLF001 -- scene-test validation
        self.test_no_overlap_without_the_opt_in("a2a3", worker)

    @pytest.mark.platforms(["a2a3"])
    def test_no_overlap_without_the_opt_in(self, st_platform, st_worker):
        """The default is unchanged: the same two submissions produce no joined launch at all.

        What makes the default serial is the **launch** budget, not storage scarcity. An A2/A3
        onboard endpoint whose published pipeline depth is at least two negotiates two task frames
        at *any* launch depth, so by default a successor is still prepared and still admitted into
        its own frame — it is simply never authorized to reach the device. Asserting one frame here
        would be asserting a different route than the default one, which is why the capacity below
        is required to be the ordinary two.
        """
        if st_platform != "a2a3":
            pytest.skip("a2a3 TMR early enqueue is gated to the a2a3 onboard tensormap_and_ringbuffer route")
        assert st_worker._launch_depth == 1, (  # noqa: SLF001 -- scene-test validation
            f"this control needs a Worker at the default launch_depth, got {st_worker._launch_depth}"  # noqa: SLF001
        )
        # The default route, with its ordinary capacity intact: two frames and two resource sets,
        # and no authorization to launch into the second. A control that ran with one frame would
        # pin frame scarcity instead of the launch budget this capability actually gates on.
        negotiated = _negotiated_frame_counts(st_worker)[0]
        assert negotiated == _FRAME_COUNT, (
            f"the default negotiated {negotiated} task frames on a2a3 TMR rather than {_FRAME_COUNT}, so this "
            f"control is not observing the default route it exists to pin"
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
                with tempfile.TemporaryDirectory(prefix="simpler-a2a3-tmr-early-enqueue-control-") as output_prefix:
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
                        drain_failures = _drain_handles(handles, "a2a3 TMR depth-one control")
                    assert not drain_failures, f"a run of this case did not finish: {drain_failures}"
                    _assert_results(expectations, trace, case_runs, "a2a3 TMR depth-one control")
            finally:
                # The sampler reads the frame views, so it stops before they are released.
                stop.set()
                sampler.join(timeout=5.0)
                observed = len(case_runs)

        # Both submissions were seen in the negotiated frames, so the absence below is an absence
        # about runs this case is shown to have run rather than about nothing having been sampled.
        assert observed >= 2, (
            f"only {observed} of this case's runs were ever seen in the negotiated frames above dispatch floor "
            f"{case_runs.floor}, so no joined-launch absence can be attributed to this case"
        )
        records: list[dict] = []
        _await_records(trace, records, _EVIDENCE_BUDGET_S, lambda seen: bool(_established_pairs(seen, case_runs, pid)))
        assert not _established_pairs(records, case_runs, pid), (
            f"a joined launch was established with launch_depth unset: {records}"
        )


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestA2a3TmrEarlyEnqueueLifetimes(_A2a3TmrEarlyEnqueueBase):
    """What admission does at capacity, and who still owns the caller's storage across the overlap.

    The overlap changes *how long* a predecessor's state stays live, not who owns it. Every
    assertion here is on a public API:

    bounded  with two runs holding the two sets, a third ``submit`` parks in the orchestrator's
             admission wait and is released when one of them retires. Read from the orchestrator's
             own count of parked callers, which is a positive observable — a caller is counted only
             once the FIFO bound has stopped it. Nothing is inferred from a thread having been
             started, and no fixed delay is asserted.
    owned    a caller device allocation this runtime passes straight through to the device
             (``copy_in_device_args`` forwards an ``is_device_memory()`` tensor unchanged) is
             *borrowed* by the run that names it, so ``Worker.free`` is refused while that run is
             in flight and accepted once it retires. The allocation is still the caller's
             afterwards: the runtime adopts nothing. A caller host buffer an in-flight run named is
             refused by ``Worker.release_buffer`` on the same terms.
    through  a caller host output buffer is this run's D2H destination, so its bytes must still be
             this run's own result when it is read only after the successor has joined and both
             have retired. The join for *these two runs* is established from the children's own
             records first, so a route that fell back to serial submission cannot satisfy it.
    """

    CASES = [
        {
            "name": "a2a3_tmr_early_enqueue_lifetimes",
            "platforms": ["a2a3"],
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
        self.test_a_third_submission_parks_until_one_retires("a2a3", worker)
        self.test_caller_storage_stays_caller_owned_across_the_overlap("a2a3", worker)
        self.test_a_predecessor_output_read_after_the_overlap_is_its_own("a2a3", worker)

    @staticmethod
    def _admission_waiters(worker):
        """How many submissions are parked in this Worker's admission wait at this instant.

        The orchestrator's own count of callers inside ``begin_run`` that the run FIFO's bound has
        stopped. A caller is counted only once the bound has stopped it, so being counted *is*
        having met backpressure — and a counted caller has not reached its graph callback, because
        ``begin_run`` returns before the callback is built.
        """
        orch = worker._orch  # noqa: SLF001 -- the admission boundary is native, and this is its seam
        assert orch is not None, "this Worker has no orchestrator, so it was never initialized"
        return orch._o._begin_run_waiter_count_for_test()  # noqa: SLF001

    def _await_parked_admission(self, worker, timeout):
        """Block until exactly one submission is parked in admission, or report the absence."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            waiters = self._admission_waiters(worker)
            if waiters == 1:
                return True
            assert waiters <= 1, f"{waiters} submissions are parked in admission, and this case made one"
            time.sleep(0.001)
        return False

    @pytest.mark.platforms(["a2a3"])
    def test_a_third_submission_parks_until_one_retires(self, st_platform, st_worker):
        """Admission is bounded at the granted sets, and a retirement is what releases it.

        The two holders carry the long device spin, so the window in which a third caller is parked
        is bounded below by that spin rather than by a sleep this case chose. Never observing the
        park is reported as an evidence failure, not passed over.
        """
        if st_platform != "a2a3":
            pytest.skip("a2a3 TMR early enqueue is gated to the a2a3 onboard tensormap_and_ringbuffer route")
        handles: list = []
        deferred: list = []
        thread_errors: list = []
        submitter = None
        expectations: list = []
        failures: list = []
        with tempfile.TemporaryDirectory(prefix="simpler-a2a3-tmr-backpressure-") as output_prefix:
            try:
                for a_value, b_value in ((2.0, 3.0), (5.0, 7.0)):
                    group, out, expected = self._three_buffers(st_worker, a_value, b_value)
                    expectations.append((out, expected))
                    handles.append(self._submit_vector(st_worker, group, output_prefix, spin_iters=_DEVICE_SPIN_ITERS))

                third_group, third_out, third_expected = self._three_buffers(st_worker, 11.0, 13.0)

                def submit_third():
                    try:
                        deferred.append(self._submit_vector(st_worker, third_group, output_prefix, spin_iters=0))
                    except BaseException as error:  # noqa: BLE001 -- carried back rather than lost in the thread
                        thread_errors.append(error)

                submitter = threading.Thread(target=submit_third, name="a2a3-tmr-third-submit", daemon=True)
                submitter.start()
                parked = self._await_parked_admission(st_worker, _PARK_BUDGET_S)
                assert parked, (
                    f"no submission was ever observed parked in admission within {_PARK_BUDGET_S}s while two runs "
                    f"held the two granted sets, so the bound this case exists to show was never exercised"
                )

                # A run retires on its own; waiting is how this process observes that, not what
                # causes it. What the wait establishes here is that both holders are terminal,
                # which is the only state in which the parked caller can have been admitted.
                for handle in handles:
                    handle.wait()
                submitter.join(timeout=_THREAD_SETTLE_S)
                assert not submitter.is_alive(), "the third submission never returned after the two holders retired"
                assert not thread_errors, f"the third submission raised: {thread_errors}"
                assert deferred, "the third submission returned no handle"
                deferred[0].wait()
                expectations.append((third_out, third_expected))
            finally:
                # Guaranteed, and while the buffers these runs name are still alive.
                failures = _drain_handles(handles, "a2a3 TMR backpressure")
                if submitter is not None:
                    submitter.join(timeout=_THREAD_SETTLE_S)
                    if submitter.is_alive():
                        failures.append("the deferred submission thread never returned")
                    else:
                        failures.extend(_drain_handles(deferred, "a2a3 TMR backpressure deferred"))
                failures.extend(f"the deferred submission raised {error!r}" for error in thread_errors)
        assert not failures, f"this case's runs did not settle: {failures}"
        for out, expected in expectations:
            torch.testing.assert_close(out, expected)

    @pytest.mark.platforms(["a2a3"])
    def test_caller_storage_stays_caller_owned_across_the_overlap(self, st_platform, st_worker):
        """A device allocation and a host buffer a live run names are refused, then released.

        The device allocation is the ``b`` argument of the first run, so the run borrows it and
        computes from its bytes — which is what makes the refusal below about a real consumer
        rather than about a handle the runtime never saw.
        """
        if st_platform != "a2a3":
            pytest.skip("a2a3 TMR early enqueue is gated to the a2a3 onboard tensormap_and_ringbuffer route")
        handles: list = []
        failures: list = []
        device_addend = st_worker.alloc_child_tensor(0, (_SIZE,), DataType.FLOAT32)
        # The device allocation's contents come from a host buffer the caller fills, through the
        # public H2D entry. `copy_to` is the documented way to load an `alloc_child_tensor`.
        addend_source, addend_tensor = self._tensor_from_host_buffer(st_worker, 3.0)
        st_worker.copy_to(device_addend, addend_source)
        first_a, first_a_tensor = self._tensor_from_host_buffer(st_worker, 2.0)
        first_out, first_out_tensor = self._tensor_from_host_buffer(st_worker, 0.0)
        first_expected = self._expected(first_a_tensor, addend_tensor)
        second_group, second_out, second_expected = self._three_buffers(st_worker, 5.0, 7.0)
        with _frame_views(st_worker, 0) as frames:
            with tempfile.TemporaryDirectory(prefix="simpler-a2a3-tmr-owned-") as output_prefix:
                try:
                    handles.append(
                        self._submit_vector(
                            st_worker,
                            (first_a, device_addend, first_out),
                            output_prefix,
                            spin_iters=_DEVICE_SPIN_ITERS,
                        )
                    )
                    # On the device, so the borrow its arguments took is live and the refusals
                    # below are about a run that can still read those bytes.
                    _wait_for_one_launched_frame(frames, 20.0)
                    handles.append(
                        self._submit_vector(st_worker, second_group, output_prefix, spin_iters=_DEVICE_SPIN_ITERS)
                    )

                    with pytest.raises(RuntimeError, match="still referenced by an in-flight"):
                        st_worker.free(device_addend)
                    with pytest.raises(RuntimeError, match="still referenced by an in-flight"):
                        st_worker.release_buffer(first_a)

                    for handle in handles:
                        handle.wait()
                finally:
                    failures = _drain_handles(handles, "a2a3 TMR caller ownership")
        assert not failures, f"this case's runs did not settle: {failures}"

        # Only now, with every run of this case drained: the same allocation the run borrowed is
        # released by the caller, because the caller is still its owner. The runtime adopted
        # nothing, and the refusal above was a window rather than a hold the caller can never
        # discharge. This is before the value comparisons on purpose — an acceptance that only
        # happened on the happy path would not be the ownership claim. Should anything above this
        # line fail instead, the allocation stays with the Worker's own teardown: it is the only
        # owner that knows the runs stopped, and freeing on a path where retirement is unknown is
        # the thing the refusal exists to prevent.
        st_worker.free(device_addend)
        st_worker.release_buffer(first_a)

        # Computed from the bytes the caller put in the device allocation, so the pass-through
        # route carried a real value rather than being merely accepted.
        torch.testing.assert_close(first_out_tensor, first_expected)
        torch.testing.assert_close(second_out, second_expected)

    @pytest.mark.platforms(["a2a3"])
    def test_a_predecessor_output_read_after_the_overlap_is_its_own(self, st_platform, st_worker):
        """A host output buffer still carries its own D2H bytes once its successor has joined and gone.

        The join for *these two runs* is established from the children's own records before the
        values are compared, so a route that fell back to serial submission fails this case instead
        of passing it.
        """
        if st_platform != "a2a3":
            pytest.skip("a2a3 TMR early enqueue is gated to the a2a3 onboard tensormap_and_ringbuffer route")
        handles: list = []
        failures: list = []
        trace = _RunTrace(st_worker)
        pid = self._endpoint_pid(st_worker)
        first_group, first_out, first_expected = self._three_buffers(st_worker, 2.0, 3.0)
        second_group, second_out, second_expected = self._three_buffers(st_worker, 5.0, 7.0)
        with _frame_views(st_worker, 0) as frames:
            case_runs = _CaseRuns(frames)
            stop = threading.Event()

            def sample():
                while not stop.is_set():
                    case_runs.note(_coherent_snapshot(frames))
                    time.sleep(0.001)

            sampler = threading.Thread(target=sample, name="a2a3-tmr-lifetimes-sampler", daemon=True)
            sampler.start()
            with tempfile.TemporaryDirectory(prefix="simpler-a2a3-tmr-lifetimes-") as output_prefix:
                try:
                    handles.append(
                        self._submit_vector(st_worker, first_group, output_prefix, spin_iters=_DEVICE_SPIN_ITERS)
                    )
                    _wait_for_one_launched_frame(frames, 20.0)
                    # Submitted while the predecessor is on the device, and before anything of the
                    # predecessor's is read.
                    handles.append(
                        self._submit_vector(st_worker, second_group, output_prefix, spin_iters=_DEVICE_SPIN_ITERS)
                    )
                    for handle in handles:
                        handle.wait()
                finally:
                    stop.set()
                    sampler.join(timeout=5.0)
                    failures = _drain_handles(handles, "a2a3 TMR lifetimes")
        assert not failures, f"this case's runs did not settle: {failures}"

        # The join for this pair, from the children's own records, before any value is read: a
        # serial fallback records none and fails here.
        records: list[dict] = []
        _await_records(trace, records, _EVIDENCE_BUDGET_S, lambda seen: bool(_established_pairs(seen, case_runs, pid)))
        pairs = _established_pairs(records, case_runs, pid)
        assert pairs, (
            f"no record established a joined pair of this case's own two runs, so the late read below would "
            f"not be reading across an overlap: records={records}, case runs={case_runs.sorted_keys()} above "
            f"floor {case_runs.floor}"
        )

        # Read in reverse: the predecessor last, after its successor had joined, reset the shared
        # arena and retired. A copy-back or shared-arena ordering fault shows up here as the
        # predecessor's buffer carrying the successor's numbers.
        torch.testing.assert_close(second_out, second_expected)
        torch.testing.assert_close(first_out, first_expected)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestA2a3TmrEarlyEnqueueOutstandingClose(_A2a3TmrEarlyEnqueueBase):
    """``close()`` drains a run this case deliberately never waits on.

    Its own class because the observable *is* the teardown: the case ends with a submitted,
    unwaited run still registered on the Worker, and the fixture's ``close()`` is then what has to
    drain it. A close that could not would fail this case in teardown rather than silently leaving
    device work behind.

    The outstanding state is asserted rather than assumed — ``_accepted_run_handles`` is non-empty
    when the case returns — so ``close()`` is known to have had work to do. The run carries no long
    spin: the point is that it is outstanding, not that it is slow.
    """

    CASES = [
        {
            "name": "a2a3_tmr_early_enqueue_outstanding_close",
            "platforms": ["a2a3"],
            "config": {"device_count": 1, "num_sub_workers": 0, "launch_depth": 2},
            "params": {},
        },
    ]

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        del kwargs
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        assert str(worker._config["platform"]) in case["platforms"]  # noqa: SLF001 -- scene-test validation
        self.test_close_drains_a_run_this_case_never_waited_on("a2a3", worker)

    @pytest.mark.platforms(["a2a3"])
    def test_close_drains_a_run_this_case_never_waited_on(self, st_platform, st_worker):
        """Submit two runs, wait neither, and leave the drain to the Worker's own close."""
        if st_platform != "a2a3":
            pytest.skip("a2a3 TMR early enqueue is gated to the a2a3 onboard tensormap_and_ringbuffer route")
        assert st_worker._launch_depth == 2, (  # noqa: SLF001 -- scene-test validation
            f"this class needs a Worker at launch_depth=2, got {st_worker._launch_depth}"  # noqa: SLF001
        )
        # The directory has to outlive this case, because a run still in flight owns the prefix its
        # child is writing to — so it must be one this case does not own. `build_output_prefix`
        # creates a durable, gitignored `outputs/<label>_<timestamp>/` that nothing afterwards
        # removes, which is what makes it safe here and is how every other case's diagnostic
        # artifacts are retained. A `mkdtemp` would either be removed under a live writer or be an
        # unexplained leak.
        output_prefix = str(build_output_prefix("a2a3_tmr_outstanding_close"))
        for a_value, b_value in ((2.0, 3.0), (5.0, 7.0)):
            group, _out, _expected = self._three_buffers(st_worker, a_value, b_value)
            # Deliberately not retained and deliberately not waited: the handles stay registered on
            # the Worker, which is exactly the state close has to drain.
            self._submit_vector(st_worker, group, output_prefix, spin_iters=_DEVICE_SPIN_ITERS)

        outstanding = len(st_worker._accepted_run_handles)  # noqa: SLF001 -- the state close must drain
        assert outstanding >= 1, (
            "no run was left outstanding, so this case's close would have nothing to drain and the "
            "obligation it exists to check would go untested"
        )
