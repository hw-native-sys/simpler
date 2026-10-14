#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Onboard validation for early enqueue on a5, over the public ``Worker.submit`` route.

a5 already prepared a successor's graph and arguments while its predecessor executed, but the
successor's native submission waited for the predecessor to end. At ``launch_depth=2`` it no
longer does. Four things are checked here, on the real submit route, for exactly the runs the
case submitted — the same four the a2a3 class checks, because the claim is the same one:

  admitted   two mailbox frames hold two *different* runs, both TASK_LAUNCHED with both sticky
             acceptance words set. At depth one that state is unreachable at any instant, and not
             for timing reasons: the endpoint negotiates a single task frame, so the parent has
             nowhere to publish a successor while the run ahead of it is live, and a successor's
             launch is not attempted until it is the lane's front. Two coexisting launched runs
             is the admission change itself.
  ordered    for a named (predecessor, successor) pair, the successor's native submission
             *completed* while the identified predecessor's whole-operator boundary was still
             unfired. That places the completed enqueue before the predecessor's completion,
             which no host-side state can — a frame stays TASK_LAUNCHED after the device has
             finished, and a run's output only reaches the host at its finalize.
  serial     and the device then ran that same pair one whole operator at a time, in that order,
             read from two passive device-timestamp markers rather than inferred from host
             overlap: the successor's `aicore_start` against the predecessor's
             `whole_operator_end`. On a5 both markers are new with this capability, and both ride
             the existing ``chip.run.runner_run.device_boundaries`` span.
  retained   both runs then produce their own correct results, so the overlap is a capability
             rather than two runs treading on each other.

**No device stream waits on the markers.** Recording an event is not a wait, so they add no
ordering to the streams they sit in and cannot make a missing production wait look satisfied. The
production completion fence keeps its own completion-only events and its own contract.

Every counted edge belongs entirely to the case that counted it: the records are read only from
this worker's own chip children's log files through a per-file byte cursor opened at each file's
current size, and both ends of a pair must be runs this case submitted, which :class:`_CaseRuns`
decides from a dispatch-id floor taken before the case submits anything.

The evidence reader below is deliberately a copy of the a2a3 class's, not an import of it:
importing a module that defines ``SceneTestCase`` classes would collect those classes here too.
Consolidating the two into one arch-neutral helper is a follow-up, and is noted as such.
"""

import contextlib
import ctypes
import shutil
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
    TensorTransfer,
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

_CHAIN_LENGTH = 512
# Long enough that the host can observe the overlap window, bounded so a regression fails on an
# assertion rather than on the op-execute timeout.
_DEVICE_SPIN_ITERS = 200_000_000
# Shorter per run: the refill case runs sixteen back to back and each only needs to outlast one
# sampling pass rather than a whole host observation window.
_REFILL_SPIN_ITERS = 40_000_000
_SIZE = 128 * 128
#: The task frames a5's endpoint negotiates with the opt-in set: one per run-resource set its
#: published depth of two grants. Each case asserts the count it actually got rather than assuming
#: this one, because the count is what decides whether a successor has anywhere to go.
_FRAME_COUNT = 2
#: The task frames a5's endpoint negotiates when three run-resource sets were requested and
#: granted. The three-run class asserts the count it actually got rather than assuming this one.
_THREE_FRAMES = 3
#: Consecutive submissions the refill case makes.
_REFILL_RUNS = 16
#: How long to keep draining the children's log files after their runs have finished. The writers
#: are asynchronous, so this covers the lag between a record being accepted and written, not the
#: run itself. Exhausting it is an absence, which the caller's assertion reports as one.
_EVIDENCE_BUDGET_S = 10.0
#: How long a case waits for a helper thread it started to return before calling it unsettled.
#: Long enough to cover a submission parked in admission that a retirement has just released,
#: and bounded so a thread that never returns is a reported failure rather than a hang.
_THREAD_SETTLE_S = 60.0
#: Lines of each child's own log a failing numeric check carries, from the end of the sequence.
#: The sixteen-run case emitted 412 lines at the previous head, so a 400-line tail had already
#: begun to cut into the first run's own records — which are the ones a first-run failure needs.
#: Sized to hold that sequence plus the per-overlap and per-publication records with headroom,
#: and still bounded.
_DIAGNOSTIC_LOG_LINES = 900

#: `SIMPLER_ERROR_INVALID_ARGS` from `src/common/host_build_graph/runtime_status.h`, which is the
#: code `require_host_tensor` reports. Named here because the fragments below are what tells this
#: refusal apart from any other preparation failure.
_INVALID_ARGS_CODE = 5
#: One child log line carries all three when a device value reaches a host read while the graph is
#: being built: the entry that made the read, the orchestrator's wrapper, and the refusal itself,
#: verbatim from `src/common/host_build_graph/host/runtime_core.cpp`. `HostLogger::emit` writes
#: `[mono_ns=…][T0x…][ERROR] <func>: <message>`, and `OrchestratorState::report_fatal` supplies
#: `FATAL(code=%d): %s` as that message.
_HOST_TENSOR_REFUSAL = (
    "get_tensor_data",
    f"FATAL(code={_INVALID_ARGS_CODE})",
    "host tensor access requires an explicit HOST/NONE argument",
)

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
                "signature": [D.IN, D.IN, D.OUT, D.IN],
            },
            "incores": [
                {
                    "func_id": 0,
                    "source": f"{_KERNELS}/aiv/delayed_add.cpp",
                    "core_type": "aiv",
                    "signature": [D.IN, D.IN, D.OUT],
                },
                {
                    "func_id": 1,
                    "source": f"{_KERNELS}/aiv/kernel_add_scalar.cpp",
                    "core_type": "aiv",
                    "signature": [D.IN, D.OUT],
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
    args.add_tensor(handles[1].tensor((_SIZE,), DataType.FLOAT32), TensorArgType.INPUT, transfer=TensorTransfer.NONE)
    for value in scalars:
        args.add_scalar(value)
    return args


def _negotiated_frame_count(worker):
    """The task frames this Worker registered its chip endpoint with.

    The child decides the count after init — only an initialized ``ChipWorker`` can answer
    whether its runtime joins native launches — and publishes it; the parent registers the
    endpoint with exactly that number. Reading the parent's record is therefore reading the route
    a case actually runs on, not the frames the mailbox happens to be laid out for.
    """
    counts = worker._chip_task_frame_counts  # noqa: SLF001 -- white-box negotiation observation
    assert len(counts) == 1, f"these cases drive one chip endpoint, got {len(counts)} frame-count records"
    return int(counts[0])


@contextlib.contextmanager
def _frame_views(worker):
    """Each negotiated task frame's ``(address, buffer)``, every view released before this exits.

    A view sliced out of the chip mailbox's shared memory is an *export* of it, and an export is
    exactly what makes ``SharedMemory.close()`` raise ``BufferError``. A failing assertion keeps
    the frame locals of everything on the stack alive into the fixture's ``Worker.close()``, so a
    view this module does not release itself becomes a teardown error that replaces the case's own
    finding. Releasing them here is what keeps a failure readable.

    ``_CaseRuns`` and ``_RunTrace`` deliberately keep none of these: the first stores an integer
    floor and identity tuples, the second file offsets, so a view's lifetime is this scope's alone.
    """
    shm_buf = worker._chip_shms[0].buf  # noqa: SLF001 -- white-box mailbox observation
    assert shm_buf is not None
    # Left unnamed on purpose: a ctypes object built from a buffer holds an export for as long as
    # it lives, so binding it would reintroduce exactly the leak this function exists to close.
    mailbox_addr = ctypes.addressof(ctypes.c_char.from_buffer(shm_buf))
    # One view per run-state entry, which is where a run's whole life is
    # published: the 64 KiB request frames carry only the payload and go back to
    # the parent as soon as the child has copied it out.
    entry_count = mailbox_run_entry_count(_negotiated_frame_count(worker))
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

    Both halves reach the native descriptor unchanged from this frame, so a record's identities
    are comparable with what the parent can see. A frame that has never carried a run reads as
    dispatch zero and is not a key.
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


def _all_distinct_runs_launched(snapshot, expected):
    """Whether the snapshot holds at least `expected` different runs, each launched and accepted.

    Counted over the entries that carry a launched run, never over the whole table. The table is
    `M = granted + K` entries so that a hand-off can be taken before a set frees, so entries
    beyond the runs in flight are idle by construction — a predicate requiring every entry
    launched could never be satisfied at any grant.
    """
    keys = _launched_run_keys(snapshot)
    if keys is None:
        return False
    return len(keys) >= expected


def _two_distinct_runs_launched(snapshot):
    """Whether the snapshot holds two different runs, both launched and both accepted."""
    return _all_distinct_runs_launched(snapshot, _FRAME_COUNT)


def _dispatch_floor(frames):
    """The highest dispatch id this Worker issued before the caller submitted anything.

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
    read what its children wrote, which :meth:`child_log_window` is for — the numeric checks below
    report a mismatch with the child's own account of the run, since nothing else in a CI job log
    carries it.
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

        The emitter writes 0 for an unavailable position and the rc that says why, so a reader
        must check the rc rather than the value — a zero could not otherwise be told from an
        absence.
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

        Every line each chip child wrote since this cursor opened, unfiltered. A child's file
        holds only that child's own records — the parent's spans go to its stderr — so the
        `[STRACE]` lines here are `chip.run.bind`, `chip.run.publish_image`,
        `chip.run.stage_inputs` and the rest of the per-run phases, which is exactly what a
        failure needs and what nothing else in a CI job log carries. Dropping them, as an earlier
        revision did, left a window whose emptiness excluded nothing.

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


def _established_pairs(records, case_runs):
    """The pairs whose ordering the records actually establish.

    A record counts only when the query answered *and* the predecessor's whole-operator boundary
    had not fired: that pair's successor completed its native submission before its predecessor
    completed. A failed query counts for neither side. Both ends must be runs this case submitted,
    and must differ.
    """
    return {
        (record["predecessor"], record["successor"])
        for record in records
        if record.get("observed")
        and record.get("unfired")
        and record["successor"] != record["predecessor"]
        and record["successor"] in case_runs
        and record["predecessor"] in case_runs
    }


def _whole_operator_order(trace, records, case_runs):
    """Split the established pairs by the *whole-operator* relation on the device's own clock.

    Returns ``(ordered, overlapping, within_tick, unmeasured)``:

    * **ordered** — the successor's AICore stream was released strictly after the predecessor's
      own AICore kernel had returned. This is the whole-operator serialization the queued wait
      constructs, measured rather than inferred.
    * **overlapping** — the successor was released before that instant. A real failure of the
      ordering edge.
    * **within_tick** — the two readings are the same tick. The predecessor's marker is queued
      ahead of the boundary event the successor's wait consumes, but the device clock can assign
      those distinct operations one value. This shows no measurable overlap, not strict ordering
      below one tick, so the claim is held against ``ordered | within_tick``.
    * **unmeasured** — a position was unavailable, the runs report different devices, or the
      readings are on different tick rates. Never a pass.
    """
    pids = {record["pid"] for record in records if not record.get("malformed")}
    ordered, overlapping, within_tick, unmeasured = set(), set(), set(), set()
    for pair in _established_pairs(records, case_runs):
        decided = False
        for pid in pids:
            ahead = trace.boundaries_of_run(pid, pair[0])
            behind = trace.boundaries_of_run(pid, pair[1])
            if ahead is None or behind is None:
                continue
            if ahead["device"] != behind["device"] or ahead["hz"] != behind["hz"]:
                continue
            end = ahead["whole_operator_end"]
            start = behind["aicore_start"]
            if end is None or start is None:
                continue
            decided = True
            if start > end:
                ordered.add(pair)
            elif start == end:
                within_tick.add(pair)
            else:
                overlapping.add(pair)
        if not decided:
            unmeasured.add(pair)
    return ordered, overlapping, within_tick, unmeasured


def _boundary_detail(trace, pairs):
    """Per-pair marker readings, for a failure message that names why rather than only that."""
    detail = {}
    for pid in trace.pids:
        for pair in pairs:
            detail[pair] = {
                "predecessor": trace.boundaries_of_run(pid, pair[0]),
                "successor": trace.boundaries_of_run(pid, pair[1]),
            }
    return detail


def _await_records(cursor, records, budget_s, satisfied):
    """Drain into ``records`` until ``satisfied``, or until the budget runs out.

    The child's writer is asynchronous and the parent's flush drains only the parent's own sink,
    so a read taken straight after ``wait()`` is no guarantee the child's record has been
    written. This polls instead, and returns on the budget rather than raising, so the caller's
    own assertion names which half was missing.
    """
    deadline = time.monotonic() + budget_s
    while True:
        records.extend(cursor.take())
        if satisfied(records):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def _await_child_log_line(trace, fragments, budget_s):
    """The first line a chip child wrote since ``trace`` opened that carries every fragment.

    Returns ``(pid, line)``, or ``None`` when the budget ran out. The same asynchrony
    :func:`_await_records` exists for applies here — a child's writer has its own lag — so this
    polls the per-file window rather than reading once.

    The window begins where the cursor was created, so a line an earlier case on the same Worker
    wrote cannot satisfy a later one. Exhausting the budget is an absence, and the caller reports
    it with the window it searched.
    """
    deadline = time.monotonic() + budget_s
    while True:
        for pid, lines in trace.child_log_window().items():
            for line in lines:
                if all(fragment in line for fragment in fragments):
                    return pid, line
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.02)


def _wait_for_one_launched_frame(frames, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = _coherent_snapshot(frames)
        if snapshot is not None and any(state == _TASK_LAUNCHED for _, state, _ in snapshot):
            return
        time.sleep(0.001)
    raise AssertionError("no run reached its device launch fence")


def _admission_waiters(worker):
    """How many submissions are parked in this Worker's admission wait at this instant.

    The orchestrator's own count of callers inside ``begin_run`` that the run FIFO's bound has
    stopped. It is a positive observable of backpressure, which a started thread is not: a thread
    that has merely been started may be anywhere, while a thread counted here is inside admission
    and has therefore not reached its graph callback — ``begin_run`` returns before the callback is
    built.
    """
    orch = worker._orch  # noqa: SLF001 -- the admission boundary is native, and this is its seam
    assert orch is not None, "this Worker has no orchestrator, so it was never initialized"
    return orch._o._begin_run_waiter_count_for_test()  # noqa: SLF001


def _live_run_handles(worker):
    """Every run this Worker still counts as accepted and un-retired, at one instant.

    A run joins this set when its submission is admitted and leaves it in ``RunHandle.wait()``,
    on the thread that calls that — nothing retires a run in the background. So for a caller that
    has waited nothing, this is exactly the set of submissions it has made.

    What that is evidence *of* is the parent's bookkeeping: a handle here still names the run, so
    the run's argument references are still borrowed. It is **not** a count of occupied native
    resource sets — a run's native side can have retired while the handle that names it has not
    been waited, so the two numbers are not interchangeable.

    Read under the lifecycle lock the Worker publishes those transitions beneath, so the list is
    one instant rather than a walk across several.
    """
    with worker._hierarchical_start_cv:  # noqa: SLF001 -- the lifecycle lock the refusal scan takes
        return [handle for handle in worker._accepted_run_handles if not handle._cleanup_published]  # noqa: SLF001


def _live_runs_naming(worker, identity):
    """The ``id()`` of every live run whose declared arguments name ``identity``.

    The same scan ``Worker.free`` runs, over the same two collections and under the same lock, so
    a non-empty answer here entails the refusal rather than predicting it: a run is a borrower
    exactly while it is accepted with its cleanup unpublished, or abandoned.

    Identities rather than handles, so a caller can match its own submissions without holding a
    second reference to anything the Worker owns.
    """
    with worker._hierarchical_start_cv:  # noqa: SLF001 -- the lifecycle lock the refusal scan takes
        live = {
            id(handle)
            for handle in worker._accepted_run_handles  # noqa: SLF001
            if not handle._cleanup_published and identity in handle._resources.touched_identities
        }
        live |= {
            id(handle)
            for handle in worker._abandoned_run_handles  # noqa: SLF001
            if identity in handle._resources.touched_identities
        }
    return live


def _drain_handles(handles, what):
    """Wait for every submitted run before the caller's resources leave scope.

    No failure is raised from here, so a run that failed cannot replace the caller's own finding
    while it is unwinding. The failures are returned instead: a caller that reaches its normal
    end with a non-empty list has no primary error to preserve and reports these itself.
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
    log_window = "\n".join(
        f"--- child {pid} ---\n" + "\n".join(lines) for pid, lines in trace.child_log_window().items()
    )
    raise AssertionError(
        f"{what}: {len(mismatches)} of {len(expectations)} runs produced the wrong output. {summary}.\n"
        f"This case's runs (dispatch id, slot) above floor {case_runs.floor}: {runs}\n"
        f"Device boundary markers per child: {boundaries}\n"
        f"First failure in full:\n{mismatches[0][1]}\n"
        f"Children's own log window:\n{log_window}"
    )


class _A5EarlyEnqueueBase(SceneTestCase):
    """The two-run sequence, shared by the opt-in class and its depth-one control."""

    CALLABLE = _CALLABLES

    def test_run(self):
        """Not the standard single-case golden path.

        These classes have no single orchestration callback and no generated argument set: each
        drives its own multi-run sequence and checks the results itself. Requesting no fixtures
        keeps the skip from allocating a device for nothing.
        """
        pytest.skip("early enqueue drives its own multi-run sequences; see the cases below")

    @staticmethod
    def _tensor_from_host_buffer(worker, value):
        buffer = worker.create_buffer(_SIZE * torch.float32.itemsize)
        tensor = torch.frombuffer(buffer.shm.buf, dtype=torch.float32, count=_SIZE)
        tensor.fill_(value)
        return buffer, tensor

    def _submit_vector(self, worker, arg_buffers, output_prefix, *, spin_iters=0):
        vector_handle = type(self)._st_chip_handles["vector"]
        vector_signature = type(self)._st_chip_handles["vector_sig"]
        # A non-empty output prefix is what makes the chip child bind its host log to a file
        # instead of leaving it on the inherited stderr, which is what gives the joined-launch
        # spans a destination this process can read by child pid. No diagnostic flag is set, so
        # nothing else is written there.
        config = self._build_config(self.CASES[0]["config"], output_prefix=output_prefix)

        def graph(orch, _args, _cfg):
            chip_args = _chip_args(arg_buffers, vector_signature, spin_iters)
            orch.submit_next_level(vector_handle, chip_args, config, worker=0)

        return worker.submit(graph)

    @staticmethod
    def _expected(a, b):
        """The chain's result for one run's inputs, read before the run overwrites its output."""
        return a + b + _CHAIN_LENGTH * b[0]

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


@scene_test(level=3, runtime="host_build_graph")
class TestA5EarlyEnqueueDepthTwo(_A5EarlyEnqueueBase):
    """At ``launch_depth=2`` a successor's work reaches a5's device while its predecessor runs."""

    CASES = [
        {
            "name": "a5_early_enqueue",
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
            pytest.skip("a5 early enqueue is gated to the a5 onboard host_build_graph route")
        # The route this case needs: a second task frame for a successor to occupy while the run
        # ahead of it is live. The child negotiates it only where its runtime resolved that one
        # native submission may be ordered behind another, so this is the opt-in reaching the
        # endpoint rather than a mailbox layout constant.
        negotiated = _negotiated_frame_count(st_worker)
        assert negotiated == _FRAME_COUNT, (
            f"a5 negotiated {negotiated} task frame(s) at launch_depth=2, so no successor can occupy one "
            f"while its predecessor runs and the overlap below is unreachable by construction"
        )
        trace = _RunTrace(st_worker)
        with _frame_views(st_worker) as frames:
            case_runs = _CaseRuns(frames)
            with tempfile.TemporaryDirectory(prefix="simpler-a5-early-enqueue-") as output_prefix:
                handles: list = []
                buffers: list = []
                expectations: list = []
                try:
                    tensors = []
                    for value in (2.0, 3.0, 0.0, 5.0, 7.0, 0.0):
                        buffer, tensor = self._tensor_from_host_buffer(st_worker, value)
                        buffers.append(buffer)
                        tensors.append(tensor)
                    for index in range(2):
                        group = buffers[index * 3 : index * 3 + 3]
                        a, b, out = tensors[index * 3 : index * 3 + 3]
                        expectations.append((out, self._expected(a, b)))
                        handles.append(
                            self._submit_vector(st_worker, group, output_prefix, spin_iters=_DEVICE_SPIN_ITERS)
                        )
                        if index == 0:
                            _wait_for_one_launched_frame(frames, 20.0)
                    together = self._observe_two_launched_runs(frames, 60.0, case_runs)
                finally:
                    # Before anything of this case's leaves scope, whatever the submission or the
                    # observation did: a run still in flight owns its arguments and its output
                    # buffer, and the temporary output directory is this case's too.
                    drain_failures = _drain_handles(handles, "a5 two-run window")
                # Reached only with no primary error, so these have nothing to hide behind.
                assert not drain_failures, f"a run of this case did not finish: {drain_failures}"
                _assert_results(expectations, trace, case_runs, "a5 two-run window")

                assert together, (
                    "two different runs were never observed launched and accepted at the same time, so a "
                    "successor's submission never reached the device while its predecessor was executing"
                )

                records: list[dict] = []
                _await_records(
                    trace,
                    records,
                    _EVIDENCE_BUDGET_S,
                    lambda seen: bool(_established_pairs(seen, case_runs)),
                )
                assert records, (
                    f"no joined-launch record reached this process from children {trace.pids}, so nothing "
                    f"carried the ordering of the two accepted runs"
                )
                pairs = _established_pairs(records, case_runs)
                assert pairs, (
                    f"no record established a pair of this case's runs: records={records}, "
                    f"case runs={case_runs.sorted_keys()} above floor {case_runs.floor}"
                )

                ordered, overlapping, within_tick, unmeasured = _whole_operator_order(trace, records, case_runs)
                assert not overlapping, (
                    f"a joined successor's AICore stream was released before its predecessor's whole operator "
                    f"had finished: {sorted(overlapping)}; {_boundary_detail(trace, overlapping)}"
                )
                assert ordered | within_tick, (
                    f"no established pair has comparable whole-operator boundary readings: "
                    f"unmeasured={sorted(unmeasured)}; {_boundary_detail(trace, pairs)}"
                )

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
            pytest.skip("a5 early enqueue is gated to the a5 onboard host_build_graph route")
        negotiated = _negotiated_frame_count(st_worker)
        assert negotiated == _FRAME_COUNT, (
            f"a5 negotiated {negotiated} task frame(s) at launch_depth=2, so the refill below has only one "
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
                with tempfile.TemporaryDirectory(prefix="simpler-a5-early-enqueue-refill-") as output_prefix:
                    expectations = []
                    try:
                        for index in range(_REFILL_RUNS):
                            tensors = []
                            group = []
                            for value in (float(index + 2), float(index + 3), 0.0):
                                buffer, tensor = self._tensor_from_host_buffer(st_worker, value)
                                buffers.append(buffer)
                                group.append(buffer)
                                tensors.append(tensor)
                            a, b, out = tensors
                            expectations.append((out, self._expected(a, b)))
                            handles.append(
                                self._submit_vector(st_worker, group, output_prefix, spin_iters=_REFILL_SPIN_ITERS)
                            )
                    finally:
                        # Every submitted run is drained before this case's buffers or its output
                        # directory leave scope, whether the loop finished or a submission failed
                        # part way.
                        drain_failures = _drain_handles(handles, "a5 sixteen-run refill")
                    assert not drain_failures, f"a run of this case did not finish: {drain_failures}"
                    _assert_results(expectations, trace, case_runs, "a5 sixteen-run refill")
            finally:
                # The sampler reads the frame views, so it stops before they are released — which
                # is this scope's exit, whether the body finished or raised.
                stop.set()
                sampler.join(timeout=5.0)

        assert len(handles) == _REFILL_RUNS, f"only {len(handles)} of {_REFILL_RUNS} runs were submitted"
        assert widest == _FRAME_COUNT, (
            f"the widest instant held {widest} launched run(s), so two runs were never in flight over the "
            f"two sets a5 grants: per-set dispatch ids {dispatches_by_slot}"
        )
        frontiers = [max(dispatch_id for dispatch_id, _ in pair) for pair in pairs_seen]
        assert len(pairs_seen) >= 2 and max(frontiers) > min(frontiers), (
            f"two runs were never seen in flight together a second time over later identities, so the "
            f"second set was filled once rather than refilled: frontiers={frontiers}, "
            f"pairs={[sorted(pair) for pair in pairs_seen]}"
        )
        regressed = {slot: ids for slot, ids in dispatches_by_slot.items() if ids != sorted(ids)}
        assert not regressed, (
            f"a resource set was seen carrying an earlier run after a later one, so the sets are not "
            f"passing from one run to its successor: {regressed}"
        )


@scene_test(level=3, runtime="host_build_graph")
class TestA5EarlyEnqueueDepthOneControl(_A5EarlyEnqueueBase):
    """With ``launch_depth`` unset, a5 keeps the serial path: no successor reaches the device.

    The control reads the route it is actually on. With the opt-in unset a5's child negotiates one
    task frame and its endpoint runs the single-frame loop, which publishes ``_TASK_READY`` then
    ``_TASK_DONE`` and no ``_TASK_LAUNCHED`` at all — so the two-frames-at-one-instant observation
    the opt-in class makes is not merely absent here, it is not a state this route publishes. The
    same claim is therefore established from what this route does publish: the negotiated frame
    count, and the absence of any joined-launch record for runs this case is shown to have run.
    """

    CASES = [
        {
            "name": "a5_early_enqueue_depth_one",
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
            pytest.skip("a5 early enqueue is gated to the a5 onboard host_build_graph route")
        assert st_worker._launch_depth == 1, (  # noqa: SLF001 -- scene-test validation
            f"this control needs a Worker at the default launch_depth, got {st_worker._launch_depth}"  # noqa: SLF001
        )
        # One frame is the whole reason a successor cannot reach the device by default: with a
        # single frame the parent has nowhere to publish a second run while the first is live, so
        # the second is dispatched only once the first has left. The opt-in class asserts the
        # other value of this same negotiated number.
        negotiated = _negotiated_frame_count(st_worker)
        assert negotiated == 1, (
            f"the default negotiated {negotiated} task frames on a5, so this control is not observing the "
            f"serial route it exists to pin"
        )
        trace = _RunTrace(st_worker)
        stop = threading.Event()
        with _frame_views(st_worker) as frames:
            case_runs = _CaseRuns(frames)

            def sample():
                while not stop.is_set():
                    case_runs.note(_coherent_snapshot(frames))
                    time.sleep(0.001)

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            try:
                with tempfile.TemporaryDirectory(prefix="simpler-a5-early-enqueue-control-") as output_prefix:
                    handles: list = []
                    buffers: list = []
                    expectations: list = []
                    try:
                        tensors = []
                        for value in (2.0, 3.0, 0.0, 5.0, 7.0, 0.0):
                            buffer, tensor = self._tensor_from_host_buffer(st_worker, value)
                            buffers.append(buffer)
                            tensors.append(tensor)
                        for index in range(2):
                            group = buffers[index * 3 : index * 3 + 3]
                            a, b, out = tensors[index * 3 : index * 3 + 3]
                            expectations.append((out, self._expected(a, b)))
                            handles.append(
                                self._submit_vector(st_worker, group, output_prefix, spin_iters=_DEVICE_SPIN_ITERS)
                            )
                    finally:
                        drain_failures = _drain_handles(handles, "a5 depth-one control")
                    assert not drain_failures, f"a run of this case did not finish: {drain_failures}"
                    _assert_results(expectations, trace, case_runs, "a5 depth-one control")
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
        # Drained on the budget rather than asserted immediately: an absence has to be given the
        # same window a presence would get.
        records: list[dict] = []
        _await_records(trace, records, _EVIDENCE_BUDGET_S, lambda seen: bool(_established_pairs(seen, case_runs)))
        assert not _established_pairs(records, case_runs), (
            f"a joined launch was established with launch_depth unset: {records}"
        )


@scene_test(level=3, runtime="host_build_graph")
class TestA5ThreeRunCapacity(_A5EarlyEnqueueBase):
    """At ``pipeline_depth=3`` a third run reaches a5's device while the first is still executing.

    ``launch_depth`` alone cannot produce this: a launched run holds its resource set for its whole
    life, so with two sets the third submission waits for a retirement however deep the launch
    budget is. Requesting three sets is what lets the third run be prepared and accepted, and the
    device still runs one whole operator at a time — each launch is queued behind the boundary of
    the run immediately ahead of it.

    What each check can see:

    granted    the endpoint negotiated three task frames, so a third run has somewhere to go.
    accepted   three different runs were observed launched and accepted at one instant, read from
               a coherent mailbox snapshot, while none of them had finished.
    ordered    the whole-operator boundaries the children recorded do not overlap, so three
               accepted runs did not become three concurrent operators.
    distinct   every run's own output is correct, so the three sets held three runs' parameters,
               graphs and results rather than sharing any of them.
    refilled   a later instant showed a different triple, which is a set freed by one retirement
               carrying a later run rather than one opening window.
    bounded    a device value in the host build's control position is refused rather than served,
               and the explicit host copy the contract names is accepted — so the capacity does not
               quietly widen what an early preparation may read.
    """

    CASES = [
        {
            "name": "a5_three_run_capacity",
            "platforms": ["a5"],
            "config": {
                "device_count": 1,
                "num_sub_workers": 0,
                "launch_depth": 3,
                "pipeline_depth": 3,
            },
            "params": {},
        },
    ]

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        del kwargs
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        assert str(worker._config["platform"]) in case["platforms"]  # noqa: SLF001 -- scene-test validation
        self._require_three_sets(worker)
        self.test_three_runs_are_launched_and_accepted_at_once("a5", worker)
        self.test_runs_keep_refilling_the_third_set("a5", worker)
        self.test_a_device_result_feeds_the_next_two_runs("a5", worker)
        self.test_a_device_control_tensor_is_refused_until_it_is_copied_to_the_host("a5", worker)

    @staticmethod
    def _require_three_sets(worker):
        # Both paths give this class its own Worker: pytest builds one per class, and the
        # standalone path partitions its groups by requested capacity. A Worker here at any other
        # budget means this case is reading a Worker that is not the one it asked for.
        assert worker._launch_depth == 3, (  # noqa: SLF001 -- scene-test validation
            f"this class needs a Worker at launch_depth=3, got {worker._launch_depth}"  # noqa: SLF001
        )
        assert worker._pipeline_depth_request == 3, (  # noqa: SLF001 -- scene-test validation
            f"this class needs a Worker at pipeline_depth=3, got {worker._pipeline_depth_request}"  # noqa: SLF001
        )
        # The grant, not the request: a declaration that stayed at two would leave this class
        # asserting a capacity the child never gave it.
        negotiated = _negotiated_frame_count(worker)
        assert negotiated == _THREE_FRAMES, (
            f"the a5 endpoint negotiated {negotiated} task frames for a three-set request, so the third set "
            f"was never granted and nothing below could observe it"
        )

    def _submit_three(self, worker, output_prefix, spin_iters, frames, handles, buffers):
        """Three runs with no wait between them, each with its own buffers and expectation.

        ``handles`` and ``buffers`` belong to the caller and are appended to as each run is made,
        so a submission that fails part way still leaves the caller holding every run already in
        flight and the memory those runs are reading.
        """
        tensors = []
        for value in (2.0, 3.0, 0.0, 5.0, 7.0, 0.0, 11.0, 13.0, 0.0):
            buffer, tensor = self._tensor_from_host_buffer(worker, value)
            buffers.append(buffer)
            tensors.append(tensor)
        expectations = []
        for index in range(3):
            group = buffers[index * 3 : index * 3 + 3]
            a, b, out = tensors[index * 3 : index * 3 + 3]
            expectations.append((out, self._expected(a, b)))
            handles.append(self._submit_vector(worker, group, output_prefix, spin_iters=spin_iters))
            if index == 0:
                # The first run has to be on the device before the next two are made, or all three
                # could be admitted into an idle pipeline and nothing would be held open.
                _wait_for_one_launched_frame(frames, 20.0)
        return expectations

    def _observe_three_launched_runs(self, frames, timeout, case_runs):
        """Watch for three different runs launched and accepted at once.

        Returns the keys of the runs seen together, or None when the window closed without three
        of them ever being visible at one instant. Every sampled frame's runs are recorded in
        ``case_runs`` so the ordering records can be held to this case's own runs.
        """
        deadline = time.monotonic() + timeout
        saw_any_launched = False
        while time.monotonic() < deadline:
            snapshot = _coherent_snapshot(frames)
            case_runs.note(snapshot)
            if _all_distinct_runs_launched(snapshot, _THREE_FRAMES):
                return _launched_run_keys(snapshot)
            if snapshot is not None:
                launched = any(state == _TASK_LAUNCHED for _, state, _ in snapshot)
                if launched:
                    saw_any_launched = True
                elif saw_any_launched:
                    return None
            time.sleep(0.001)
        raise AssertionError(f"no run stayed launched long enough to sample within {timeout}s")

    @pytest.mark.platforms(["a5"])
    def test_three_runs_are_launched_and_accepted_at_once(self, st_platform, st_worker):
        """Three accepted submissions, one device order, three correct results."""
        if st_platform != "a5":
            pytest.skip("a5 three-run capacity is gated to the a5 onboard host_build_graph route")
        self._require_three_sets(st_worker)
        trace = _RunTrace(st_worker)
        with _frame_views(st_worker) as frames:
            case_runs = _CaseRuns(frames)
            with tempfile.TemporaryDirectory(prefix="simpler-a5-three-run-") as output_prefix:
                # Held by this scope, not by the helper: a submission that fails after making one
                # or two of the runs still leaves them drainable here, with their arguments alive.
                handles: list = []
                buffers: list = []
                expectations: list = []
                try:
                    expectations = self._submit_three(
                        st_worker, output_prefix, _DEVICE_SPIN_ITERS, frames, handles, buffers
                    )
                    together = self._observe_three_launched_runs(frames, 60.0, case_runs)
                finally:
                    # Before anything of this case's leaves scope, whatever the submission or the
                    # observation did: a run still in flight owns its arguments and its output
                    # buffer, and the temporary output directory is this case's too.
                    drain_failures = _drain_handles(handles, "a5 three-run window")
                # Reached only with no primary error, so these have nothing to hide behind.
                assert not drain_failures, f"a run of this case did not finish: {drain_failures}"
                _assert_results(expectations, trace, case_runs, "a5 three-run window")

                assert together is not None, (
                    "three different runs were never observed launched and accepted at the same time, so a "
                    "third resource set never carried a run"
                )
                assert len(together) == _THREE_FRAMES, f"fewer than three runs were named together: {sorted(together)}"

        # What the device then did with the pairs the children recorded: three accepted runs must
        # not have become three concurrent operators.
        records: list[dict] = []
        _await_records(
            trace,
            records,
            _EVIDENCE_BUDGET_S,
            lambda seen: len(_established_pairs(seen, case_runs)) >= 2,
        )
        assert records, (
            f"no joined-launch record reached this process from children {trace.pids}, so nothing carried the "
            f"ordering of the three accepted runs"
        )
        ordered, overlapping, within_tick, unmeasured = _whole_operator_order(trace, records, case_runs)
        assert not overlapping, (
            f"a joined successor's AICore stream was released before its predecessor's whole operator had "
            f"finished: {sorted(overlapping)}; {_boundary_detail(trace, overlapping)}"
        )
        # Both adjacent pairs, not just one: three accepted runs are only shown serial if the
        # second was ordered behind the first *and* the third behind the second.
        comparable = ordered | within_tick
        assert len(comparable) >= 2, (
            f"the three accepted runs were not shown ordered end to end: comparable={sorted(comparable)}, "
            f"unmeasured={sorted(unmeasured)}; {_boundary_detail(trace, _established_pairs(records, case_runs))}"
        )
        assert len({key for pair in comparable for key in pair}) == _THREE_FRAMES, (
            f"the ordered pairs do not span all three runs: comparable={sorted(comparable)}"
        )

    @pytest.mark.platforms(["a5"])
    def test_runs_keep_refilling_the_third_set(self, st_platform, st_worker):
        """Consecutive submissions that keep all three sets occupied as they turn over.

        Correct results alone would pass under ordinary serial execution, so they are not the
        evidence. What is: two different instants each showed three different runs launched and
        accepted at once, over this case's own dispatch floor, so a set freed by an earlier
        retirement carried a later run.

        No count here bounds concurrency from above: the mailbox this reads has three frames, so a
        fourth simultaneous run would not be visible in it at all.
        """
        if st_platform != "a5":
            pytest.skip("a5 three-run capacity is gated to the a5 onboard host_build_graph route")
        self._require_three_sets(st_worker)
        trace = _RunTrace(st_worker)
        stop = threading.Event()
        triples: list[frozenset] = []
        with _frame_views(st_worker) as frames:
            case_runs = _CaseRuns(frames)

            def sample():
                while not stop.is_set():
                    snapshot = _coherent_snapshot(frames)
                    case_runs.note(snapshot)
                    if _all_distinct_runs_launched(snapshot, _THREE_FRAMES):
                        keys = frozenset(_launched_run_keys(snapshot))
                        if keys not in triples:
                            triples.append(keys)
                    time.sleep(0.001)

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            try:
                with tempfile.TemporaryDirectory(prefix="simpler-a5-three-run-refill-") as output_prefix:
                    handles: list = []
                    buffers: list = []
                    expectations: list = []
                    try:
                        for index in range(_REFILL_RUNS):
                            group = []
                            tensors = []
                            for value in (float(index + 2), float(index + 3), 0.0):
                                buffer, tensor = self._tensor_from_host_buffer(st_worker, value)
                                buffers.append(buffer)
                                group.append(buffer)
                                tensors.append(tensor)
                            a, b, out = tensors
                            expectations.append((out, self._expected(a, b)))
                            handles.append(
                                self._submit_vector(st_worker, group, output_prefix, spin_iters=_REFILL_SPIN_ITERS)
                            )
                    finally:
                        drain_failures = _drain_handles(handles, "a5 three-run refill")
                    assert not drain_failures, f"a run of this case did not finish: {drain_failures}"
                    _assert_results(expectations, trace, case_runs, "a5 three-run refill")
            finally:
                # The sampler reads the frame views, so it stops before they are released.
                stop.set()
                sampler.join(timeout=5.0)

        assert triples, (
            f"over {_REFILL_RUNS} consecutive runs no instant ever showed three different runs launched and "
            f"accepted together above dispatch floor {case_runs.floor}, so the third set never carried a run"
        )
        assert len(triples) >= 2, (
            f"only one distinct triple was ever seen ({[sorted(keys) for keys in triples]}), so the depth was "
            f"reached once rather than refilled as runs retired"
        )

    @staticmethod
    def _link(value, control):
        """One run's arithmetic: the orchestration adds the control tensor, then its first element
        once per chain step. Composing this is what a chained device result has to show."""
        return value + control + _CHAIN_LENGTH * control

    def _submit_link(self, worker, source, control, target, output_prefix, *, spin_iters):
        """One link, ``target = f(source, control)``, built exactly as the host cases build theirs.

        ``source`` and ``target`` are whichever handles the caller passes — a host buffer for the
        chain's first input and its device allocations for everything after it.
        """
        return self._submit_vector(worker, (source, control, target), output_prefix, spin_iters=spin_iters)

    @staticmethod
    def _settle_chain(
        worker, handles, intermediates, what, *, deferred=(), submitter=None, thread_errors=(), tolerated=()
    ):
        """Drain everything this case submitted, settle its thread, and report what is unproven.

        Called from a ``finally``, so it raises nothing: a run that failed must not replace the
        assertion that was unwinding. Everything it could not establish is returned as text, which
        a caller reaching its normal end reports itself.

        The order is the only one that can terminate. A submission thread parked in admission is
        released by a *retirement*, so the already-admitted runs are drained first; joining before
        that would time out by construction.

        It frees nothing. The caller's device allocations are released only where safe completion
        has been established — after this returns clean and after the caller has finished reading
        them — so a drain that could not prove completion leaves them held and says so. `worker`
        is taken for that report's sake, not to release anything.

        ``tolerated`` names runs whose terminal refusal the caller has **already verified** — both
        that it arrived at the handle and that the child reported the specific refusal the case is
        about. Those are waited once so none is left in flight, and that one wait's failure is not
        reported again. A run whose refusal was not verified does not belong here: the caller
        passes it in ``handles`` instead, where an unexpected failure or a timeout is reported and
        the allocations are retained. Nothing is retried, here or anywhere in this helper.
        """
        del worker
        problems = list(_drain_handles(handles, what))
        for handle in tolerated:
            with contextlib.suppress(Exception):
                handle.wait()
        settled = True
        if submitter is not None:
            submitter.join(timeout=_THREAD_SETTLE_S)
            if submitter.is_alive():
                settled = False
                problems.append(
                    f"{what}: the deferred submission thread never returned within {_THREAD_SETTLE_S}s, so it "
                    f"may still submit into this Worker and nothing it owns can be accounted for"
                )
        if settled:
            # Only once the thread has returned is this list final. A still-running submitter can
            # append to it, and draining a list that is still growing proves nothing about it.
            problems.extend(_drain_handles(list(deferred), f"{what} deferred submission"))
        problems.extend(f"{what}: the deferred submission raised {error!r}" for error in list(thread_errors))
        live = [handle for handle in intermediates if handle is not None]
        if problems and live:
            problems.append(
                f"{what}: {len(live)} device allocation(s) retained — no completion proof for the runs that "
                f"may still name them"
            )
        return problems

    @staticmethod
    def _report_cleanup(problems, what):
        """Put the cleanup findings somewhere a *failing* case shows them.

        A caller that reaches its normal end asserts on these itself. A caller unwinding a primary
        failure never reaches that assert, and must not have it replace the failure either — so the
        findings are printed, which pytest attaches to the report of whichever case failed.
        """
        if problems:
            print(f"[{what} cleanup] " + "; ".join(problems))

    @staticmethod
    def _await_parked_admission(worker, timeout, what):
        """Block until exactly one submission is parked in admission.

        Positive evidence, and the only kind available here: a caller is counted only once the run
        FIFO's bound has stopped it, so being counted *is* having met backpressure. Nothing is
        inferred from a thread having been started, and nothing is inferred from a flag staying
        clear for a while.

        Exactly one, not at least one: this case has a single deferred submission, so a second
        waiter would mean something else is submitting into the Worker it is reading.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            waiters = _admission_waiters(worker)
            if waiters == 1:
                return
            assert waiters <= 1, f"{what}: {waiters} submissions are parked in admission, and this case made one"
            time.sleep(0.001)
        raise AssertionError(
            f"{what}: no submission was ever observed parked in admission within {timeout}s, so the "
            f"capacity bound a fourth run meets was never exercised"
        )

    @staticmethod
    def _observe_triple_while_admission_parked(worker, frames, case_runs, timeout, what):
        """The three runs seen launched and accepted at one instant, with the fourth still parked.

        The parked reading brackets the snapshot — taken once before it and once after — and this
        case has a single deferred submission, which parks at most once and never parks again after
        it is released. One submission counted on both sides of the snapshot is therefore the same
        park spanning it, so the conjunction holds over the interval rather than over two readings a
        retirement could have fallen between. No new atomicity is involved: both halves are readings
        that already exist.

        Never observing the conjunction is an evidence failure, not a pass.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _admission_waiters(worker) != 1:
                time.sleep(0.001)
                continue
            snapshot = _coherent_snapshot(frames)
            case_runs.note(snapshot)
            if _all_distinct_runs_launched(snapshot, _THREE_FRAMES) and _admission_waiters(worker) == 1:
                return _launched_run_keys(snapshot)
            time.sleep(0.001)
        raise AssertionError(
            f"{what}: three distinct runs were never seen launched and accepted across an interval in "
            f"which the fourth submission stayed parked in admission, within {timeout}s"
        )

    @pytest.mark.platforms(["a5"])
    def test_a_device_result_feeds_the_next_two_runs(self, st_platform, st_worker):
        """``A(x) -> y -> B(y) -> z -> C(z)`` at three sets, with the fourth held back.

        The same capability as the host triple above, with the intermediates staying on the
        device: each successor names its predecessor's output by device address alone, so the
        caller neither waits nor copies between links.

        What each check can see, and what it cannot:

        chained     the final value is the arithmetic of all three links composed, and each
                    intermediate is read back separately. Composition is the only way the last
                    value can be right, which is what makes this a chain rather than three runs
                    that happened to agree.
        on device   ``y``, ``z`` and ``w`` are ``alloc_child_tensor`` allocations; the only host
                    transfers are the initial fill and the closing read-backs.
        held        while a run naming ``y`` is in flight, ``Worker.free(y)`` is refused, and the
                    same address releases cleanly once the chain has finished. That pair is the
                    borrow — the caller keeps the right to release, and only *now* is unsafe.
        backpressed a fourth submission, made from its own thread, was observed *parked in the
                    orchestrator's admission wait* across an interval in which three distinct runs
                    held the three granted sets — and its graph callback had run by the time they
                    had all retired. Both halves are positive readings. What this does **not**
                    establish is which retirement released it, or the instant it was released: that
                    ordering is proven deterministically in
                    ``tests/ut/cpp/common/hierarchical/test_three_run_capacity.cpp``, where
                    admission is driven synchronously with no device in it.
        ordered     the device still ran one whole operator at a time, over **both** adjacent
                    edges of this chain's three runs; the deferred fourth run is excluded by
                    identity, so a two-run chain or an unrelated pair cannot satisfy it.

        The orchestration reads its control tensor's first element on the host, so that argument
        stays a host buffer here. A device tensor in that position is refused outright, which is
        the case below.
        """
        if st_platform != "a5":
            pytest.skip("a5 three-run capacity is gated to the a5 onboard host_build_graph route")
        self._require_three_sets(st_worker)
        worker = st_worker
        trace = _RunTrace(worker)

        x_buffer, _ = self._tensor_from_host_buffer(worker, 2.0)
        control_buffer, _ = self._tensor_from_host_buffer(worker, 0.5)
        readback_buffer, readback = self._tensor_from_host_buffer(worker, 0.0)
        # Every intermediate up front, which is what lets the chain pass addresses: no allocation
        # happens between submissions.
        intermediates = [worker.alloc_child_tensor(0, (_SIZE,), DataType.FLOAT32) for _ in range(3)]
        y, z, w = intermediates
        fourth_callback_ran = threading.Event()
        handles: list = []
        fourth: list = []
        thread_errors: list = []
        submitter = None
        cleanup_problems: list = []
        together = None
        output_prefix = tempfile.mkdtemp(prefix="simpler-a5-three-run-device-")
        with _frame_views(worker) as frames:
            case_runs = _CaseRuns(frames)
            stop = threading.Event()

            def sample():
                while not stop.is_set():
                    case_runs.note(_coherent_snapshot(frames))
                    time.sleep(0.001)

            sampler = threading.Thread(target=sample, name="a5-device-chain-sampler", daemon=True)
            sampler.start()
            try:
                # Appended one at a time: a submission that raises after an earlier one succeeded
                # must still leave the earlier run reachable for the drain below.
                handles.append(
                    self._submit_link(worker, x_buffer, control_buffer, y, output_prefix, spin_iters=_DEVICE_SPIN_ITERS)
                )
                handles.append(
                    self._submit_link(worker, y, control_buffer, z, output_prefix, spin_iters=_DEVICE_SPIN_ITERS)
                )
                handles.append(
                    self._submit_link(worker, z, control_buffer, w, output_prefix, spin_iters=_DEVICE_SPIN_ITERS)
                )

                # A launched run names `y`, so the caller's release of it is refused. Taken before
                # any wait, which is the only point at which that window is open.
                _wait_for_one_launched_frame(frames, 30.0)
                with pytest.raises(RuntimeError, match="still referenced by an in-flight"):
                    worker.free(y)

                # The fourth submission is made from a thread because its graph callback is what
                # must not begin: with three sets occupied, admission blocks before invoking it.
                def submit_fourth():
                    try:

                        def graph(orch, _args, _cfg):
                            fourth_callback_ran.set()
                            chip_args = _chip_args(
                                (x_buffer, control_buffer, readback_buffer),
                                type(self)._st_chip_handles["vector_sig"],
                                0,
                            )
                            orch.submit_next_level(
                                type(self)._st_chip_handles["vector"],
                                chip_args,
                                self._build_config(self.CASES[0]["config"], output_prefix=output_prefix),
                                worker=0,
                            )

                        fourth.append(worker.submit(graph))
                    except BaseException as error:  # noqa: BLE001 -- carried back rather than lost in the thread
                        thread_errors.append(error)

                submitter = threading.Thread(target=submit_fourth, name="a5-fourth-submit", daemon=True)
                submitter.start()
                # Where that thread actually got to, before anything is claimed about it: parked in
                # admission, which is the bound a fourth run meets while three sets are held.
                self._await_parked_admission(worker, 60.0, "a5 three-run device chain")
                # The capacity, over an interval in which that park still holds.
                together = self._observe_triple_while_admission_parked(
                    worker, frames, case_runs, 60.0, "a5 three-run device chain"
                )
                for handle in handles:
                    handle.wait()
                submitter.join(timeout=_THREAD_SETTLE_S)
                assert not submitter.is_alive(), "the fourth submission never returned after the chain retired"
                assert not thread_errors, f"the fourth submission raised: {thread_errors}"
                assert fourth_callback_ran.is_set(), (
                    "the fourth graph callback never ran even after every chained run retired"
                )
            finally:
                stop.set()
                sampler.join(timeout=5.0)
                # Guaranteed: drains the chain, settles the submitter, drains the fourth run and
                # releases the device allocations only when nothing is left outstanding.
                cleanup_problems = self._settle_chain(
                    worker,
                    handles,
                    intermediates,
                    "a5 three-run device chain",
                    deferred=fourth,
                    submitter=submitter,
                    thread_errors=thread_errors,
                )
                self._report_cleanup(cleanup_problems, "a5 three-run device chain")

        assert not cleanup_problems, (
            f"this case's runs or resources did not settle: {cleanup_problems}; trace kept at {output_prefix}"
        )

        first = self._link(2.0, 0.5)
        second = self._link(first, 0.5)
        third = self._link(second, 0.5)
        for handle, expected, label in ((y, first, "y"), (z, second, "z"), (w, third, "w")):
            worker.copy_from(readback_buffer, handle)
            assert torch.allclose(readback, torch.full((_SIZE,), expected)), (
                f"device result {label} is not its link of the chain: got {readback[0].item()}, "
                f"expected {expected}; trace kept at {output_prefix}"
            )

        # Only now, with every run drained and every intermediate read: the same address that was
        # refused above releases cleanly, which is what shows the refusal was a window rather than
        # a hold the caller can never discharge.
        worker.free(y)
        intermediates[0] = None
        for leftover in intermediates:
            if leftover is not None:
                worker.free(leftover)

        records: list[dict] = []
        # Restricted to the three runs seen holding the sets together, so the deferred fourth run
        # — admitted only after a retirement — cannot stand in for one of this chain's edges.
        chain = set(together)

        def chain_pairs(pairs):
            return {pair for pair in pairs if set(pair) <= chain}

        _await_records(
            trace,
            records,
            _EVIDENCE_BUDGET_S,
            lambda seen: len(chain_pairs(_established_pairs(seen, case_runs))) >= 2,
        )
        established = chain_pairs(_established_pairs(records, case_runs))
        assert len(established) >= 2, (
            f"the chain's two adjacent edges were not both recorded with the successor's submission complete "
            f"and its predecessor's whole-operator boundary still unfired: established={sorted(established)}, "
            f"chain={sorted(chain)}; trace kept at {output_prefix}"
        )
        assert len({key for pair in established for key in pair}) == _THREE_FRAMES, (
            f"the recorded edges do not span all three chained runs: {sorted(established)}; "
            f"trace kept at {output_prefix}"
        )
        ordered, overlapping, within_tick, unmeasured = _whole_operator_order(trace, records, case_runs)
        assert not chain_pairs(overlapping), (
            f"a chained successor's AICore stream was released before its predecessor's whole operator had "
            f"finished: {sorted(chain_pairs(overlapping))}; {_boundary_detail(trace, chain_pairs(overlapping))}; "
            f"trace kept at {output_prefix}"
        )
        comparable = chain_pairs(ordered | within_tick)
        assert len(comparable) >= 2, (
            f"both adjacent edges of this chain need comparable whole-operator boundary readings: "
            f"comparable={sorted(comparable)}, unmeasured={sorted(chain_pairs(unmeasured))}; "
            f"trace kept at {output_prefix}"
        )
        shutil.rmtree(output_prefix, ignore_errors=True)

    @pytest.mark.platforms(["a5"])
    def test_a_device_control_tensor_is_refused_until_it_is_copied_to_the_host(self, st_platform, st_worker):
        """The host build's control tensor must be a HOST value, and a DEVICE one is refused.

        ``get_tensor_data`` is a *host* read the orchestration makes while building the graph, and
        ``require_host_tensor`` (``src/common/host_build_graph/host/runtime_core.cpp``) admits only
        an explicit HOST/NONE argument: "DEVICE values need an explicit copy before submit". That
        is a contract refusal on the argument's address space. It is **not** timing-dependent, not
        a decline that retries at the FIFO head, and not something a predecessor's completion can
        satisfy — where the run sits in the FIFO never enters into it.

        The lane keeps those two outcomes apart by exception type:
        ``PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE`` raises ``PreparedRunIncompatible`` and *is* the
        depth-one fallback, while every other preparation failure raises
        ``prepare_native_run failed with code …`` and is terminal for that run
        (``chip_worker.cpp`` ≈:1071-1078, ``chip_run_lane.cpp::prepare_successor_if_eligible``).
        That split places the failure, and it is **all** it does: the terminal branch is every
        preparation error there is. Which one this was comes from the child's own log, below.

        Four things, on the real ``Worker.submit`` route:

        refused   the run is admitted by ``submit`` and fails at its own ``wait``, during its
                  preparation and before any device work of its own — **and the child's own log
                  says it was this refusal**. The handle's message alone would not: every
                  preparation failure that is not the depth-one fallback reaches the caller as
                  ``prepare_native_run failed with code …``, so an allocation failure or a
                  different invalid argument would satisfy it. The orchestrator's
                  ``FATAL(code=5)`` line, under the name of the entry that made the read and
                  carrying ``require_host_tensor``'s own words, is what names the rejection.
        intact    the predecessor completes on its own terms and its result is still correct, so
                  the refusal is the successor's alone.
        released  a refused preparation reached no device submission, so it discharges its caller
                  references rather than keeping them — ``Worker.free`` on the very allocation it
                  named is accepted. That is the opposite of the in-flight refusal the device-chain
                  case above asserts, and the difference is exactly whether device work exists.
        accepted  the same bytes, copied to the host first, are a legal control tensor on the same
                  Worker — so the refusal is about the address space, not about the value, and the
                  Worker is still usable afterwards.

        No automatic copy is expected or wanted: the explicit ``copy_from`` is the caller's, and
        nothing in the runtime inserts one.
        """
        if st_platform != "a5":
            pytest.skip("a5 three-run capacity is gated to the a5 onboard host_build_graph route")
        self._require_three_sets(st_worker)
        worker = st_worker

        x_buffer, _ = self._tensor_from_host_buffer(worker, 2.0)
        control_buffer, _ = self._tensor_from_host_buffer(worker, 0.5)
        # The host destination of the explicit copy, and separately the read-back buffer, so a
        # later read cannot overwrite the control value the accepted run is still naming.
        copied_control_buffer, _ = self._tensor_from_host_buffer(worker, 0.0)
        readback_buffer, readback = self._tensor_from_host_buffer(worker, 0.0)
        intermediates = [worker.alloc_child_tensor(0, (_SIZE,), DataType.FLOAT32) for _ in range(3)]
        y, z, w = intermediates
        expected_y = self._link(2.0, 0.5)
        expected_w = self._link(2.0, expected_y)
        handles: list = []
        refused_handles: list = []
        # Set only once both halves of the refusal have been established. Until then the refused
        # run is drained as an ordinary handle, so an unexpected failure is reported rather than
        # absorbed.
        refusal_verified = False
        cleanup_problems: list = []
        # Opened before anything is submitted, so the window it searches holds only this case's
        # own lines.
        trace = _RunTrace(worker)
        output_prefix = tempfile.mkdtemp(prefix="simpler-a5-device-control-refusal-")
        try:
            handles.append(
                self._submit_link(worker, x_buffer, control_buffer, y, output_prefix, spin_iters=_DEVICE_SPIN_ITERS)
            )
            # `y` — a device allocation — in the control position, which is the argument the host
            # build reads. Admitted here; refused when this run prepares.
            refused_handles.append(self._submit_link(worker, x_buffer, y, z, output_prefix, spin_iters=0))

            # The predecessor first: it owns device work, and the refusal below must not be read
            # off a Worker whose earlier run had not finished.
            handles[0].wait()

            with pytest.raises(RuntimeError) as refusal:
                refused_handles[0].wait()
            # The boundary: it failed preparing, not at submit and not on the device. This is the
            # branch that is terminal — `PreparedRunIncompatible` carries "requires depth-one
            # fallback" instead and would leave the run to retry at the front.
            assert "prepare_native_run failed" in str(refusal.value), (
                f"the refused run did not fail at its own preparation: {refusal.value}; trace kept at {output_prefix}"
            )
            # The oracle, from this child's own account of this run: which rejection it was. No
            # other preparation failure writes these three on one line.
            reported = _await_child_log_line(trace, _HOST_TENSOR_REFUSAL, _EVIDENCE_BUDGET_S)
            assert reported is not None, (
                f"the run failed its preparation, but no chip child reported the host-tensor refusal "
                f"{_HOST_TENSOR_REFUSAL} within {_EVIDENCE_BUDGET_S}s — so this case has not shown the "
                f"failure was the DEVICE control tensor rather than some other preparation error. "
                f"Handle reported: {refusal.value}. Children's own log window: "
                f"{trace.child_log_window()}; trace kept at {output_prefix}"
            )
            refusal_verified = True

            worker.copy_from(readback_buffer, y)
            assert torch.allclose(readback, torch.full((_SIZE,), expected_y)), (
                f"the predecessor's own result is wrong, so the successor's refusal cannot be read from "
                f"this case: got {readback[0].item()}, expected {expected_y}; trace kept at {output_prefix}"
            )

            # The same value, through the copy the contract names. This is the whole difference: a
            # HOST argument carrying the predecessor's produced bytes. Registered before it is
            # waited, so a wait that raises still leaves this run reachable for the drain below.
            worker.copy_from(copied_control_buffer, y)
            handles.append(self._submit_link(worker, x_buffer, copied_control_buffer, w, output_prefix, spin_iters=0))
            handles[-1].wait()
            worker.copy_from(readback_buffer, w)
            assert torch.allclose(readback, torch.full((_SIZE,), expected_w)), (
                f"the run whose control tensor was copied to the host first did not compute from those "
                f"bytes: got {readback[0].item()}, expected {expected_w}; trace kept at {output_prefix}"
            )
        finally:
            # Guaranteed, and while the buffers these runs name are still alive. The refused run is
            # tolerated only once its refusal has been established; otherwise it is drained as an
            # ordinary handle, so an unexpected failure is reported and the allocations retained.
            cleanup_problems = self._settle_chain(
                worker,
                handles if refusal_verified else [*handles, *refused_handles],
                intermediates,
                "a5 device control refusal",
                tolerated=refused_handles if refusal_verified else (),
            )
            self._report_cleanup(cleanup_problems, "a5 device control refusal")

        assert not cleanup_problems, (
            f"this case's runs or resources did not settle: {cleanup_problems}; trace kept at {output_prefix}"
        )

        # Only now, with every run of this case settled. The refused run named `z` and never
        # reached the device, so the caller's right to release it is not withheld: a refusal here
        # would mean a run that did no device work still held a borrow over the caller's pages.
        worker.free(z)
        intermediates[1] = None
        for leftover in intermediates:
            if leftover is not None:
                worker.free(leftover)
        shutil.rmtree(output_prefix, ignore_errors=True)


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


def _submit_wide_window(case, worker, output_prefix, spin_iters, frames, handles, buffers):
    """`_WIDE_SETS` runs with no wait between them, each with its own buffers and expectation.

    ``handles`` and ``buffers`` belong to the caller and are appended to as each run is made, so a
    submission that fails part way still leaves the caller holding every run already in flight and
    the memory those runs are reading.
    """
    tensors = []
    for index in range(_WIDE_SETS):
        for value in (float(2 * index + 2), float(2 * index + 3), 0.0):
            buffer, tensor = case._tensor_from_host_buffer(worker, value)  # noqa: SLF001 -- suite-local helper
            buffers.append(buffer)
            tensors.append(tensor)
    expectations = []
    for index in range(_WIDE_SETS):
        group = buffers[index * 3 : index * 3 + 3]
        a, b, out = tensors[index * 3 : index * 3 + 3]
        expectations.append((out, case._expected(a, b)))  # noqa: SLF001 -- suite-local helper
        handles.append(case._submit_vector(worker, group, output_prefix, spin_iters=spin_iters))  # noqa: SLF001
        if index == 0:
            # The first run has to be on the device before the rest are made, or every one could
            # be admitted into an idle pipeline with nothing held open.
            _wait_for_one_launched_frame(frames, 20.0)
    return expectations


@scene_test(level=3, runtime="host_build_graph")
class TestA5RunCapacityBeyondTheOldCeiling(_A5EarlyEnqueueBase):
    """At ``pipeline_depth=5`` five runs are in flight at once, which no fixed layout could carry.

    Three was the width every per-set owner used to be built for, so a request of five is the
    claim this capacity has no number in it any more: the arena, the mailbox, the per-run runner
    storage, the wait table and the workspace ledger are each sized to what this Worker asked for.

    What each check can see:

    granted    the endpoint negotiated five task frames, so five runs have somewhere to go.
    accepted   five different runs were observed launched and accepted at one coherent instant.
    distinct   every run's own output is correct, so five sets held five runs' parameters, graphs
               and results rather than sharing any of them.
    """

    CASES = [
        {
            "name": "a5_capacity_beyond_the_old_ceiling",
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
        # The grant, not the request: a backend that clamped would leave this class asserting a
        # capacity the child never gave it, which is exactly the state this case exists to refuse.
        negotiated = _negotiated_frame_count(worker)
        assert negotiated == _WIDE_SETS, (
            f"the a5 endpoint negotiated {negotiated} task frames for a {_WIDE_SETS}-set request, so the "
            f"capacity was clamped somewhere and nothing below could observe it"
        )

    @pytest.mark.platforms(["a5"])
    def test_five_runs_are_launched_and_accepted_at_once(self, st_platform, st_worker):
        """Five accepted submissions over five sets, each with its own correct result."""
        if st_platform != "a5":
            pytest.skip("this capacity case is gated to the a5 onboard host_build_graph route")
        self._require_wide_sets(st_worker)
        trace = _RunTrace(st_worker)
        with _frame_views(st_worker) as frames:
            case_runs = _CaseRuns(frames)
            with tempfile.TemporaryDirectory(prefix="simpler-a5-wide-capacity-") as output_prefix:
                handles: list = []
                buffers: list = []
                expectations: list = []
                observer = _WideWindowObserver(st_worker, frames, case_runs, _WIDE_SETS)
                try:
                    # Sampling spans the submission window rather than starting after it.
                    with observer:
                        expectations = _submit_wide_window(
                            self, st_worker, output_prefix, _DEVICE_SPIN_ITERS, frames, handles, buffers
                        )
                        deadline = time.monotonic() + _WIDE_WINDOW_BUDGET_S
                        while observer.together is None and time.monotonic() < deadline:
                            time.sleep(0.001)
                    together = observer.together
                finally:
                    # Before anything of this case's leaves scope, whatever the submission or the
                    # observation did: a run still in flight owns its arguments and its output.
                    drain_failures = _drain_handles(handles, "a5 wide-capacity window")
                assert not drain_failures, f"a run of this case did not finish: {drain_failures}"
                _assert_results(expectations, trace, case_runs, "a5 wide-capacity window")

                assert together is not None, (
                    f"{_WIDE_SETS} different runs were never observed launched and accepted at the same time, "
                    f"so the capacity above the old three-set layout never carried runs"
                )
                assert len(together) == _WIDE_SETS, (
                    f"fewer than {_WIDE_SETS} runs were named together: {sorted(together)}"
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


@scene_test(level=3, runtime="host_build_graph")
class TestA5WideDeviceResultChain(_A5EarlyEnqueueBase):
    """One device result feeds four later runs, and the caller keeps the right to release it.

    The three-link sibling establishes the borrow at the width every per-set owner used to be laid
    out for, where the tested allocation has exactly one later consumer. Here one producer's output
    `y` is named as the **input of four separate later runs**, so the address is borrowed across
    more live consumers than that layout could carry, and the last of them is the last submission
    this case makes.

    What each check can see:

    consumed    each of the four consumers' own output equals `y`'s value composed with that
                consumer's own control scalar. The four controls differ, so four correct and
                *different* results can only come from four runs that each read the same `y`.
    on device   `y` and every consumer output are `alloc_child_tensor` allocations; the only host
                transfers are the initial fill and the closing read-backs.
    borrowed    at the instant of the refusal, the Worker's own accepted-run registry is read and
                every one of the four consumers is found still naming `y` — including the last one
                — with its cleanup unpublished. That registry *is* what `Worker.free` scans, so the
                refusal that follows is entailed rather than raced for.
    discharged  once every run has been waited and every result read, the same address releases
                cleanly. The borrow was a window, not a hold the caller can never discharge.

    Why the witness is a registry read and not a device observation: a run leaves that registry
    only in `RunHandle.wait()`, on the thread that calls it — nothing retires a run in the
    background. This case waits nothing until after the refusal, so every submitted run is
    necessarily still a borrower. A launched-frame sample would have been weaker, because it says
    some run is live without saying which, and "some run is live" does not entail that a run naming
    *this* allocation is.

    Not covered here: the whole-operator ordering of adjacent edges, and admission backpressure
    behind the grant. Both are the three-link sibling's subject over the same production path, and
    neither is re-proven at a second width. Nor is native resource occupancy: this reads the
    parent's borrow bookkeeping, and a handle can outlive the native run it names. How many sets
    are occupied together is what the worked-capacity examples read, from the entry table itself.
    """

    CASES = [
        {
            "name": "a5_wide_device_result_chain",
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
        self.test_a_device_result_feeds_four_later_runs("a5", worker)

    @staticmethod
    def _link(value, control):
        """One run's arithmetic, matching what the orchestration computes for this control."""
        return value + control + _CHAIN_LENGTH * control

    @pytest.mark.platforms(["a5"])
    def test_a_device_result_feeds_four_later_runs(self, st_platform, st_worker):
        """``x -> y``, then four runs each reading ``y`` with a control of their own."""
        if st_platform != "a5":
            pytest.skip("this capacity case is gated to the a5 onboard host_build_graph route")
        negotiated = _negotiated_frame_count(st_worker)
        assert negotiated == _WIDE_SETS, (
            f"the a5 endpoint negotiated {negotiated} task frames for a {_WIDE_SETS}-set request, so the four "
            f"consumers of one device result could not all be admitted at once"
        )

        consumer_count = _WIDE_SETS - 1
        x_buffer, _ = self._tensor_from_host_buffer(st_worker, 2.0)
        # One control buffer per run, each with its own value: the orchestration reads this
        # tensor's first element on the host, so it stays a host buffer, and a different value per
        # consumer is what makes four correct results four *different* numbers.
        producer_control, _ = self._tensor_from_host_buffer(st_worker, 0.5)
        controls = [self._tensor_from_host_buffer(st_worker, 0.25 * (index + 1))[0] for index in range(consumer_count)]
        readback_buffer, readback = self._tensor_from_host_buffer(st_worker, 0.0)
        # Every device allocation up front, which is what lets the submissions pass addresses: no
        # allocation happens between them.
        shared = st_worker.alloc_child_tensor(0, (_SIZE,), DataType.FLOAT32)
        outputs = [st_worker.alloc_child_tensor(0, (_SIZE,), DataType.FLOAT32) for _ in range(consumer_count)]
        allocations = [shared, *outputs]
        handles: list = []
        witness = None
        refusal_seen = False
        unexpected_release = False
        drain_failures: list = []
        output_prefix = tempfile.mkdtemp(prefix="simpler-a5-wide-device-chain-")
        try:
            # The producer first, then its four consumers. Appended one at a time: a submission
            # that raises after an earlier one succeeded must still leave the earlier run
            # reachable for the drain below.
            handles.append(
                self._submit_vector(
                    st_worker, (x_buffer, producer_control, shared), output_prefix, spin_iters=_DEVICE_SPIN_ITERS
                )
            )
            for control, target in zip(controls, outputs):
                handles.append(
                    self._submit_vector(
                        st_worker, (shared, control, target), output_prefix, spin_iters=_DEVICE_SPIN_ITERS
                    )
                )

            # The borrow window, read from the registry the refusal itself scans and before any
            # wait. `witness` names the live runs that hold `y`; the four consumers must all be
            # among them, or the refusal below would be about a different borrower than this case
            # claims.
            witness = _live_runs_naming(st_worker, shared.identity)
            assert len(witness) == _WIDE_SETS, (
                f"{len(witness)} live run(s) name the shared device result, not {_WIDE_SETS}: {witness}. The "
                f"producer and all four consumers must still hold it for the refusal below to be the "
                f"last-consumer boundary this case is about"
            )
            for index, handle in enumerate(handles):
                assert id(handle) in witness, (
                    f"run #{index} is no longer a live borrower of the shared device result, so the window this "
                    f"case tests had already closed before it was tested"
                )
            # Width, stated for what it is: more than three runs are live *borrowers* of this
            # allocation at one instant, which is a parent-side fact and the one the refusal turns
            # on. It is not a reading of how many native resource sets are occupied — a parent
            # handle outlives the native run it names, since only its wait retires it. The
            # native-occupancy claim at width belongs to the worked-capacity examples, whose
            # coherent entry-table snapshot reads the sets themselves.
            outstanding = _live_run_handles(st_worker)
            assert len(outstanding) > 3, (
                f"only {len(outstanding)} run(s) are live borrowers at the refusal, so this case does not "
                f"exercise a borrow above the old three-set layout"
            )
            assert _admission_waiters(st_worker) == 0, (
                "a submission is still parked in admission, so one of this case's runs had not been admitted "
                "when the borrow window was read"
            )

            try:
                st_worker.free(shared)
                unexpected_release = True
            except RuntimeError as error:
                assert "still referenced by an in-flight" in str(error), (
                    f"freeing a borrowed device result was refused for an unrelated reason: {error}"
                )
                refusal_seen = True
        finally:
            # Every submitted run is drained before anything of this case's leaves scope, whether
            # the body finished or a submission failed part way. Nothing is released here: an
            # allocation whose run could not be proved finished stays held.
            drain_failures = _drain_handles(handles, "a5 wide device chain")

        assert not drain_failures, f"a run of this case did not finish: {drain_failures}; trace kept at {output_prefix}"
        # Reported after the drain, so an unexpected release is never left double-freed below.
        assert not unexpected_release, (
            "Worker.free released a device result that four live runs still named; the allocations it was "
            f"handed are gone, so nothing further is released here. Trace kept at {output_prefix}"
        )
        assert refusal_seen, "the in-flight release refusal was never reached, so the borrow window went untested"
        assert len(handles) == _WIDE_SETS, f"only {len(handles)} of {_WIDE_SETS} runs were submitted"

        shared_value = self._link(2.0, 0.5)
        st_worker.copy_from(readback_buffer, shared)
        assert torch.allclose(readback, torch.full((_SIZE,), shared_value)), (
            f"the shared device result is not its producer's output: got {readback[0].item()}, expected "
            f"{shared_value}; trace kept at {output_prefix}"
        )
        for index, target in enumerate(outputs):
            expected = self._link(shared_value, 0.25 * (index + 1))
            st_worker.copy_from(readback_buffer, target)
            assert torch.allclose(readback, torch.full((_SIZE,), expected)), (
                f"consumer {index} did not read the shared device result: got {readback[0].item()}, expected "
                f"{expected}; trace kept at {output_prefix}"
            )

        # Only now, with every run drained and every result read: the address refused above
        # releases cleanly, which is what shows the refusal was a window rather than a hold the
        # caller can never discharge.
        for leftover in allocations:
            st_worker.free(leftover)


@scene_test(level=3, runtime="host_build_graph")
class TestA5WorkedCapacityExample(_A5EarlyEnqueueBase):
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
            "name": "a5_worked_capacity_example",
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
        negotiated = _negotiated_frame_count(worker)
        assert negotiated == _WORKED_SETS, (
            f"the endpoint negotiated {negotiated} sets for a {_WORKED_SETS}-set request, so the capacity was "
            f"clamped somewhere and nothing below could observe it"
        )

    @pytest.mark.platforms(["a5"])
    def test_sixty_four_runs_retire_through_eight_sets(self, st_platform, st_worker):
        """Sustained submission at the contracted budgets, with the launch bound held."""
        if st_platform != "a5":
            pytest.skip("the worked capacity example is gated to the a5 onboard host_build_graph route")
        self._require_worked_budgets(st_worker)
        with _frame_views(st_worker) as entries:
            case_runs = _CaseRuns(entries)

            handles: list = []
            buffers: list = []
            expectations: list = []
            # Sampling spans the whole body, and the scope is what proves the sampler stopped
            # before the entry views it reads are released — on every path, including the
            # readiness handshake failing and a submission raising. See `_EntrySampler`.
            with _WorkedCapacityObserver(st_worker, entries, case_runs) as observer:
                with tempfile.TemporaryDirectory(prefix="simpler-a5-hbg-worked-") as output_prefix:
                    try:
                        for index in range(_WORKED_SUBMISSIONS):
                            group = []
                            tensors = []
                            for value in (float(index % 7 + 2), float(index % 5 + 3), 0.0):
                                buffer, tensor = self._tensor_from_host_buffer(st_worker, value)
                                group.append(buffer)
                                tensors.append(tensor)
                            buffers.extend(group)
                            expectations.append((tensors[2], self._expected(tensors[0], tensors[1])))
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
                    for out, expected in expectations:
                        torch.testing.assert_close(out, expected)

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
