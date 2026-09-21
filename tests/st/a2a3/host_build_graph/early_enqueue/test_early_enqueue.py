#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Onboard validation for early enqueue on the public ``Worker.submit`` route.

``launch_depth=2`` lets a staged successor's native submission reach the device while the run
ahead of it is still executing. Four things are checked, on the real submit route, for exactly
the runs the case submitted.

  admitted   two mailbox frames hold two *different* runs, both TASK_LAUNCHED with both sticky
             acceptance words set. At depth one that state is unreachable at any instant, and not
             for timing reasons: the successor's launch is not attempted until it is the lane's
             front, which requires its predecessor to be terminal and its frame retired. So two
             coexisting launched runs is the admission change itself.
  ordered    for a named (predecessor, successor) pair, the successor's native submission
             *completed* and the identified predecessor's whole-operator boundary was still
             unfired. That places the completed enqueue before the predecessor's completion,
             which no host-side state can: a frame stays TASK_LAUNCHED after the device has
             finished — the child publishes that word and `continue`s — and a run's output only
             reaches the host at its finalize.
  serial     and the device then ran that same pair one whole operator at a time, in that order.
             Each run records two passive device-timestamp markers: one on its AICore stream
             after any cross-run wait and before its kernel launch, one on its AICPU stream
             behind the wait on its *own* AICore boundary. So a successor's
             `aicore_start > predecessor.whole_operator_end` says its AICore stream was released
             only after the predecessor's AICore kernel had returned. Both are
             `aclrtEventGetTimestamp` readings — "syscnt when event recorded", one chip-wide
             counter — which is what makes a reading on one stream comparable with one on
             another. The AICPU run wall is kept alongside as bounded supporting evidence.
  retained   both runs then produce their own correct results, so the overlap is a capability
             rather than two runs treading on each other: each keeps its own inputs, output
             buffer and outcome across the window.

The ordering observation is taken inside the child, immediately after its own submission returned
success, and published as a ``chip.run.joined_launch`` host span — the trace mechanism the run's
other markers already use, at the default-visible timing tier, so it needs no new API, no mailbox
field, no protocol change and no log-level flag. A query that could not answer is carried as
``observed=0`` rather than folded into the answer, and the span is emitted whatever it says, so a
reader can tell an unfavourable observation from a missing one. The device times ride the same
mechanism: ``chip.run.runner_run.device_wall`` is the product's own, and
``chip.run.runner_run.device_boundaries`` carries the two marker readings under the same
non-diagnostic capture gate, each with the rc that says why a position is unavailable.

**No device stream waits on the markers.** Recording an event is not a wait and no stream waits on
one, so they add no ordering to the streams they sit in and cannot make a missing production wait
look satisfied. The production completion fence keeps its own completion-only events and its own
contract; these are separate handles with a separate creation flag, because a timestamp needs a
capability the completion flag does not promise.

Retrieving a reading is this event's own bounded completion step and then the timestamp call: the
device having passed the record does not by itself make the timestamp retrievable. The first step
blocks the host thread that reads, and queues nothing, so it adds no device ordering; it is taken
at that run's finalize, once the run's own completion has been established. A reading is then
accepted only when it can be tied to the run that asked — that run recorded that position, exactly
one record has happened since the last attributed reading, and the value advances. Anything else is
unavailable.

Every counted edge belongs entirely to the case that counted it. Three independent filters:

* the records are read only from this worker's own chip children's log files, through a per-file
  byte cursor opened at each file's current size;
* both ends must be runs the case itself submitted, which :class:`_CaseRuns` decides from a
  **dispatch-id floor** taken before the case submits anything — the frames still hold a previous
  case's terminal identities until this case overwrites them, and that case's writer is
  asynchronous, so neither a frame snapshot nor a byte cursor excludes an older edge on its own;
* the four span families are joined on the invocation id every span of one run shares.

**What is not claimed.** The timestamp unit is device-uptime microseconds on a2a3, so a pair whose
two readings land in the same tick is counted separately and never as ordered: the ordering held
to the resolution available, and no sub-microsecond exclusion follows from a 1 MHz reading. A
position this run did not record, or whose reading could not be attributed to it, is *unmeasured*
and can never make a case pass.

The depth-one class below runs the same sequence with the key absent. It is the control for both
the "admitted" observation and the records: neither may appear there.

Onboard only, a2a3 host_build_graph: the ordering is queued onto per-run device streams, and the
runtime capability gate answers for exactly that pairing.

Run under pytest, which builds one Worker per class and so gives this class its depth of two. The
standalone ``run_module`` path shares one Worker across every class in the file at the *minimum*
requested depth, so there it needs ``--case early_enqueue`` to select this class on its own.
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
from simpler_setup.tools.strace_timing import parse_spans

#: The point-in-time host span one joined launch publishes. Its attributes carry both run
#: identities, so a claim belongs to an exact pair, and carry `observed` separately from
#: `unfired` so a failed query reads as a failed query.
_JOIN_SPAN = "chip.run.joined_launch"
#: The run's own host span, whose attributes name the run its invocation belongs to.
_RUN_SPAN = "chip.run"
#: That run's AICPU device wall, whose attributes carry absolute device-counter bounds. Emitted
#: by default: `device_phase_capture_enabled()` is on unless SIMPLER_DEVICE_STRACE_ENABLE=0, and
#: the span tier is the default-visible one.
_DEVICE_WALL_SPAN = "chip.run.runner_run.device_wall"
#: That run's two passive device-boundary marker times, under the same gate. `aic_start` is the
#: instant its AICore stream was released to begin its kernel; `wo_end` is the instant its own
#: AICore kernel had returned. Both are `aclrtEventGetTimestamp` readings — the device's own
#: syscnt at the record — so they are comparable across runs and streams on one device.
_BOUNDARY_SPAN = "chip.run.runner_run.device_boundaries"

_KERNELS = "../worker_async_fifo/kernels"

_CHAIN_LENGTH = 512
# Long enough that the host can observe the overlap window, bounded so a
# regression fails on an assertion rather than on the op-execute timeout.
_DEVICE_SPIN_ITERS = 200_000_000
# Shorter per run, because the refill case runs four of them back to back and only needs each to
# outlast one sampling pass rather than a whole host observation window.
_REFILL_SPIN_ITERS = 40_000_000
_SIZE = 128 * 128
_FRAME_COUNT = 2
# How long to keep draining the children's log files after their runs have finished. The writers
# are asynchronous, so this covers the lag between a record being accepted and written — not the
# run itself, which has already been waited for. Exhausting it is an absence, and the caller's
# own assertion reports it as one.
_EVIDENCE_BUDGET_S = 10.0

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
    for value in scalars:
        args.add_scalar(value)
    return args


def _frames(worker):
    """Each task frame's ``(address, buffer)``, for identity-carrying reads."""
    shm_buf = worker._chip_shms[0].buf  # noqa: SLF001 -- white-box mailbox observation
    assert shm_buf is not None
    mailbox_addr = ctypes.addressof(ctypes.c_char.from_buffer(shm_buf))
    return [
        (
            mailbox_addr + (1 + index) * MAILBOX_FRAME_SIZE,
            shm_buf[(1 + index) * MAILBOX_FRAME_SIZE : (2 + index) * MAILBOX_FRAME_SIZE],
        )
        for index in range(_FRAME_COUNT)
    ]


def _coherent_snapshot(frames):
    """Both frames' ``(identity, state, accepted)``, or None when they moved mid-read.

    The two frames are separate words, so reading them one after another can mix two instants.
    Re-reading each identity after the state settles that: a pair whose identities are unchanged
    describes two named runs at one point in their lives, which is what makes a claim about the
    *pair* attributable at all.
    """
    before = [_read_task_frame_identity(buf) for _, buf in frames]
    samples = [(_mailbox_load_i32(addr + _OFF_STATE), _mailbox_load_i32(addr + _OFF_ACCEPTED)) for addr, _ in frames]
    after = [_read_task_frame_identity(buf) for _, buf in frames]
    if before != after:
        return None
    return [(identity, state, accepted) for identity, (state, accepted) in zip(after, samples)]


def _two_distinct_runs_launched(snapshot):
    """Whether the snapshot holds two different runs, both launched and both accepted."""
    if snapshot is None:
        return False
    if not all(state == _TASK_LAUNCHED and accepted == _TASK_ACCEPTED for _, state, accepted in snapshot):
        return False
    identities = {identity for identity, _, _ in snapshot}
    return len(identities) == _FRAME_COUNT


def _run_key(identity):
    """The ``(dispatch id, pipeline slot)`` pair a chip child gives its native run.

    Both halves reach the native descriptor unchanged from this frame, so a record's identities
    are comparable with what the parent can see. A frame that has never carried a run reads as
    dispatch zero and is not a key.
    """
    _protocol, _run_id, slot_id, _generation, dispatch_id, *_rest = identity
    return None if dispatch_id == 0 else (dispatch_id, slot_id)


def _dispatch_floor(frames):
    """The highest dispatch id this Worker has issued before the caller submits anything.

    Read from a coherent snapshot, so the two frames are not mixed across an instant. A frame
    that has never carried a run reads as zero, so a fresh Worker's floor is zero.
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
    higher id, and that floor is what keeps a **previous** case's record out of the evidence.

    A per-file byte cursor cannot do that on its own: the earlier case's writer is asynchronous,
    so its record may only be appended after the cursor's baseline — but its dispatch id is below
    the floor either way. Nor can a bare frame snapshot: the frames still hold the previous case's
    terminal identities until this case overwrites them, so "seen in a frame" is *observed ever*,
    not *submitted here*.

    Runs are sampled rather than declared, because the parent has no path from a submitted handle
    to the dispatch id the scheduler gave it. So the set is what was observed at or above the
    floor, and a record counts only when **both** of its ends are in it.
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
    """The directory this worker's chip children append their host logs to.

    The children inherited the parent's logger state at fork, so either the parent was already
    bound and they write where it writes, or neither was and they bind the same process-tree
    spool the parent's own module constant names. One call covers both.
    """
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
            "successor_epoch": int(attributes["s_epoch"]),
            "predecessor_epoch": int(attributes["p_epoch"]),
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
    file that sorts earlier appending between two reads shifts every later record, which both
    admits records from before the window and hides the ones inside it.

    Reading only this worker's own children, and only whole lines, is the other half of
    attribution — a record still has to name a run the case submitted, which
    :func:`_established_pairs` requires.

    Each cursor opens at its files' current size, so its window begins where it was created: an
    earlier case on the same worker, or an earlier session that happened to reuse a pid, cannot
    supply a record to a later one.

    Four span families are collected, joined on ``(pid, invocation)``, which every span of one
    run shares:

    ``chip.run``                              the run's identity, so an invocation can be named.
    ``chip.run.runner_run.device_wall``       that run's absolute AICPU-wall bounds.
    ``chip.run.runner_run.device_boundaries`` that run's AICore release instant and the instant
                                              its own AICore kernel had returned.
    ``chip.run.joined_launch``                which pairs were joined, and what the successor read
                                              of its predecessor's boundary once its submission
                                              had succeeded.
    """

    def __init__(self, worker):
        directory = Path(_child_log_directory())
        self._pids = list(worker._chip_pids)  # noqa: SLF001 -- white-box child attribution
        self._paths = {pid: directory / f"host.{pid}.log" for pid in self._pids}
        self._offsets = {pid: self._size(pid) for pid in self._pids}
        #: ``(pid, inv)`` -> the ``(dispatch id, pipeline slot)`` that invocation ran.
        self.run_of_invocation: dict[tuple, tuple] = {}
        #: ``(pid, inv)`` -> that run's AICPU device wall, as absolute counter ticks.
        self.wall_of_invocation: dict[tuple, dict] = {}
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

        The other three families are folded into this cursor's own state rather than returned:
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
                elif span.name == _DEVICE_WALL_SPAN:
                    self._note_wall(pid, span)
                elif span.name == _BOUNDARY_SPAN:
                    self._note_boundaries(pid, span)
        return fresh

    def _note_identity(self, pid, span):
        attributes = _attributes(span)
        with contextlib.suppress(KeyError, ValueError):
            key = (int(attributes["dispatch_id"]), int(attributes["slot_id"]))
            self.run_of_invocation[pid, span.inv] = key

    def _note_wall(self, pid, span):
        attributes = _attributes(span)
        with contextlib.suppress(KeyError, ValueError):
            self.wall_of_invocation[pid, span.inv] = {
                "device": int(attributes["dev_id"]),
                "start": int(attributes["dev_start_cycle"]),
                "end": int(attributes["dev_end_cycle"]),
                "hz": int(attributes["dev_cnt_hz"]),
            }

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

    def _of_run(self, table, pid, run_key):
        for (row_pid, inv), row in table.items():
            if row_pid == pid and self.run_of_invocation.get((row_pid, inv)) == run_key:
                return row
        return None

    def wall_of_run(self, pid, run_key):
        """That run's AICPU device wall on this child, or None while it has not been published."""
        return self._of_run(self.wall_of_invocation, pid, run_key)

    def boundaries_of_run(self, pid, run_key):
        """That run's marker times on this child, or None while they have not been published."""
        return self._of_run(self.boundary_of_invocation, pid, run_key)


def _established_pairs(records, case_runs):
    """The pairs whose ordering the records actually establish.

    A record counts only when the query answered *and* the predecessor's whole-operator boundary
    had not fired: that pair's successor completed its native submission before its predecessor
    completed. A failed query counts for neither side.

    Both ends must also be runs *this case* submitted, and must differ. A dispatch id is unique
    within one worker thread but restarts at 1 in each, and a previous case on the same Worker
    leaves both its terminal identities in the frames — so without :class:`_CaseRuns` an older
    edge could be counted towards this case's refill.
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

    * **ordered** — the successor's AICore stream was released *strictly after* the predecessor's
      own AICore kernel had returned. This is the whole-operator serialization the queued wait
      constructs, measured rather than inferred.
    * **overlapping** — the successor was released before that instant. A real failure of the
      ordering edge, whichever tier it happened on.
    * **within_tick** — the two readings are the same tick. The ordering held to the resolution
      available and is *not* an exclusion below it, so it is counted separately and never as
      ordered. On a2a3 the timestamp unit is device-uptime microseconds, so a tick is 1 µs; no
      sub-microsecond claim is made from it.
    * **unmeasured** — a position was unavailable, the two runs report different devices, or the
      two readings are on different tick rates. Never a pass.

    The two readings come from separate passive markers: the successor's on its AICore stream
    after any cross-run wait and before its kernel launch, the predecessor's on its AICPU stream
    behind the wait on its *own* AICore boundary. Both are `aclrtEventGetTimestamp` values, which
    the SDK documents as "get syscnt when event recorded" — one chip-wide counter read at the
    instant the stream reached the record — which is what makes a reading taken on one stream
    comparable with one taken on another. No stream waits on either marker, so neither can make a
    missing production wait appear satisfied, and retrieving one queues nothing.

    A position is unavailable when this run did not record it, when its event could not be created
    or recorded, when its completion step or the timestamp call failed, when a recorded generation
    went unread, or when the value did not advance — the last three being how a value belonging to
    another run on the same re-recorded event is excluded. `aic_rc` / `wo_rc` carry which, and an
    unavailable position never becomes a time.
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
            ahead = trace.boundaries_of_run(pid, pair[0])
            behind = trace.boundaries_of_run(pid, pair[1])
            detail[pair] = {"predecessor": ahead, "successor": behind}
    return detail


def _device_execution_order(trace, records, case_runs):
    """Split the established pairs by what the device's own clock says about them.

    Returns ``(ordered, overlapping, unmeasured)``. A pair is *ordered* when the successor's
    device wall begins no earlier than the predecessor's ends, *overlapping* when it begins
    earlier, and *unmeasured* when either wall has not been published.

    ``dev_start_cycle`` / ``dev_end_cycle`` are absolute ticks of the device's free-running
    system counter — CNTVCT_EL0 rescaled into the platform's profiling-counter unit — which is
    never reset per run. Two runs' ticks on one device are therefore directly comparable, which
    is what makes this a statement about device execution rather than about host submission
    order, and it is the comparison the emitter documents as its purpose. A pair whose two runs
    report different ``dev_id`` is not compared at all.

    **This is the AICPU run wall, and it is bounded supporting evidence.** The wall brackets
    `aicpu_execute`, which returns once the AICore workers have answered the handshake — not once
    their kernels have returned. So an ordered pair here establishes that the device ran the two
    runs' AICPU work serially and in that order, and it **cannot falsify** a missing
    successor-AICore wait. An *overlap* is still a real failure, which is why the cases assert on
    it. The whole-operator relation is :func:`_whole_operator_order`, which is what decides.
    """
    pids = {record["pid"] for record in records if not record.get("malformed")}
    ordered, overlapping, unmeasured = set(), set(), set()
    for pair in _established_pairs(records, case_runs):
        walls = [
            (trace.wall_of_run(pid, pair[0]), trace.wall_of_run(pid, pair[1]))
            for pid in pids
            if trace.wall_of_run(pid, pair[0]) is not None and trace.wall_of_run(pid, pair[1]) is not None
        ]
        comparable = [(ahead, behind) for ahead, behind in walls if ahead["device"] == behind["device"]]
        if not comparable:
            unmeasured.add(pair)
            continue
        for ahead, behind in comparable:
            bucket = ordered if behind["start"] >= ahead["end"] else overlapping
            bucket.add(pair)
    return ordered, overlapping, unmeasured


def _chained(pairs):
    """Whether some pair's predecessor is another pair's successor.

    What separates one chain of refills from two unrelated openings: a run that was the
    successor of one early enqueue going on to be the predecessor of the next.
    """
    successors = {successor for _, successor in pairs}
    return any(predecessor in successors for predecessor, _ in pairs)


def _await_records(cursor, records, budget_s, satisfied):
    """Drain into ``records`` until ``satisfied``, or until the budget runs out.

    The child's writer is asynchronous and the parent's flush drains only the parent's own sink,
    so a read taken straight after ``wait()`` is no guarantee that the child's record has been
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


def _wait_for_one_launched_frame(worker, timeout):
    frames = _frames(worker)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = _coherent_snapshot(frames)
        if snapshot is not None and any(state == _TASK_LAUNCHED for _, state, _ in snapshot):
            return
        time.sleep(0.001)
    raise AssertionError("no run reached its device launch fence")


class _EarlyEnqueueBase(SceneTestCase):
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

    def _observe_two_launched_runs(self, worker, timeout, case_runs):
        """Watch for two different runs launched and accepted at once.

        Returns whether that state was ever observed, and records every run a sampled frame
        names in ``case_runs`` so the ordering records can be held to this case's own runs. The
        watch ends when no frame is launched any more, which is as long as there is anything
        left to observe; it does not claim the predecessor's device boundary was unfinished at
        the sampled instant.
        """
        frames = _frames(worker)
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

    def _run_two_and_observe(self, worker, output_prefix, case_runs):
        """Submit a long predecessor and a trivial successor; report whether they overlapped.

        Both runs' results are checked afterwards, which is what makes an observed overlap a
        capability rather than a corruption: each run keeps its own inputs, output and outcome
        across the window in which both were on the device.
        """
        buffers = []
        tensors = []
        for value in (2.0, 3.0, 0.0, 5.0, 7.0, 0.0):
            buffer, tensor = self._tensor_from_host_buffer(worker, value)
            buffers.append(buffer)
            tensors.append(tensor)
        first_a, first_b, first_out, second_a, second_b, second_out = tensors
        first_bufs, second_bufs = buffers[:3], buffers[3:]
        first_expected = self._expected(first_a, first_b)
        second_expected = self._expected(second_a, second_b)

        first = self._submit_vector(worker, first_bufs, output_prefix, spin_iters=_DEVICE_SPIN_ITERS)
        _wait_for_one_launched_frame(worker, 20.0)
        second = self._submit_vector(worker, second_bufs, output_prefix)

        overlapped = self._observe_two_launched_runs(worker, 60.0, case_runs)
        first.wait()
        second.wait()
        torch.testing.assert_close(first_out, first_expected)
        torch.testing.assert_close(second_out, second_expected)
        return overlapped


@scene_test(level=3, runtime="host_build_graph")
class TestEarlyEnqueueDepthTwo(_EarlyEnqueueBase):
    """At ``launch_depth=2`` a successor's work reaches the device while its predecessor runs."""

    CASES = [
        {
            "name": "early_enqueue",
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
        # The standalone path shares one Worker across every class in the file at the minimum
        # requested depth, so this class only gets its two when it is selected on its own. Say
        # so here rather than letting the admission assertion report a depth-one run as a
        # regression.
        assert worker._launch_depth == 2, (  # noqa: SLF001 -- scene-test validation
            "this class needs a Worker at launch_depth=2; run it under pytest, or standalone "
            "with --case early_enqueue so no depth-one class shares the Worker"
        )
        self.test_two_runs_are_launched_and_accepted_at_once("a2a3", worker)
        self.test_sustained_refill_overlaps_successive_run_pairs("a2a3", worker)

    def test_two_runs_are_launched_and_accepted_at_once(self, st_platform, st_worker):
        """Admission, the ordering the admission exists for, and what the device then did."""
        if st_platform != "a2a3":
            pytest.skip("early enqueue is gated to a2a3 onboard host_build_graph")
        trace = _RunTrace(st_worker)
        # Taken before anything is submitted, so the frames' residual identities from an earlier
        # case fall below the floor and cannot be counted as this case's runs.
        case_runs = _CaseRuns(_frames(st_worker))
        with tempfile.TemporaryDirectory(prefix="simpler-early-enqueue-") as output_prefix:
            assert self._run_two_and_observe(st_worker, output_prefix, case_runs), (
                "two different runs were never observed launched and accepted at the same time"
            )

            # The mailbox observation above shows two runs launched at once. This shows *when*: a
            # record taken after the successor's native submission returned success, reporting
            # the identified predecessor's whole-operator boundary still unfired. That places
            # the completed enqueue before the predecessor's completion, which no host-side
            # signal can. The child's writer is asynchronous, so the record is waited for — and
            # so is the device wall, which is published later, at each run's finalize.
            records: list[dict] = []
            _await_records(
                trace,
                records,
                _EVIDENCE_BUDGET_S,
                lambda seen: bool(_whole_operator_order(trace, seen, case_runs)[0]),
            )

        assert len(case_runs) <= 2, (
            f"more runs than this case submitted were attributed to it: {case_runs.sorted_keys()} "
            f"above dispatch floor {case_runs.floor}"
        )
        assert records, (
            f"no joined-launch record reached this process from children {trace.pids}, so nothing carried the ordering"
        )
        established = _established_pairs(records, case_runs)
        assert established, (
            f"no joined launch found its predecessor's boundary unfired for a pair of runs this "
            f"case submitted; this case's runs={case_runs.sorted_keys()} above dispatch floor "
            f"{case_runs.floor}, records={records}"
        )

        # What the device itself did with the pair whose submissions overlapped. This is the
        # whole-operator relation: the successor's AICore stream was released only after the
        # predecessor's own AICore kernel had returned.
        ordered, overlapping, within_tick, unmeasured = _whole_operator_order(trace, records, case_runs)
        assert not overlapping, (
            f"a joined successor's AICore stream was released before its predecessor's whole "
            f"operator had finished, so the queued ordering edge did not hold: {sorted(overlapping)}; "
            f"{_boundary_detail(trace, overlapping)}"
        )
        assert ordered, (
            f"whole-operator device order is not established for any pair this case submitted. "
            f"within one timestamp tick={sorted(within_tick)}, unmeasured={sorted(unmeasured)}; "
            f"{_boundary_detail(trace, established)}"
        )

        # The AICPU walls, as bounded supporting evidence: an overlap there is a failure too.
        _, wall_overlapping, _ = _device_execution_order(trace, records, case_runs)
        assert not wall_overlapping, (
            f"a joined pair's AICPU device walls overlap: {sorted(wall_overlapping)}; walls={trace.wall_of_invocation}"
        )

    def test_sustained_refill_overlaps_successive_run_pairs(self, st_platform, st_worker):
        """N, S, T, R: each retirement must free one admission *and* one authorization.

        Correct results from four runs would pass under ordinary serial execution, so they are
        not the evidence here. What is: the *set of distinct run pairs* ever seen launched
        together. One pair is the first overlap; a second pair names a later predecessor and a
        later successor, which is renewed early enqueue rather than a single opening one.
        """
        if st_platform != "a2a3":
            pytest.skip("early enqueue is gated to a2a3 onboard host_build_graph")

        trace = _RunTrace(st_worker)
        # Taken before the sampler starts and before anything is submitted. Without it the
        # sampler's first pass would adopt the previous case's two terminal frame identities, and
        # an older established edge could be counted towards this case's refill.
        case_runs = _CaseRuns(_frames(st_worker))
        pairs: set[tuple] = set()
        stop = threading.Event()

        def sample():
            frames = _frames(st_worker)
            while not stop.is_set():
                snapshot = _coherent_snapshot(frames)
                case_runs.note(snapshot)
                if _two_distinct_runs_launched(snapshot):
                    assert snapshot is not None
                    pairs.add(tuple(sorted(identity for identity, _, _ in snapshot)))
                time.sleep(0.0005)

        # Every run spins, so each is long enough to be sampled while the next is admitted
        # behind it; the sampler runs alongside because `wait` blocks.
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        handles = []
        outs = []
        expected = []
        records: list[dict] = []
        try:
            with tempfile.TemporaryDirectory(prefix="simpler-early-enqueue-refill-") as output_prefix:
                for index in range(4):
                    a_buf, a = self._tensor_from_host_buffer(st_worker, float(index + 1))
                    b_buf, b = self._tensor_from_host_buffer(st_worker, float(index + 2))
                    out_buf, out = self._tensor_from_host_buffer(st_worker, 0.0)
                    expected.append(self._expected(a, b))
                    outs.append(out)
                    handles.append(
                        self._submit_vector(
                            st_worker, [a_buf, b_buf, out_buf], output_prefix, spin_iters=_REFILL_SPIN_ITERS
                        )
                    )
                for handle in handles:
                    handle.wait()
                _await_records(
                    trace,
                    records,
                    _EVIDENCE_BUDGET_S,
                    lambda seen: _chained(_whole_operator_order(trace, seen, case_runs)[0]),
                )
        finally:
            stop.set()
            sampler.join(timeout=10.0)

        for index, (out, want) in enumerate(zip(outs, expected)):
            assert torch.equal(out, want), f"run {index} produced the wrong result"
        assert len(pairs) >= 2, (
            f"early enqueue did not renew across the four runs: distinct overlapping run pairs seen = {len(pairs)}"
        )
        assert len(case_runs) <= 4, (
            f"more runs than this case submitted were attributed to it: {case_runs.sorted_keys()} "
            f"above dispatch floor {case_runs.floor}"
        )

        # And the same for the device-side relation, which is what makes the refill *renewed*
        # rather than one opening: two different pairs, chained, each with its successor's
        # submission completed while that pair's predecessor had not. Every end of every pair is
        # one of this case's own four runs, so an older case's edge cannot supply either.
        established = _established_pairs(records, case_runs)
        assert len(established) >= 2, (
            f"early enqueue was established for fewer than two distinct run pairs: {sorted(established)}; "
            f"this case's runs={case_runs.sorted_keys()} above dispatch floor {case_runs.floor}, "
            f"records={records}"
        )
        assert _chained(established), (
            f"the established pairs are not consecutive — no pair's predecessor is another's "
            f"successor, so this is not one chain of refills: {sorted(established)}"
        )

        # Each of those overlaps still executed serially on the device, whole operator against
        # whole operator — and the *measured* edges must themselves chain. Requiring a chain of
        # the established edges and separately two measured ones is not the same claim: with
        # established = {N->S, S->T, T->R} and S->T unmeasured, ordered = {N->S, T->R} satisfies
        # both and contains no two consecutive measured edges. So the chain is required of
        # `ordered`, which subsumes the count.
        ordered, overlapping, within_tick, unmeasured = _whole_operator_order(trace, records, case_runs)
        assert not overlapping, (
            f"a joined successor's AICore stream was released before its predecessor's whole "
            f"operator had finished, so the queued ordering edge did not hold: {sorted(overlapping)}; "
            f"{_boundary_detail(trace, overlapping)}"
        )
        assert _chained(ordered), (
            f"the measured whole-operator edges are not consecutive, so sustained refill is not "
            f"established on the device: ordered={sorted(ordered)}, "
            f"within one timestamp tick={sorted(within_tick)}, unmeasured={sorted(unmeasured)}; "
            f"{_boundary_detail(trace, established)}"
        )

        # The AICPU walls, as bounded supporting evidence: an overlap there is a failure too.
        _, wall_overlapping, _ = _device_execution_order(trace, records, case_runs)
        assert not wall_overlapping, (
            f"a joined pair's AICPU device walls overlap: {sorted(wall_overlapping)}; walls={trace.wall_of_invocation}"
        )


@scene_test(level=3, runtime="host_build_graph")
class TestEarlyEnqueueDepthOneControl(_EarlyEnqueueBase):
    """Without the key, the same sequence keeps the successor off the device until promotion."""

    CASES = [
        {
            "name": "serial_control",
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

    def test_no_overlap_without_the_opt_in(self, st_platform, st_worker):
        """The control for the admission observation, and for the records that follow it.

        No overlap is expected, so no joined launch is either: the same spans that carry the
        depth-two evidence must be absent here. An assertion on their absence is what makes
        them evidence rather than a marker the lane emits regardless of depth.
        """
        if st_platform != "a2a3":
            pytest.skip("early enqueue is gated to a2a3 onboard host_build_graph")
        trace = _RunTrace(st_worker)
        case_runs = _CaseRuns(_frames(st_worker))
        with tempfile.TemporaryDirectory(prefix="simpler-early-enqueue-control-") as output_prefix:
            assert not self._run_two_and_observe(st_worker, output_prefix, case_runs), (
                "two runs were launched and accepted at once with launch_depth unset, so the depth-two "
                "observation measures nothing the serial path did not already do"
            )
            # Any record at all is a failure here, so no case-run filter is applied: this class
            # has its own Worker and its own children, and the cursor reads only their files from
            # the size they had before the case began.
            records: list[dict] = []
            _await_records(trace, records, _EVIDENCE_BUDGET_S, lambda seen: bool(seen))
        assert not records, f"a joined launch was recorded with launch_depth unset: {records}"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
