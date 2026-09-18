#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A completed run's result outlives its successor, and reading it does not wait (#2267).

Removing the drain's whole-stream synchronize rests on exactly one property:
with run N complete and run N+1 executing on the same streams, the host can
still read N's result, and that read does not wait for N+1. The production path
cannot demonstrate it -- its drain synchronizes the pair, so it waits for the
successor by construction -- and production admission refuses the state in three
separate places. `run_retention_probe.h` describes the fixture that reaches it.

What makes the observation evidence rather than an assumption is that the same
sequence runs in two arms differing in one variable:

  candidate  the record is read directly;
  retained   the whole-pair synchronize runs first, which is what
             `wait_run_fence` still appends and what node E would remove.

A read that waited for the successor could not leave the successor's completion
boundary Pending. The retained arm shows what waiting looks like -- that
boundary reads Complete -- so the candidate arm's Pending is a measurement with
a demonstrated failure mode, not an untested expectation.

Onboard only: the property is about streams, and no simulated backend has them.
"""

import os

import pytest
from _task_interface import ChipCallable, ChipStorageTaskArgs
from simpler.task_interface import CallConfig, ChipWorker

from simpler_setup import ensure_pto_isa_root
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.runtime_builder import RuntimeBuilder

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH_DIR = os.path.join(HERE, "kernels", "orchestration")

# RunCompletionFence::Completion, as reported by the fixture.
PENDING = 0
COMPLETE = 1
# RunExecutionState, as decided by `decide_run_execution`.
SUCCEEDED = 1

_PREDECESSOR_SLOT = 0
_SUCCESSOR_SLOT = 1
_GENERATION = 1

# How many times the candidate arm may re-try for an overlap window. Nothing
# pins the successor in flight, so a preempted attempt observes no overlap and
# is uninformative rather than failing. A regression misses every attempt.
_WINDOW_ATTEMPTS = 5

_ORCH = "retention_orch.cpp"


def _build_callable(platform: str, runtime: str) -> ChipCallable:
    """Compile the orchestration-only callable this test runs.

    Orchestration-only on purpose: incore kernel sources are per-architecture,
    and one arch-neutral source keeps all four platform/runtime combinations on
    the same test rather than on four copies of it.
    """
    compiler = KernelCompiler(platform=platform)
    # Onboard orchestration needs the PTO-ISA headers even with no kernels.
    ensure_pto_isa_root()
    orch_bytes = compiler.compile_orchestration(runtime_name=runtime, source_path=os.path.join(ORCH_DIR, _ORCH))
    return ChipCallable.build(
        signature=[],
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=[],
    )


class _Fixture:
    """One initialized ChipWorker with the callable registered, plus the recipe.

    The recipe is the part production refuses, so it is spelled out once: the
    successor's prepare needs an already-launched predecessor, and the
    successor's launch needs a permit the fixture mints.
    """

    def __init__(self, platform: str, runtime: str, device_id: int):
        self.worker = ChipWorker()
        self.worker.init(device_id, RuntimeBuilder(platform).get_binaries(runtime))
        handle = self.worker.register_callable(_build_callable(platform, runtime))
        self.cid = self.worker._resolve_handle(handle).slot_id
        self.config = CallConfig()
        self.config.aicpu_thread_num = 2
        # Publishing AICore code marks the run stream stale and the next launch
        # replaces it, which would destroy the overlap under measurement. One
        # ordinary run settles that before any arm runs.
        self._run_once_normally()

    def _prepare(self, slot: int):
        return self.worker._prepare_native_run_with_pipeline_lease(
            self.cid, ChipStorageTaskArgs(), slot_id=slot, generation=_GENERATION, config=self.config
        )

    def _run_once_normally(self):
        run = self._prepare(_PREDECESSOR_SLOT)
        self.worker._launch_native_run(run)
        self.worker._wait_native_run(run)
        self.worker._finalize_native_run(run)

    def arm(self, *, launch_successor: bool = True, use_retained_sync: bool = False) -> dict:
        predecessor = self._prepare(_PREDECESSOR_SLOT)
        self.worker._launch_native_run(predecessor)
        successor = self._prepare(_SUCCESSOR_SLOT)
        try:
            return self.worker._probe_run_retention(
                predecessor,
                successor,
                launch_successor=launch_successor,
                use_retained_sync=use_retained_sync,
            )
        finally:
            # Both runs are owed a finalize whatever the fixture reported, and
            # the successor goes first: it is the one the fixture already
            # drained.
            self.worker._finalize_native_run(successor)
            self.worker._finalize_native_run(predecessor)

    def close(self):
        self.worker.finalize()


def _fixture_for(request, platform, device_ids, runtime):
    fixture = _Fixture(platform, runtime, int(device_ids[0]))
    request.addfinalizer(fixture.close)
    return fixture


@pytest.fixture
def retention_tmr(request, st_platform, st_device_ids):
    return _fixture_for(request, st_platform, st_device_ids, "tensormap_and_ringbuffer")


@pytest.fixture
def retention_hbg(request, st_platform, st_device_ids):
    return _fixture_for(request, st_platform, st_device_ids, "host_build_graph")


def _assert_no_step_failed(report):
    failures = {key: value for key, value in report.items() if key.endswith("_rc") and value != 0}
    assert not failures, f"the fixture reported a failed step: {failures}"


def _overlap_established(report):
    """Whether this attempt is usable at all.

    The successor has to be executing when the read begins, and nothing pins it
    there: on fast silicon or a loaded runner it can finish between its own
    launch and the predecessor's read. That is a **precondition**, not the claim,
    so an attempt that fails it says nothing either way and is discarded rather
    than failed. Both arms share the rule; measured on a5 CI, where the
    successor's original workload was too short to survive a busy runner.
    """
    return report["successor_started"] and report["successor_completion_before_read"] == PENDING


def _assert_predecessor_decided(report):
    """The predecessor's own result must be valid, whatever the successor did.

    Independent of the overlap window, and therefore checked before any attempt
    is discarded for lacking one. A failed copy-back or a stale epoch leaves the
    record undecided, which is the retention failure this test exists to catch --
    and it reaches the caller only here, because `read_device_run_result` returns
    void and no `_rc` field carries it.
    """
    assert report["execution_state"] == SUCCEEDED, (
        f"the predecessor's record did not decide its run: state={report['execution_state']} "
        f"code={report['execution_code']} reason={report['execution_reason']!r}"
    )


def check_result_reads_while_successor_runs(retention):
    """The candidate arm: N's verdict is decided while N+1 is still executing.

    A single sample can never prove the read waited. Nothing pins the successor
    in flight, so a `Complete` boundary after the read has two possible causes --
    the read waited for the successor, or the successor simply finished on its
    own -- and **after the fact neither leaves distinguishable evidence**. Its
    remaining drain does not separate them either: a drain costs time even when
    the device is already done (the control arm measures exactly that, at 14 us),
    so no threshold on it can tell "still had work" from "already finished".

    So this asserts the property only over repeated attempts, and never calls a
    single miss a regression:

      - an attempt whose overlap never existed is discarded outright;
      - an attempt observing Pending across the read is a pass, because a read
        that waited could not have produced it;
      - an attempt observing Complete is discarded as inconclusive;
      - all attempts inconclusive is a failure, but one reported as
        "never observed" rather than as a proven regression, because that is
        what the evidence supports.

    A real regression makes the read wait every time, so it exhausts the budget
    and fails. A preempted run costs one attempt.
    """
    attempts = []
    for _ in range(_WINDOW_ATTEMPTS):
        report = retention.arm()
        _assert_no_step_failed(report)

        # Checked before any window filtering, because the predecessor's result
        # has to be valid whatever the successor did. `_assert_no_step_failed`
        # cannot cover this: `read_device_run_result` returns void and reports a
        # failed copy or a stale epoch through the terminal state instead. Letting
        # a discarded attempt skip it would hide a real retention failure behind a
        # later attempt that happened to catch the window.
        _assert_predecessor_decided(report)

        if not _overlap_established(report):
            attempts.append("no-overlap")
            continue

        if report["successor_completion_after_read"] != PENDING:
            attempts.append("completed-across-read")
            continue

        # Pending across the read: the read did not wait, and the record it read
        # decided the run.
        return

    raise AssertionError(
        f"in {_WINDOW_ATTEMPTS} attempts the successor was never still Pending after the predecessor's "
        f"read (attempts: {attempts}). Every attempt's predecessor result was valid, so this is about "
        "the overlap window, not the read: 'no-overlap' means the successor finished before the read "
        "even began, so its workload no longer outlives the read on this box; 'completed-across-read' "
        "means either the read now waits for the successor or the successor finished during it, which "
        "nothing recorded here distinguishes. The per-arm timings are diagnostics for that, not "
        "evidence: a drain costs time on an already-finished run too."
    )


def check_retained_synchronize_waits_for_the_successor(retention):
    """The control: the call node E removes does wait, so Pending above means something.

    `COMPLETE` here is what the synchronize is supposed to cause, and unlike the
    candidate arm's `PENDING` it is not self-proving: a successor that finished on
    its own would produce it too. Its duration does not settle that either --
    a synchronize costs time on an already-drained pair.

    So this arm asserts what it can: that after the synchronize the successor is
    always Complete, on every attempt whose overlap was established. That is a
    real regression barrier -- a synchronize that stopped waiting would leave a
    Pending -- and it is deliberately weaker than the candidate arm's claim. The
    quantitative separation between the arms lives in the probe report, not in an
    assertion.
    """
    usable = 0
    for _ in range(_WINDOW_ATTEMPTS):
        report = retention.arm(use_retained_sync=True)
        _assert_no_step_failed(report)
        # Before the window filter, for the reason the candidate arm gives: the
        # predecessor's result validity does not depend on the successor, and
        # waiting for the successor must not change what its record says.
        _assert_predecessor_decided(report)
        if not _overlap_established(report):
            continue
        usable += 1

        assert report["successor_completion_after_read"] == COMPLETE, (
            "the whole-pair synchronize left the successor's boundary Pending, so it did not wait "
            "for the successor and no longer controls the candidate arm's Pending observation"
        )
        # It also has to have actually run: a zero here would mean the arm
        # measured nothing.
        assert report["reference_sync_ns"] > 0

    assert usable > 0, (
        f"in {_WINDOW_ATTEMPTS} attempts the successor never reached the read still executing, so this "
        "arm controlled nothing. The successor's workload no longer outlives the read on this box."
    )


def check_record_decides_the_run_with_no_successor(retention):
    """The baseline: the same read, with nothing overlapping it."""
    report = retention.arm(launch_successor=False)
    _assert_no_step_failed(report)
    assert not report["successor_started"], "no successor was launched, so none can have started"
    assert report["execution_state"] == SUCCEEDED
    # Per-run cost, reported rather than bounded: a threshold here would be a
    # guess about this box, and the number's purpose is to be recorded.
    assert report["record_read_ns"] > 0
    assert report["candidate_drain_ns"] >= report["record_read_ns"]


# Both runtimes publish the record and both own the streams, so each check runs
# against each. The runtime marker is what the session's runtime filter reads;
# the checks themselves are runtime-agnostic.


@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_result_reads_while_successor_runs_tmr(retention_tmr):
    check_result_reads_while_successor_runs(retention_tmr)


@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_retained_synchronize_waits_for_the_successor_tmr(retention_tmr):
    check_retained_synchronize_waits_for_the_successor(retention_tmr)


@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_record_decides_the_run_with_no_successor_tmr(retention_tmr):
    check_record_decides_the_run_with_no_successor(retention_tmr)


@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("host_build_graph")
def test_result_reads_while_successor_runs_hbg(retention_hbg):
    check_result_reads_while_successor_runs(retention_hbg)


@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("host_build_graph")
def test_retained_synchronize_waits_for_the_successor_hbg(retention_hbg):
    check_retained_synchronize_waits_for_the_successor(retention_hbg)


@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("host_build_graph")
def test_record_decides_the_run_with_no_successor_hbg(retention_hbg):
    check_record_decides_the_run_with_no_successor(retention_hbg)
