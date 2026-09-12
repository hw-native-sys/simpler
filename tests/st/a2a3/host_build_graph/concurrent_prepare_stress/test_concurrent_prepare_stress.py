#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Concurrent-prepare overlap stress for host_build_graph.

Drives a 2-deep native-run pipeline over the two arena banks: run i is
*prepared* (which runs its full bind — arena build + host orchestration) while
run i-1 is still launched-but-not-finalized. That is exactly the
``overlaps_active_run`` path (a successor prepared into bank B while a
predecessor executes in bank A), which the ordinary blocking run() never hits.

Each iteration uses distinct input data, so any cross-run interference between
the two banks' runtimes/orchestrators (e.g. a shared or under-initialized
orchestrator) would corrupt a result and fail the golden check. The point is to
exercise the concurrent-prepare bind path under many alternating-bank
iterations, not to measure anything.

Four arms drive the same pipeline through :meth:`_drive_pipeline` and read the
same ``assert_native_overlap`` verdict, so each negative arm differs from the
positive one in exactly one variable:

* ``inflight_limit=2``, ordinary config — overlap is required.
* ``inflight_limit=1`` — the lane's capability is untouched and the submissions
  simply never coexist, so the verdict must be *rejected*. This is what makes
  the positive arm a detector rather than a formality: the span chain between
  the pipeline and the verdict (which spans are emitted, where their endpoints
  land, ``bind`` standing in for preparation, ``runner_run`` being a host wall
  span) could otherwise report an intersection independent of real concurrency.
* ``inflight_limit=2`` with a diagnostics config — overlap is required here too.
  A collector's pools and per-run state are built and reset under the execution
  claim, so a diagnostic flag no longer keeps a run and its successor on
  separate device windows.
* ``inflight_limit=2`` at chip-swimlane level 4 — *rejected*, and the one
  diagnostics config that still serializes. Its bind opens a clock-correlation
  session on the resident swimlane collector and samples an anchor into it,
  which is not per-run and cannot move under the claim without losing its
  meaning.

The last two are what hold that boundary in place. It is invisible from goldens:
the lane declines to stage silently rather than raising, so a run that lost its
overlap, or one that kept an overlap it should not have, looks identical from
outside.
"""

import contextlib
import tempfile

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_chip_task_args, _compare_outputs
from simpler_setup.tools.strace_timing import NativeOverlapError, assert_native_overlap, parse_spans

_VECTOR_KERNELS = "../vector_example/kernels"
_SIZE = 128 * 128
_ITERS = 40  # 20 reuses per bank
# A negative arm needs only enough adjacent pairs for the analyzer to have one
# to reject; the stress count buys nothing once the property is absent.
_CONTROL_ITERS = 3
# This class's st_worker is class-scoped (see the sibling conftest, which
# overrides the pooled default), so all four of its tests share one slot table.
# The framework's inherited test_run registers through the ordinary path and so
# holds id 0 for the rest of the class. The arms below therefore own a separate
# registry id (capacity is 64), which keeps their register/unregister pairs off
# id 0 whatever order the four tests run in.
_ARM_CALLABLE_ID = 1


@scene_test(level=2, runtime="host_build_graph")
class TestConcurrentPrepareStressHbg(SceneTestCase):
    """Prepare-while-active stress over the two pipeline banks (a2a3 onboard)."""

    CALLABLE = {
        "orchestration": {
            "source": f"{_VECTOR_KERNELS}/orchestration/example_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{_VECTOR_KERNELS}/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": f"{_VECTOR_KERNELS}/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": f"{_VECTOR_KERNELS}/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    _PLATFORMS = ["a2a3"]

    CASES = [
        {
            "name": "overlap_stress",
            "platforms": _PLATFORMS,
            # Consumed by the framework's default test_run (a plain blocking run);
            # the overlap loop below builds its own per-iteration params.
            "params": {"a": 2.0, "b": 3.0},
        },
    ]

    def generate_args(self, params):
        """Build the two constant input vectors and the zero output for ``params``."""
        a, b = params["a"], params["b"]
        return TaskArgsBuilder(
            TensorArg("a", torch.full((_SIZE,), a, dtype=torch.float32)),
            TensorArg("b", torch.full((_SIZE,), b, dtype=torch.float32)),
            TensorArg("f", torch.zeros(_SIZE, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        """Reference output: example_orch computes ``(a + b + 1) * (a + b + 2)``."""
        a, b = args.a, args.b
        args.f[:] = (a + b + 1) * (a + b + 2)

    def _chip_worker(self, worker):
        """Return the L2 ChipWorker backing ``worker``, ready to stage a successor.

        The property folds ``initialized_``, the runtime PipelineContract's
        ``pipeline_depth > 1``, and the runtime's concurrent-prepare capability
        symbol. On a2a3 host_build_graph all three hold unconditionally — the
        contract declares depth 2 (as does every onboard runtime, and
        PTO_PIPELINE_MAX_DEPTH is 2) and the capability impl returns 1 outright —
        so a false here means one of them regressed, not that this box is
        differently configured. Neither is reachable from a CallConfig, which is
        why this is an assert and not a skip: skipping would report green for the
        one state in which the arms below cannot hold.

        The sim platform hardcodes the capability to 0, so overlap never happens
        there. Each arm's platform gate runs before this assert for that reason.
        """
        chip_worker = worker._chip_worker
        assert chip_worker is not None
        assert chip_worker._impl.supports_concurrent_native_prepare, (
            f"a2a3 host_build_graph must be able to stage a successor, have pipeline_depth={chip_worker.pipeline_depth}"
        )
        return chip_worker

    @contextlib.contextmanager
    def _registered_callable(self, chip_worker, callable_id, callable_obj):
        """Hold ``callable_obj`` in one slot for the duration of the block."""
        chip_worker._register_callable_at_slot(callable_id, callable_obj)
        try:
            yield
        finally:
            chip_worker._unregister_slot(callable_id)

    def _submit_iteration(self, chip_worker, callable_id, orch_sig, config, iteration):
        """Submit run ``iteration`` with its own data; return its in-flight entry.

        Each in-flight entry keeps BOTH the encoded chip_args and the backing
        tensors (test_args) alive until the run reaches terminal: the lane copies
        inputs but the buffers back the resolved descriptors, and the output is
        copied back at finalize.
        """
        # Distinct inputs per iteration so golden catches cross-bank interference.
        params = {"a": 1.0 + iteration, "b": 2.0 + 0.5 * iteration}
        test_args = self.generate_args(params)
        chip_args, output_names = _build_chip_task_args(test_args, orch_sig)
        golden_args = test_args.clone()
        self.compute_golden(golden_args, params)
        chip_run = chip_worker._impl._submit_chip_run_direct(callable_id, chip_args, config)
        return (chip_run, test_args, golden_args, output_names, chip_args, iteration)

    def _retire(self, entry):
        """Block one run to terminal (lane finalizes + copies back) and golden-check it."""
        chip_run, test_args, golden_args, output_names, _chip_args, _iteration = entry
        # Block to the completion fence; the lane finalizes (validate +
        # copy-back) as part of reaching terminal.
        chip_run.wait(-1.0)
        # Distinct per-iteration data → a corrupted/aliased bank fails this.
        _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

    def _drive_pipeline(self, chip_worker, callable_id, orch_sig, config, iters, inflight_limit):
        """Submit ``iters`` runs, never letting more than ``inflight_limit`` coexist.

        The direct-chip lane is the sole admission authority and follows the
        runtime PipelineContract: it admits one active plus one prepared
        compatible successor. At ``inflight_limit=2`` a submission therefore
        lands while its predecessor is still in flight, and the predecessor's
        ``wait`` prepares it (its full bind — arena build + host orchestration)
        against the other bank, i.e. the overlaps_active_run path. At
        ``inflight_limit=1`` every run reaches terminal before the next is
        submitted, so no bind can coexist with a device window.
        """
        inflight = []
        for iteration in range(iters):
            inflight.append(self._submit_iteration(chip_worker, callable_id, orch_sig, config, iteration))
            if len(inflight) >= inflight_limit:
                self._retire(inflight.pop(0))
        while inflight:
            self._retire(inflight.pop(0))

    def test_concurrent_prepare_overlap(self, st_platform, st_worker, capfd, drain_host_log):
        """Golden-check a 2-deep overlapping pipeline over both arena banks.

        Each submission prepares (fully binds) its run against one bank while the
        predecessor is still active in the other, exercising the
        ``overlaps_active_run`` path many times with distinct data.
        """
        if st_platform != "a2a3":
            pytest.skip("concurrent native prepare / two-bank pipeline is an a2a3 onboard path")

        orch_sig = self.CALLABLE["orchestration"]["signature"]
        callable_obj = self.build_callable(st_platform)
        config = self._build_config({})
        chip_worker = self._chip_worker(st_worker)
        callable_id = _ARM_CALLABLE_ID

        with self._registered_callable(chip_worker, callable_id, callable_obj):
            drain_host_log(capfd)  # only this arm's spans reach the verdict
            self._drive_pipeline(chip_worker, callable_id, orch_sig, config, _ITERS, inflight_limit=2)
            captured = drain_host_log(capfd)

        # Each pair must prove prepare(N+1) ran concurrently with device(N) — the
        # property this pipeline exists for. Not that prepare *finished* inside
        # that window (`require_hidden=True`): that additionally depends on how
        # much host CPU this shared machine gives the preparing thread.
        checks = assert_native_overlap(parse_spans(captured.splitlines()))
        assert len(checks) == _ITERS - 1

    def test_serial_submission_is_rejected_as_not_overlapping(self, st_platform, st_worker, capfd, drain_host_log):
        """One run at a time must fail the same assertion the overlapping arm passes.

        The single-variable control for the positive arm: same lane, same
        capability, same config, one fewer run in flight. Nothing can then
        prepare while a device window is open, so an assertion that still
        reported overlap would be measuring the span chain rather than the
        pipeline. Matching the message is load-bearing — it separates a real
        rejection from the vacuous "no pairs to check" one.

        This arm names no mechanism, so it outlives any of them: one run at a
        time cannot overlap under any admission policy. Contrast the diagnostics
        arm below, which is tied to a fallback with an expiry date.
        """
        if st_platform != "a2a3":
            pytest.skip("concurrent native prepare / two-bank pipeline is an a2a3 onboard path")

        orch_sig = self.CALLABLE["orchestration"]["signature"]
        callable_obj = self.build_callable(st_platform)
        config = self._build_config({})
        chip_worker = self._chip_worker(st_worker)
        callable_id = _ARM_CALLABLE_ID

        with self._registered_callable(chip_worker, callable_id, callable_obj):
            drain_host_log(capfd)
            self._drive_pipeline(chip_worker, callable_id, orch_sig, config, _CONTROL_ITERS, inflight_limit=1)
            captured = drain_host_log(capfd)

        with pytest.raises(NativeOverlapError, match="did not overlap"):
            assert_native_overlap(parse_spans(captured.splitlines()))

    def test_diagnostics_config_still_overlaps_the_native_lane(self, st_platform, st_worker, capfd, drain_host_log):
        """A diagnostic flag does not serialize the lane, and the log must overlap.

        ``allow_prepared_successor`` used to fold in ``CallConfig::diagnostics_any()``
        — the OR of all five diagnostic flags — because a collector's setup wrote
        runner-global state during preparation, which a prepared successor would
        have done while its predecessor was still running against it. The pools
        and that per-run state are now built and reset by
        ``arm_collectors_for_run()`` under the execution claim, so the successor's
        preparation touches neither and the configuration overlaps like any other.

        This arm is the only thing holding that open. The property is invisible
        from goldens — the lane declines to stage silently rather than raising, so
        a regression would let every diagnostic run fall back to depth one with
        the whole suite still green.

        Host spans are gated separately (compile-time ``SIMPLER_HOST_STRACE``),
        so the trace still comes out with device diagnostics on.
        """
        if st_platform != "a2a3":
            pytest.skip("concurrent native prepare / two-bank pipeline is an a2a3 onboard path")

        orch_sig = self.CALLABLE["orchestration"]["signature"]
        callable_obj = self.build_callable(st_platform)
        chip_worker = self._chip_worker(st_worker)
        callable_id = _ARM_CALLABLE_ID

        with tempfile.TemporaryDirectory(prefix="simpler-diagnostics-lane-") as output_dir:
            # scope_stats is the lightest of the five flags; which one is set does
            # not matter, only that diagnostics_any() becomes true. output_prefix
            # is required by CallConfig::validate() whenever one of them is.
            config = self._build_config({}, enable_scope_stats=True, output_prefix=output_dir)
            with self._registered_callable(chip_worker, callable_id, callable_obj):
                drain_host_log(capfd)
                self._drive_pipeline(chip_worker, callable_id, orch_sig, config, _CONTROL_ITERS, inflight_limit=2)
                captured = drain_host_log(capfd)

        assert_native_overlap(parse_spans(captured.splitlines()))

    def test_orch_phase_swimlane_serializes_the_native_lane(self, st_platform, st_worker, capfd, drain_host_log):
        """Level-4 swimlane is the one diagnostics config that still serializes.

        A host-orchestrating runtime opens a clock-correlation session on the
        *resident* swimlane collector, and samples its ``HostOrchestrationBegin``
        anchor into it, from inside bind — that is, from inside preparation,
        which a prepared successor performs while its predecessor is still
        executing. Unlike the collector pools, that state cannot move under the
        execution claim: the anchor's whole meaning is when host orchestration
        began. Until it is per-run, a level-4 run neither carries a prepared
        successor nor is one, and the verdict must be *rejected*.

        This is the arm that would go quiet if the exclusion were dropped without
        making the session per-run: the lane declines to stage rather than
        raising, so the runs would still succeed and their goldens still pass
        while the predecessor's clock anchors were destroyed by the successor's
        bind. It retires when that state becomes per-run, and then it flips to
        the same `assert_native_overlap` its sibling arm above already makes.
        """
        if st_platform != "a2a3":
            pytest.skip("concurrent native prepare / two-bank pipeline is an a2a3 onboard path")

        orch_sig = self.CALLABLE["orchestration"]["signature"]
        callable_obj = self.build_callable(st_platform)
        chip_worker = self._chip_worker(st_worker)
        callable_id = _ARM_CALLABLE_ID

        with tempfile.TemporaryDirectory(prefix="simpler-orch-phase-lane-") as output_dir:
            # 4 is ChipSwimlaneLevel::ORCH_PHASES, the only level whose bind arms
            # host-orchestration phase state on the resident collector.
            config = self._build_config({}, enable_chip_swimlane=4, output_prefix=output_dir)
            with self._registered_callable(chip_worker, callable_id, callable_obj):
                drain_host_log(capfd)
                self._drive_pipeline(chip_worker, callable_id, orch_sig, config, _CONTROL_ITERS, inflight_limit=2)
                captured = drain_host_log(capfd)

        with pytest.raises(NativeOverlapError, match="did not overlap"):
            assert_native_overlap(parse_spans(captured.splitlines()))
