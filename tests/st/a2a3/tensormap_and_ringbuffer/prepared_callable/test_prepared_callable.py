#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end test for Worker.prepare_callable / run / unregister_callable.

Reuses the vector_example orchestration + AIV kernels. Exercises:
  - prepare_callable once, then run twice (second run proves the
    AICPU-side dlopen cache / host-side orch SO dedup is working — no re-upload).
  - Two distinct handles sharing the same callable identity: verifies both
    produce correct output independently.
  - unregister_callable after runs complete: should not raise.
  - aicpu_dlopen_count assertions covering: same-handle repeat, duplicate
    handle interleaving, deduped duplicate prepare, and unregister + re-prepare.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _build_chip_task_args, _compare_outputs

_VECTOR_KERNELS = "../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"

@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestPreparedCallable(SceneTestCase):
    """Exercise prepare_callable / run / unregister_callable ABI.

    Requires an isolated L2 ``Worker`` (cid table starts empty); this is
    provided by the directory-local ``conftest.py`` overriding ``st_worker``
    with a class-scope fixture.
    """

    CALLABLE = {
        "orchestration": {
            "source": f"{_VECTOR_KERNELS}/orchestration/example_orchestration.cpp",
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

    _COMMON_CONFIG = {"aicpu_thread_num": 4, "block_dim": 3}
    _PLATFORMS = ["a2a3sim", "a2a3"]

    CASES = [
        {
            "name": "prepare_run_twice",
            "platforms": _PLATFORMS,
            "config": _COMMON_CONFIG,
            "params": {"a": 2.0, "b": 3.0},
        },
    ]

    def generate_args(self, params):
        size = 128 * 128
        a, b = params["a"], params["b"]
        return TaskArgsBuilder(
            Tensor("a", torch.full((size,), a, dtype=torch.float32)),
            Tensor("b", torch.full((size,), b, dtype=torch.float32)),
            Tensor("f", torch.zeros(size, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.f[:] = (args.a + args.b + 1) * (args.a + args.b + 2) + (args.a + args.b)

    def _run_and_validate_l2(  # noqa: PLR0913
        self,
        worker,
        callable_obj,
        case,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=False,
        enable_dump_tensor=False,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
        output_prefix="",
    ):
        params = case.get("params", {})
        config_dict = case.get("config", {})
        orch_sig = self.CALLABLE.get("orchestration", {}).get("signature", [])

        config = self._build_config(config_dict)

        # 1) prepare two handles with the SAME callable identity.
        primary = worker.prepare_callable(callable_obj)
        secondary = worker.prepare_callable(callable_obj)

        # 2) run primary handle twice (second run proves dedup/cache hit)
        for _ in range(2):
            test_args = self.generate_args(params)
            chip_args, output_names = _build_chip_task_args(test_args, orch_sig)
            golden_args = test_args.clone()
            self.compute_golden(golden_args, params)

            worker.run(primary, chip_args, config=config)
            _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

        # 3) run secondary handle, which shares the same callable identity.
        test_args = self.generate_args(params)
        chip_args, output_names = _build_chip_task_args(test_args, orch_sig)
        golden_args = test_args.clone()
        self.compute_golden(golden_args, params)

        worker.run(secondary, chip_args, config=config)
        _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

        # 4) unregister both — should not raise
        worker.unregister_callable(primary)
        worker.unregister_callable(secondary)

    # ------------------------------------------------------------------
    # aicpu_dlopen_count assertions.
    #
    # The class-scope L2 worker is shared across test methods in this
    # class (see ./conftest.py), so the counter can be non-zero on entry
    # from prior methods. Each test below snapshots the counter on entry,
    # asserts the *delta* introduced by the scenario, then unregisters
    # everything it staged. unregister_callable does NOT decrement the
    # counter (the counter is monotonic — see test_dlopen_count_unregister_re_prepare).
    # ------------------------------------------------------------------

    def _setup_dlopen_count_test(self, st_worker, st_platform):
        """Common fixture: build callable + config, return (callable, config, case)."""
        case = self.CASES[0]
        callable_obj = self.build_callable(st_platform)
        config = self._build_config(case["config"])
        return callable_obj, config, case

    def _run_one(self, worker, handle, callable_obj, config, case):
        params = case["params"]
        orch_sig = self.CALLABLE["orchestration"]["signature"]
        test_args = self.generate_args(params)
        chip_args, output_names = _build_chip_task_args(test_args, orch_sig)
        golden_args = test_args.clone()
        self.compute_golden(golden_args, params)
        worker.run(handle, chip_args, config=config)
        _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

    def test_dlopen_count_same_cid_repeated_runs(self, st_platform, st_worker):
        """Case A: prepare(primary) + run x5 -> dlopen_count delta == 1."""
        callable_obj, config, case = self._setup_dlopen_count_test(st_worker, st_platform)
        baseline = st_worker.aicpu_dlopen_count
        primary = None
        try:
            primary = st_worker.prepare_callable(callable_obj)
            for _ in range(5):
                self._run_one(st_worker, primary, callable_obj, config, case)
            assert st_worker.aicpu_dlopen_count - baseline == 1, (
                f"expected exactly 1 new dlopen for 5 runs of primary handle, "
                f"got delta {st_worker.aicpu_dlopen_count - baseline}"
            )
        finally:
            if primary is not None:
                st_worker.unregister_callable(primary)

    def test_dlopen_count_two_cids_alternating(self, st_platform, st_worker):
        """Case B: duplicate handles share one prepared callable identity."""
        callable_obj, config, case = self._setup_dlopen_count_test(st_worker, st_platform)
        baseline = st_worker.aicpu_dlopen_count
        primary = None
        secondary = None
        try:
            primary = st_worker.prepare_callable(callable_obj)
            secondary = st_worker.prepare_callable(callable_obj)
            for _ in range(5):
                self._run_one(st_worker, primary, callable_obj, config, case)
                self._run_one(st_worker, secondary, callable_obj, config, case)
            assert st_worker.aicpu_dlopen_count - baseline == 1, (
                f"expected exactly 1 new dlopen for duplicate handle interleaving, "
                f"got delta {st_worker.aicpu_dlopen_count - baseline}"
            )
        finally:
            if secondary is not None:
                st_worker.unregister_callable(secondary)
            if primary is not None:
                st_worker.unregister_callable(primary)

    def test_dlopen_count_duplicate_prepare_dedups(self, st_platform, st_worker):
        """Case C: duplicate prepare returns another handle for the same identity."""
        callable_obj, _config, _case = self._setup_dlopen_count_test(st_worker, st_platform)
        first = None
        second = None
        try:
            first = st_worker.prepare_callable(callable_obj)
            second = st_worker.prepare_callable(callable_obj)
            assert first is not second
            assert first.hashid == second.hashid
        finally:
            if second is not None:
                st_worker.unregister_callable(second)
            if first is not None:
                st_worker.unregister_callable(first)

    def test_dedup_shared_so_independent_unregister(self, st_platform, st_worker):
        """Case E: two handles on the same ChipCallable share one device orch SO buffer.

        Build-ID-keyed dedup in DeviceRunner refcounts the buffer; unregistering
        one handle must not invalidate the other. Run-after-unregister proves the
        shared buffer is still alive (a missing refcount would either crash or
        produce incorrect results when the second handle dispatches into a freed
        device region).
        """
        callable_obj, config, case = self._setup_dlopen_count_test(st_worker, st_platform)
        primary = None
        secondary = None
        try:
            primary = st_worker.prepare_callable(callable_obj)
            secondary = st_worker.prepare_callable(callable_obj)
            # Sanity: both handles work before any unregister.
            self._run_one(st_worker, primary, callable_obj, config, case)
            self._run_one(st_worker, secondary, callable_obj, config, case)
            # Drop primary; secondary's run must still succeed because
            # the dedup refcount is still > 0.
            st_worker.unregister_callable(primary)
            primary = None
            self._run_one(st_worker, secondary, callable_obj, config, case)
        finally:
            if secondary is not None:
                st_worker.unregister_callable(secondary)
            if primary is not None:
                st_worker.unregister_callable(primary)

    def test_dlopen_count_unregister_re_prepare(self, st_platform, st_worker):
        """Case D: prepare+run+unregister+prepare+run -> delta == 2.

        unregister erases the slot from aicpu_seen_callable_ids_, so the second
        prepare/run pair sets register_new_callable_id_ again and the AICPU
        does a fresh dlopen. The counter is monotonic (does NOT decrement on
        unregister), so the delta after the second cycle is 2.
        """
        callable_obj, config, case = self._setup_dlopen_count_test(st_worker, st_platform)
        baseline = st_worker.aicpu_dlopen_count
        handle = None
        try:
            handle = st_worker.prepare_callable(callable_obj)
            self._run_one(st_worker, handle, callable_obj, config, case)
            assert st_worker.aicpu_dlopen_count - baseline == 1
            st_worker.unregister_callable(handle)
            handle = None
            after_unreg = st_worker.aicpu_dlopen_count
            assert after_unreg - baseline == 1, (
                f"unregister must NOT decrement the dlopen counter; baseline={baseline}, after_unreg={after_unreg}"
            )
            handle = st_worker.prepare_callable(callable_obj)
            self._run_one(st_worker, handle, callable_obj, config, case)
            assert st_worker.aicpu_dlopen_count - baseline == 2, (
                f"after re-prepare expected counter +2 (two distinct AICPU dlopens), "
                f"got delta {st_worker.aicpu_dlopen_count - baseline}"
            )
        finally:
            if handle is not None:
                st_worker.unregister_callable(handle)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
