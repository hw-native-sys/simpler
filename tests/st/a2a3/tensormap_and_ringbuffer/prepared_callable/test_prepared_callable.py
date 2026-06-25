#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end white-box test for the private L2 prepared-callable ABI.

Reuses the vector_example orchestration + AIV kernels. Exercises:
  - prepare one private callable slot, then run twice (prepare proves prewarm;
    both runs prove the
    AICPU-side dlopen cache / host-side orch SO dedup is working — no re-upload).
  - Two distinct private callable slots sharing the same callable: verifies both
    produce correct output independently.
  - private slot unregister after runs complete: should not raise.
  - aicpu_dlopen_count assertions covering: same-slot repeat, multi-slot
    interleaving, double-prepare rejection, and unregister + re-prepare.
"""

import copy

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _build_chip_task_args, _compare_outputs

_VECTOR_KERNELS = "../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"
_SLOT_PRIMARY = 0
_SLOT_SECONDARY = 1


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestPreparedCallable(SceneTestCase):
    """Exercise private prepare / run / unregister slot ABI.

    Requires an isolated L2 ``Worker`` (private slot table starts empty); this is
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

    def _chip_worker(self, worker):
        chip_worker = worker._chip_worker
        assert chip_worker is not None
        return chip_worker

    def _run_and_validate_l2(  # noqa: PLR0913
        self,
        worker,
        callable_obj,
        case,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=False,
        enable_dump_args=False,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
        output_prefix="",
    ):
        params = case.get("params", {})
        config_dict = case.get("config", {})
        orch_sig = self.CALLABLE.get("orchestration", {}).get("signature", [])

        config = self._build_config(config_dict)
        chip_worker = self._chip_worker(worker)

        # 1) prepare two private slots with the SAME callable.
        chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, callable_obj)
        chip_worker._prepare_callable_at_slot(_SLOT_SECONDARY, callable_obj)

        # 2) run primary slot twice (second run proves dedup/cache hit)
        for _ in range(2):
            test_args = self.generate_args(params)
            chip_args, output_names = _build_chip_task_args(test_args, orch_sig)
            golden_args = test_args.clone()
            self.compute_golden(golden_args, params)

            chip_worker._run_slot(_SLOT_PRIMARY, chip_args, config=config)
            _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

        # 3) run secondary slot — different slot, same SO, must also work.
        test_args = self.generate_args(params)
        chip_args, output_names = _build_chip_task_args(test_args, orch_sig)
        golden_args = test_args.clone()
        self.compute_golden(golden_args, params)

        chip_worker._run_slot(_SLOT_SECONDARY, chip_args, config=config)
        _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

        # 4) unregister both — should not raise.
        chip_worker._unregister_slot(_SLOT_PRIMARY)
        chip_worker._unregister_slot(_SLOT_SECONDARY)

    # ------------------------------------------------------------------
    # aicpu_dlopen_count assertions.
    #
    # The class-scope L2 worker is shared across test methods in this
    # class (see ./conftest.py), so the counter can be non-zero on entry
    # from prior methods. Each test below snapshots the counter on entry,
    # asserts the *delta* introduced by the scenario, then unregisters
    # everything it staged. unregister does NOT decrement the
    # counter (the counter is monotonic — see test_dlopen_count_unregister_re_prepare).
    # ------------------------------------------------------------------

    def _setup_dlopen_count_test(self, st_worker, st_platform):
        """Common fixture: build callable + config, return (callable, config, case)."""
        case = self.CASES[0]
        callable_obj = self.build_callable(st_platform)
        config = self._build_config(case["config"])
        return callable_obj, config, case

    def _run_one(self, worker, slot, config, case):
        params = case["params"]
        orch_sig = self.CALLABLE["orchestration"]["signature"]
        test_args = self.generate_args(params)
        chip_args, output_names = _build_chip_task_args(test_args, orch_sig)
        golden_args = test_args.clone()
        self.compute_golden(golden_args, params)
        self._chip_worker(worker)._run_slot(slot, chip_args, config=config)
        _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

    def test_dlopen_count_same_slot_repeated_runs(self, st_platform, st_worker):
        """Case A: prepare(primary) prewarms once; run x5 adds no dlopens."""
        callable_obj, config, case = self._setup_dlopen_count_test(st_worker, st_platform)
        baseline = st_worker.aicpu_dlopen_count
        prepared = False
        chip_worker = self._chip_worker(st_worker)
        try:
            chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, callable_obj)
            prepared = True
            assert st_worker.aicpu_dlopen_count - baseline == 1
            for _ in range(5):
                self._run_one(st_worker, _SLOT_PRIMARY, config, case)
                assert st_worker.aicpu_dlopen_count - baseline == 1
            assert st_worker.aicpu_dlopen_count - baseline == 1, (
                f"expected exactly 1 new dlopen for 5 runs of primary slot, "
                f"got delta {st_worker.aicpu_dlopen_count - baseline}"
            )
        finally:
            if prepared:
                chip_worker._unregister_slot(_SLOT_PRIMARY)

    def test_dlopen_count_two_slots_alternating(self, st_platform, st_worker):
        """Case B: two prepares prewarm two slots; alternating runs add none."""
        callable_obj, config, case = self._setup_dlopen_count_test(st_worker, st_platform)
        baseline = st_worker.aicpu_dlopen_count
        primary_prepared = False
        secondary_prepared = False
        chip_worker = self._chip_worker(st_worker)
        try:
            chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, callable_obj)
            primary_prepared = True
            assert st_worker.aicpu_dlopen_count - baseline == 1
            chip_worker._prepare_callable_at_slot(_SLOT_SECONDARY, callable_obj)
            secondary_prepared = True
            assert st_worker.aicpu_dlopen_count - baseline == 2
            for _ in range(5):
                self._run_one(st_worker, _SLOT_PRIMARY, config, case)
                self._run_one(st_worker, _SLOT_SECONDARY, config, case)
                assert st_worker.aicpu_dlopen_count - baseline == 2
            assert st_worker.aicpu_dlopen_count - baseline == 2, (
                f"expected exactly 2 new dlopens for two slots interleaved, "
                f"got delta {st_worker.aicpu_dlopen_count - baseline}"
            )
        finally:
            if secondary_prepared:
                chip_worker._unregister_slot(_SLOT_SECONDARY)
            if primary_prepared:
                chip_worker._unregister_slot(_SLOT_PRIMARY)

    def test_dlopen_count_double_prepare_raises(self, st_platform, st_worker):
        """Case C: prepare(primary) + prepare(primary) -> second call raises RuntimeError."""
        callable_obj, _config, _case = self._setup_dlopen_count_test(st_worker, st_platform)
        prepared = False
        chip_worker = self._chip_worker(st_worker)
        try:
            chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, callable_obj)
            prepared = True
            with pytest.raises(RuntimeError):
                chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, callable_obj)
        finally:
            if prepared:
                chip_worker._unregister_slot(_SLOT_PRIMARY)

    def test_dedup_shared_so_independent_unregister(self, st_platform, st_worker):
        """Case E: two slots on the same ChipCallable share one device orch SO buffer.

        Build-ID-keyed dedup in DeviceRunner refcounts the buffer; unregistering
        one slot must not invalidate the other. Run-after-unregister proves the
        shared buffer is still alive (a missing refcount would either crash or
        produce incorrect results when the second slot dispatches into a freed
        device region).
        """
        callable_obj, config, case = self._setup_dlopen_count_test(st_worker, st_platform)
        primary_prepared = False
        secondary_prepared = False
        chip_worker = self._chip_worker(st_worker)
        try:
            chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, callable_obj)
            primary_prepared = True
            chip_worker._prepare_callable_at_slot(_SLOT_SECONDARY, callable_obj)
            secondary_prepared = True
            # Sanity: both slots work before any unregister.
            self._run_one(st_worker, _SLOT_PRIMARY, config, case)
            self._run_one(st_worker, _SLOT_SECONDARY, config, case)
            # Drop primary; secondary's run must still succeed because the
            # dedup refcount is still > 0.
            chip_worker._unregister_slot(_SLOT_PRIMARY)
            primary_prepared = False
            self._run_one(st_worker, _SLOT_SECONDARY, config, case)
        finally:
            if secondary_prepared:
                chip_worker._unregister_slot(_SLOT_SECONDARY)
            if primary_prepared:
                chip_worker._unregister_slot(_SLOT_PRIMARY)

    def test_prewarm_failure_rolls_back_registration(self, st_platform, st_worker):
        """Case F: missing AICPU orch entry fails prepare and leaves no prepared slot."""

        class MissingEntryCallable(type(self)):
            CALLABLE = copy.deepcopy(type(self).CALLABLE)

        MissingEntryCallable.CALLABLE["orchestration"]["function_name"] = "missing_aicpu_orchestration_entry"
        bad_callable = MissingEntryCallable().build_callable(st_platform)
        config = self._build_config(self.CASES[0]["config"])
        case = self.CASES[0]
        baseline = st_worker.aicpu_dlopen_count
        chip_worker = self._chip_worker(st_worker)

        with pytest.raises(RuntimeError):
            chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, bad_callable)
        assert st_worker.aicpu_dlopen_count == baseline

        with pytest.raises(RuntimeError):
            self._run_one(st_worker, _SLOT_PRIMARY, config, case)
        assert st_worker.aicpu_dlopen_count == baseline

        chip_worker._unregister_slot(_SLOT_PRIMARY)

    def test_dlopen_count_unregister_re_prepare(self, st_platform, st_worker):
        """Case D: prepare+unregister+prepare prewarms twice -> delta == 2.

        unregister erases the slot from aicpu_seen_callable_ids_, so the second
        prepare prewarms the reused slot with a fresh AICPU dlopen. The counter
        is monotonic (does NOT decrement on unregister), so the delta after the
        second cycle is 2.
        """
        callable_obj, config, case = self._setup_dlopen_count_test(st_worker, st_platform)
        baseline = st_worker.aicpu_dlopen_count
        prepared = False
        chip_worker = self._chip_worker(st_worker)
        try:
            chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, callable_obj)
            prepared = True
            assert st_worker.aicpu_dlopen_count - baseline == 1
            self._run_one(st_worker, _SLOT_PRIMARY, config, case)
            assert st_worker.aicpu_dlopen_count - baseline == 1
            chip_worker._unregister_slot(_SLOT_PRIMARY)
            prepared = False
            after_unreg = st_worker.aicpu_dlopen_count
            assert after_unreg - baseline == 1, (
                f"unregister must NOT decrement the dlopen counter; baseline={baseline}, after_unreg={after_unreg}"
            )
            chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, callable_obj)
            prepared = True
            assert st_worker.aicpu_dlopen_count - baseline == 2
            self._run_one(st_worker, _SLOT_PRIMARY, config, case)
            assert st_worker.aicpu_dlopen_count - baseline == 2, (
                f"after re-prepare expected counter +2 (two distinct AICPU dlopens), "
                f"got delta {st_worker.aicpu_dlopen_count - baseline}"
            )
        finally:
            if prepared:
                chip_worker._unregister_slot(_SLOT_PRIMARY)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
