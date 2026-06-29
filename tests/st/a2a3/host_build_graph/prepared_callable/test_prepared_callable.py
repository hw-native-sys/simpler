#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end white-box test for the private L2 prepared-callable ABI on host_build_graph.

Mirrors tests/st/a2a3/tensormap_and_ringbuffer/prepared_callable for the hbg
variant: instead of the AICPU dlopening the orch SO, hbg dlopens on the host
inside private slot preparation and replays the cached handle/fn pointer
on every run. The dlopen counter to assert is `host_dlopen_count`,
not `aicpu_dlopen_count` (which stays 0 — AICPU never sees the orch SO).
"""

from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _build_chip_task_args, _compare_outputs

_VECTOR_KERNELS = "../vector_example/kernels"
_SLOT_PRIMARY = 0
_SLOT_SECONDARY = 1
_ORCH_SO_MAP_TOKEN = "/tmp/orch_so_"


def _count_live_orch_so_mappings():
    try:
        lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        pytest.skip(f"/proc/self/maps unavailable: {exc}")

    paths = set()
    for line in lines:
        if _ORCH_SO_MAP_TOKEN not in line:
            continue
        parts = line.split(None, 5)
        if len(parts) == 6:
            paths.add(parts[5])
    return len(paths)


@scene_test(level=2, runtime="host_build_graph")
class TestPreparedCallableHbg(SceneTestCase):
    """Exercise private prepare / run / unregister slot ABI on hbg.

    Requires an isolated L2 ``Worker`` (private slot table starts empty); this is
    provided by the directory-local ``conftest.py`` overriding ``st_worker``
    with a class-scope fixture.
    """

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
                "arg_index": [0, 1, 2],
            },
            {
                "func_id": 1,
                "source": f"{_VECTOR_KERNELS}/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
                "arg_index": [0, 1],
            },
            {
                "func_id": 2,
                "source": f"{_VECTOR_KERNELS}/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
                "arg_index": [0, 1, 2],
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
        # vector_example orchestration computes (a + b + 1) * (a + b + 2)
        a, b = args.a, args.b
        args.f[:] = (a + b + 1) * (a + b + 2)

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

        chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, callable_obj)
        chip_worker._prepare_callable_at_slot(_SLOT_SECONDARY, callable_obj)

        for _ in range(2):
            test_args = self.generate_args(params)
            chip_args, output_names = _build_chip_task_args(test_args, orch_sig)
            golden_args = test_args.clone()
            self.compute_golden(golden_args, params)

            chip_worker._run_slot(_SLOT_PRIMARY, chip_args, config=config)
            _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

        test_args = self.generate_args(params)
        chip_args, output_names = _build_chip_task_args(test_args, orch_sig)
        golden_args = test_args.clone()
        self.compute_golden(golden_args, params)

        chip_worker._run_slot(_SLOT_SECONDARY, chip_args, config=config)
        _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

        chip_worker._unregister_slot(_SLOT_PRIMARY)
        chip_worker._unregister_slot(_SLOT_SECONDARY)

    # ------------------------------------------------------------------
    # host_dlopen_count assertions (hbg path).
    #
    # hbg increments host_dlopen_count on every register_callable_host_orch
    # invocation (i.e. each private slot prepare), independent of how many
    # times run is invoked afterwards. AICPU never dlopens the orch
    # SO on this variant, so aicpu_dlopen_count stays at 0.
    # ------------------------------------------------------------------

    def _setup_dlopen_count_test(self, st_worker, st_platform):
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
        """prepare(primary) + run x5 -> host_dlopen delta == 1, aicpu == 0."""
        callable_obj, config, case = self._setup_dlopen_count_test(st_worker, st_platform)
        baseline = st_worker.host_dlopen_count
        baseline_aicpu = st_worker.aicpu_dlopen_count
        prepared = False
        chip_worker = self._chip_worker(st_worker)
        try:
            chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, callable_obj)
            prepared = True
            for _ in range(5):
                self._run_one(st_worker, _SLOT_PRIMARY, config, case)
            assert st_worker.host_dlopen_count - baseline == 1, (
                f"expected exactly 1 new host dlopen for 5 runs of primary slot, "
                f"got delta {st_worker.host_dlopen_count - baseline}"
            )
            assert st_worker.aicpu_dlopen_count == baseline_aicpu, "hbg must not trigger any AICPU orch SO dlopens"
        finally:
            if prepared:
                chip_worker._unregister_slot(_SLOT_PRIMARY)

    def test_dlopen_count_two_slots_alternating(self, st_platform, st_worker):
        """prepare(primary)+prepare(secondary) + alternating runs x5 -> host_dlopen delta == 2."""
        callable_obj, config, case = self._setup_dlopen_count_test(st_worker, st_platform)
        baseline = st_worker.host_dlopen_count
        baseline_aicpu = st_worker.aicpu_dlopen_count
        primary_prepared = False
        secondary_prepared = False
        chip_worker = self._chip_worker(st_worker)
        try:
            chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, callable_obj)
            primary_prepared = True
            chip_worker._prepare_callable_at_slot(_SLOT_SECONDARY, callable_obj)
            secondary_prepared = True
            for _ in range(5):
                self._run_one(st_worker, _SLOT_PRIMARY, config, case)
                self._run_one(st_worker, _SLOT_SECONDARY, config, case)
            assert st_worker.host_dlopen_count - baseline == 2, (
                f"expected exactly 2 new host dlopens for two slots interleaved, "
                f"got delta {st_worker.host_dlopen_count - baseline}"
            )
            assert st_worker.aicpu_dlopen_count == baseline_aicpu
        finally:
            if secondary_prepared:
                chip_worker._unregister_slot(_SLOT_SECONDARY)
            if primary_prepared:
                chip_worker._unregister_slot(_SLOT_PRIMARY)

    def test_dlopen_count_double_prepare_raises(self, st_platform, st_worker):
        """prepare(primary) twice -> second call raises RuntimeError."""
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

    def test_failed_double_prepare_closes_unpublished_host_handle(self, st_platform, st_worker):
        if not st_platform.endswith("sim"):
            pytest.skip("unpublished host dlopen guard is validated on sim")

        callable_obj, _config, _case = self._setup_dlopen_count_test(st_worker, st_platform)
        baseline_count = st_worker.host_dlopen_count
        baseline_maps = _count_live_orch_so_mappings()
        prepared = False
        chip_worker = self._chip_worker(st_worker)
        try:
            chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, callable_obj)
            prepared = True
            after_first_maps = _count_live_orch_so_mappings()
            assert st_worker.host_dlopen_count - baseline_count == 1
            assert after_first_maps == baseline_maps + 1

            with pytest.raises(RuntimeError):
                chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, callable_obj)

            assert st_worker.host_dlopen_count - baseline_count == 1
            assert _count_live_orch_so_mappings() == after_first_maps
        finally:
            if prepared:
                chip_worker._unregister_slot(_SLOT_PRIMARY)

        assert _count_live_orch_so_mappings() == baseline_maps

    def test_dlopen_count_unregister_re_prepare(self, st_platform, st_worker):
        """prepare+run+unregister+prepare+run -> host_dlopen delta == 2.

        Counter is monotonic — re-prepare always counts a fresh dlopen.
        """
        callable_obj, config, case = self._setup_dlopen_count_test(st_worker, st_platform)
        baseline = st_worker.host_dlopen_count
        prepared = False
        chip_worker = self._chip_worker(st_worker)
        try:
            chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, callable_obj)
            prepared = True
            self._run_one(st_worker, _SLOT_PRIMARY, config, case)
            assert st_worker.host_dlopen_count - baseline == 1
            chip_worker._unregister_slot(_SLOT_PRIMARY)
            prepared = False
            after_unreg = st_worker.host_dlopen_count
            assert after_unreg - baseline == 1, (
                f"unregister must NOT decrement the host dlopen counter; baseline={baseline}, after_unreg={after_unreg}"
            )
            chip_worker._prepare_callable_at_slot(_SLOT_PRIMARY, callable_obj)
            prepared = True
            self._run_one(st_worker, _SLOT_PRIMARY, config, case)
            assert st_worker.host_dlopen_count - baseline == 2, (
                f"after re-prepare expected counter +2 (two distinct host dlopens), "
                f"got delta {st_worker.host_dlopen_count - baseline}"
            )
        finally:
            if prepared:
                chip_worker._unregister_slot(_SLOT_PRIMARY)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
