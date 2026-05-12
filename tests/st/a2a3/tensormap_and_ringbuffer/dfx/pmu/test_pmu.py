#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PMU profiling smoke — capture pipeline produces a usable ``pmu.csv``.

Re-uses ``vector_example`` (5 submit_task calls). With ``--enable-pmu N``
the AICore counters land in ``<output_prefix>/pmu.csv``, one data row per
task. The schema is fixed (see docs/dfx/pmu-profiling.md and
src/a2a3/platform/src/host/pmu_collector.cpp's "Build CSV header"
block). Smoke asserts: file exists, header starts with the documented
prefix, at least one data row present.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

KERNELS_BASE = "../../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"
# Required leading columns — keep in sync with build_csv_header() in
# pmu_collector.cpp. Counter columns follow these and vary per event_type.
_REQUIRED_HEADER_PREFIX = ("thread_id", "core_id", "task_id", "func_id", "core_type", "pmu_total_cycles")


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestPmu(SceneTestCase):
    """Vector example with --enable-pmu, then assert pmu.csv."""

    CALLABLE = {
        "orchestration": {
            "source": f"{KERNELS_BASE}/orchestration/example_orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{KERNELS_BASE}/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": f"{KERNELS_BASE}/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": f"{KERNELS_BASE}/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {},
        },
    ]

    def generate_args(self, params):
        SIZE = 128 * 128
        return TaskArgsBuilder(
            Tensor("a", torch.full((SIZE,), 2.0, dtype=torch.float32)),
            Tensor("b", torch.full((SIZE,), 3.0, dtype=torch.float32)),
            Tensor("f", torch.zeros(SIZE, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.f[:] = (args.a + args.b + 1) * (args.a + args.b + 2) + (args.a + args.b)

    def test_run(self, st_platform, st_worker, request):
        super().test_run(st_platform, st_worker, request)
        if not request.config.getoption("--enable-pmu", default=0):
            return
        for case in self.CASES:
            if st_platform in case["platforms"]:
                self._validate_pmu_artifact(case)

    def _validate_pmu_artifact(self, case):
        safe_label = _sanitize_for_filename(f"TestPmu_{case['name']}")
        matches = sorted(_outputs_dir().glob(f"{safe_label}_*"), key=lambda p: p.stat().st_mtime)
        if not matches:
            return
        csv = matches[-1] / "pmu.csv"
        assert csv.exists(), f"pmu.csv missing under {matches[-1]} — PMU capture failed?"
        lines = csv.read_text().splitlines()
        assert lines, "pmu.csv is empty"
        header_cols = lines[0].split(",")
        for col in _REQUIRED_HEADER_PREFIX:
            assert col in header_cols, f"header missing required column '{col}': {header_cols}"
        # At least one data row — sim runs all 5 vector_example tasks; expect ≥1
        # to keep the assertion robust if a future scheduler change collapses
        # / batches per-task PMU sampling.
        assert len(lines) >= 2, f"pmu.csv has no data rows (only header): {lines}"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
