#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L2 swimlane profiling smoke — capture pipeline produces a usable
``l2_perf_records.json``.

Re-uses ``vector_example`` as a known-good 5-task workload. When the
``--enable-l2-swimlane`` flag is on, asserts that the perf record file lands
under the per-case output_prefix and parses as the documented schema
(``version`` + ``tasks[]`` with one entry per submit_task). Without the flag
the assertions are skipped — the test still runs the case so the default
``pytest tests/st`` invocation doesn't pay an extra step.
"""

import json

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

KERNELS_BASE = "../../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"
# example_orchestration.cpp issues 5 submit_task calls.
_EXPECTED_TASK_COUNT = 5


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestL2Swimlane(SceneTestCase):
    """Vector example with --enable-l2-swimlane, then assert l2_perf_records.json."""

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
        if not request.config.getoption("--enable-l2-swimlane", default=False):
            return
        for case in self.CASES:
            if st_platform in case["platforms"]:
                self._validate_perf_artifact(case)

    def _validate_perf_artifact(self, case):
        safe_label = _sanitize_for_filename(f"TestL2Swimlane_{case['name']}")
        matches = sorted(_outputs_dir().glob(f"{safe_label}_*"), key=lambda p: p.stat().st_mtime)
        if not matches:
            return
        perf = matches[-1] / "l2_perf_records.json"
        assert perf.exists(), f"l2_perf_records.json missing under {matches[-1]} — swimlane capture failed?"
        with perf.open() as f:
            data = json.load(f)
        assert data.get("version") in (1, 2), f"unexpected version: {data.get('version')}"
        tasks = data.get("tasks")
        assert isinstance(tasks, list), "tasks field missing or not a list"
        assert len(tasks) == _EXPECTED_TASK_COUNT, (
            f"got {len(tasks)} perf records, expected {_EXPECTED_TASK_COUNT} "
            f"(vector_example issues 5 submit_task calls)"
        )
        # Spot-check a single record's required fields — guards against drift in
        # the swimlane schema that swimlane_converter.py / perf_to_mermaid.py rely on.
        first = tasks[0]
        for key in ("task_id", "func_id", "core_id", "core_type", "start_time_us", "end_time_us", "fanout"):
            assert key in first, f"perf record missing required field '{key}': {first}"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
