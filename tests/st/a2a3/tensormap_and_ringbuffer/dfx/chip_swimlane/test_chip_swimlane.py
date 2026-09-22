#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""chip swimlane profiling smoke — capture pipeline produces a usable
``chip_swimlane_records.json``.

Re-uses ``vector_example`` as a known-good 5-task AIV-only workload. When the
``--enable-chip-swimlane`` flag is on, the helper in :mod:`_swimlane_validate`
asserts schema, runs the converter / sched_overhead tool smokes, and fires a
differential gate over Pop / Fanout / Fanin. Without the flag the assertions
are skipped — the test still runs the case so the default ``pytest tests/st``
invocation doesn't pay an extra step.

A mixed AIC+AIV companion lives in ``test_chip_swimlane_mixed.py`` —
that variant exercises the per-task dedup branch in
``compute_dag_stats_from_deps`` which this AIV-only workload doesn't.
"""

import json
import time
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import build_output_prefix

from ._swimlane_validate import validate_perf_artifact

KERNELS_BASE = "../../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"
# example_orchestration.cpp issues 5 submit_task calls.
_EXPECTED_TASK_COUNT = 5


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestChipSwimlane(SceneTestCase):
    """Vector example with --enable-chip-swimlane, then assert chip_swimlane_records.json."""

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

    # The host's drain/collector thread count follows aicpu_thread_num, so the
    # low-thread cases exercise a shard layout the 4-thread default never
    # reaches. The task-count assertions below are exact, so a record lost to a
    # mis-sharded buffer fails the run rather than degrading it silently.
    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "manual": ["a2a3sim"],
            "params": {},
        },
        {
            "name": "aicpu_threads_2",
            "platforms": ["a2a3sim", "a2a3"],
            "manual": ["a2a3sim"],
            "config": {"aicpu_thread_num": 2},
            "params": {},
        },
    ]

    def generate_args(self, params):
        SIZE = 128 * 128
        return TaskArgsBuilder(
            TensorArg("a", torch.full((SIZE,), 2.0, dtype=torch.float32)),
            TensorArg("b", torch.full((SIZE,), 3.0, dtype=torch.float32)),
            TensorArg("f", torch.zeros(SIZE, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.f[:] = (args.a + args.b + 1) * (args.a + args.b + 2) + (args.a + args.b)

    def test_run(self, st_platform, st_worker, request):
        # Marker taken before the run so validate_perf_artifact can bind to this
        # invocation's output dir rather than a stale same-label leftover.
        run_marker = int(time.time())  # floor to whole seconds: safe if outputs/ ever lands on a coarse-mtime fs
        super().test_run(st_platform, st_worker, request)
        if not request.config.getoption("--enable-chip-swimlane", default=0):
            return
        for case in self._matching_cases(st_platform, request):
            validate_perf_artifact(
                f"TestChipSwimlane_{case['name']}", since=run_marker, expected_task_count=_EXPECTED_TASK_COUNT
            )


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestDeviceOrchestrationClockCapture(TestChipSwimlane):
    """Clock captures and disabled runs alternate on the same worker."""

    _swimlane_level = 4
    _clock_capture_paths: tuple[Path, ...] = ()

    def test_run(self, st_platform, st_worker, request):
        for level in (4, 0, 4, 0):
            self._swimlane_level = level
            self._clock_capture_paths = ()
            SceneTestCase.test_run(self, st_platform, st_worker, request)
            # Artifact checks cover every case even when golden comparison is disabled.
            for path in self._clock_capture_paths:
                assert path.is_file(), f"missing device orchestration capture: {path}"
                records = json.loads(path.read_text())
                assert any(records.get("aicpu_orchestrator_phases", [])), "AICPU orchestration records missing"
                assert "orchestrator_source" not in records["metadata"]

    def _build_config(self, config_dict, *args, **kwargs):
        config = super()._build_config(config_dict, *args, **kwargs)
        config.enable_chip_swimlane = self._swimlane_level
        if not config.output_prefix:
            config.output_prefix = str(build_output_prefix(f"{type(self).__name__}_{time.monotonic_ns()}"))
        if self._swimlane_level:
            path = Path(config.output_prefix) / "chip_swimlane_records.json"
            # CLI-generated prefixes can repeat for the same case within one second.
            path.unlink(missing_ok=True)
            self._clock_capture_paths += (path,)
        return config


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
