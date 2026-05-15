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
import re
import subprocess
import sys

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
        if not request.config.getoption("--enable-l2-swimlane", default=0):
            return
        for case in self.CASES:
            if st_platform in case["platforms"]:
                self._validate_perf_artifact(case, st_platform)

    def _validate_perf_artifact(self, case, st_platform):
        safe_label = _sanitize_for_filename(f"TestL2Swimlane_{case['name']}")
        matches = sorted(_outputs_dir().glob(f"{safe_label}_*"), key=lambda p: p.stat().st_mtime)
        if not matches:
            return
        perf = matches[-1] / "l2_perf_records.json"
        assert perf.exists(), f"l2_perf_records.json missing under {matches[-1]} — swimlane capture failed?"
        with perf.open() as f:
            data = json.load(f)
        assert data.get("version") in (1, 2, 3, 4), f"unexpected version: {data.get('version')}"
        tasks = data.get("tasks")
        assert isinstance(tasks, list), "tasks field missing or not a list"
        assert len(tasks) == _EXPECTED_TASK_COUNT, (
            f"got {len(tasks)} perf records, expected {_EXPECTED_TASK_COUNT} "
            f"(vector_example issues 5 submit_task calls)"
        )
        # Spot-check a single record's required fields — guards against drift in
        # the swimlane schema that swimlane_converter.py / deps_to_graph.py rely on.
        first = tasks[0]
        for key in ("task_id", "func_id", "core_id", "core_type", "start_time_us", "end_time_us", "fanout"):
            assert key in first, f"perf record missing required field '{key}': {first}"

        # ---- Tool smoke: swimlane_converter ----
        # Exit-code-only check; we don't validate the Perfetto JSON content. A
        # schema change that breaks the converter fires here in the same CI
        # step that produced the artifact.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "simpler_setup.tools.swimlane_converter",
                str(perf),
                "-o",
                str(matches[-1] / "_smoke_swimlane.json"),
            ],
            check=True,
            timeout=60,
        )

        # ---- Tool smoke: sched_overhead_analysis ----
        # The analysis joins l2_perf_records.json with AICPU cycle counters
        # (dispatch_time / finish_time, populated in l2_perf_collector.cpp).
        # No #ifdef gates the capture — the fields are unconditionally written
        # when the value is non-zero, so coverage hinges on the AICPU actually
        # writing real cycle counts. That happens on hardware; on sim the
        # cycle counters may be 0 or absent.
        #
        # Sim: smoke is `--help` — verifies the module imports, argparse is
        # wired, the entry point hasn't bit-rotted. Cheap, doesn't false-
        # positive on missing device data.
        # Hardware: full run with stdout assertions — section headers must
        # all print AND at least one numeric metric must be non-zero. The
        # "all zeros" failure mode (e.g., capture path was silently dropped
        # in a refactor) would print `0.0 us` everywhere; the regex check
        # below catches that.
        if st_platform.endswith("sim"):
            subprocess.run(
                [sys.executable, "-m", "simpler_setup.tools.sched_overhead_analysis", "--help"],
                check=True,
                timeout=10,
                stdout=subprocess.DEVNULL,
            )
        else:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "simpler_setup.tools.sched_overhead_analysis",
                    "--l2-perf-records-json",
                    str(perf),
                ],
                check=True,
                timeout=120,
                capture_output=True,
                text=True,
            )
            for header in ("Part 1:", "Part 2:", "Part 3:"):
                assert header in result.stdout, (
                    f"sched_overhead missing section header '{header}'\nstdout:\n{result.stdout}"
                )
            # Bad pattern: AICPU didn't capture real cycle counters → tool
            # "succeeds" but every metric is 0. Match the line that's printed
            # unconditionally in Part 2 and assert its value is non-zero.
            m = re.search(r"Avg scheduler loop iteration:\s+([\d.]+)\s+us", result.stdout)
            assert m, f"sched_overhead stdout missing 'Avg scheduler loop iteration'\nstdout:\n{result.stdout}"
            assert float(m.group(1)) > 0.0, (
                f"sched_overhead reports zero loop iteration (avg_loop_us={m.group(1)}). "
                f"AICPU likely didn't capture dispatch_time/finish_time cycle counters — "
                f"the L2 perf collector path may have regressed.\nstdout:\n{result.stdout}"
            )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
