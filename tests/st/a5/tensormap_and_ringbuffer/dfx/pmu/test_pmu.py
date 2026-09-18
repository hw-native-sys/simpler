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
src/a5/platform/shared/host/pmu_collector.cpp's "Build CSV header"
block). Smoke asserts: file exists, header starts with the documented
prefix (ordered — the header literal and the positional row writer are
separate code paths, so the header must stay in the row order or every
column is silently mislabeled), at least one data row present.
"""

import time

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

KERNELS_BASE = "../../../../../../examples/a5/tensormap_and_ringbuffer/vector_example/kernels"
# Required leading columns — keep in sync with build_csv_header() in
# pmu_collector.cpp. Counter columns follow these and vary per event_type.
_REQUIRED_HEADER_PREFIX = ("thread_id", "core_id", "task_id", "func_id", "core_type", "pmu_total_cycles")
# Run identity is the last column, after the variable-length counter block. It
# says which run produced the row; without it, attribution rests entirely on the
# host clearing its state between runs.
_REQUIRED_TRAILING_COLUMNS = ("event_type", "run_epoch")


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
            "platforms": ["a5sim", "a5"],
            "manual": ["a5sim"],
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
        # Marker taken before the run so _validate_pmu_artifact can distinguish
        # this invocation's output directory from stale same-label leftovers.
        run_marker = int(time.time())  # floor to whole seconds: safe if outputs/ ever lands on a coarse-mtime fs
        super().test_run(st_platform, st_worker, request)
        if not request.config.getoption("--enable-pmu", default=0):
            return
        for case in self._matching_cases(st_platform, request):
            self._validate_pmu_artifact(case, run_marker)

    def _validate_pmu_artifact(self, case, run_marker):
        safe_label = _sanitize_for_filename(f"TestPmu_{case['name']}")
        # Only directories created by this invocation (mtime past the pre-run
        # marker); a stale dir from a prior run must not stand in for a missing
        # current-run output and mask a capture regression.
        matches = [p for p in _outputs_dir().glob(f"{safe_label}_*") if p.stat().st_mtime >= run_marker]
        assert matches, f"no PMU output directory for '{safe_label}' created this run — PMU capture failed?"
        run_dir = max(matches, key=lambda p: p.stat().st_mtime)
        csv = run_dir / "pmu.csv"
        assert csv.exists(), f"pmu.csv missing under {run_dir} — PMU capture failed?"
        lines = csv.read_text().splitlines()
        assert lines, "pmu.csv is empty"
        header_cols = lines[0].split(",")
        prefix = list(_REQUIRED_HEADER_PREFIX)
        assert header_cols[: len(prefix)] == prefix, f"header must start with {prefix} in order: {header_cols}"
        assert tuple(header_cols[-len(_REQUIRED_TRAILING_COLUMNS) :]) == _REQUIRED_TRAILING_COLUMNS, (
            f"header must end with {_REQUIRED_TRAILING_COLUMNS} in order: {header_cols}"
        )
        # At least one data row — sim runs all 5 vector_example tasks; expect ≥1
        # to keep the assertion robust if a future scheduler change collapses
        # / batches per-task PMU sampling.
        assert len(lines) >= 2, f"pmu.csv has no data rows (only header): {lines}"

        # Every row carries a run identity, and it is the one the producer
        # stamped rather than the zero a buffer starts life with. A row whose
        # column count disagrees with the header would make this read the wrong
        # field, so check the shape first.
        epoch_index = header_cols.index("run_epoch")
        epochs = set()
        for row in lines[1:]:
            cols = row.split(",")
            assert len(cols) == len(header_cols), f"row has {len(cols)} columns, header has {len(header_cols)}: {row}"
            epochs.add(cols[epoch_index])
        assert epochs, "no data rows to check run_epoch on"
        assert "0" not in epochs, f"a row carries run_epoch=0, i.e. no identity was stamped: {sorted(epochs)}"
        assert len(epochs) == 1, f"a single run produced more than one run_epoch: {sorted(epochs)}"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
