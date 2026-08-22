#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-orchestration scope-stats capture and JSONL export."""

import json
import time

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

KERNELS_BASE = "../../../../../../examples/a5/tensormap_and_ringbuffer/vector_example/kernels"


@scene_test(level=2, runtime="host_build_graph")
class TestA5ScopeStatsHostBuildGraph(SceneTestCase):
    """Capture the executor scope plus one nested orchestration scope."""

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

    CASES = [{"name": "nested", "platforms": ["a5sim", "a5"], "params": {}}]

    def generate_args(self, params):
        size = 128 * 128
        return TaskArgsBuilder(
            TensorArg("a", torch.full((size,), 2.0, dtype=torch.float32)),
            TensorArg("b", torch.full((size,), 3.0, dtype=torch.float32)),
            TensorArg("f", torch.zeros(size, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.f[:] = (args.a + args.b + 1) * (args.a + args.b + 2) + (args.a + args.b)

    def test_run(self, st_platform, st_worker, request):
        run_marker = int(time.time())
        super().test_run(st_platform, st_worker, request)
        if not request.config.getoption("--enable-scope-stats", default=False):
            return
        self._validate_artifact(run_marker)

    def _validate_artifact(self, run_marker):
        safe_label = _sanitize_for_filename("TestA5ScopeStatsHostBuildGraph_nested")
        matches = [p for p in _outputs_dir().glob(f"{safe_label}_*") if p.stat().st_mtime >= run_marker]
        assert matches, "scope-stats run produced no output directory"
        out_dir = max(matches, key=lambda p: p.stat().st_mtime)
        path = out_dir / "scope_stats" / "scope_stats.jsonl"
        assert path.exists(), f"host scope-stats did not produce {path}"

        lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(lines) == 5, f"expected metadata plus four boundaries, got {lines!r}"
        meta, *records = lines
        assert meta["version"] == 6
        assert meta["fatal"] is False
        assert meta["dropped"] == 0
        assert meta["total"] == 4
        assert meta["task_window_max"][0] > 0
        assert meta["heap_max"][0] > 0
        assert meta["dep_pool_max"][0] == 0
        assert meta["tensormap_max"] > 0

        assert [(record["depth"], record["phase"]) for record in records] == [
            (0, "begin"),
            (1, "begin"),
            (1, "end"),
            (0, "end"),
        ]
        inner_begin, inner_end = records[1], records[2]
        assert inner_begin["site"].startswith("example_orchestration.cpp:")
        assert inner_end["site"] == inner_begin["site"]
        assert inner_end["task_window_end"] - inner_begin["task_window_end"] == 4
        assert inner_end["heap_end"] > inner_begin["heap_end"]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
