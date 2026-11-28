#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SPMD sync_start edge: boundary block_num values. Output: 126 CL = 2016 float32.

The cohort widths are fractions of the run's core count, derived on device and
reported back in `layout` — a run always takes the whole device, and that width
differs between sim and silicon. The golden is rebuilt from the reported
geometry; `output` is sized for the widest device the platform allows.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 3
NUM_TASKS = 5
# Widest the platform allows: 1 + c/3 + c/3+1 + c-1 + 1 at 24 clusters.
MAX_CLUSTERS = 24
MAX_TOTAL_CL = (1 + MAX_CLUSTERS // 3 + MAX_CLUSTERS // 3 + 1 + MAX_CLUSTERS - 1 + 1) * SLOTS_PER_BLOCK


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdSyncStartEdge(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_sync_start_edge_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SPMD_MIX_AIC",
                "source": "../spmd_multiblock_mix/kernels/aic/kernel_spmd_mix.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "name": "SPMD_MIX_AIV0",
                "source": "../spmd_multiblock_mix/kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "name": "SPMD_MIX_AIV1",
                "source": "../spmd_multiblock_mix/kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "manual": ["a2a3"],
            "platforms": ["a2a3sim", "a2a3"],
            "params": {},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("output", torch.zeros(MAX_TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            TensorArg("layout", torch.zeros(2 * NUM_TASKS, dtype=torch.int32)),
        )

    def compute_golden(self, args, params):
        # Both outputs are checked against the reported layout in compare_outputs.
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        layout = [int(v) for v in test_args.layout]
        expected = torch.zeros(MAX_TOTAL_CL, dtype=torch.float32)
        for i in range(NUM_TASKS):
            block_num, base_cl = layout[2 * i], layout[2 * i + 1]
            assert block_num >= 1, f"task {i} reported block_num {block_num}"
            assert base_cl + block_num * 3 <= MAX_TOTAL_CL, (
                f"task {i} layout ({block_num}, {base_cl}) overflows {MAX_TOTAL_CL} cache lines"
            )
            for block_idx in range(block_num):
                for slot in range(3):
                    expected[base_cl + block_idx * 3 + slot] = float(block_idx)
        actual = test_args.output.reshape(MAX_TOTAL_CL, FLOATS_PER_CACHE_LINE)[:, 0]
        assert torch.equal(actual, expected), (
            f"block slots disagree with the reported layout {layout}: "
            f"got {actual.tolist()}, expected {expected.tolist()}"
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
