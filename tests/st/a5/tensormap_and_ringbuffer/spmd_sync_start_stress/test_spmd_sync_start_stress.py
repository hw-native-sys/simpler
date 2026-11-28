#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SPMD sync_start stress: 54 tasks over 6 rounds with mixed shapes (MIX + AIV).

The four cohort widths scale with the run's cluster count and are reported back
in `layout` — a run always takes the whole device, and that width differs
between sim and silicon. The host replays the same round structure from the
reported widths; `output` is sized for the widest device the platform allows.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
ROUNDS = 6
SHAPE_MIX, SHAPE_AIV = "MIX", "AIV"
MIX_SLOTS, AIV_SLOTS = 3, 1
# Widest the platform allows: cohort divisors 6/2/3/6 at 36 clusters.
MAX_CLUSTERS = 36


def _build_tasks(normal_mix_bn, sync_mix_bn, sync_aiv_bn, normal_aiv_bn):
    """Replay the orchestration's round structure from its reported widths."""
    tasks, cl = [], 0
    for _ in range(ROUNDS):
        for _ in range(4):
            tasks.append((normal_mix_bn, cl, SHAPE_MIX))
            cl += normal_mix_bn * MIX_SLOTS
        for _ in range(2):
            tasks.append((sync_mix_bn, cl, SHAPE_MIX))
            cl += sync_mix_bn * MIX_SLOTS
        for _ in range(2):
            tasks.append((sync_aiv_bn, cl, SHAPE_AIV))
            cl += sync_aiv_bn * AIV_SLOTS
        tasks.append((normal_aiv_bn, cl, SHAPE_AIV))
        cl += normal_aiv_bn * AIV_SLOTS
    return tasks


def _total_cl(tasks):
    return sum(bn * (MIX_SLOTS if s == SHAPE_MIX else AIV_SLOTS) for bn, _, s in tasks)


MAX_TOTAL_CL = _total_cl(_build_tasks(MAX_CLUSTERS // 6, MAX_CLUSTERS // 2, MAX_CLUSTERS // 3, MAX_CLUSTERS // 6))


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdSyncStartStress(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_sync_start_stress_orch.cpp",
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
            {
                "func_id": 3,
                "name": "SPMD_WRITE_AIV",
                "source": "../spmd_multiblock_aiv/kernels/aiv/kernel_spmd_write.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a5sim", "a5"],
            "manual": ["a5sim", "a5"],
            "params": {},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("output", torch.zeros(MAX_TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            TensorArg("layout", torch.zeros(4, dtype=torch.int32)),
        )

    def compute_golden(self, args, params):
        # Both outputs are checked against the reported widths in compare_outputs.
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        widths = [int(v) for v in test_args.layout]
        assert all(w >= 1 for w in widths), f"reported cohort widths {widths}"
        tasks = _build_tasks(*widths)
        assert _total_cl(tasks) <= MAX_TOTAL_CL, f"widths {widths} overflow {MAX_TOTAL_CL} cache lines"
        expected = torch.zeros(MAX_TOTAL_CL, dtype=torch.float32)
        for block_num, base_cl, shape in tasks:
            for block_idx in range(block_num):
                if shape == SHAPE_MIX:
                    for slot in range(MIX_SLOTS):
                        expected[base_cl + block_idx * MIX_SLOTS + slot] = float(block_idx)
                else:
                    expected[base_cl + block_idx] = float(block_idx)
        actual = test_args.output.reshape(MAX_TOTAL_CL, FLOATS_PER_CACHE_LINE)[:, 0]
        assert torch.equal(actual, expected), f"slots disagree with the reported cohort widths {widths}"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
