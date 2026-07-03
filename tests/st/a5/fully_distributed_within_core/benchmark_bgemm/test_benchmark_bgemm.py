#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Benchmark BGEMM: runtime-configurable tiled matmul C = sum(k) A[k] @ B[k]."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="fully_distributed_within_core")
class TestBenchmarkBgemm(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/bgemm_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "GEMM",
                "source": "kernels/aic/kernel_gemm_tile.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "ADD",
                "source": "kernels/aiv/kernel_tile_add.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "Case0",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"matmul_add_task_num": 500, "incore_data_size": 128, "incore_loop": 4, "grid_k": 2},
        },
        {
            "name": "Bgemm64",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {"matmul_add_task_num": 32, "incore_data_size": 64, "incore_loop": 1, "grid_k": 4},
        },
        {
            # Non-oversubscribed balance control: block_dim=3 → 3 AIC + 6 AIV = 9
            # workers, ~1.1x the 8-core sim host (vs FullCore36's 13.5x). Same
            # workload shape as FullCore36. Lets the execute-first claim race run
            # on (nearly) truly-parallel threads, isolating the ENGINE's intrinsic
            # balance from host oversubscription. GEMM ideal 180/3=60/AIC, ADD
            # 180/6=30/AIV. Capture like FullCore36 with --case Balanced9.
            "name": "Balanced9",
            "manual": True,
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {"matmul_add_task_num": 180, "incore_data_size": 128, "incore_loop": 4, "grid_k": 2},
        },
        {
            # Full-core swimlane visualization: block_dim == a5sim capacity
            # (PLATFORM_MAX_BLOCKDIM=36 → 36 AIC + 72 AIV = 108 cores). 360 GEMM
            # (1C) + 360 ADD (1V) tasks so every one of the 36 AIC blocks gets
            # ~10 GEMMs, filling all lanes. Manual (opt-in) so it does not slow
            # the default suite.
            #
            # Uniform fake per-kernel cost (PTO_DIST_FAKE_EXEC_NS) makes every
            # kernel "run" for an equal synthetic duration instead of the real
            # (variable / near-zero) compute, so the swimlane reflects genuine
            # scheduling balance rather than a skip-exec artifact. Capture with:
            #   PTO_DIST_SWIMLANE=$PWD/outputs/dist_swimlane/bgemm_fullcore_a5.json \
            #   PTO_DIST_FAKE_EXEC_NS=1000 \
            #     python test_benchmark_bgemm.py -p a5sim --case FullCore36 --manual include
            #   python -m simpler_setup.tools.dist_swimlane_render \
            #     outputs/dist_swimlane/bgemm_fullcore_a5.json --names 0=GEMM,1=ADD -v
            # NOTE: in fake-exec mode the real GEMM is skipped, so the golden
            # comparison is not meaningful — this case is for the swimlane only.
            "name": "FullCore36",
            "manual": True,
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"matmul_add_task_num": 360, "incore_data_size": 128, "incore_loop": 4, "grid_k": 2},
        },
        {
            # Batch-size sweep companion to FullCore36: 3x the tasks (1080), same
            # 108 cores. Tests whether a larger batch improves per-core balance.
            "name": "FullCore36Big",
            "manual": True,
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"matmul_add_task_num": 1080, "incore_data_size": 128, "incore_loop": 4, "grid_k": 2},
        },
    ]

    def generate_args(self, params):
        tile_size = params["incore_data_size"]
        incore_loop = params["incore_loop"]
        grid_k = params["grid_k"]
        num_groups = params["matmul_add_task_num"] // grid_k
        A = torch.randn(num_groups, grid_k, incore_loop, tile_size, tile_size, dtype=torch.float32) * 0.01
        B = torch.randn(num_groups, grid_k, incore_loop, tile_size, tile_size, dtype=torch.float32) * 0.01
        C = torch.zeros(incore_loop * num_groups, tile_size, tile_size, dtype=torch.float32)
        config = torch.tensor([tile_size, grid_k, num_groups, incore_loop], dtype=torch.int64)
        return TaskArgsBuilder(
            Tensor("A", A.flatten()), Tensor("B", B.flatten()), Tensor("C", C.flatten()), Tensor("config", config)
        )

    def compute_golden(self, args, params):
        tile_size = params["incore_data_size"]
        incore_loop = params["incore_loop"]
        grid_k = params["grid_k"]
        num_groups = params["matmul_add_task_num"] // grid_k
        A = args.A.reshape(num_groups, grid_k, incore_loop, tile_size, tile_size)
        B = args.B.reshape(num_groups, grid_k, incore_loop, tile_size, tile_size)
        C = args.C.reshape(incore_loop * num_groups, tile_size, tile_size)
        C[:] = 0.0
        for group in range(num_groups):
            for k_idx in range(grid_k):
                for i in range(incore_loop):
                    C[group * incore_loop + i] += torch.matmul(A[group, k_idx, i], B[group, k_idx, i])


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
