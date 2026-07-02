#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MIX co-ownership test for fully_distributed_within_core.

Each group submits a 1C+2V MIX task (Cmm=A@B on AIC, V0=A+B on AIV0, V1=A+B on
AIV1) plus a consumer (Vfinal=V0+V1). This exercises the block.won anchor->
follower deposit/drain path and the single joint completion flag.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="fully_distributed_within_core")
class TestMixCoown(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/mix_coown_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "MM",
                "source": "kernels/aic/kernel_mm.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT, D.OUT, D.OUT, D.IN],
            },
            {
                "func_id": 1,
                "name": "ADD_V0",
                "source": "kernels/aiv/kernel_add_v0.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT, D.OUT, D.OUT, D.IN],
            },
            {
                "func_id": 2,
                "name": "ADD_V1",
                "source": "kernels/aiv/kernel_add_v1.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT, D.OUT, D.OUT, D.IN],
            },
            {
                "func_id": 3,
                "name": "SUM",
                "source": "kernels/aiv/kernel_sum.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT, D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "Mix12",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {"num_groups": 12, "tile_size": 64},
        },
        {
            "name": "Mix24",
            "manual": True,
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"num_groups": 48, "tile_size": 64},
        },
    ]

    def generate_args(self, params):
        n = params["num_groups"]
        t = params["tile_size"]
        A = torch.randn(n, t, t, dtype=torch.float32) * 0.01
        B = torch.randn(n, t, t, dtype=torch.float32) * 0.01
        Cmm = torch.zeros(n, t, t, dtype=torch.float32)
        Vfinal = torch.zeros(n, t, t, dtype=torch.float32)
        # config: [tile_size, grid_k(unused), num_groups, num_tiles_per_group]
        config = torch.tensor([t, 1, n, 1], dtype=torch.int64)
        return TaskArgsBuilder(
            Tensor("A", A.flatten()),
            Tensor("B", B.flatten()),
            Tensor("Cmm", Cmm.flatten()),
            Tensor("Vfinal", Vfinal.flatten()),
            Tensor("config", config),
        )

    def compute_golden(self, args, params):
        n = params["num_groups"]
        t = params["tile_size"]
        A = args.A.reshape(n, t, t)
        B = args.B.reshape(n, t, t)
        Cmm = args.Cmm.reshape(n, t, t)
        Vfinal = args.Vfinal.reshape(n, t, t)
        for g in range(n):
            Cmm[g] = torch.matmul(A[g], B[g])
            Vfinal[g] = 2.0 * (A[g] + B[g])


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
