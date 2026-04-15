#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Matmul diamond — host_build_graph runtime with AIC+AIV mixed execution.

Computation: F = exp(sqrt(log(A)) @ W1 + sqrt(log(A)) @ W2)
Diamond topology: t0(AIV) -> t1(AIC), t2(AIC) -> t3(AIV).
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="host_build_graph")
class TestMatmulHostBuildGraph(SceneTestCase):
    """Matmul diamond: F = exp(sqrt(log(A)) @ W1 + sqrt(log(A)) @ W2)."""

    RTOL = 1e-2
    ATOL = 1e-2

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/matmul_orch.cpp",
            "function_name": "build_matmul_graph",
            "signature": [D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/kernel_log_sqrt.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aic/kernel_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": "kernels/aiv/kernel_add_exp.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 3, "block_dim": 3},
            "params": {},
        },
    ]

    def generate_args(self, params):
        ROWS = 128
        COLS = 128
        SIZE = ROWS * COLS

        input_value = torch.exp(torch.tensor(4.0)).item()
        weight_value = 1.0 / (2 * COLS)

        a = torch.full((SIZE,), input_value, dtype=torch.float16)
        w1 = torch.full((SIZE,), weight_value, dtype=torch.float16)
        w2 = torch.full((SIZE,), weight_value, dtype=torch.float16)
        f = torch.zeros(SIZE, dtype=torch.float32)

        return TaskArgsBuilder(
            Tensor("a", a),
            Tensor("w1", w1),
            Tensor("w2", w2),
            Tensor("f", f),
        )

    def compute_golden(self, args, params):
        ROWS = 128
        COLS = 128

        a = args.a.reshape(ROWS, COLS).to(torch.float32)
        w1 = args.w1.reshape(ROWS, COLS).to(torch.float32)
        w2 = args.w2.reshape(ROWS, COLS).to(torch.float32)

        b = torch.sqrt(torch.log(a))
        c = torch.matmul(b, w1)
        d = torch.matmul(b, w2)
        args.f[:] = torch.exp(c + d).flatten().to(torch.float32)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
