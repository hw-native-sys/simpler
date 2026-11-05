#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Alternating matmul + add benchmark for the host-build-graph runtime."""

import ctypes

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestAlternatingMatmulAddHostBuildGraph(SceneTestCase):
    """Alternating matmul + add with scalar parameters."""

    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/alternating_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/kernel_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a5sim", "a5"],
            "params": {"batch": 1, "M": 1, "N": 1, "matmul_batch": 1, "add_batch": 1},
        },
        {
            "name": "Case1",
            "platforms": ["a5"],
            "params": {"batch": 500, "M": 4, "N": 4, "matmul_batch": 4, "add_batch": 4},
            "manual": True,
        },
        {
            "name": "Case2",
            "platforms": ["a5"],
            "params": {"batch": 512, "M": 2, "N": 5, "matmul_batch": 4, "add_batch": 5},
            "manual": True,
        },
    ]

    def generate_args(self, params):
        batch = params["batch"]
        M = params["M"]
        N = params["N"]
        matmul_batch = params.get("matmul_batch", 1)
        add_batch = params.get("add_batch", 1)
        matmul_size = 128
        add_rows = 128
        add_cols = 128

        torch.manual_seed(42)
        A = torch.randn(batch, M, matmul_size, matmul_size, dtype=torch.float32) * 0.01
        B = torch.randn(batch, M, matmul_size, matmul_size, dtype=torch.float32) * 0.01
        C = torch.zeros(batch, M, matmul_size, matmul_size, dtype=torch.float32)
        X = torch.randn(batch, N, add_rows, add_cols, dtype=torch.float32) * 0.01
        Y = torch.randn(batch, N, add_rows, add_cols, dtype=torch.float32) * 0.01
        Z = torch.zeros(batch, N, add_rows, add_cols, dtype=torch.float32)

        return TaskArgsBuilder(
            TensorArg("A", A.flatten()),
            TensorArg("B", B.flatten()),
            TensorArg("C", C.flatten()),
            TensorArg("X", X.flatten()),
            TensorArg("Y", Y.flatten()),
            TensorArg("Z", Z.flatten()),
            Scalar("batch", ctypes.c_int64(batch)),
            Scalar("M_val", ctypes.c_int64(M)),
            Scalar("N_val", ctypes.c_int64(N)),
            Scalar("matmul_batch", ctypes.c_int64(matmul_batch)),
            Scalar("add_batch", ctypes.c_int64(add_batch)),
        )

    def compute_golden(self, args, params):
        batch = params["batch"]
        M = params["M"]
        N = params["N"]
        matmul_size = 128
        add_rows = 128
        add_cols = 128

        A = args.A.reshape(batch, M, matmul_size, matmul_size)
        B = args.B.reshape(batch, M, matmul_size, matmul_size)
        C = args.C.reshape(batch, M, matmul_size, matmul_size)
        X = args.X.reshape(batch, N, add_rows, add_cols)
        Y = args.Y.reshape(batch, N, add_rows, add_cols)
        Z = args.Z.reshape(batch, N, add_rows, add_cols)

        for b in range(batch):
            for m in range(M):
                C[b, m] = torch.matmul(A[b, m], B[b, m])
        for b in range(batch):
            for n in range(N):
                Z[b, n] = X[b, n] + Y[b, n]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
