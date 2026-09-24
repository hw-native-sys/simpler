#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-block Mix scheduling coverage for dependent and independent DAGs."""

import ctypes

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

MATMUL_SIZE = 128
TILE_ELEMS = MATMUL_SIZE * MATMUL_SIZE
TASK_COUNT = 16


@scene_test(level=2, runtime="host_build_graph")
class TestSingleBlockMixHostBuildGraphA5(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/single_block_mix_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.INOUT, D.IN, D.IN, D.INOUT, D.IN, D.IN, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "MATMUL",
                "source": "../../tensormap_and_ringbuffer/mixed_example/kernels/aic/kernel_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "ADD",
                "source": "../../tensormap_and_ringbuffer/mixed_example/kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "name": "MUL",
                "source": "../../tensormap_and_ringbuffer/mixed_example/kernels/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {"name": f"{kind}16_mask{mask}", "platforms": ["a5sim", "a5"], "params": {"graph_case": case, "mask": mask}}
        for kind, case in [("chain", 0), ("burst", 1)]
        for mask in [3, 5, 6, 7]
    ] + [{"name": "ordinary_mix_interleaved", "platforms": ["a5sim", "a5"], "params": {"graph_case": 2, "mask": 7}}]

    def generate_args(self, params):
        torch.manual_seed(42)
        a = torch.randn(MATMUL_SIZE, MATMUL_SIZE, dtype=torch.float32) * 0.01
        b = torch.randn(MATMUL_SIZE, MATMUL_SIZE, dtype=torch.float32) * 0.01
        d = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
        e = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
        g = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
        h = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01

        def output():
            return torch.zeros(TASK_COUNT * TILE_ELEMS, dtype=torch.float32)

        return TaskArgsBuilder(
            TensorArg("a", a.flatten()),
            TensorArg("b", b.flatten()),
            TensorArg("c", output()),
            TensorArg("d", d),
            TensorArg("e", e),
            TensorArg("f", output()),
            TensorArg("g", g),
            TensorArg("h", h),
            TensorArg("i", output()),
            Scalar("graph_case", ctypes.c_int64(params["graph_case"])),
            Scalar("mask", ctypes.c_int64(params["mask"])),
        )

    def compute_golden(self, args, params):
        c = args.c.reshape(TASK_COUNT, TILE_ELEMS)
        f = args.f.reshape(TASK_COUNT, TILE_ELEMS)
        i = args.i.reshape(TASK_COUNT, TILE_ELEMS)
        for task in range(TASK_COUNT):
            mask = params["mask"]
            if params["graph_case"] == 2 and task % 2:
                mask = 1 if task % 4 == 1 else 2
            chained = params["graph_case"] == 0 and task != 0
            if mask & 1:
                a = c[task - 1] if chained else args.a
                c[task] = torch.matmul(
                    a.reshape(MATMUL_SIZE, MATMUL_SIZE), args.b.reshape(MATMUL_SIZE, MATMUL_SIZE)
                ).flatten()
            if mask & 2:
                f[task] = (f[task - 1] if chained else args.d) + args.e
            if mask & 4:
                i[task] = (i[task - 1] if chained else args.g) * args.h


@scene_test(level=2, runtime="host_build_graph")
class TestMixKernelRendezvous(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/rendezvous_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": lane,
                "name": f"RENDEZVOUS_{lane}",
                "source": f"kernels/rendezvous_{lane}.cpp",
                "core_type": "aic" if lane == 0 else "aiv",
                "signature": [D.INOUT],
            }
            for lane in range(3)
        ],
    }
    CASES = [
        {"name": f"burst256_mask{mask}", "platforms": ["a5sim", "a5"], "params": {"mask": mask, "task_count": 256}}
        for mask in [6, 7]
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("state", torch.zeros(params["task_count"] * 48, dtype=torch.int64)),
            Scalar("mask", ctypes.c_int64(params["mask"])),
            Scalar("task_count", ctypes.c_int64(params["task_count"])),
        )

    def compute_golden(self, args, params):
        for lane in range(3):
            if params["mask"] & (1 << lane):
                args.state.reshape(params["task_count"], 48)[:, lane * 16] = 2


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
