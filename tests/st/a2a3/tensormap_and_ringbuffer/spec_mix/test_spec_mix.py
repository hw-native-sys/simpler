#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Speculative early-dispatch — MIX consumer coverage.

DAG: t0 (AIV, flagged) c = a + b -> t1 (MIX) d = c + b (aiv0), e = c * b (aiv1).
While t0 runs, the single-block MIX consumer t1 is pre-staged onto an idle
cluster (both AIV cores gated on the doorbell) and released on t0's completion,
exercising the multi-doorbell (staged_count == 2) release path.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

SIZE = 128 * 128


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpecMix(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spec_mix_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "ADD",
                "source": "kernels/aiv/kernel_add_standalone.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "MUL",
                "source": "kernels/aiv/kernel_mul_standalone.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 8},
            "params": {},
        },
    ]

    def generate_args(self, params):
        torch.manual_seed(42)
        a = torch.randn(SIZE, dtype=torch.float32) * 0.01
        b = torch.randn(SIZE, dtype=torch.float32) * 0.01
        return TaskArgsBuilder(
            Tensor("a", a),
            Tensor("b", b),
            Tensor("d", torch.zeros(SIZE, dtype=torch.float32)),
            Tensor("e", torch.zeros(SIZE, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        c = args.a + args.b
        args.d[:] = c + args.b
        args.e[:] = c * args.b


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
