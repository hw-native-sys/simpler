#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Speculative early-dispatch — SPMD producer coverage.

DAG: t0 (SPMD AIV, block_num=4, flagged) block i writes out_p[i]=i+1 ->
t1 (single-block AIV) out_c[i]=2*out_p[i]. The flagged producer is multi-block;
its consumer is pre-staged while the blocks run and released only after the last
block completes, so the result is correct iff the doorbell fires once, late.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

BLOCK_NUM = 4


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpecSpmdProducer(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spec_spmd_producer_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "FILL",
                "source": "kernels/aiv/kernel_spmd_fill.cpp",
                "core_type": "aiv",
                "signature": [D.OUT],
            },
            {
                "func_id": 1,
                "name": "DOUBLE",
                "source": "kernels/aiv/kernel_double.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
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
        return TaskArgsBuilder(
            Tensor("out_c", torch.zeros(BLOCK_NUM, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        # block i fills out_p[i] = i + 1; consumer doubles it.
        out_p = torch.arange(1, BLOCK_NUM + 1, dtype=torch.float32)
        args.out_c[:] = 2.0 * out_p


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
