#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Speculative early-dispatch — SPMD consumer coverage.

DAG: t0 (single-block AIV, flagged) out_p[i]=i+1 -> t1 (SPMD AIV, block_num=4)
out_c[i*CL]=out_p[i]+10. The multi-block consumer is pre-staged block-by-block:
three blocks fit the doorbell budget (gated + released by doorbell) and the
fourth dispatches normally off the ready queue. Each block writes its own cache
line (block i -> out_c[i*CL]); golden out_c[i*CL]=i+11, other slots 0.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

BLOCK_NUM = 4
# Each SPMD consumer block writes its own cache line — two AICore cores must
# never write the same line on silicon (see docs/aicore-kernel-programming.md).
# CL = 64B / sizeof(float) on a2a3, so block i writes out_c[i * CL].
CL = 16


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpecSpmdConsumer(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spec_spmd_consumer_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "FILL",
                "source": "kernels/aiv/kernel_fill_all.cpp",
                "core_type": "aiv",
                "signature": [D.OUT],
            },
            {
                "func_id": 1,
                "name": "ADDK",
                "source": "kernels/aiv/kernel_spmd_addk.cpp",
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
            Tensor("out_c", torch.zeros(BLOCK_NUM * CL, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        # SPMD block i reads out_p[i]=i+1 and writes out_c[i*CL]=(i+1)+10.
        args.out_c.zero_()
        for i in range(BLOCK_NUM):
            args.out_c[i * CL] = float(i + 1 + 10)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
