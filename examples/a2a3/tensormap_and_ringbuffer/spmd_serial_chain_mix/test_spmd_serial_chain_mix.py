#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SPMD MIX serial chain of 50us busy-wait tasks.

Sibling to spmd_serial_chain_spin. Same 4-task chain (input counts 0, 4, 8, 12),
same block_num=24, same ~50us busy-wait — but every task is a MIX submit
(AIC + AIV0 + AIV1), so each of the 24 blocks occupies a full cluster (1 AIC +
2 AIV cores = 72 AICore cores in flight per task).
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdSerialChainMix(SceneTestCase):
    """4 chained MIX tasks, block_dim=24, ~50us each."""

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_serial_chain_mix_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SPIN_AIC",
                "source": "kernels/aic/kernel_spin50us_aic.cpp",
                "core_type": "aic",
            },
            {
                "func_id": 1,
                "name": "SPIN_AIV",
                "source": "kernels/aiv/kernel_spin50us_aiv.cpp",
                "core_type": "aiv",
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {},
        },
    ]

    def generate_args(self, params):
        SIZE = 128
        return TaskArgsBuilder(
            Tensor("out", torch.zeros(SIZE, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        # Kernels intentionally do not touch memory.
        args.out[:] = 0.0


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
