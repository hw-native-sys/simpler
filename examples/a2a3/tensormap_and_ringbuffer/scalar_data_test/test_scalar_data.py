#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scalar data dependency test: GetTensorData, SetTensorData, add_inout.

Tests orchestration-level data manipulation: scalar initialization,
Get/Set round-trips, WAW+WAR dependency auto-wait, and external tensor WAR.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestScalarData(SceneTestCase):
    """Scalar data dependency: Get/SetTensorData, add_inout with initial value."""

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/scalar_data_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_noop.cpp",
                "core_type": "aiv",
                "signature": [],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {},
        },
    ]

    def generate_args(self, params):
        SIZE = 128 * 128
        return TaskArgsBuilder(
            Tensor("a", torch.full((SIZE,), 2.0, dtype=torch.float32)),
            Tensor("b", torch.arange(SIZE, dtype=torch.float32)),
            Tensor("result", torch.zeros(SIZE, dtype=torch.float32)),
            Tensor("check", torch.zeros(10, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        # result = a + b (computed by kernel_add)
        args.result[:] = args.a + args.b

        # check values written by orchestration via SetTensorData
        args.check[0] = 2.0  # GetTensorData(c, {0}): c = a + b, c[0] = 2.0+0.0
        args.check[1] = 102.0  # GetTensorData(c, {100}): c[100] = 2.0+100.0
        args.check[2] = 77.0  # runtime-created scalar output initialized to 77.0
        args.check[3] = 77.0  # second noop via add_inout preserves the value
        args.check[4] = 79.0  # orchestration arithmetic: 2.0 + 77.0
        args.check[5] = 42.0  # Orch set->get round-trip: SetTensorData then GetTensorData
        args.check[6] = 12.0  # Orch->AICore RAW: SetTensorData(d,10.0) + kernel_add(d,a) -> 10.0+2.0
        args.check[7] = 88.0  # WAW+WAR: kernel reads c, SetTensorData(c,88.0) auto-waits
        args.check[8] = 55.0  # External WAR: noop(ext_b INOUT) -> SetTensorData(ext_b,55.0) auto-waits


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
