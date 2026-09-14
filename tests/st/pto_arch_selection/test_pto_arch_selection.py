#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end regression for selecting the PTO ISA architecture in CPU sim."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="host_build_graph")
class TestPtoArchSelection(SceneTestCase):
    """Use an FP32 row sum whose result distinguishes A2/A3 from A5."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/row_sum_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/row_sum.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            }
        ],
    }

    CASES = [
        {
            "name": "a2a3_reduction",
            "platforms": ["a2a3sim"],
            "params": {"expected": 1.0},
        },
        {
            "name": "a5_reduction",
            "platforms": ["a5sim"],
            "params": {"expected": 0.0},
        },
    ]

    def generate_args(self, params):
        del params
        pattern = torch.tensor([1e8, 1.0, -1e8, 1.0], dtype=torch.float32)
        return TaskArgsBuilder(
            TensorArg("x", pattern.repeat(8, 16)),
            TensorArg("out", torch.zeros((8, 1), dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.out.fill_(params["expected"])


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
