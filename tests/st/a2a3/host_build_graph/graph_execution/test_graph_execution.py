#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Graph Execution records once and replays topology with dynamic TaskArgs."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="host_build_graph")
class TestGraphExecutionHostBuildGraph(SceneTestCase):
    RTOL = 1e-5
    ATOL = 1e-5

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/graph_execution_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.OUT, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "../vector_example/kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "../vector_example/kernels/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": "../vector_example/kernels/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "record_then_replay",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {},
        },
    ]

    def generate_args(self, params):
        size = 128 * 128
        return TaskArgsBuilder(
            Tensor("a", torch.full((size,), 2.0, dtype=torch.float32)),
            Tensor("b", torch.full((size,), 3.0, dtype=torch.float32)),
            Tensor("output_1", torch.zeros(size, dtype=torch.float32)),
            Tensor("output_3", torch.zeros(size, dtype=torch.float32)),
            Tensor("output_5", torch.zeros(size, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        base = args.a + args.b
        args.output_1[:] = (base + 1.0) * (base + 2.0)
        args.output_3[:] = (base + 3.0) * (base + 2.0)
        args.output_5[:] = (base + 5.0) * (base + 2.0)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
