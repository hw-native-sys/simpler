#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Same-boundary colliding configurations each record once and replay independently."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="host_build_graph")
class TestGraphCollision(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/graph_collision_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.OUT, D.OUT, D.OUT, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "../vector_example/kernels/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            }
        ],
    }
    CASES = [{"name": "same_boundary_abab", "platforms": ["a2a3sim", "a2a3"], "params": {}}]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("input", torch.full((16384,), 2.0)),
            *(TensorArg(f"out{i}", torch.zeros(16384)) for i in range(4)),
        )

    def compute_golden(self, args, params):
        for i, delta in enumerate((1.0, 20.0, 3.0, 40.0)):
            getattr(args, f"out{i}")[:] = args.input + delta


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
