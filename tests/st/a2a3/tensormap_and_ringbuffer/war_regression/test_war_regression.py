#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 loop-carried and cross-ring write-after-read regressions."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestWarRegressionTmrA2A3(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/war_regression_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.OUT],
        },
        "incores": [
            {"func_id": 0, "source": "kernels/aiv/writer.cpp", "core_type": "aiv", "signature": [D.INOUT]},
            {"func_id": 1, "source": "kernels/aiv/reader.cpp", "core_type": "aiv", "signature": [D.IN, D.OUT]},
        ],
    }
    CASES = [
        {
            "name": "LoopCarried",
            "platforms": ["a2a3sim", "a2a3"],
            "params": {"cross_ring": False},
        },
        {
            "name": "CrossRing",
            "platforms": ["a2a3sim", "a2a3"],
            "params": {"cross_ring": True},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("buffer", torch.zeros(16, dtype=torch.float32)),
            TensorArg("outputs", torch.zeros((2, 16), dtype=torch.float32)),
            Scalar("spin_iters", 100000),
            Scalar("cross_ring", int(params["cross_ring"])),
        )

    def compute_golden(self, args, params):
        args.buffer[:] = 2.0
        args.outputs[0] = 1.0
        args.outputs[1] = 2.0


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
