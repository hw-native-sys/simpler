# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Waiters must release AIV execution resources before their signal arrives."""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestDeferWaitProgress(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/defer_wait_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/defer_wait.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            }
        ],
    }
    CASES = [
        {"name": "c_minus_one", "platforms": ["a5sim", "a5"], "params": {"delta": -1}},
        {"name": "c", "platforms": ["a5sim", "a5"], "params": {"delta": 0}},
        {"name": "c_plus_one", "platforms": ["a5sim", "a5"], "params": {"delta": 1}},
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("out", torch.zeros(96, dtype=torch.int32)),
            Scalar("delta", params["delta"]),
        )

    def compute_golden(self, args, params):
        args.out[0] = 1
        args.out[32] = 1
        args.out[64] = 1


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
