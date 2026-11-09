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
            "signature": [D.INOUT, D.IN, D.IN],
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
        {"name": "negative_signal", "platforms": ["a5sim", "a5"], "params": {"delta": 1, "initial": -1}},
        {
            "name": "minimum_signal",
            "platforms": ["a5sim", "a5"],
            "params": {"delta": 1, "initial": -(2**31)},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("out", torch.zeros(96, dtype=torch.int32)),
            Scalar("delta", params["delta"]),
            Scalar("initial", params.get("initial", 0)),
        )

    def compute_golden(self, args, params):
        args.out[0] = 1
        args.out[32] = 1
        args.out[64] = 1


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestDeferWaitReadyFairness(SceneTestCase):
    """Observe bounded progress on nonpreemptive AICore workers.

    A simulator host can deschedule a CHECK/FINISH owner while other host
    threads exhaust the finite workload. That is not an AICore service-budget
    violation; CPU tests drive the production service steps deterministically.
    """

    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/ready_fairness_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/ready_fairness.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN, D.IN, D.IN],
            }
        ],
    }
    CASES = [{"name": "continuous_ready", "platforms": ["a5"], "params": {"count": 2048}}]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("out", torch.zeros((params["count"] + 4) * 32, dtype=torch.int32)),
            Scalar("count", params["count"]),
        )

    def compute_golden(self, args, params):
        args.out[: 4 * 32 : 32] = 1
        args.out[4 * 32 :: 32] = torch.arange(1, params["count"] + 1, dtype=torch.int32)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
