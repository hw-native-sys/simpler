# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Deferred waiters release execution resources to later notification tasks."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from tests.st.a5.host_build_graph.defer_wait_shapes.test_defer_wait_shapes import TestDeferWaitShapes as _Shapes


@scene_test(level=2, runtime="percounter")
class TestPercounterDeferShapes(_Shapes):
    pytestmark = []
    CASES = [dict(case, manual=True, params=dict(case["params"], condition_capacity=4)) for case in _Shapes.CASES]


@scene_test(level=2, runtime="percounter")
class TestPercounterDeferProgress(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/progress.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.SCALAR, D.SCALAR],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/progress.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.SCALAR, D.SCALAR],
            }
        ],
    }
    CASES = [
        {
            "name": "more_waiters_than_lane_slots",
            "manual": True,
            "platforms": ["a5sim", "a5"],
            "params": {"waiters": 256, "consumers": 0},
        },
        {
            "name": "notifier_after_blocked_windows",
            "manual": True,
            "platforms": ["a5sim", "a5"],
            "params": {"waiters": 1, "consumers": 1024},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("out", torch.zeros((params["waiters"] + params["consumers"]) * 32, dtype=torch.int32)),
            Scalar("waiters", params["waiters"]),
            Scalar("consumers", params["consumers"]),
        )

    def compute_golden(self, args, params):
        args.out[::32] = 1


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
