# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Async payload aliases and predicate-only inputs across ordinary and Graph tasks."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

OUT_WORDS = 4224


@scene_test(level=2, runtime="host_build_graph")
class TestDeferWaitVisibility(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/visibility/orchestration/visibility.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT] * 4 + [D.SCALAR] * 3,
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/visibility/aic/body.cpp",
                "core_type": "aic",
                "signature": [D.INOUT] * 3 + [D.SCALAR] * 2,
            },
            {
                "func_id": 1,
                "source": "kernels/visibility/aiv/body.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT] * 3 + [D.SCALAR] * 2,
            },
            {"func_id": 2, "source": "kernels/visibility/aiv/waiter.cpp", "core_type": "aiv", "signature": [D.IN]},
        ],
    }
    CASES = [
        {
            "name": f"{('aic', 'aiv')[core]}_{'graph' if graph else 'ordinary'}_{'take' if target else 'skip'}",
            "platforms": ["a5", "a5sim"],
            "params": {"core": core, "graph": graph, "target": target},
        }
        for core in (0, 1)
        for graph in (False, True)
        for target in (0, 1)
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            *(TensorArg(name, torch.zeros(64, dtype=torch.int32)) for name in ("signal", "predicate", "payload")),
            TensorArg("out", torch.zeros(2 * OUT_WORDS, dtype=torch.int32)),
            *(Scalar(name, int(params[name])) for name in ("core", "graph", "target")),
        )

    def compute_golden(self, args, params):
        for instance in range(2):
            row = args.out[instance * OUT_WORDS : (instance + 1) * OUT_WORDS]
            if params["target"]:
                row[:16] = torch.arange(16, dtype=torch.int32) * 2 + instance * 200 + 8
                row[16] = 1
            row[17] = 1
            row[18] = instance * 200 + 8

    def compare_outputs(self, actual, expected, output_names, params):
        for instance in range(2):
            row = actual.out[instance * OUT_WORDS : (instance + 1) * OUT_WORDS]
            golden = expected.out[instance * OUT_WORDS : (instance + 1) * OUT_WORDS].clone()
            blocks = int(row[64])
            assert 0 < blocks <= 128, "missing full-pool cache warmup"
            golden[64] = blocks
            golden[128 : 128 + blocks * 32 : 32] = -17 * 32 + 1 - params["target"]
            torch.testing.assert_close(row, golden, rtol=0, atol=0)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
