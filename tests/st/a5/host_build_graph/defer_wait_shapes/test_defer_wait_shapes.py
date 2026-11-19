# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Physical returns and all condition groups jointly release one logical task."""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestDeferWaitShapes(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/shapes.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.SCALAR, D.SCALAR, D.SCALAR, D.SCALAR],
        },
        "incores": [
            {
                "func_id": func_id,
                "source": f"kernels/{core_type}.cpp",
                "core_type": core_type,
                "signature": [D.INOUT, D.INOUT, D.SCALAR, D.SCALAR, D.SCALAR, D.SCALAR, D.SCALAR],
            }
            for func_id, core_type in enumerate(["aic", "aiv"])
        ],
    }
    CASES = [
        {
            "name": f"mask_{mask}_{name}",
            "platforms": ["a5sim", "a5"],
            "params": {"mask": mask, "mode": mode, "graph": graph},
        }
        for mask in [1, 2, 3, 5, 6, 7]
        for name, mode, graph in [
            ("single", 0, False),
            ("waves", 1, False),
            ("sync_start", 2, False),
            ("graph_instances", 0, True),
        ]
    ] + [
        {
            "name": f"mask_{mask}_predicate_{mode == 4}",
            "platforms": ["a5sim", "a5"],
            "params": {"mask": mask, "mode": mode, "graph": False},
        }
        for mask in [1, 2]
        for mode in [3, 4]
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("out", torch.zeros(128, dtype=torch.int32)),
            Scalar("mask", params["mask"]),
            Scalar("mode", params["mode"]),
            Scalar("graph", int(params["graph"])),
            Scalar("condition_capacity", params.get("condition_capacity", 64)),
        )

    def compute_golden(self, args, params):
        for instance in range(2 if params["graph"] else 1):
            args.out[instance * 64] = 1
            args.out[instance * 64 + 32] = 1


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
