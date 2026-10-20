#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Mixed-core DAG correctness, queue-wrap, and topology pressure matrix."""

import ctypes

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


def _case(name, graph_case, task_count):
    case = {
        "name": name,
        "platforms": ["a5sim", "a5"],
        "params": {"graph_case": graph_case, "task_count": task_count},
    }
    if name != "mixed_chain_65":
        case["manual"] = True
    return case


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestHbgMultiCoreDag(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/multi_core_dag_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/check_stress.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "source": "kernels/check_stress.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        *[_case(f"mixed_chain_{count}", 0, count) for count in (1, 2, 63, 65, 1024)],
        *[_case(f"mixed_fanin32_{count}", 1, count) for count in (64, 128, 256, 1024)],
        _case("mixed_random_1024", 3, 1024),
        _case("mixed_multi_root_4096", 2, 4096),
    ]

    def generate_args(self, params):
        task_count = params["task_count"]
        return TaskArgsBuilder(
            TensorArg("task_state", torch.zeros((task_count + 1) * 8, dtype=torch.int64)),
            Scalar("graph_case", ctypes.c_int64(params["graph_case"])),
            Scalar("task_count", ctypes.c_int64(task_count)),
        )

    def compute_golden(self, args, params):
        task_count = params["task_count"]
        args.task_state[8 : (task_count + 1) * 8 : 8] = torch.arange(1, task_count + 1, dtype=torch.int64)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
