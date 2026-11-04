#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Graph gates and wide joins retain every child dependency on resident AICore scheduling."""

import ctypes

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestGraphNativeTopology(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/graph_topology_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {"func_id": 0, "source": "kernels/check_graph.cpp", "core_type": "aic", "signature": [D.INOUT]},
            {"func_id": 1, "source": "kernels/check_graph.cpp", "core_type": "aiv", "signature": [D.INOUT]},
        ],
    }
    CASES = [
        {
            "name": name,
            "platforms": ["a5sim", "a5"],
            "params": {"nodes": nodes, "kind": kind, "instances": instances},
            **({"manual": ["a5sim"]} if nodes == 1024 else {}),
        }
        for name, nodes, kind, instances in (
            ("roots_1", 1, 0, 1),
            ("roots_1024", 1024, 0, 1),
            ("fanin_1", 2, 1, 1),
            ("fanin_128", 129, 1, 1),
            ("fanin_129", 130, 1, 1),
            ("fanin_1023", 1024, 1, 1),
            ("dummy_chain", 1024, 2, 1),
            ("dummy_only", 1024, 3, 1),
            ("shared_definition", 33, 0, 3),
        )
    ]

    def generate_args(self, params):
        size = params["instances"] * (params["nodes"] + 2) * 8
        return TaskArgsBuilder(
            TensorArg("state", torch.zeros(size, dtype=torch.int64)),
            Scalar("nodes", ctypes.c_int64(params["nodes"])),
            Scalar("kind", ctypes.c_int64(params["kind"])),
            Scalar("instances", ctypes.c_int64(params["instances"])),
        )

    def compute_golden(self, args, params):
        nodes = params["nodes"]
        for instance in range(params["instances"]):
            base = instance * (nodes + 2)
            token = 10000 + instance * 1000
            args.state[base * 8] = token
            if params["kind"] < 2:
                args.state[(base + 1) * 8 : (base + nodes + 1) * 8 : 8] = torch.arange(
                    token + 1, token + nodes + 1, dtype=torch.int64
                )
            elif params["kind"] == 2:
                args.state[(base + nodes) * 8] = token + nodes
            args.state[(base + nodes + 1) * 8] = token + nodes + 1


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
