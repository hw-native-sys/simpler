#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paged attention unroll manual-scope benchmark for host_build_graph."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.goldens.paged_attention import compute_golden as _pa_compute_golden
from simpler_setup.goldens.paged_attention import generate_inputs as _pa_generate_inputs


@scene_test(level=2, runtime="host_build_graph")
class TestPagedAttentionUnrollManualScopeHostBuildGraph(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/paged_attention_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "QK",
                "source": "kernels/aic/aic_qk_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "SF",
                "source": "kernels/aiv/aiv_softmax_prepare.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT, D.OUT, D.OUT],
            },
            {
                "func_id": 2,
                "name": "PV",
                "source": "kernels/aic/aic_pv_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 3,
                "name": "UP",
                "source": "kernels/aiv/aiv_online_update.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a2a3"],
            "params": {
                "batch": 256,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 128,
                "context_len": 8192,
                "max_model_len": 32768,
                "dtype": "bfloat16",
            },
        },
        {
            "name": "Case2",
            "platforms": ["a2a3"],
            "manual": True,
            "params": {
                "batch": 64,
                "num_heads": 64,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 64,
                "context_len": 8192,
                "max_model_len": 32768,
                "dtype": "bfloat16",
            },
        },
        {
            "name": "Case3",
            "platforms": ["a2a3"],
            "manual": True,
            "params": {
                "batch": 64,
                "num_heads": 64,
                "kv_head_num": 1,
                "head_dim": 256,
                "block_size": 64,
                "context_len": 8192,
                "max_model_len": 32768,
                "dtype": "bfloat16",
            },
        },
    ]

    # Matched A/B for the child_memory declaration: same workload, same fixtures,
    # only the memory policy differs.
    CASES += [
        {
            "name": name,
            "platforms": ["a2a3"],
            "manual": True,
            "params": {
                "batch": 16,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 128,
                "context_len": 1024,
                "max_model_len": 2048,
                "dtype": "bfloat16",
                "child_memory": child_memory,
            },
        }
        for name, child_memory in (("HostStaged", False), ("ChildMemory", True))
    ]

    def generate_args(self, params):
        params = {**params, "variant": "paged_attention_unroll"}
        result = _pa_generate_inputs(params)
        specs = []
        for name, value in result:
            if isinstance(value, torch.Tensor):
                # The orchestration reads these two on the host to shape the graph,
                # so they stay host-staged; only the bulk tensors become child memory.
                control = name in ("context_lens", "block_table")
                child_memory = params.get("child_memory", False) and not control
                specs.append(TensorArg(name, value, child_memory=child_memory))
            else:
                specs.append(Scalar(name, value))
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        tensors = {s.name: s.value for s in args.specs if isinstance(s, TensorArg)}
        _pa_compute_golden(tensors, params)
        for s in args.specs:
            if isinstance(s, TensorArg) and s.name in tensors:
                getattr(args, s.name)[:] = tensors[s.name]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
