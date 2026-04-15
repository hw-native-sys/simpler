#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paged attention — host_build_graph runtime.

Tests host_build_graph runtime with AIC+AIV mixed execution and INOUT tensors.
Templated kernels support variable tile sizes via runtime dispatch.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.goldens.paged_attention import compute_golden as _pa_compute_golden  # noqa: PLC0415
from simpler_setup.goldens.paged_attention import generate_inputs as _pa_generate_inputs  # noqa: PLC0415


@scene_test(level=2, runtime="host_build_graph")
class TestPagedAttentionHostBuildGraph(SceneTestCase):
    """Paged attention with host_build_graph runtime."""

    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/paged_attention_orch.cpp",
            "function_name": "build_paged_attention_graph",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/aic_qk_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": "kernels/aic/aic_pv_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/aiv_softmax_prepare.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT, D.OUT, D.OUT],
            },
            {
                "func_id": 3,
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
            "config": {"aicpu_thread_num": 3, "block_dim": 24},
            "params": {
                "batch": 256,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 128,
                "context_len": 8100,
                "max_model_len": 32768,
                "dtype": "bfloat16",
            },
        },
        {
            "name": "Case2",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 3, "block_dim": 24},
            "manual": True,
            "params": {
                "batch": 64,
                "num_heads": 64,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 64,
                "context_len": 8150,
                "max_model_len": 32768,
                "dtype": "bfloat16",
            },
        },
        {
            "name": "small1",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 3, "block_dim": 3},
            "params": {
                "batch": 1,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 16,
                "block_size": 16,
                "context_len": 16,
                "max_model_len": 256,
                "dtype": "bfloat16",
            },
        },
        {
            "name": "small2",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 3, "block_dim": 3},
            "manual": True,
            "params": {
                "batch": 1,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 16,
                "block_size": 16,
                "context_len": 64,
                "max_model_len": 256,
                "dtype": "bfloat16",
            },
        },
    ]

    def generate_args(self, params):
        inputs = _pa_generate_inputs(params)
        specs = []
        for name, val in inputs:
            if isinstance(val, torch.Tensor):
                specs.append(Tensor(name, val))
            else:
                specs.append(Scalar(name, val))
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        tensors = {s.name: s.value for s in args.specs if isinstance(s, Tensor)}
        _pa_compute_golden(tensors, params)
        for s in args.specs:
            if isinstance(s, Tensor) and s.name in tensors:
                getattr(args, s.name)[:] = tensors[s.name]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
