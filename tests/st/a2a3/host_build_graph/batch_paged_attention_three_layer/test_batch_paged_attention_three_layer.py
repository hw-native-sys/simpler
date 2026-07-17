#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Three-layer batch paged attention baseline for host_build_graph."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.goldens.paged_attention import compute_golden as _pa_compute_golden
from simpler_setup.goldens.paged_attention import generate_inputs as _pa_generate_inputs


@scene_test(level=2, runtime="host_build_graph")
class TestBatchPagedAttentionThreeLayer(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/paged_attention_three_layer_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "QK",
                "source": "kernels/aic/aic_qk_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
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
                "signature": [D.IN, D.IN, D.OUT],
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
            "name": "SmallThreeLayer",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 9},
            "params": {
                "batch": 1,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 16,
                "block_size": 16,
                "context_len": 33,
                "max_model_len": 256,
                "dtype": "bfloat16",
                "layer_count": 3,
            },
        },
        {
            "name": "PipelineStressThreeLayer",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "manual": True,
            "params": {
                "batch": 16,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 128,
                "context_len": 8192,
                "max_model_len": 8192,
                "dtype": "bfloat16",
                "layer_count": 3,
            },
        },
        {
            "name": "PipelineStressTwoTokenFortyLayer",
            "platforms": ["a2a3"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 24,
                "runtime_env": {
                    "ring_task_window": 32768,
                    "ring_heap": 1024 * 1024 * 1024,
                    "ring_dep_pool": 32768,
                },
            },
            "manual": True,
            "params": {
                "batch": 16,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 128,
                "context_len": 8192,
                "max_model_len": 8192,
                "dtype": "bfloat16",
                "layer_count": 80,
                "layers_per_epoch": 40,
            },
        },
        {
            "name": "PipelineStressThreeTokenFortyLayer",
            "platforms": ["a2a3"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 24,
                "runtime_env": {
                    "ring_task_window": 32768,
                    "ring_heap": 1024 * 1024 * 1024,
                    "ring_dep_pool": 32768,
                },
            },
            "manual": True,
            "params": {
                "batch": 16,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 128,
                "context_len": 8192,
                "max_model_len": 8192,
                "dtype": "bfloat16",
                "layer_count": 120,
                "layers_per_epoch": 40,
            },
        },
    ]

    def generate_args(self, params):
        specs = []
        for name, value in _pa_generate_inputs(params):
            if isinstance(value, torch.Tensor):
                specs.append(Tensor(name, value))
            else:
                specs.append(Scalar(name, value))
        specs.append(Scalar("layer_count", params["layer_count"]))
        specs.append(Scalar("layers_per_epoch", params.get("layers_per_epoch", 1)))
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        tensors = {spec.name: spec.value for spec in args.specs if isinstance(spec, Tensor)}
        _pa_compute_golden(tensors, params)
        for spec in args.specs:
            if isinstance(spec, Tensor) and spec.name in tensors:
                getattr(args, spec.name)[:] = tensors[spec.name]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
