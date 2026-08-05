#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paged attention: online softmax with AIC/AIV subgraph splitting (bfloat16), host_build_graph runtime.

The orchestration and kernel sources are the same as the tensormap_and_ringbuffer
variant: `pto_orchestration_api.h` is identical between the two runtimes, and the
framework compiles the orchestration against the include dirs of whichever
runtime `@scene_test` names.

host_build_graph populates the whole task graph on the host before the device
begins scheduling, so a ring slot cannot be reclaimed mid-orchestration: the
ring window and GM heap must hold every task of a case at once. Cases above the
default window carry explicit `runtime_env` sizing below.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.goldens.paged_attention import compute_golden as _pa_compute_golden
from simpler_setup.goldens.paged_attention import generate_inputs as _pa_generate_inputs

# tasks = batch * (4 * ceil(context_len / block_size) + 1). ring_task_window must
# be a power of two in [4, INT32_MAX] and is rejected otherwise by the runtime.
_RING_65K = {"ring_task_window": 131072, "ring_heap": 2 * 1024 * 1024 * 1024}
_RING_33K = {"ring_task_window": 65536, "ring_heap": 1024 * 1024 * 1024}


@scene_test(level=2, runtime="host_build_graph")
class TestPagedAttentionHostBuildGraph(SceneTestCase):
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
            # 65 792 tasks.
            "name": "Case1",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "runtime_env": _RING_65K},
            "manual": True,
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
            # 32 832 tasks.
            "name": "Case2",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "runtime_env": _RING_33K},
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
            "name": "CaseSmall1",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4},
            "params": {
                "batch": 1,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 16,
                "block_size": 16,
                "context_len": 33,
                "max_model_len": 256,
                "dtype": "bfloat16",
            },
        },
        {
            "name": "CaseSmall2",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4},
            "manual": True,
            "params": {
                "batch": 1,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 16,
                "block_size": 16,
                "context_len": 128,
                "max_model_len": 256,
                "dtype": "bfloat16",
            },
        },
        {
            # context_lens_list makes the per-batch block counts differ, so the
            # graph is ragged rather than a uniform batch * blocks grid.
            "name": "CaseVarSeq2",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4},
            "manual": True,
            "params": {
                "batch": 2,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 16,
                "block_size": 16,
                "context_len": 33,
                "context_lens_list": [33, 17],
                "max_model_len": 256,
                "dtype": "bfloat16",
            },
        },
        {
            "name": "CaseVarSeq4",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4},
            "manual": True,
            "params": {
                "batch": 4,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 16,
                "block_size": 16,
                "context_len": 128,
                "context_lens_list": [33, 64, 128, 15],
                "max_model_len": 256,
                "dtype": "bfloat16",
            },
        },
    ]

    def generate_args(self, params):
        result = _pa_generate_inputs(params)
        specs = []
        for name, value in result:
            if isinstance(value, torch.Tensor):
                specs.append(Tensor(name, value))
            else:
                specs.append(Scalar(name, value))
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        tensors = {s.name: s.value for s in args.specs if isinstance(s, Tensor)}
        _pa_compute_golden(tensors, params)
        for s in args.specs:
            if isinstance(s, Tensor) and s.name in tensors:
                getattr(args, s.name)[:] = tensors[s.name]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
