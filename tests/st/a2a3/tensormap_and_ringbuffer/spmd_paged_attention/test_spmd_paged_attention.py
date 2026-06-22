#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paged attention unroll with TPUSH/TPOP: MIX kernel AIC+AIV cooperative pipeline."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.goldens.paged_attention import compute_golden as _pa_compute_golden
from simpler_setup.goldens.paged_attention import generate_inputs as _pa_generate_inputs


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestPagedAttentionUnrollTpushPop(SceneTestCase):
    # Tolerances relaxed (2e-3 -> 5e-3 in #825, then 5e-3 -> 1e-2 for #848)
    # to absorb hardware numerical drift in the AIC/AIV cooperative TPUSH/TPOP
    # pipeline; observed max_diff ~5.5e-3.
    RTOL = 1e-2
    ATOL = 1e-2

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_paged_attention_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "PA_AIC",
                "source": "kernels/mix/paged_attention_parallel.cpp",
                "core_type": "aic",
                # Declare the full 9-tensor layout here (AIV entry left empty)
                # so the tensor dump — which sums per-subtask signature tensors
                # and matches them to the payload — captures all args under
                # func_id 0. Consumed only by the dump; dispatch ignores it.
                "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.INOUT, D.OUT, D.OUT, D.OUT],
            },
            {
                "func_id": 1,
                "name": "PA_AIV",
                "source": "kernels/mix/paged_attention_parallel.cpp",
                "core_type": "aiv",
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
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
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
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
            # Intra-core trace target only (--case SmallCase1; manual -> not in
            # the default onboard CI sweep). batch=24 == the orchestration's
            # hardcoded SPMD_BLOCK_NUM, so every hw block gets one logical block
            # (fewer stalls in the AIC<->AIV handshake). Same q_tile=16 path as
            # Case1; passes golden at context_len=8192.
            "name": "SmallCase1",
            "platforms": ["a2a3"],
            "manual": True,
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {
                "batch": 24,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 128,
                "context_len": 8192,
                "max_model_len": 32768,
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
