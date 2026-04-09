#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paged attention with small ring buffer sizes — stress test for ring rotation/reclamation.

Tests RUNTIME_ENV (PTO2_RING_TASK_WINDOW, PTO2_RING_HEAP, PTO2_RING_DEP_POOL),
INOUT tensors, bfloat16, and AIC+AIV mixed execution.
"""

import torch
from paged_attention_golden import compute_golden as _pa_compute_golden  # noqa: PLC0415
from paged_attention_golden import generate_inputs as _pa_generate_inputs  # noqa: PLC0415
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test

PA_KERNELS = "../../../../tests/st/a2a3/tensormap_and_ringbuffer/paged_attention/kernels"


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestPagedAttentionRingbuffer(SceneTestCase):
    """Paged attention with small ring buffer sizes for stress testing."""

    RTOL = 1e-3
    ATOL = 1e-3
    RUNTIME_ENV = {
        "PTO2_RING_TASK_WINDOW": "64",
        "PTO2_RING_HEAP": "2621440",
        "PTO2_RING_DEP_POOL": "256",
    }

    CALLABLE = {
        "orchestration": {
            "source": f"{PA_KERNELS}/orchestration/paged_attention_orch.cpp",
            "function_name": "build_paged_attention_graph",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{PA_KERNELS}/aic/aic_qk_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": f"{PA_KERNELS}/aic/aic_pv_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": f"{PA_KERNELS}/aiv/aiv_softmax_prepare.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT, D.OUT, D.OUT],
            },
            {
                "func_id": 3,
                "source": f"{PA_KERNELS}/aiv/aiv_online_update.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "ringbuffer_stress",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {
                "batch": 32,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 128,
                "context_len": 4096,
                "max_model_len": 32768,
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
