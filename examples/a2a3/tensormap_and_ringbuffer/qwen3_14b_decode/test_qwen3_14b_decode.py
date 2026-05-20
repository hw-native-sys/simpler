#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B single-layer decode — tensormap_and_ringbuffer SceneTestCase.

A single fused decode step (21 kernels: 8 AIC + 13 AIV) covering
RMSNorm → QKV → per-head Q/K RMS → RoPE → paged KV-cache write → paged
attention (online softmax) → output projection + residual → post-RMSNorm
→ SwiGLU FFN → down-proj + residual, against the production Qwen3-14B
hidden/intermediate/head shapes (HIDDEN=5120, INTERMEDIATE=17408,
NUM_HEADS=40 / NUM_KV_HEADS=8, HEAD_DIM=128, BLOCK_SIZE=256).
"""

from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, scene_test
from simpler_setup.goldens.qwen3_14b_decode import (
    compute_golden as _decode_golden,
)
from simpler_setup.goldens.qwen3_14b_decode import (
    generate_inputs as _decode_generate_inputs,
)


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestQwen314BDecode(SceneTestCase):
    """Single-layer Qwen3-14B decode against a torch reference."""

    # Bf16 deep-transformer drift over 21 kernels in series — paged attention
    # plus FFN accumulate, so values O(10) settle in the ~1e-1 absolute range.
    RTOL = 5e-2
    ATOL = 1e-1

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/qwen3_decode.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [
                D.IN,  # 0  hidden_states
                D.IN,  # 1  input_rms_weight
                D.IN,  # 2  wq
                D.IN,  # 3  wk
                D.IN,  # 4  wv
                D.IN,  # 5  q_norm_weight
                D.IN,  # 6  k_norm_weight
                D.IN,  # 7  seq_lens
                D.IN,  # 8  block_table
                D.IN,  # 9  slot_mapping
                D.IN,  # 10 rope_cos
                D.IN,  # 11 rope_sin
                D.INOUT,  # 12 k_cache
                D.INOUT,  # 13 v_cache
                D.IN,  # 14 wo
                D.IN,  # 15 post_rms_weight
                D.IN,  # 16 w_gate
                D.IN,  # 17 w_up
                D.IN,  # 18 w_down
                D.OUT,  # 19 out
            ],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "copy_hidden",
                "source": "kernels/aiv/copy_hidden.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN, D.SCALAR, D.SCALAR],
            },
            {
                "func_id": 1,
                "name": "rmsnorm",
                "source": "kernels/aiv/rmsnorm.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT, D.IN, D.SCALAR, D.SCALAR],
            },
            {
                "func_id": 2,
                "name": "q_proj",
                "source": "kernels/aic/q_proj.cpp",
                "core_type": "aic",
                "signature": [D.OUT, D.IN, D.IN, D.SCALAR, D.SCALAR],
            },
            {
                "func_id": 3,
                "name": "kv_proj",
                "source": "kernels/aic/kv_proj.cpp",
                "core_type": "aic",
                "signature": [D.OUT, D.OUT, D.IN, D.IN, D.IN, D.SCALAR, D.SCALAR],
            },
            {
                "func_id": 4,
                "name": "qk_norm",
                "source": "kernels/aiv/qk_norm.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN, D.IN, D.OUT, D.IN, D.IN, D.SCALAR],
            },
            {
                "func_id": 5,
                "name": "q_pad",
                "source": "kernels/aiv/q_pad.cpp",
                "core_type": "aiv",
                "signature": [D.OUT],
            },
            {
                "func_id": 6,
                "name": "rope_kv_cache",
                "source": "kernels/aiv/rope_kv_cache.cpp",
                "core_type": "aiv",
                "signature": [
                    D.INOUT,
                    D.OUT,
                    D.OUT,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.SCALAR,
                    D.SCALAR,
                    D.SCALAR,
                    D.SCALAR,
                ],
            },
            {
                "func_id": 7,
                "name": "qk_matmul",
                "source": "kernels/aic/qk_matmul.cpp",
                "core_type": "aic",
                "signature": [D.OUT, D.IN, D.IN, D.IN, D.SCALAR, D.SCALAR, D.SCALAR, D.SCALAR],
            },
            {
                "func_id": 8,
                "name": "softmax",
                "source": "kernels/aiv/softmax.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.OUT, D.OUT, D.IN, D.SCALAR, D.SCALAR, D.SCALAR],
            },
            {
                "func_id": 9,
                "name": "sv_matmul",
                "source": "kernels/aic/sv_matmul.cpp",
                "core_type": "aic",
                "signature": [D.OUT, D.IN, D.IN, D.IN, D.SCALAR, D.SCALAR, D.SCALAR],
            },
            {
                "func_id": 10,
                "name": "online_softmax",
                "source": "kernels/aiv/online_softmax.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN, D.IN, D.IN, D.SCALAR],
            },
            {
                "func_id": 11,
                "name": "attention_writeback",
                "source": "kernels/aiv/attention_writeback.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN],
            },
            {
                "func_id": 12,
                "name": "out_proj",
                "source": "kernels/aic/out_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT, D.SCALAR, D.SCALAR],
            },
            {
                "func_id": 13,
                "name": "out_proj_residual",
                "source": "kernels/aiv/out_proj_residual.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT, D.SCALAR, D.SCALAR, D.SCALAR],
            },
            {
                "func_id": 14,
                "name": "post_rmsnorm",
                "source": "kernels/aiv/post_rmsnorm.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT, D.IN],
            },
            {
                "func_id": 15,
                "name": "gate_proj",
                "source": "kernels/aic/gate_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT, D.SCALAR],
            },
            {
                "func_id": 16,
                "name": "up_proj",
                "source": "kernels/aic/up_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT, D.SCALAR],
            },
            {
                "func_id": 17,
                "name": "silu",
                "source": "kernels/aiv/silu.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT, D.SCALAR],
            },
            {
                "func_id": 18,
                "name": "down_proj",
                "source": "kernels/aic/down_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT, D.SCALAR],
            },
            {
                "func_id": 19,
                "name": "down_proj_residual",
                "source": "kernels/aiv/down_proj_residual.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT, D.SCALAR, D.SCALAR],
            },
            {
                "func_id": 20,
                "name": "copy_out",
                "source": "kernels/aiv/copy_out.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN, D.SCALAR, D.SCALAR],
            },
        ],
    }

    CASES = [
        {
            "name": "SmallSingle",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"user_batch": 1, "seq_len": 8},
        },
    ]

    def generate_args(self, params):
        return _decode_generate_inputs(params["user_batch"], params["seq_len"])

    def compute_golden(self, args, params):
        _decode_golden(args, params["user_batch"], params["seq_len"])


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
