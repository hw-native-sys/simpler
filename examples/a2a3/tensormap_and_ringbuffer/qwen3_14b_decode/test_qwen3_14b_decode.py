#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B 2-layer decode (load-balanced fused attention) — SceneTestCase.

Self-contained port of pypto-lib ``models/qwen3/14b/decode_layer.py`` entry
``decode_fwd_layers`` with ``_CHUNK_NLAYERS == 2``: a fused chunk of two
consecutive Qwen3-14B decode layers (hidden -> hidden, no LM head), with FP32
inter-layer residual carry. The 36 C++ sources under ``kernels/`` (orchestration
+ 35 incores: 8 AIC + 27 AIV) and ``simpler_setup/goldens/qwen3_14b_decode.py``
are the pypto codegen for that entry, harvested verbatim so simpler developers
run it directly — no descent through pypto-lib / the JIT.

Parameter regime matches ``stress_profile.py`` (vLLM serving stress): BATCH=16,
MAX_SEQ=5500 (= max_model_len), fixed decode seq_len=3500 (the ~3500-token
prompt). Weights + paged KV pool are stacked x2 (one slice per layer; the two
layers reuse layer-0 weights, per the lib const-layer-0 stacked-fwd reference).
"""

from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, scene_test
from simpler_setup.goldens.qwen3_14b_decode import (
    compute_golden as _decode_golden,
)
from simpler_setup.goldens.qwen3_14b_decode import (
    generate_inputs as _decode_generate_inputs,
)


# Validates the fused 2-layer decode against a torch reference.
@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestQwen314BDecode(SceneTestCase):
    """Two-layer Qwen3-14B decode (decode_fwd_layers N=2) against a torch reference."""

    RTOL = 5e-2
    ATOL = 1e-1

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/decode_fwd_layers.cpp",
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
                D.IN,  # 15 w_gate
                D.IN,  # 16 w_up
                D.IN,  # 17 w_down
                D.IN,  # 18 post_rms_weight
                D.OUT,  # 19 out
            ],
        },
        # 35 incores (func_id 0..34), transcribed from the pypto codegen
        # kernel_config.py for decode_fwd_layers (N=2). fa_fused is the
        # codegen-split mixed kernel (fa_fused_aic + fa_fused_aiv).
        "incores": [
            {
                "func_id": 0,
                "name": "copy_hidden",
                "source": "kernels/aiv/copy_hidden.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN],
            },
            {
                "func_id": 1,
                "name": "x_gamma",
                "source": "kernels/aiv/x_gamma.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 2,
                "name": "rms_recip",
                "source": "kernels/aiv/rms_recip.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT],
            },
            {
                "func_id": 3,
                "name": "q_seed",
                "source": "kernels/aiv/q_seed.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 4,
                "name": "q_proj",
                "source": "kernels/aic/q_proj.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 5,
                "name": "k_seed",
                "source": "kernels/aiv/k_seed.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 6,
                "name": "k_proj",
                "source": "kernels/aic/k_proj.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 7,
                "name": "v_seed",
                "source": "kernels/aiv/v_seed.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 8,
                "name": "v_proj",
                "source": "kernels/aic/v_proj.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 9,
                "name": "fa_work_build",
                "source": "kernels/aiv/fa_work_build.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT, D.INOUT],
            },
            {
                "func_id": 10,
                "name": "qk_gamma",
                "source": "kernels/aiv/qk_gamma.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 11,
                "name": "qk_recip",
                "source": "kernels/aiv/qk_recip.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 12,
                "name": "rope_qkv",
                "source": "kernels/aiv/rope_qkv.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.IN, D.IN, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 13,
                "name": "down_seed",
                "source": "kernels/aiv/down_seed.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 14,
                "name": "gate_seed",
                "source": "kernels/aiv/gate_seed.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 15,
                "name": "up_seed",
                "source": "kernels/aiv/up_seed.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 16,
                "name": "fa_fused_aic",
                "source": "kernels/aic/fa_fused_aic.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.INOUT, D.INOUT, D.INOUT, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 17,
                "name": "fa_fused_aiv",
                "source": "kernels/aiv/fa_fused_aiv.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT, D.INOUT, D.INOUT, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 18,
                "name": "online_softmax",
                "source": "kernels/aiv/online_softmax.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 19,
                "name": "out_seed",
                "source": "kernels/aiv/out_seed.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 20,
                "name": "attn_fence",
                "source": "kernels/aiv/attn_fence.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT],
            },
            {
                "func_id": 21,
                "name": "out_proj",
                "source": "kernels/aic/out_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 22,
                "name": "residual_rms_cast",
                "source": "kernels/aiv/residual_rms_cast.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 23,
                "name": "residual_rms_cast_0",
                "source": "kernels/aiv/residual_rms_cast_0.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 24,
                "name": "residual_rms_cast_1",
                "source": "kernels/aiv/residual_rms_cast_1.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 25,
                "name": "residual_rms_cast_2",
                "source": "kernels/aiv/residual_rms_cast_2.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 26,
                "name": "residual_rms_cast_3",
                "source": "kernels/aiv/residual_rms_cast_3.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 27,
                "name": "post_rms_reduce",
                "source": "kernels/aiv/post_rms_reduce.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 28,
                "name": "gate_proj",
                "source": "kernels/aic/gate_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 29,
                "name": "up_proj",
                "source": "kernels/aic/up_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 30,
                "name": "silu",
                "source": "kernels/aiv/silu.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 31,
                "name": "down_proj",
                "source": "kernels/aic/down_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 32,
                "name": "down_cast_residual",
                "source": "kernels/aiv/down_cast_residual.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 33,
                "name": "out_consolidate",
                "source": "kernels/aiv/out_consolidate.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN],
            },
            {
                "func_id": 34,
                "name": "copy_out",
                "source": "kernels/aiv/copy_out.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "StressBatch16Seq3500",
            "platforms": ["a2a3"],
            # block_dim=0 -> auto (stream max capacity), matching the lib default.
            "config": {"aicpu_thread_num": 4, "block_dim": 0},
            "params": {"seed": 1234, "seq_len": 3500},
        },
    ]

    def generate_args(self, params):
        return _decode_generate_inputs(params.get("seed", 1234), params.get("seq_len", 3500))

    def compute_golden(self, args, params):
        _decode_golden(args)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
