#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B 40-layer decode under TensorMap-derived dependencies — SceneTestCase.

The AUTO-scope twin of ``../qwen3_14b_decode/``: same network, same fixture,
same golden, but ``SIMPLER_SCOPE(ScopeMode::AUTO)`` instead of MANUAL, so every
WAIT edge is derived from tensor overlap rather than declared by hand. Its
``INOUT`` / ``INPUT`` views are ``.slice()``-narrowed to the column band each
task indexes, because a whole-buffer declaration under AUTO serializes bands
that do not overlap.

Four split-K accumulators write private partials that a ``partials_reduce_*``
task sums, rather than each atomically adding into a shared band: the atomic form
is commutative but ``INOUT`` cannot say so, so TensorMap had to serialize the
writers.

Only the 13 incores that had to change live here; the ``CALLABLE`` reaches the
other 27 as ``../qwen3_14b_decode/kernels/...`` so the harvested codegen has one
home. See README.md for the measured A/B and what the residual critical path is
made of; see the sibling's README for model, provenance and regen.

Parameter regime matches ``stress_profile.py`` (vLLM serving stress): BATCH=16,
MAX_SEQ=5500 (= max_model_len), fixed decode seq_len=3500. Weights and the paged
KV pool are stacked x40 (one slice per layer); every layer reuses layer-0
weights, per the lib const-layer-0 stacked-fwd reference, while each layer reads
and writes its own KV pool.
"""

from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, scene_test
from simpler_setup.goldens.qwen3_14b_decode import (
    compute_golden as _decode_golden,
)
from simpler_setup.goldens.qwen3_14b_decode import (
    generate_inputs as _decode_generate_inputs,
)

# CANN devkit headers for the attention extern, which builds on AscendC and the
# vendored FusedInferAttentionScore under kernels/paged_attention_cce/vendor/.
# `vendor/.../attn_infra/base_defs.hpp` selects its AscendC entry header under
# `#if ASC_DEVKIT_MAJOR >= 9`, which ccec predefines from the installed devkit,
# so a CANN 9 box must be able to resolve `basic_api/kernel_basic_intf.h` from
# one of these.
#
# `$ASCEND_HOME_PATH` keeps the paths machine-independent and is expanded at
# compile time, not import time — this file is collected on sim and macOS runners
# that have no CANN at all. The devkit's arch subdirectory is named differently
# across installs, so both layouts are listed; missing ones are dropped, and the
# scene-test resolver raises if *every* entry is missing.
_CANN_SUBDIRS = (
    "include",
    "asc",
    "asc/impl/adv_api",
    "asc/impl/basic_api",
    "asc/impl/basic_api/reg_compute",
    "asc/impl/c_api",
    "asc/impl/simt_api",
    "asc/impl/utils",
    "asc/include",
    "asc/include/adv_api",
    "asc/include/aicpu_api",
    "asc/include/basic_api",
    "asc/include/basic_api/reg_compute",
    "asc/include/c_api",
    "asc/include/interface",
    "asc/include/simt_api",
    "asc/include/utils",
    "tikcpp/tikcfw",
    "tikcpp/tikcfw/impl",
    "tikcpp/tikcfw/interface",
)

_CANN_INCLUDE_DIRS = [f"$ASCEND_HOME_PATH/{prefix}{sub}" for prefix in ("aarch64-linux/", "") for sub in _CANN_SUBDIRS]


# Validates the full 40-layer fused decode against a torch reference.
@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestQwen314BDecodeAuto(SceneTestCase):
    """Qwen3-14B decode, all 40 layers in one dispatch, against a torch reference."""

    RTOL = 5e-2
    ATOL = 1e-1

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/decode_fwd_layers.cpp",
            "function_name": "aicpu_orchestration_entry",
            # decode_fwd_layers takes k_cache / v_cache as plain inputs, but the
            # attention extern writes the current token's KV into them. Declaring
            # them INOUT here has simpler copy the pools back, so the golden can
            # check all 40 layers' KV writes and not just the hidden output.
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
        # func_id 0..36 are transcribed from the pypto codegen kernel_config.py
        # for decode_fwd_layers (N=40); 0/11/12 are the CANN attention externs, and
        # 11 and 12 are the same source dispatched as the AIC and AIV halves of one
        # mixed task. func_id 37..39 are this example's own split-K partial reducers
        # and have no counterpart in the codegen. 13/31/32/34 keep their codegen
        # func_id but store a private partial, so their last argument is OUT.
        "incores": [
            {
                "func_id": 0,
                "name": "paged_attention_tiling_cce",
                "source": "../qwen3_14b_decode/kernels/vendor/paged_attention_cce/tiling/entry.cpp",
                "core_type": "aiv",
                "extra_include_dirs": _CANN_INCLUDE_DIRS,
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "copy_hidden",
                "source": "../qwen3_14b_decode/kernels/aiv/copy_hidden.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN],
            },
            {
                "func_id": 2,
                "name": "x_gamma0",
                "source": "../qwen3_14b_decode/kernels/aiv/x_gamma0.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN, D.IN],
            },
            {
                "func_id": 3,
                "name": "attn_out_seed",
                "source": "../qwen3_14b_decode/kernels/aiv/attn_out_seed.cpp",
                "core_type": "aiv",
                "signature": [D.IN],
            },
            {
                "func_id": 4,
                "name": "rms_recip",
                "source": "../qwen3_14b_decode/kernels/aiv/rms_recip.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT],
            },
            {
                "func_id": 5,
                "name": "q_seed",
                "source": "../qwen3_14b_decode/kernels/aiv/q_seed.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 6,
                "name": "q_proj",
                "source": "../qwen3_14b_decode/kernels/aic/q_proj.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 7,
                "name": "kv_seed",
                "source": "../qwen3_14b_decode/kernels/aiv/kv_seed.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT],
            },
            {
                "func_id": 8,
                "name": "mlp_out_seed",
                "source": "../qwen3_14b_decode/kernels/aiv/mlp_out_seed.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.INOUT, D.INOUT],
            },
            {
                "func_id": 9,
                "name": "k_proj",
                "source": "../qwen3_14b_decode/kernels/aic/k_proj.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 10,
                "name": "v_proj",
                "source": "../qwen3_14b_decode/kernels/aic/v_proj.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 11,
                "name": "paged_attention_rope_cce_aic",
                "source": "../qwen3_14b_decode/kernels/vendor/paged_attention_cce/attention_rope/entry.cpp",
                "core_type": "aic",
                "extra_include_dirs": _CANN_INCLUDE_DIRS,
                "signature": [
                    D.INOUT,
                    D.INOUT,
                    D.INOUT,
                    D.INOUT,
                    D.IN,
                    D.INOUT,
                    D.INOUT,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                ],
            },
            {
                "func_id": 12,
                "name": "paged_attention_rope_cce_aiv",
                "source": "../qwen3_14b_decode/kernels/vendor/paged_attention_cce/attention_rope/entry.cpp",
                "core_type": "aiv",
                "extra_include_dirs": _CANN_INCLUDE_DIRS,
                "signature": [
                    D.INOUT,
                    D.INOUT,
                    D.INOUT,
                    D.INOUT,
                    D.IN,
                    D.INOUT,
                    D.INOUT,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                ],
            },
            {
                "func_id": 13,
                "name": "out_proj",
                "source": "kernels/aic/out_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 14,
                "name": "out_proj_0",
                "source": "../qwen3_14b_decode/kernels/aic/out_proj_0.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 15,
                "name": "residual_rms_cast",
                "source": "kernels/aiv/residual_rms_cast.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 16,
                "name": "residual_rms_cast_0",
                "source": "kernels/aiv/residual_rms_cast_0.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 17,
                "name": "residual_rms_cast_1",
                "source": "kernels/aiv/residual_rms_cast_1.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 18,
                "name": "residual_rms_cast_2",
                "source": "kernels/aiv/residual_rms_cast_2.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 19,
                "name": "residual_rms_cast_3",
                "source": "kernels/aiv/residual_rms_cast_3.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 20,
                "name": "post_rms_reduce",
                "source": "../qwen3_14b_decode/kernels/aiv/post_rms_reduce.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 21,
                "name": "gate_proj",
                "source": "../qwen3_14b_decode/kernels/aic/gate_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 22,
                "name": "up_proj",
                "source": "../qwen3_14b_decode/kernels/aic/up_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 23,
                "name": "gate_proj_0",
                "source": "../qwen3_14b_decode/kernels/aic/gate_proj_0.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 24,
                "name": "up_proj_0",
                "source": "../qwen3_14b_decode/kernels/aic/up_proj_0.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 25,
                "name": "gate_proj_1",
                "source": "../qwen3_14b_decode/kernels/aic/gate_proj_1.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 26,
                "name": "up_proj_1",
                "source": "../qwen3_14b_decode/kernels/aic/up_proj_1.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 27,
                "name": "gate_proj_2",
                "source": "../qwen3_14b_decode/kernels/aic/gate_proj_2.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 28,
                "name": "up_proj_2",
                "source": "../qwen3_14b_decode/kernels/aic/up_proj_2.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 29,
                "name": "gate_proj_3",
                "source": "../qwen3_14b_decode/kernels/aic/gate_proj_3.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 30,
                "name": "up_proj_3",
                "source": "../qwen3_14b_decode/kernels/aic/up_proj_3.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 31,
                "name": "gate_proj_4",
                "source": "kernels/aic/gate_proj_4.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 32,
                "name": "up_proj_4",
                "source": "kernels/aic/up_proj_4.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 33,
                "name": "silu",
                "source": "kernels/aiv/silu.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 34,
                "name": "down_proj",
                "source": "kernels/aic/down_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 37,
                "name": "partials_reduce_hidden",
                "source": "kernels/aiv/partials_reduce_hidden.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 38,
                "name": "partials_reduce_hidden_add",
                "source": "kernels/aiv/partials_reduce_hidden_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT],
            },
            {
                "func_id": 39,
                "name": "partials_reduce_inter",
                "source": "kernels/aiv/partials_reduce_inter.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 35,
                "name": "dcr_xgamma",
                "source": "../qwen3_14b_decode/kernels/aiv/dcr_xgamma.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT, D.IN, D.INOUT],
            },
            {
                "func_id": 36,
                "name": "copy_out",
                "source": "../qwen3_14b_decode/kernels/aiv/copy_out.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "AutoDepBatch16Seq3500",
            "platforms": ["a2a3"],
            "manual": True,
            # A run takes the whole device, matching the lib default.
            "params": {"seed": 1234, "seq_len": 3500},
        },
    ]

    def generate_args(self, params):
        return _decode_generate_inputs(params.get("seed", 1234), params.get("seq_len", 3500))

    def compute_golden(self, args, params):
        _decode_golden(args)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
