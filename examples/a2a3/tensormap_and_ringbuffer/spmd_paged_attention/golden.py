# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SPMD Paged Attention Golden - tensormap_and_ringbuffer example (small scale, bfloat16).

Uses SPMD parallelism: each block handles one (batch, q_tile) position.
Kernels use get_block_idx() to determine their work slice.
"""

from paged_attention_golden import (
    compute_golden,  # noqa: F401
    run_golden_test,
)
from paged_attention_golden import generate_inputs as _generate_inputs

__outputs__ = ["out"]

RTOL = 1e-2
ATOL = 1e-2

ALL_CASES = {
    "Case1": {
        "batch": 256,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 128,
        "block_size": 128,
        "context_len": 256,
        "max_model_len": 32768,
        "dtype": "bfloat16",
    },
    "Case2": {
        "batch": 64,
        "num_heads": 64,
        "kv_head_num": 1,
        "head_dim": 128,
        "block_size": 64,
        "context_len": 8192,
        "max_model_len": 32768,
        "dtype": "bfloat16",
    },
    "Case3": {
        "batch": 64,
        "num_heads": 64,
        "kv_head_num": 1,
        "head_dim": 256,
        "block_size": 64,
        "context_len": 8192,
        "max_model_len": 32768,
        "dtype": "bfloat16",
    },
}

DEFAULT_CASE = "Case1"


def generate_inputs(params: dict) -> list:
    return _generate_inputs(params)


if __name__ == "__main__":
    run_golden_test(ALL_CASES, DEFAULT_CASE, generate_inputs, label="SPMD Paged Attention")
