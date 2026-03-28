# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paged Attention Golden - tensormap_and_ringbuffer example (small scale, float16)."""

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
        "batch": 1,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 16,
        "block_size": 16,
        "context_len": 33,
        "max_model_len": 256,
        "dtype": "float16",
    },
    "Case2": {
        "batch": 1,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 16,
        "block_size": 16,
        "context_len": 128,
        "max_model_len": 256,
        "dtype": "float16",
    },
    "CaseVarSeq2": {
        "batch": 2,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 16,
        "block_size": 16,
        "context_len": 33,
        "context_lens_list": [33, 17],
        "max_model_len": 256,
        "dtype": "float16",
    },
    "CaseVarSeq4": {
        "batch": 4,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 16,
        "block_size": 16,
        "context_len": 128,
        "context_lens_list": [33, 64, 128, 15],
        "max_model_len": 256,
        "dtype": "float16",
    },
}

DEFAULT_CASE = "Case1"


def generate_inputs(params: dict) -> list:
    return _generate_inputs(params)


if __name__ == "__main__":
    run_golden_test(ALL_CASES, DEFAULT_CASE, generate_inputs)
