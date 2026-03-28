# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paged Attention Golden - host_build_graph example (small scale, float16).

Args layout: [query, key_cache, value_cache, block_table, context_lens, out, scale]
  - Tensors retain original multi-dimensional shapes (ContinuousTensor metadata carries shape/dtype)
  - scale is a scalar float parameter
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
        "batch": 1,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 16,
        "block_size": 16,
        "context_len": 16,
        "max_model_len": 256,
        "dtype": "float16",
    },
    "Case2": {
        "batch": 1,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 16,
        "block_size": 16,
        "context_len": 64,
        "max_model_len": 256,
        "dtype": "float16",
    },
}

DEFAULT_CASE = "Case1"


def generate_inputs(params: dict) -> list:
    return _generate_inputs(params)


if __name__ == "__main__":
    run_golden_test(ALL_CASES, DEFAULT_CASE, generate_inputs)
