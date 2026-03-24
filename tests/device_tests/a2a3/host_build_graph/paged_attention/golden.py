"""Paged Attention Golden - host_build_graph test (production scale, bfloat16).

Args layout: [query, key_cache, value_cache, block_table, context_lens, out, scale]
  - Tensors retain original multi-dimensional shapes (OrchArg metadata carries shape/dtype)
  - scale is a scalar float parameter
"""

from paged_attention_golden import (
    generate_inputs as _generate_inputs,
    compute_golden,
    run_golden_test,
)

__outputs__ = ["out"]

RTOL = 1e-3
ATOL = 1e-3

ALL_CASES = {
    "Case1": {
        "batch": 256,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 128,
        "block_size": 128,
        "context_len": 8100,
        "max_model_len": 32768,
        "dtype": "bfloat16",
    },
    "Case2": {
        "batch": 64,
        "num_heads": 64,
        "kv_head_num": 1,
        "head_dim": 128,
        "block_size": 64,
        "context_len": 8150,
        "max_model_len": 32768,
        "dtype": "bfloat16",
    },
}

DEFAULT_CASE = "Case1"


def generate_inputs(params: dict) -> list:
    return _generate_inputs(params)


if __name__ == "__main__":
    run_golden_test(ALL_CASES, DEFAULT_CASE, generate_inputs)
