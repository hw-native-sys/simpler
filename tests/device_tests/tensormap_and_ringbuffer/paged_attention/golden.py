"""Paged Attention Golden - tensormap_and_ringbuffer test (production scale, bfloat16)."""

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
        "batch": 64,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 128,
        "block_size": 128,
        "context_len": 8193,
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
}

DEFAULT_CASE = "Case1"


def generate_inputs(params: dict) -> list:
    return _generate_inputs(params, return_all_sizes=False)


if __name__ == "__main__":
    run_golden_test(ALL_CASES, DEFAULT_CASE, generate_inputs)
