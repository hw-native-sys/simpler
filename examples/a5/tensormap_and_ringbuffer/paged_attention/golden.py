"""Paged Attention Golden - tensormap_and_ringbuffer example (small scale, float16)."""

from paged_attention_golden import (
    generate_inputs as _generate_inputs,
    compute_golden,
    run_golden_test,
)

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
    return _generate_inputs(params, return_all_sizes=False)


if __name__ == "__main__":
    run_golden_test(ALL_CASES, DEFAULT_CASE, generate_inputs)
