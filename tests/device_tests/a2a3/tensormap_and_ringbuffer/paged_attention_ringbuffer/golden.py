"""
Paged Attention Ring Buffer Stress Test Golden

Tests paged attention with small ring buffer sizes (TW=1024, HP=1MB, DP=1024)
to guard the ring buffer rotation/reclamation logic.
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
        "batch": 32,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 128,
        "block_size": 128,
        "context_len": 4096,
        "max_model_len": 32768,
        "dtype": "bfloat16",
    },
}

DEFAULT_CASE = "Case1"


def generate_inputs(params: dict) -> list:
    return _generate_inputs(params)


if __name__ == "__main__":
    run_golden_test(ALL_CASES, DEFAULT_CASE, generate_inputs)
