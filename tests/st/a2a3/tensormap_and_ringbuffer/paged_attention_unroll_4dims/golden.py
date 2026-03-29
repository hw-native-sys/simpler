"""Paged Attention Unroll Golden - tensormap_and_ringbuffer test (production scale, bfloat16).

Input shapes use 4D format: (batch, seq_len, num_heads, head_dim) for query and out.
"""

from paged_attention_golden import (
    generate_inputs as _generate_inputs,
    compute_golden as _compute_golden,
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
        "context_len": 8192,
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
    result = _generate_inputs(params)
    batch = params["batch"]
    num_heads = params["num_heads"]
    head_dim = params["head_dim"]
    reshaped = []
    for name, val in result:
        if name in ("query", "out"):
            val = val.reshape(batch, 1, num_heads, head_dim)
        reshaped.append((name, val))
    return reshaped


def compute_golden(tensors: dict, params: dict) -> None:
    batch = params["batch"]
    num_heads = params["num_heads"]
    head_dim = params["head_dim"]
    out_4d = tensors["out"]
    tensors["out"] = out_4d.reshape(batch, num_heads, head_dim)
    _compute_golden(tensors, params)
    tensors["out"] = out_4d


if __name__ == "__main__":
    run_golden_test(ALL_CASES, DEFAULT_CASE, generate_inputs, label="Paged Attention Unroll")
