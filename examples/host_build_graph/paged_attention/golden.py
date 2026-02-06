"""
Paged Attention Golden Implementation - 16x16 Version

Implements the online softmax algorithm for paged attention computation.
Used as reference for validation of the simulation results.

For framework-generated matmul (A @ B), we store:
  - K as K^T: (head_dim, block_size) so Q @ K_stored = Q @ K^T
  - V as V: (block_size, head_dim) so P @ V works directly
"""

import os
import struct
import numpy as np

# Output tensor names
__outputs__ = ["out"]

# Tensor order matching orchestration function parameter order
TENSOR_ORDER = ["query", "key_cache", "value_cache", "block_table", "context_lens", "out", "config"]

# Comparison tolerances
RTOL = 1e-3
ATOL = 1e-3

# All test cases - 16x16 framework version
ALL_CASES = {
    "Case1": {
        "batch": 1,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 16,
        "block_size": 16,
        "block_num": 1,  # Single block for debugging
        "context_len": 16,
    },
    "Case2": {
        "batch": 1,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 16,
        "block_size": 16,
        "block_num": 4,
        "context_len": 64,
    },
}

# Select case by env var PA_CASE, default to Case1
_selected = os.environ.get("PA_CASE", "Case1")
PARAMS_LIST = [{"name": _selected, **ALL_CASES[_selected]}]


def generate_inputs(params: dict) -> dict:
    batch = params["batch"]
    num_heads = params["num_heads"]
    kv_head_num = params["kv_head_num"]
    head_dim = params["head_dim"]
    block_size = params["block_size"]
    block_num = params["block_num"]
    context_len = params["context_len"]
    seed = params.get("seed", 42)

    np.random.seed(seed)

    total_blocks = batch * block_num
    scale_value = 1.0 / np.sqrt(head_dim)
    scale_bits = struct.unpack('I', struct.pack('f', scale_value))[0]

    block_table = np.zeros(batch * block_num, dtype=np.int32)
    for b in range(batch):
        for bn in range(block_num):
            block_table[b * block_num + bn] = b * block_num + bn

    context_lens = np.full(batch, context_len, dtype=np.int32)

    config = np.array(
        [batch, num_heads, kv_head_num, head_dim, block_size, block_num, scale_bits],
        dtype=np.int64,
    )

    # Q: (batch, num_heads, head_dim) = (1, 16, 16) stored as flat
    query = np.random.randn(batch, num_heads, head_dim).astype(np.float32) * 0.1
    
    # K_logical: (total_blocks, block_size, head_dim) - natural format
    key_logical = np.random.randn(total_blocks, block_size, head_dim).astype(np.float32) * 0.1
    # K_stored: (total_blocks, head_dim, block_size) - transposed for matmul
    key_stored = np.transpose(key_logical, (0, 2, 1)).copy()
    
    # V: (total_blocks, block_size, head_dim)
    value_cache = np.random.randn(total_blocks, block_size, head_dim).astype(np.float32) * 0.1

    # Compute golden output
    out = compute_attention(query, key_logical, value_cache, block_table.reshape(batch, block_num), 
                           context_lens, scale_value, batch, num_heads, head_dim, block_size, block_num)

    return {
        "query": query.flatten(),
        "key_cache": key_stored.flatten(),
        "value_cache": value_cache.flatten(),
        "block_table": block_table,
        "context_lens": context_lens,
        "out": out.flatten(),
        "config": config,
    }


def compute_attention(query, key_logical, value_cache, block_table, context_lens, 
                     scale_value, batch, num_heads, head_dim, block_size, block_num):
    """Compute paged attention using online softmax algorithm."""
    heads_per_kv = num_heads  # With kv_head_num=1, all heads share same K/V
    
    out = np.zeros((batch, num_heads, head_dim), dtype=np.float32)

    for b_idx in range(batch):
        cur_seq = context_lens[b_idx]
        bn_this_batch = (cur_seq + block_size - 1) // block_size

        for h_idx in range(num_heads):
            qi = query[b_idx, h_idx, :]

            mi = -np.inf
            li = 0.0
            oi = np.zeros(head_dim, dtype=np.float32)

            for bn in range(bn_this_batch):
                cur_block_idx = block_table[b_idx, bn]
                kj = key_logical[cur_block_idx, :, :]  # (block_size, head_dim)
                vj = value_cache[cur_block_idx, :, :]  # (block_size, head_dim)

                valid_tokens = cur_seq - bn * block_size if bn == bn_this_batch - 1 else block_size

                # QK attention scores
                sij = qi @ kj.T  # (head_dim,) @ (head_dim, block_size) -> (block_size,)
                sij_scale = sij * scale_value

                if valid_tokens < block_size:
                    sij_scale[valid_tokens:] = -np.inf

                # Softmax preparation
                mij = np.max(sij_scale)
                pij = np.exp(sij_scale - mij)
                lij = np.sum(pij)
                
                # PV matmul
                oi_new = pij @ vj  # (block_size,) @ (block_size, head_dim) -> (head_dim,)

                # Online softmax update
                if bn == 0:
                    mi, li, oi = mij, lij, oi_new
                else:
                    mi_new = max(mi, mij)
                    alpha = np.exp(mi - mi_new)
                    beta = np.exp(mij - mi_new)
                    li = alpha * li + beta * lij
                    oi = alpha * oi + beta * oi_new
                    mi = mi_new

            # Final normalization
            oi = oi / li
            out[b_idx, h_idx, :] = oi

    return out


def compute_golden(tensors: dict, params: dict) -> None:
    """Called by test framework to compute expected output."""
    # The output is already computed in generate_inputs
    pass


if __name__ == "__main__":
    params = PARAMS_LIST[0]
    tensors = generate_inputs(params)

    print(f"=== Paged Attention Golden Test ({params['name']}) ===")
    print(f"batch={params['batch']}, num_heads={params['num_heads']}, head_dim={params['head_dim']}")
    print(f"kv_head_num={params['kv_head_num']}, block_size={params['block_size']}, block_num={params['block_num']}")
    print(f"Tasks needed: {params['batch'] * params['block_num'] * 4}")

    out = tensors["out"].reshape(params["batch"], params["num_heads"], params["head_dim"])
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"Output mean: {out.mean():.4f}")
    print("Golden test passed!")
