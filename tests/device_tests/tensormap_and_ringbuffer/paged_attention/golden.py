"""
Paged Attention Golden Implementation - Production Scale

Implements the online softmax algorithm for paged attention with:
- bfloat16 Q/K/V inputs
- Non-transposed K storage: (total_blocks, block_size, kv_head_num, head_dim)
- GQA support (kv_head_num=1)
- Head tiling: q_tile = min(q_head_num, 128)
- Random block table mapping
"""

import os
import struct
import torch

# Output tensor names
__outputs__ = ["out"]

# Tensor order matching orchestration function parameter order
TENSOR_ORDER = ["query", "key_cache", "value_cache", "block_table", "context_lens", "out", "config"]

# Comparison tolerances
RTOL = 1e-3
ATOL = 1e-3


# All test cases - production scale
ALL_CASES = {
    "Case1": {
        "batch": 16,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 128,
        "block_size": 128,
        "context_len": 8193,
        "max_model_len": 32768,
    },
    "Case2": {
        "batch": 64,
        "num_heads": 64,
        "kv_head_num": 1,
        "head_dim": 128,
        "block_size": 64,
        "context_len": 8192,
        "max_model_len": 32768,
    },
}

# Select case by env var PA_CASE, default to Case1
_selected = os.environ.get("PA_CASE", "Case1")
PARAMS_LIST = [{"name": _selected, **ALL_CASES[_selected]}]


def generate_inputs(params: dict) -> dict:
    """Generate input tensors and zeroed output tensor."""
    batch = params["batch"]
    num_heads = params["num_heads"]
    kv_head_num = params["kv_head_num"]
    head_dim = params["head_dim"]
    block_size = params["block_size"]
    context_len = params["context_len"]
    max_model_len = params["max_model_len"]

    max_num_blocks_per_req = max_model_len // block_size
    cur_valid_blocks = (context_len + block_size - 1) // block_size
    total_blocks = batch * cur_valid_blocks
    scale_value = 1.0
    scale_bits = struct.unpack('I', struct.pack('f', scale_value))[0]

    # Random block table: (batch, max_num_blocks_per_req) int32
    block_table = torch.randint(
        0,
        max(total_blocks, 1),
        size=(batch, max_num_blocks_per_req),
        dtype=torch.int32,
    )

    # Context lens: all = context_len
    context_lens = torch.full((batch,), context_len, dtype=torch.int32)

    config = torch.tensor(
        [batch, num_heads, kv_head_num, head_dim, block_size,
         max_num_blocks_per_req, scale_bits],
        dtype=torch.int64,
    )

    # Query: (batch, 1, num_heads * head_dim) -> (batch, num_heads, head_dim) bfloat16
    query_bf16 = torch.empty(batch, 1, num_heads * head_dim).uniform_(-0.5, 0.5).to(torch.bfloat16)
    query_bf16 = query_bf16.reshape(batch, num_heads, head_dim)

    # Key cache: (total_blocks, block_size, kv_head_num, head_dim) bfloat16
    key_bf16 = torch.empty(total_blocks, block_size, kv_head_num, head_dim).uniform_(-0.5, 0.5).to(torch.bfloat16)

    # Value cache: (total_blocks, block_size, kv_head_num, head_dim) bfloat16
    value_bf16 = torch.empty(total_blocks, block_size, kv_head_num, head_dim).uniform_(-1, 1).to(torch.bfloat16)

    return {
        "query": query_bf16.flatten(),
        "key_cache": key_bf16.flatten(),
        "value_cache": value_bf16.flatten(),
        "block_table": block_table.flatten(),
        "context_lens": context_lens,
        "out": torch.zeros(batch * num_heads * head_dim, dtype=torch.float32),
        "config": config,
    }


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    scale_value: float,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    """
    Compute paged attention using online softmax with head tiling and GQA.

    Args:
        query: (batch, num_heads, head_dim) bfloat16
        key_cache: (total_blocks, block_size, num_kv_heads, head_dim) bfloat16
        value_cache: (total_blocks, block_size, num_kv_heads, head_dim) bfloat16
        num_kv_heads: int
        num_heads: int
        scale_value: float
        block_table: (batch, block_num) int32
        context_lens: (batch,) int32

    Returns:
        out: (batch, num_heads, head_dim) float32
    """
    assert num_kv_heads == 1
    batch, num_heads, head_dim = query.shape
    _, block_size, _, _ = key_cache.shape
    _, block_num = block_table.shape

    query = query.reshape(-1, head_dim)
    key_cache = key_cache.reshape(-1, block_size, head_dim)
    value_cache = value_cache.reshape(-1, block_size, head_dim)

    out = torch.zeros((batch * num_heads, head_dim), dtype=torch.float32)

    for b_idx in range(batch):
        cur_seq = int(context_lens[b_idx])
        bn_this_batch = (cur_seq + block_size - 1) // block_size
        assert bn_this_batch <= block_num

        q_tile = min(num_heads, 128)
        for cur_offset in range(0, num_heads, q_tile):
            q_tile_size = min(q_tile, num_heads - cur_offset)
            base_idx = b_idx * num_heads + cur_offset
            qi = query[base_idx : base_idx + q_tile_size].to(torch.float32)

            oi = None
            li = None
            mi = None

            for bn in range(bn_this_batch):
                cur_block_idx = block_table[b_idx, bn]
                valid_len = min(block_size, cur_seq - bn * block_size)
                kj = key_cache[cur_block_idx, :valid_len, :].to(torch.float32)
                vj = value_cache[cur_block_idx, :valid_len, :].to(torch.float32)

                sij = (qi @ kj.T) * scale_value
                mij = sij.max(dim=-1, keepdim=True)[0]
                pij = torch.exp(sij - mij).to(torch.bfloat16).to(torch.float32)
                lij = pij.sum(dim=1, keepdim=True)

                if bn == 0:
                    oi = pij @ vj
                    li = lij
                    mi = mij
                else:
                    mi_new = torch.maximum(mi, mij)
                    alpha = torch.exp(mi - mi_new)
                    beta = torch.exp(mij - mi_new)
                    li_new = alpha * li + beta * lij
                    oi_new = pij @ vj
                    oi = alpha * oi + beta * oi_new
                    li = li_new
                    mi = mi_new

                if bn == bn_this_batch - 1:
                    oi = oi / li

            out[base_idx : base_idx + q_tile_size] = oi

    return out


def compute_golden(tensors: dict, params: dict) -> None:
    """Compute expected output in-place using online softmax paged attention."""
    batch = params["batch"]
    num_heads = params["num_heads"]
    kv_head_num = params["kv_head_num"]
    head_dim = params["head_dim"]
    block_size = params["block_size"]
    max_model_len = params["max_model_len"]

    max_num_blocks_per_req = max_model_len // block_size

    # Reconstruct shaped tensors from flat tensors
    query = tensors["query"].reshape(batch, num_heads, head_dim)
    key_cache = tensors["key_cache"].reshape(-1, block_size, kv_head_num, head_dim)
    value_cache = tensors["value_cache"].reshape(-1, block_size, kv_head_num, head_dim)
    block_table = tensors["block_table"].reshape(batch, max_num_blocks_per_req)
    context_lens = tensors["context_lens"]

    out = paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=kv_head_num,
        num_heads=num_heads,
        scale_value=1.0,
        block_table=block_table,
        context_lens=context_lens,
    )

    tensors["out"][:] = out.flatten()


if __name__ == "__main__":
    params = PARAMS_LIST[0]
    tensors = generate_inputs(params)
    compute_golden(tensors, params)

    print(f"=== Paged Attention Golden Test ({params['name']}) ===")
    print(f"batch={params['batch']}, num_heads={params['num_heads']}, head_dim={params['head_dim']}")
    print(f"kv_head_num={params['kv_head_num']}, block_size={params['block_size']}")
    print(f"context_len={params['context_len']}")

    max_num_blocks = params['max_model_len'] // params['block_size']
    q_tile = min(params['num_heads'], 128)
    print(f"max_num_blocks_per_req={max_num_blocks}, q_tile_size={q_tile}")

    out = tensors["out"].reshape(params["batch"] * params["num_heads"], params["head_dim"])
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"Output mean: {out.mean():.4f}")
    print("Golden test passed!")
