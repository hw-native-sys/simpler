"""
Batch Paged Attention Golden Implementation - Production Scale

Implements the online softmax algorithm for batched paged attention with:
- bfloat16 Q/K/V inputs
- Non-transposed K storage: (total_blocks, block_size, kv_head_num, head_dim)
- GQA support (kv_head_num=1)
- Head tiling: q_tile = min(q_head_num, 128)
- Variable sequence lengths per batch (CaseVarSeq)

Args layout: [ptr_query, ..., ptr_config, size_query, size_key_cache, size_value_cache]
"""

import ctypes
import struct
import torch

__outputs__ = ["out"]

RTOL = 1e-3
ATOL = 1e-3


# All test cases - production scale
ALL_CASES = {
    "Case1": {
        "batch": 64,
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
    # Variable sequence length cases
    "CaseVarSeq": {
        "batch": 64,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 128,
        "block_size": 128,
        "context_len": 8193,
        "context_lens_list": [8193, 4096, 1024, 256, 8000, 512, 2048, 7777],
        "max_model_len": 32768,
    },
}

DEFAULT_CASE = "Case1"


def generate_inputs(params: dict) -> list:
    """Generate input tensors and zeroed output tensor."""
    batch = params["batch"]
    num_heads = params["num_heads"]
    kv_head_num = params["kv_head_num"]
    head_dim = params["head_dim"]
    block_size = params["block_size"]
    context_len = params["context_len"]
    max_model_len = params["max_model_len"]
    context_lens_list = params.get("context_lens_list")

    assert context_len >= 1, "context_len must be >= 1 to avoid division by zero in attention"

    max_num_blocks_per_req = max_model_len // block_size
    scale_value = 1.0
    scale_bits = struct.unpack('I', struct.pack('f', scale_value))[0]

    # Build per-batch context_lens tensor
    if context_lens_list is not None:
        seq_vals = context_lens_list
        if len(seq_vals) < batch:
            seq_vals = (seq_vals * ((batch + len(seq_vals) - 1) // len(seq_vals)))[:batch]
        elif len(seq_vals) > batch:
            seq_vals = seq_vals[:batch]
        context_lens = torch.tensor(seq_vals, dtype=torch.int32)
    else:
        context_lens = torch.full((batch,), context_len, dtype=torch.int32)

    max_ctx = int(context_lens.max().item())
    cur_valid_blocks = (max_ctx + block_size - 1) // block_size
    total_blocks = batch * cur_valid_blocks

    # Random block table: (batch, max_num_blocks_per_req) int32
    block_table = torch.randint(
        0,
        max(total_blocks, 1),
        size=(batch, max_num_blocks_per_req),
        dtype=torch.int32,
    )

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

    query = query_bf16.flatten()
    key_cache = key_bf16.flatten()
    value_cache = value_bf16.flatten()
    block_table_flat = block_table.flatten()
    out = torch.zeros(batch * num_heads * head_dim, dtype=torch.float32)

    return [
        ("query", query),
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("block_table", block_table_flat),
        ("context_lens", context_lens),
        ("out", out),
        ("config", config),
        ("size_query", ctypes.c_int64(query.nbytes)),
        ("size_key_cache", ctypes.c_int64(key_cache.nbytes)),
        ("size_value_cache", ctypes.c_int64(value_cache.nbytes)),
    ]


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

    Vectorized across the batch dimension for performance.
    Supports different context_lens per batch via masking.

    Args:
        query: (batch, num_heads, head_dim) bfloat16
        key_cache: (total_blocks, block_size, num_kv_heads, head_dim) bfloat16
        value_cache: (total_blocks, block_size, num_kv_heads, head_dim) bfloat16
        num_kv_heads: int
        num_heads: int
        scale_value: float
        block_table: (batch, block_num) int32 (non-negative)
        context_lens: (batch,) int32

    Returns:
        out: (batch * num_heads, head_dim) float32
    """
    assert num_kv_heads == 1
    batch, num_heads_dim, head_dim = query.shape
    _, block_size, _, _ = key_cache.shape

    # Reshape for batched computation
    key_cache_flat = key_cache.reshape(-1, block_size, head_dim)
    value_cache_flat = value_cache.reshape(-1, block_size, head_dim)

    out = torch.zeros((batch, num_heads_dim, head_dim), dtype=torch.float32)

    q_tile = min(num_heads_dim, 128)

    # Max blocks across all batches (each batch may have different context_len)
    max_bn = int(((context_lens.max().item()) + block_size - 1) // block_size)

    for q_offset in range(0, num_heads_dim, q_tile):
        q_tile_size = min(q_tile, num_heads_dim - q_offset)
        # qi: (batch, q_tile_size, head_dim)
        qi = query[:, q_offset:q_offset + q_tile_size, :].to(torch.float32)

        oi = None  # (batch, q_tile_size, head_dim)
        li = None  # (batch, q_tile_size, 1)
        mi = None  # (batch, q_tile_size, 1)

        for bn in range(max_bn):
            # valid_len per batch for this block position
            valid_lens = torch.clamp(context_lens - bn * block_size, min=0, max=block_size)
            active_mask = valid_lens > 0  # (batch,)

            if not active_mask.any():
                break

            # Gather block indices for all batches
            block_indices = block_table[:, bn]  # (batch,)

            # Gather K and V: (batch, block_size, head_dim)
            kj_all = key_cache_flat[block_indices].to(torch.float32)
            vj_all = value_cache_flat[block_indices].to(torch.float32)

            # QK matmul: (batch, q_tile_size, block_size)
            sij = torch.bmm(qi, kj_all.transpose(1, 2)) * scale_value

            # Mask out invalid positions (beyond valid_len per batch)
            pos = torch.arange(block_size, device=sij.device).unsqueeze(0)  # (1, block_size)
            valid_mask = pos < valid_lens.unsqueeze(1)  # (batch, block_size)
            valid_mask = valid_mask.unsqueeze(1)  # (batch, 1, block_size)
            sij = sij.masked_fill(~valid_mask, float('-inf'))

            # Also mask inactive batches (no blocks at this position)
            batch_mask = active_mask.view(-1, 1, 1)  # (batch, 1, 1)
            sij = sij.masked_fill(~batch_mask, float('-inf'))

            mij = sij.max(dim=-1, keepdim=True)[0]  # (batch, q_tile_size, 1)
            mij = mij.clamp(min=-1e30)
            pij = torch.exp(sij - mij)
            pij = pij.masked_fill(~valid_mask, 0.0)
            pij = pij.masked_fill(~batch_mask, 0.0)
            pij = pij.to(torch.bfloat16).to(torch.float32)
            lij = pij.sum(dim=-1, keepdim=True)  # (batch, q_tile_size, 1)

            # PV matmul: (batch, q_tile_size, head_dim)
            oi_new = torch.bmm(pij, vj_all)

            if bn == 0:
                oi = oi_new
                li = lij
                mi = mij
            else:
                mi_new = torch.maximum(mi, mij)
                alpha = torch.exp(mi - mi_new)
                beta = torch.exp(mij - mi_new)
                li = alpha * li + beta * lij
                oi = alpha * oi + beta * oi_new
                mi = mi_new

        # Final normalization
        out[:, q_offset:q_offset + q_tile_size, :] = oi / li

    return out.reshape(-1, head_dim)


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
    params = {"name": DEFAULT_CASE, **ALL_CASES[DEFAULT_CASE]}
    result = generate_inputs(params)
    tensors = {name: tensor for name, tensor in result if isinstance(tensor, torch.Tensor)}
    compute_golden(tensors, params)

    print(f"=== Batch Paged Attention Golden Test ({params['name']}) ===")
    print(f"batch={params['batch']}, num_heads={params['num_heads']}, head_dim={params['head_dim']}")
    print(f"kv_head_num={params['kv_head_num']}, block_size={params['block_size']}")
    if params.get('context_lens_list'):
        print(f"context_lens (variable): {params['context_lens_list'][:8]}{'...' if len(params['context_lens_list']) > 8 else ''}")
    else:
        print(f"context_len={params['context_len']}")

    max_num_blocks = params['max_model_len'] // params['block_size']
    q_tile = min(params['num_heads'], 128)
    print(f"max_num_blocks_per_req={max_num_blocks}, q_tile_size={q_tile}")

    out = tensors["out"].reshape(params["batch"] * params["num_heads"], params["head_dim"])
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"Output mean: {out.mean():.4f}")
    print("Golden test passed!")
