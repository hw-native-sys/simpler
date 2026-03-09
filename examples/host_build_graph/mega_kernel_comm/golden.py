"""
Mega Kernel + Communication: Paged Attention → TGATHER.

Each rank independently computes paged attention on its own Q/K/V data,
then the first GATHER_COUNT elements of each rank's output are gathered to root.

Args layout:
  [ptr_query, ..., ptr_config, size_query, ..., size_config,
   device_ctx_ptr, win_in_base, win_out_base, n_ranks, root, rank_id]
"""

import ctypes
import struct
import torch
import numpy as np

GATHER_COUNT = 64

BATCH = 1
NUM_HEADS = 16
KV_HEAD_NUM = 1
HEAD_DIM = 16
BLOCK_SIZE = 16
CONTEXT_LEN = 16
MAX_MODEL_LEN = 256

__outputs__ = ["attn_out", "gather_out"]
RTOL = 1e-2
ATOL = 1e-2

ALL_CASES = {"Default": {}}
DEFAULT_CASE = "Default"


def _make_block_table_and_context():
    """Rank-independent block table and context lens (fixed seed)."""
    max_num_blocks_per_req = MAX_MODEL_LEN // BLOCK_SIZE
    cur_valid_blocks = (CONTEXT_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = BATCH * cur_valid_blocks

    torch.manual_seed(100)
    block_table = torch.randint(
        0, max(total_blocks, 1),
        size=(BATCH, max_num_blocks_per_req), dtype=torch.int32,
    )
    context_lens = torch.full((BATCH,), CONTEXT_LEN, dtype=torch.int32)
    return block_table, context_lens, total_blocks, max_num_blocks_per_req


def _make_qkv(rank_id, total_blocks):
    """Per-rank Q, K, V with deterministic seed."""
    torch.manual_seed(42 + rank_id)
    q = (torch.rand(BATCH, 1, NUM_HEADS * HEAD_DIM) - 0.5).to(torch.float16)
    q = q.reshape(BATCH, NUM_HEADS, HEAD_DIM)
    k = (torch.rand(total_blocks, BLOCK_SIZE, KV_HEAD_NUM, HEAD_DIM) - 0.5).to(torch.float16)
    v = (torch.rand(total_blocks, BLOCK_SIZE, KV_HEAD_NUM, HEAD_DIM) * 2 - 1).to(torch.float16)
    return q, k, v


def generate_inputs(params: dict) -> list:
    """Generate input tensors for mega_kernel_comm."""
    rank_id = params.get("rank_id", 0)
    n_ranks = params.get("n_ranks", 2)
    root = params.get("root", 0)

    block_table, context_lens, total_blocks, max_num_blocks_per_req = (
        _make_block_table_and_context()
    )
    query_fp16, key_fp16, value_fp16 = _make_qkv(rank_id, total_blocks)

    scale_value = 1.0
    scale_bits = struct.unpack('I', struct.pack('f', scale_value))[0]

    config = torch.tensor(
        [BATCH, NUM_HEADS, KV_HEAD_NUM, HEAD_DIM, BLOCK_SIZE,
         max_num_blocks_per_req, scale_bits],
        dtype=torch.int64,
    )

    query = query_fp16.flatten()
    key_cache = key_fp16.flatten()
    value_cache = value_fp16.flatten()
    block_table_flat = block_table.flatten()

    attn_out = torch.zeros(BATCH * NUM_HEADS * HEAD_DIM, dtype=torch.float32)
    gather_out = torch.zeros(n_ranks * GATHER_COUNT, dtype=torch.float32)

    result = [
        ("query", query),
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("block_table", block_table_flat),
        ("context_lens", context_lens),
        ("attn_out", attn_out),
        ("gather_out", gather_out),
        ("config", config),
        ("size_query", ctypes.c_int64(query.nbytes)),
        ("size_key_cache", ctypes.c_int64(key_cache.nbytes)),
        ("size_value_cache", ctypes.c_int64(value_cache.nbytes)),
        ("size_block_table", ctypes.c_int64(block_table_flat.nbytes)),
        ("size_context_lens", ctypes.c_int64(context_lens.nbytes)),
        ("size_attn_out", ctypes.c_int64(attn_out.nbytes)),
        ("size_gather_out", ctypes.c_int64(gather_out.nbytes)),
        ("size_config", ctypes.c_int64(config.nbytes)),
    ]

    if "device_ctx_ptr" in params and "win_in_base" in params and "win_out_base" in params:
        result.extend([
            ("device_ctx_ptr", ctypes.c_uint64(params["device_ctx_ptr"])),
            ("win_in_base", ctypes.c_uint64(params["win_in_base"])),
            ("win_out_base", ctypes.c_uint64(params["win_out_base"])),
            ("n_ranks", ctypes.c_int32(n_ranks)),
            ("root", ctypes.c_int32(root)),
            ("rank_id", ctypes.c_int32(rank_id)),
        ])

    return result


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
    """Online softmax paged attention (same algorithm as paged_attention/golden.py)."""
    assert num_kv_heads == 1
    batch, num_heads_dim, head_dim = query.shape
    _, block_size, _, _ = key_cache.shape

    key_cache_flat = key_cache.reshape(-1, block_size, head_dim)
    value_cache_flat = value_cache.reshape(-1, block_size, head_dim)

    out = torch.zeros((batch, num_heads_dim, head_dim), dtype=torch.float32)
    q_tile = min(num_heads_dim, 128)
    max_bn = int(((context_lens.max().item()) + block_size - 1) // block_size)

    for q_offset in range(0, num_heads_dim, q_tile):
        q_tile_size = min(q_tile, num_heads_dim - q_offset)
        qi = query[:, q_offset:q_offset + q_tile_size, :].to(torch.float32)

        oi = None
        li = None
        mi = None

        for bn in range(max_bn):
            valid_lens = torch.clamp(context_lens - bn * block_size, min=0, max=block_size)
            active_mask = valid_lens > 0
            if not active_mask.any():
                break

            block_indices = block_table[:, bn]
            kj_all = key_cache_flat[block_indices].to(torch.float32)
            vj_all = value_cache_flat[block_indices].to(torch.float32)

            sij = torch.bmm(qi, kj_all.transpose(1, 2)) * scale_value

            pos = torch.arange(block_size, device=sij.device).unsqueeze(0)
            valid_mask = pos < valid_lens.unsqueeze(1)
            valid_mask = valid_mask.unsqueeze(1)
            sij = sij.masked_fill(~valid_mask, float('-inf'))

            batch_mask = active_mask.view(-1, 1, 1)
            sij = sij.masked_fill(~batch_mask, float('-inf'))

            mij = sij.max(dim=-1, keepdim=True)[0]
            mij = mij.clamp(min=-1e30)
            pij = torch.exp(sij - mij)
            pij = pij.masked_fill(~valid_mask, 0.0)
            pij = pij.masked_fill(~batch_mask, 0.0)
            pij = pij.to(torch.bfloat16).to(torch.float32)
            lij = pij.sum(dim=-1, keepdim=True)

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

        out[:, q_offset:q_offset + q_tile_size, :] = oi / li

    return out.reshape(-1, head_dim)


def _compute_rank_attn(rank_id, block_table, context_lens, total_blocks):
    """Compute paged attention output for a specific rank."""
    q, k, v = _make_qkv(rank_id, total_blocks)
    return paged_attention(
        q, k.reshape(-1, BLOCK_SIZE, KV_HEAD_NUM, HEAD_DIM),
        v.reshape(-1, BLOCK_SIZE, KV_HEAD_NUM, HEAD_DIM),
        KV_HEAD_NUM, NUM_HEADS, 1.0, block_table, context_lens,
    )


def compute_golden(tensors: dict, params: dict) -> None:
    """Compute expected output: paged attention per rank, then gather to root."""
    rank_id = params.get("rank_id", 0)
    n_ranks = params.get("n_ranks", 2)
    root = params.get("root", 0)

    max_num_blocks_per_req = MAX_MODEL_LEN // BLOCK_SIZE
    total_blocks = BATCH * ((CONTEXT_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE)

    block_table = tensors["block_table"].reshape(BATCH, max_num_blocks_per_req)
    context_lens_t = tensors["context_lens"]

    # This rank's attention output
    query = tensors["query"].reshape(BATCH, NUM_HEADS, HEAD_DIM)
    key_cache = tensors["key_cache"].reshape(-1, BLOCK_SIZE, KV_HEAD_NUM, HEAD_DIM)
    value_cache = tensors["value_cache"].reshape(-1, BLOCK_SIZE, KV_HEAD_NUM, HEAD_DIM)

    attn_result = paged_attention(
        query, key_cache, value_cache,
        KV_HEAD_NUM, NUM_HEADS, 1.0, block_table, context_lens_t,
    )
    tensors["attn_out"][:] = attn_result.flatten()

    # Gather first GATHER_COUNT elements from each rank's attn output to root
    if rank_id == root:
        gather_np = tensors["gather_out"].cpu().numpy()
        for r in range(n_ranks):
            attn_r = _compute_rank_attn(r, block_table, context_lens_t, total_blocks)
            flat_r = attn_r.flatten().numpy()
            gather_np[r * GATHER_COUNT : (r + 1) * GATHER_COUNT] = flat_r[:GATHER_COUNT]


if __name__ == "__main__":
    params = {"name": DEFAULT_CASE, **ALL_CASES[DEFAULT_CASE]}
    result = generate_inputs(params)
    tensors = {name: tensor for name, tensor in result if isinstance(tensor, torch.Tensor)}
    compute_golden(tensors, params)

    out = tensors["attn_out"]
    print(f"=== Mega Kernel Comm Golden Test ===")
    print(f"attn_out shape: {out.shape}, range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"gather_out shape: {tensors['gather_out'].shape}")
    print("Golden test passed!")
