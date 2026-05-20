# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Golden reference for the Qwen3-14B 1-layer decode SceneTestCase.

Mirrors the device pipeline in ``qwen3_14b_decode/kernels/orchestration/qwen3_decode.cpp``:
RMSNorm → QKV projection → per-head Q/K RMS → RoPE → KV-cache write → paged
attention (online softmax) → output projection + residual → post-RMSNorm →
SwiGLU FFN → down-proj + residual.

KV cache layout (must match the kernels byte-for-byte):
    row = (block_idx * NUM_KV_HEADS + kv_head_idx) * BLOCK_SIZE + pos_in_block

where ``block_idx`` is what ``block_table`` contains and ``slot_mapping[b]``
encodes the write position as ``slot_block * BLOCK_SIZE + slot_offset``.

Weight matrices are stored in ``[in_features, out_features]`` layout — the
kernels consume them directly without a transpose — so the matmul is
``y = x @ w`` rather than the more common ``y = x @ w.T``.
"""

import math

import torch

from simpler_setup.scene_test import TaskArgsBuilder, Tensor

# Qwen3-14B architectural constants — baked into the kernels.
HIDDEN = 5120
KV_HIDDEN = 1024
HEAD_DIM = 128
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS  # 5
INTERMEDIATE = 17408
BLOCK_SIZE = 256
BLOCKS_PER_BATCH = 2  # qwen3_decode.cpp pins block_table to 2 entries per batch
MAX_SEQ = 4096
EPS = 1e-6
ROPE_THETA = 1_000_000.0


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Standard LLaMA-style RMSNorm: ``x * rsqrt(mean(x**2) + eps) * weight``.

    ``weight`` may be ``[N]`` or ``[1, N]`` (the kernels' FP32 gammas are stored
    as ``[1, HIDDEN]`` / ``[1, HEAD_DIM]``); both broadcast against a 1-D ``x``.
    """
    x32 = x.float()
    inv = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    w = weight.float().reshape(-1)
    return (x32 * inv * w).to(x.dtype)


def _build_rope_tables(max_seq: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard RoPE precomputed cos/sin tables, shape ``[max_seq, head_dim]``.

    Within a head_dim row the first ``head_dim // 2`` entries pair against the
    second half, matching the split that ``rope_kv_cache.cpp`` performs on the
    Q/K tiles (``cos_lo / cos_hi`` halves).
    """
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(max_seq, dtype=torch.float32).unsqueeze(1)  # [max_seq, 1]
    angles = pos * freqs.unsqueeze(0)  # [max_seq, half]
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos_row: torch.Tensor, sin_row: torch.Tensor) -> torch.Tensor:
    """Rotate the (head_dim) vector by the position-specific cos/sin row.

    Matches the lo/hi split in ``rope_kv_cache.cpp``: pair element ``i`` with
    element ``i + head_dim/2`` and apply the 2-D rotation.
    """
    half = x.shape[-1] // 2
    x_lo = x[..., :half]
    x_hi = x[..., half:]
    cos_lo = cos_row[..., :half]
    sin_lo = sin_row[..., :half]
    cos_hi = cos_row[..., half:]
    sin_hi = sin_row[..., half:]
    rot_lo = x_lo * cos_lo - x_hi * sin_lo
    rot_hi = x_hi * cos_hi + x_lo * sin_hi
    return torch.cat([rot_lo, rot_hi], dim=-1)


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _cache_row(block_idx: int, kv_head: int, pos_in_block: int) -> int:
    """Flat-cache row index used by ``rope_kv_cache`` (write) and ``qk_matmul`` (read)."""
    return (block_idx * NUM_KV_HEADS + kv_head) * BLOCK_SIZE + pos_in_block


def generate_inputs(user_batch: int, seq_len: int) -> TaskArgsBuilder:
    """Produce the 20 ordered tensors the orchestration consumes.

    Cache sizing assumes ``seq_len <= BLOCK_SIZE`` (single block per batch). The
    block table is one block per batch (with the second slot unused but
    present, to match the orchestration's hardcoded ``block_table_base = b * 2``).
    """
    if seq_len < 1 or seq_len > BLOCK_SIZE:
        raise ValueError(f"seq_len must be in [1, {BLOCK_SIZE}], got {seq_len}")

    num_blocks = user_batch  # one block per batch
    kv_cache_rows = num_blocks * NUM_KV_HEADS * BLOCK_SIZE

    hidden_states = torch.randn(user_batch, HIDDEN, dtype=torch.bfloat16) * 0.02
    # Norm gammas are read as FP32 by rmsnorm/qk_norm/post_rmsnorm — see the
    # `float*` arg type in each kernel's static function signature.
    input_rms_weight = torch.randn(1, HIDDEN, dtype=torch.float32)
    wq = torch.randn(HIDDEN, NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16) * 0.02
    wk = torch.randn(HIDDEN, KV_HIDDEN, dtype=torch.bfloat16) * 0.02
    wv = torch.randn(HIDDEN, KV_HIDDEN, dtype=torch.bfloat16) * 0.02
    q_norm_weight = torch.randn(1, HEAD_DIM, dtype=torch.float32)
    k_norm_weight = torch.randn(1, HEAD_DIM, dtype=torch.float32)

    seq_lens = torch.full((user_batch,), seq_len, dtype=torch.int32)
    block_table = torch.zeros((user_batch, BLOCKS_PER_BATCH), dtype=torch.int32)
    for b in range(user_batch):
        block_table[b, 0] = b  # one unique block per batch

    slot_mapping = torch.zeros(user_batch, dtype=torch.int32)
    for b in range(user_batch):
        slot_mapping[b] = b * BLOCK_SIZE + (seq_len - 1)

    cos_table, sin_table = _build_rope_tables(MAX_SEQ, HEAD_DIM, ROPE_THETA)
    rope_cos = cos_table.contiguous()  # FP32
    rope_sin = sin_table.contiguous()  # FP32

    k_cache = (torch.randn(kv_cache_rows, HEAD_DIM, dtype=torch.bfloat16) * 0.02).contiguous()
    v_cache = (torch.randn(kv_cache_rows, HEAD_DIM, dtype=torch.bfloat16) * 0.02).contiguous()

    wo = torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN, dtype=torch.bfloat16) * 0.02
    post_rms_weight = torch.randn(1, HIDDEN, dtype=torch.float32)
    w_gate = torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16) * 0.02
    w_up = torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16) * 0.02
    w_down = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16) * 0.02

    out = torch.zeros(user_batch, HIDDEN, dtype=torch.bfloat16)

    return TaskArgsBuilder(
        Tensor("hidden_states", hidden_states),
        Tensor("input_rms_weight", input_rms_weight),
        Tensor("wq", wq),
        Tensor("wk", wk),
        Tensor("wv", wv),
        Tensor("q_norm_weight", q_norm_weight),
        Tensor("k_norm_weight", k_norm_weight),
        Tensor("seq_lens", seq_lens),
        Tensor("block_table", block_table),
        Tensor("slot_mapping", slot_mapping),
        Tensor("rope_cos", rope_cos),
        Tensor("rope_sin", rope_sin),
        Tensor("k_cache", k_cache),
        Tensor("v_cache", v_cache),
        Tensor("wo", wo),
        Tensor("post_rms_weight", post_rms_weight),
        Tensor("w_gate", w_gate),
        Tensor("w_up", w_up),
        Tensor("w_down", w_down),
        Tensor("out", out),
    )


def compute_golden(args: TaskArgsBuilder, user_batch: int, seq_len: int) -> None:
    """Compute the expected ``out``, ``k_cache``, ``v_cache`` in-place on ``args``."""
    scale = 1.0 / math.sqrt(HEAD_DIM)

    hidden_states = args.hidden_states
    wq = args.wq.float()
    wk = args.wk.float()
    wv = args.wv.float()
    wo = args.wo.float()
    w_gate = args.w_gate.float()
    w_up = args.w_up.float()
    w_down = args.w_down.float()
    rope_cos = args.rope_cos
    rope_sin = args.rope_sin

    for b in range(user_batch):
        pos = seq_len - 1
        slot = int(args.slot_mapping[b].item())
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot - slot_block * BLOCK_SIZE

        residual_in = hidden_states[b].float()
        normed = _rms_norm(hidden_states[b], args.input_rms_weight).float()

        q = (normed @ wq).reshape(NUM_HEADS, HEAD_DIM)
        k = (normed @ wk).reshape(NUM_KV_HEADS, HEAD_DIM)
        v = (normed @ wv).reshape(NUM_KV_HEADS, HEAD_DIM)

        q_normed = torch.stack([_rms_norm(q[h], args.q_norm_weight).float() for h in range(NUM_HEADS)])
        k_normed = torch.stack([_rms_norm(k[h], args.k_norm_weight).float() for h in range(NUM_KV_HEADS)])

        cos_row = rope_cos[pos].float()
        sin_row = rope_sin[pos].float()
        q_roped = torch.stack([_apply_rope(q_normed[h], cos_row, sin_row) for h in range(NUM_HEADS)])
        k_roped = torch.stack([_apply_rope(k_normed[h], cos_row, sin_row) for h in range(NUM_KV_HEADS)])

        # Write the new K/V vector into the cache at slot_mapping position.
        for h in range(NUM_KV_HEADS):
            row = _cache_row(slot_block, h, slot_offset)
            args.k_cache[row] = k_roped[h].to(torch.bfloat16)
            args.v_cache[row] = v.to(torch.bfloat16)[h]

        # Gather the [seq_len] context rows for each kv_head from the block table.
        k_ctx = torch.zeros(NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=torch.float32)
        v_ctx = torch.zeros(NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=torch.float32)
        for p in range(seq_len):
            ctx_block_slot = p // BLOCK_SIZE
            ctx_in_block = p - ctx_block_slot * BLOCK_SIZE
            ctx_block = int(args.block_table[b, ctx_block_slot].item())
            for h in range(NUM_KV_HEADS):
                row = _cache_row(ctx_block, h, ctx_in_block)
                k_ctx[h, p] = args.k_cache[row].float()
                v_ctx[h, p] = args.v_cache[row].float()

        # GQA attention: each query head attends to its kv-head group.
        attn_out = torch.zeros(NUM_HEADS, HEAD_DIM, dtype=torch.float32)
        for qh in range(NUM_HEADS):
            kv_head = qh // HEADS_PER_KV
            scores = (q_roped[qh] @ k_ctx[kv_head].T) * scale  # [seq_len]
            probs = torch.softmax(scores, dim=-1)
            attn_out[qh] = probs @ v_ctx[kv_head]

        attn_flat = attn_out.reshape(NUM_HEADS * HEAD_DIM).to(torch.bfloat16)

        # Output projection + residual (residual is the original BF16 hidden_states).
        resid1 = residual_in + (attn_flat.float() @ wo)

        normed2 = _rms_norm(resid1.to(torch.bfloat16), args.post_rms_weight).float()
        gate = normed2 @ w_gate
        up = normed2 @ w_up
        mlp = (_silu(gate) * up).to(torch.bfloat16).float()
        final = resid1 + mlp @ w_down

        args.out[b] = final.to(torch.bfloat16)
