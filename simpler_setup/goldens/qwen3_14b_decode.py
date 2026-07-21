# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Golden reference + fixture for the Qwen3-14B 2-layer decode SceneTestCase.

Ported from pypto-lib ``models/qwen3/14b/decode_layer.py`` (entry
``decode_fwd_layers`` with ``_CHUNK_NLAYERS == 2``): a fused chunk of two
consecutive Qwen3-14B decode layers, hidden -> hidden, no LM head. Weights and
the paged KV pool are STACKED along dim 0 (one slice per layer); the two layers
use the same (replicated) weights — matching the lib's ``--validate-fwd``
const-layer-0 stack — but each reads/writes its own KV pool. The inter-layer
hidden is carried in FP32 (no per-layer bf16 round); the only chunk-boundary
casts are ``copy_hidden`` (bf16 input embed) and ``copy_out`` (FP32->bf16).

Parameter regime matches ``stress_profile.py`` (the vLLM serving stress run):
BATCH=16 (CONCURRENCY, aligned with decode kernel BATCH=16), MAX_SEQ=5500
(= max_model_len), and a fixed decode sequence length of 3500 (the ~3500-token
prompt). ``MAX_SEQ`` MUST match the harvested kernels' codegen-time value.

KV-cache layout per layer pool: row = (phys_page * NUM_KV_HEADS + kv_head) *
BLOCK_SIZE + pos_in_block; layer L's pool starts at L * CACHE_ROWS.
Weight matrices are ``[in_features, out_features]`` -> ``y = x @ w``.
"""

from __future__ import annotations

import torch

from simpler_setup.scene_test import TaskArgsBuilder, Tensor

# ── Model architecture (Qwen3-14B) ──
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 17408
BATCH = 16
EPS = 1e-6

HIDDEN = NUM_HEADS * HEAD_DIM  # 5120
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM  # 1024
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS  # 5
HALF_DIM = HEAD_DIM // 2  # 64
ATTN_SCALE = 1.0 / (HEAD_DIM**0.5)
ROPE_THETA = 1.0e4

# ── Chunk / paging (must match the harvested kernels) ──
N_LAYERS = 2
MAX_SEQ = 5500  # = stress_profile max_model_len; codegen-time KV-pool / RoPE sizing
SEQ_TILE = 128
BLOCK_SIZE = SEQ_TILE
MAX_CTX_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE  # 43 @ 5500
MAX_BLOCKS_PER_SEQ = MAX_CTX_BLOCKS
NUM_PAGES = BATCH * MAX_BLOCKS_PER_SEQ
CACHE_ROWS = NUM_PAGES * NUM_KV_HEADS * BLOCK_SIZE  # rows of ONE layer's paged pool

DEFAULT_SEQ_LEN = 3500  # the stress prompt length (~3500 tokens)

# Ordered entry args of decode_fwd_layers (orchestration signature order).
INPUT_NAMES = (
    "hidden_states",
    "input_rms_weight",
    "wq",
    "wk",
    "wv",
    "q_norm_weight",
    "k_norm_weight",
    "seq_lens",
    "block_table",
    "slot_mapping",
    "rope_cos",
    "rope_sin",
    "k_cache",
    "v_cache",
    "wo",
    "w_gate",
    "w_up",
    "w_down",
    "post_rms_weight",
)


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).to(torch.float32)


def _rmsnorm_inv(x: torch.Tensor) -> torch.Tensor:
    return torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)


def _rope_half(vec: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    lo, hi = vec[..., :HALF_DIM], vec[..., HALF_DIM:]
    cos_lo, cos_hi = cos[..., :HALF_DIM], cos[..., HALF_DIM:]
    sin_lo, sin_hi = sin[..., :HALF_DIM], sin[..., HALF_DIM:]
    return torch.cat([lo * cos_lo - hi * sin_lo, hi * cos_hi + lo * sin_hi], dim=-1)


def _paged_block_table_slot_mapping(seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Identity paging within a layer pool (same for every layer)."""
    block_table = torch.arange(BATCH * MAX_BLOCKS_PER_SEQ, dtype=torch.int32)
    slot_mapping = torch.empty(BATCH, dtype=torch.int32)
    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        logical_block = pos // BLOCK_SIZE
        phys_page = b * MAX_BLOCKS_PER_SEQ + logical_block
        slot_mapping[b] = phys_page * BLOCK_SIZE + (pos % BLOCK_SIZE)
    return block_table, slot_mapping


def generate_inputs(seed: int = 1234, seq_len: int = DEFAULT_SEQ_LEN) -> TaskArgsBuilder:
    """Deterministic fixture for decode_fwd_layers (N=2), stacked x2 along dim 0.

    Every lane uses sequence length ``seq_len`` (default 3500, the stress prompt).
    Per-layer weights are replicated (stack0) so layer 1 reuses layer 0's weights,
    matching the lib's const-layer-0 stacked-fwd reference; each layer still has
    its own KV pool.
    """
    if not (1 <= seq_len <= MAX_SEQ):
        raise ValueError(f"seq_len must be in [1, {MAX_SEQ}], got {seq_len}")
    g = torch.Generator().manual_seed(seed)

    def rn(shape, std=1.0, bias=0.0):
        return torch.empty(shape).normal_(0.0, std, generator=g) + bias

    def s0(t):  # replicate along dim 0 (one slice per layer)
        return torch.cat([t] * N_LAYERS, dim=0).contiguous()

    seq_lens = torch.full([BATCH], seq_len, dtype=torch.int32)
    block_table, slot_mapping = _paged_block_table_slot_mapping(seq_lens)

    posv = torch.arange(MAX_SEQ).float().unsqueeze(1)
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HALF_DIM).float() / HALF_DIM))
    ang = posv * inv_freq.unsqueeze(0)
    rope_cos = torch.cat([ang.cos(), ang.cos()], dim=1).float()
    rope_sin = torch.cat([ang.sin(), ang.sin()], dim=1).float()

    tensors = {
        "hidden_states": rn([BATCH, HIDDEN], 1.0).to(torch.bfloat16),
        "input_rms_weight": s0(rn([1, HIDDEN], 0.1, 1.0).float()),
        "wq": s0(rn([HIDDEN, HIDDEN], 0.02).to(torch.bfloat16)),
        "wk": s0(rn([HIDDEN, KV_HIDDEN], 0.02).to(torch.bfloat16)),
        "wv": s0(rn([HIDDEN, KV_HIDDEN], 0.02).to(torch.bfloat16)),
        "q_norm_weight": s0(rn([1, HEAD_DIM], 0.1, 1.0).float()),
        "k_norm_weight": s0(rn([1, HEAD_DIM], 0.1, 1.0).float()),
        "seq_lens": seq_lens,
        "block_table": block_table,
        "slot_mapping": slot_mapping,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "k_cache": s0(rn([CACHE_ROWS, HEAD_DIM], 0.01).to(torch.bfloat16)),
        "v_cache": s0(rn([CACHE_ROWS, HEAD_DIM], 0.02, 0.3).to(torch.bfloat16)),
        "wo": s0(rn([HIDDEN, HIDDEN], 0.0006).to(torch.bfloat16)),
        "w_gate": s0(rn([HIDDEN, INTERMEDIATE], 0.02).to(torch.bfloat16)),
        "w_up": s0(rn([HIDDEN, INTERMEDIATE], 0.02).to(torch.bfloat16)),
        "w_down": s0(rn([INTERMEDIATE, HIDDEN], 0.0004).to(torch.bfloat16)),
        "post_rms_weight": s0(rn([1, HIDDEN], 0.1, 1.0).float()),
    }
    specs = [Tensor(name, tensors[name]) for name in INPUT_NAMES]
    specs.append(Tensor("out", torch.zeros([BATCH, HIDDEN], dtype=torch.bfloat16)))
    return TaskArgsBuilder(*specs)


def _one_layer(args, layer: int, x: torch.Tensor) -> torch.Tensor:
    """One decode layer of the chunk. ``x`` is the FP32 layer input; returns the
    FP32 residual-stream output (down + h1, NO bf16 round). Also writes the
    current token's RoPE'd K / raw V into THIS layer's KV pool (for INOUT compare).
    """
    hb, ib, cb = layer * HIDDEN, layer * INTERMEDIATE, layer * CACHE_ROWS
    irw = args.input_rms_weight.float()[layer]  # [H]
    wq = args.wq[hb : hb + HIDDEN].float()
    wk = args.wk[hb : hb + HIDDEN].float()
    wv = args.wv[hb : hb + HIDDEN].float()
    qn = args.q_norm_weight.float()[layer]
    kn = args.k_norm_weight.float()[layer]
    wo = args.wo[hb : hb + HIDDEN].float()
    wg = args.w_gate[hb : hb + HIDDEN].float()
    wu = args.w_up[hb : hb + HIDDEN].float()
    wd = args.w_down[ib : ib + INTERMEDIATE].float()
    post_gamma = args.post_rms_weight.float()[layer]

    seq_lens = args.seq_lens
    block_table = args.block_table
    slot_mapping = args.slot_mapping
    rope_cos = args.rope_cos.float()
    rope_sin = args.rope_sin.float()
    k_cache = args.k_cache[cb : cb + CACHE_ROWS].float()  # this layer's pool
    v_cache = args.v_cache[cb : cb + CACHE_ROWS].float()

    inv_rms = _rmsnorm_inv(x)
    normed = _bf16(x * irw)
    q_proj = normed @ wq
    k_proj = normed @ wk
    v_proj = normed @ wv
    qh = (q_proj * inv_rms).reshape(BATCH, NUM_HEADS, HEAD_DIM)
    qh = qh * _rmsnorm_inv(qh) * qn
    kh = (k_proj * inv_rms).reshape(BATCH, NUM_KV_HEADS, HEAD_DIM)
    kh = kh * _rmsnorm_inv(kh) * kn
    v_heads = (v_proj * inv_rms).reshape(BATCH, NUM_KV_HEADS, HEAD_DIM)

    attn_out = torch.zeros(BATCH, HIDDEN)
    cur_k, cur_v = {}, {}
    for b in range(BATCH):
        slen = int(seq_lens[b].item())
        p = slen - 1
        cos_p, sin_p = rope_cos[p], rope_sin[p]
        q_b = _bf16(_rope_half(qh[b], cos_p, sin_p))
        k_cur = _bf16(_rope_half(kh[b], cos_p, sin_p))
        v_cur = _bf16(v_heads[b])
        n_blocks = (slen + BLOCK_SIZE - 1) // BLOCK_SIZE
        for kvh in range(NUM_KV_HEADS):
            k_lane = torch.empty(slen, HEAD_DIM)
            v_lane = torch.empty(slen, HEAD_DIM)
            for sb in range(n_blocks):
                pbid = int(block_table[b * MAX_BLOCKS_PER_SEQ + sb].item())
                row = (pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE
                lo = sb * BLOCK_SIZE
                blk = min(BLOCK_SIZE, slen - lo)
                k_lane[lo : lo + blk] = k_cache[row : row + blk]
                v_lane[lo : lo + blk] = v_cache[row : row + blk]
            k_lane[p] = k_cur[kvh]
            v_lane[p] = v_cur[kvh]
            cur_k[(b, kvh)] = k_cur[kvh]
            cur_v[(b, kvh)] = v_cur[kvh]
            for j in range(Q_PER_KV):
                hq = kvh * Q_PER_KV + j
                scores = (q_b[hq].unsqueeze(0) * k_lane).sum(-1) * ATTN_SCALE
                w = torch.softmax(scores, dim=-1)
                attn_out[b, hq * HEAD_DIM : (hq + 1) * HEAD_DIM] = (w.unsqueeze(-1) * v_lane).sum(0)
    attn_out = _bf16(attn_out)

    attn_proj = attn_out @ wo
    h1 = x + attn_proj  # FP32 residual (no bf16 round — inter-layer carry)
    post_inv = _rmsnorm_inv(h1)
    mlp_in = _bf16(h1 * post_gamma)
    gate = mlp_in @ wg
    up = mlp_in @ wu
    sg = gate * post_inv
    su = up * post_inv
    mlp = _bf16(sg * torch.sigmoid(sg) * su)
    down = mlp @ wd

    # Write the current token into THIS layer's pool (INOUT compare).
    for b in range(BATCH):
        slot = int(slot_mapping[b].item())
        sblk, soff = slot // BLOCK_SIZE, slot % BLOCK_SIZE
        for kvh in range(NUM_KV_HEADS):
            row = cb + (sblk * NUM_KV_HEADS + kvh) * BLOCK_SIZE + soff
            args.k_cache[row] = cur_k[(b, kvh)].to(torch.bfloat16)
            args.v_cache[row] = cur_v[(b, kvh)].to(torch.bfloat16)

    return down + h1  # FP32


def compute_golden(args: TaskArgsBuilder) -> None:
    """Fill ``args.out`` (and INOUT k_cache/v_cache) for the 2-layer decode chunk."""
    cur = args.hidden_states.float()  # copy_hidden: bf16 input embedded as FP32
    for layer in range(N_LAYERS):
        cur = _one_layer(args, layer, cur)
    args.out[:] = cur.to(torch.bfloat16)  # copy_out: single FP32->bf16 round
