# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ruff: noqa: PLC0415
"""Torch-only compressor epilogue for the frozen Pro B4S2 CSA fixture."""

COMPRESS_RATIO = 4
ROPE_HEAD_DIM = 64
HEAD_DIM = 512
OUT_DIM = 1024
COMPRESS_STATE_BLOCK_SIZE = 4
BLOCK_SIZE = 128
EPS = 1e-6


def reference_kv_quant_fp8(x):
    """Independent torch expression of the official group-64 cache epilog."""
    import torch

    values = x.to(torch.bfloat16).float().reshape(*x.shape[:-1], -1, 64)
    maximum = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2((maximum / 448.0).double()))).float()
    payload = (values / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (payload.float() * scale).to(torch.bfloat16).float().reshape(x.shape)


def reference_compressor_cache(
    compress_state,
    compress_state_block_table,
    norm_w,
    cos,
    sin,
    position_ids,
    cmp_slot_mapping,
    initial_cache,
):
    """Pool final paged compressor state and apply the cache epilogue in Torch."""
    import torch

    bsz, seq = position_ids.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM
    cmp_kv_cache = initial_cache.clone()
    cache_kv = torch.full((bsz, seq, HEAD_DIM), float("nan"), dtype=torch.float32, device=compress_state.device)
    pooled = torch.zeros(bsz, 1, HEAD_DIM, dtype=torch.float32, device=compress_state.device)
    should_compress_rows = torch.zeros(bsz, dtype=torch.bool, device=compress_state.device)

    def read_front_state(b, abs_pos):
        blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
        if blk_id < 0:
            return (
                torch.zeros(HEAD_DIM, dtype=torch.float32, device=compress_state.device),
                torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32, device=compress_state.device),
            )
        intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
        return (
            compress_state[blk_id, intra, :HEAD_DIM],
            compress_state[blk_id, intra, OUT_DIM : OUT_DIM + HEAD_DIM],
        )

    def read_back_state(b, abs_pos):
        blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
        if blk_id < 0:
            return (
                torch.zeros(HEAD_DIM, dtype=torch.float32, device=compress_state.device),
                torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32, device=compress_state.device),
            )
        intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
        return (
            compress_state[blk_id, intra, HEAD_DIM:OUT_DIM],
            compress_state[blk_id, intra, OUT_DIM + HEAD_DIM :],
        )

    for b in range(bsz):
        first_pos = int(position_ids[b, 0].item())
        pre_tokens = min(seq, ratio - (first_pos % ratio))
        boundary_s = ratio - 1 - (first_pos % ratio)
        should_compress = 0 <= boundary_s < seq
        boundary_end = first_pos + pre_tokens - 1
        cur_window_start = boundary_end - ratio + 1
        prev_window_start = cur_window_start - ratio

        if should_compress:
            should_compress_rows[b] = True
            kv_rows = []
            score_rows = []
            for s in range(ratio):
                abs_pos = prev_window_start + s
                if abs_pos < 0:
                    kv_rows.append(torch.zeros(HEAD_DIM, dtype=torch.float32, device=compress_state.device))
                    score_rows.append(
                        torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32, device=compress_state.device)
                    )
                    continue
                kv_row, score_row = read_front_state(b, abs_pos)
                kv_rows.append(kv_row)
                score_rows.append(score_row)
            for s in range(ratio):
                abs_pos = cur_window_start + s
                kv_row, score_row = read_back_state(b, abs_pos)
                kv_rows.append(kv_row)
                score_rows.append(score_row)
            kvs = torch.stack(kv_rows, dim=0).unsqueeze(0)
            scs = torch.stack(score_rows, dim=0).unsqueeze(0)
            pooled[b : b + 1] = (kvs * scs.softmax(dim=1)).sum(dim=1, keepdim=True)

    if not bool(should_compress_rows.any()):
        return cache_kv, cmp_kv_cache

    def rmsnorm(x, w):
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + EPS)
        return w * x

    for b in range(bsz):
        if not bool(should_compress_rows[b]):
            continue
        first_pos = int(position_ids[b, 0].item())
        boundary_s = ratio - 1 - (first_pos % ratio)
        kv_b = rmsnorm(pooled[b : b + 1], norm_w)

        x_pair = kv_b[..., -rd:].unflatten(-1, (-1, 2))
        x0, x1 = x_pair[..., 0], x_pair[..., 1]
        cos_v, sin_v = cos[b].view(-1), sin[b].view(-1)
        y0 = x0 * cos_v - x1 * sin_v
        y1 = x0 * sin_v + x1 * cos_v

        kv_b = torch.cat([kv_b[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)
        kv_b[..., :-rd] = reference_kv_quant_fp8(kv_b[..., :-rd])

        cmp_row = int(cmp_slot_mapping[b, boundary_s].item())
        if cmp_row >= 0:
            cache_kv[b : b + 1, 0:1, :] = kv_b
            blk_id = cmp_row // BLOCK_SIZE
            cmp_kv_cache[blk_id, cmp_row % BLOCK_SIZE, 0] = kv_b[0, 0]

    return cache_kv, cmp_kv_cache
