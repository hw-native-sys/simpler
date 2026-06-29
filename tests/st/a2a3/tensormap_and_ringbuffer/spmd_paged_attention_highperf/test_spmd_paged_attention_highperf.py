#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""High-performance SPMD paged attention."""

import ctypes
import math
import sys
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
sys.path.insert(0, str(KERNEL_DIR))

from pa_tiling import make_pa_nd_decode_tiling, workspace_sizes  # noqa: E402


def _pack_kv_to_paged(
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, _ = k_dense.shape
    num_blocks = seq_len // block_size
    k_page = (
        k_dense.view(batch, seq_len, num_kv_heads, head_dim)
        .view(batch, num_blocks, block_size, num_kv_heads, head_dim)
        .reshape(batch * num_blocks, block_size, num_kv_heads, head_dim)
        .contiguous()
    )
    v_page = (
        v_dense.view(batch, seq_len, num_kv_heads, head_dim)
        .view(batch, num_blocks, block_size, num_kv_heads, head_dim)
        .reshape(batch * num_blocks, block_size, num_kv_heads, head_dim)
        .contiguous()
    )
    block_table = (
        torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0).expand(batch, -1).clone()
        + torch.arange(batch, dtype=torch.int32).unsqueeze(1) * num_blocks
    )
    return k_page, v_page, block_table


def _compute_gqa_golden(
    q: torch.Tensor,
    k_page: torch.Tensor,
    v_page: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch, num_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_page.shape
    heads_per_kv = num_heads // num_kv_heads
    out = torch.empty(batch, num_heads, head_dim, dtype=q.dtype)

    for batch_idx in range(batch):
        seq_len = int(context_lens[batch_idx].item())
        block_count = (seq_len + block_size - 1) // block_size
        blocks = block_table[batch_idx, :block_count]
        for head_idx in range(num_heads):
            kv_head = head_idx // heads_per_kv
            keys = []
            values = []
            remaining = seq_len
            for block in blocks:
                valid = min(block_size, remaining)
                block_id = int(block.item())
                keys.append(k_page[block_id, :valid, kv_head, :])
                values.append(v_page[block_id, :valid, kv_head, :])
                remaining -= valid
            key = torch.cat(keys, dim=0).float()
            value = torch.cat(values, dim=0).float()
            scores = torch.mv(key, q[batch_idx, head_idx].float()) * scale
            probs = torch.softmax(scores, dim=0)
            out[batch_idx, head_idx] = torch.mv(value.t(), probs).to(q.dtype)

    return out


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdPagedAttentionHighPerf(SceneTestCase):
    RTOL = 5e-3
    ATOL = 2e-2

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/paged_attention_highperf_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.OUT,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
            ],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "PA_HIGHPERF_AIC",
                "source": "kernels/aic/paged_attention_highperf.cpp",
                "core_type": "aic",
                "signature": [
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.OUT,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                ],
            },
            {
                "func_id": 1,
                "name": "PA_HIGHPERF_AIV",
                "source": "kernels/aic/paged_attention_highperf.cpp",
                "core_type": "aiv",
                "signature": [
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.OUT,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                ],
            },
        ],
    }

    CASES = [
        {
            "name": "b1_h32_kv8_s128_bs128_fp16",
            # onboard a2a3 enabled: the 'out' golden mismatch is closed by the
            # producer-side DdrBarrierBeforeFfts cross-core DDR fence, validated
            # over 19 st-onboard-a2a3 rounds.
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {
                "batch": 1,
                "num_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "kv_seq": 128,
                "block_size": 128,
                "block_dim": 24,
                "dtype": "float16",
            },
        },
        {
            "name": "b4_h32_kv8_s512_bs128_fp16",
            "manual": True,
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {
                "batch": 4,
                "num_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "kv_seq": 512,
                "block_size": 128,
                "block_dim": 24,
                "dtype": "float16",
            },
        },
        {
            "name": "b1_h32_kv8_s16384_bs128_fp16",
            "manual": True,
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {
                "batch": 1,
                "num_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "kv_seq": 16384,
                "block_size": 128,
                "block_dim": 24,
                "dtype": "float16",
            },
        },
        {
            "name": "b1_h32_kv8_s4096_bs128_fp16",
            "manual": True,
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {
                "batch": 1,
                "num_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "kv_seq": 4096,
                "block_size": 128,
                "block_dim": 24,
                "dtype": "float16",
            },
        },
        {
            "name": "b1_h32_kv8_s6144_bs128_fp16",
            "manual": True,
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {
                "batch": 1,
                "num_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "kv_seq": 6144,
                "block_size": 128,
                "block_dim": 24,
                "dtype": "float16",
            },
        },
        {
            "name": "b1_h32_kv8_s8192_bs128_fp16",
            # enabled in CI to guard the long-sequence fix onboard.
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {
                "batch": 1,
                "num_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "kv_seq": 8192,
                "block_size": 128,
                "block_dim": 24,
                "dtype": "float16",
            },
        },
        {
            "name": "b2_h32_kv8_s4096_bs128_fp16",
            "manual": True,
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {
                "batch": 2,
                "num_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "kv_seq": 4096,
                "block_size": 128,
                "block_dim": 24,
                "dtype": "float16",
            },
        },
        {
            "name": "b2_h32_kv8_s8192_bs128_fp16",
            "manual": True,
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {
                "batch": 2,
                "num_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "kv_seq": 8192,
                "block_size": 128,
                "block_dim": 24,
                "dtype": "float16",
            },
        },
    ]

    def generate_args(self, params):
        batch = params["batch"]
        num_heads = params["num_heads"]
        num_kv_heads = params["num_kv_heads"]
        head_dim = params["head_dim"]
        kv_seq = params["kv_seq"]
        block_size = params["block_size"]
        block_dim = params["block_dim"]
        dtype = getattr(torch, params["dtype"])
        scale = 1.0 / math.sqrt(float(head_dim))

        torch.manual_seed(42)
        q = torch.randn(batch, num_heads, head_dim, dtype=dtype)
        k_dense = torch.randn(batch, kv_seq, num_kv_heads * head_dim, dtype=dtype)
        v_dense = torch.randn(batch, kv_seq, num_kv_heads * head_dim, dtype=dtype)
        k_page, v_page, block_table = _pack_kv_to_paged(k_dense, v_dense, num_kv_heads, head_dim, block_size)
        context_lens = torch.tensor([kv_seq] * batch, dtype=torch.int32)

        tiling, effective_block_dim = make_pa_nd_decode_tiling(
            batch=batch,
            kv_seq_lens=context_lens.tolist(),
            num_heads=num_heads,
            kv_heads=num_kv_heads,
            head_dim=head_dim,
            head_dim_v=head_dim,
            num_blocks=k_page.shape[0],
            block_size=block_size,
            max_blocks_per_query=block_table.shape[1],
            scale=scale,
            block_dim=block_dim,
            device="cpu",
            dtype=dtype,
        )
        ws = workspace_sizes(batch, num_heads, head_dim, head_dim, block_dim)

        return TaskArgsBuilder(
            Tensor("query", q),
            Tensor("key_cache", k_page),
            Tensor("value_cache", v_page),
            Tensor("block_table", block_table),
            Tensor("out", torch.zeros(batch, num_heads, head_dim, dtype=dtype)),
            Tensor("s_gm", torch.zeros(ws["s"], dtype=torch.uint8)),
            Tensor("p_gm", torch.zeros(ws["p"], dtype=torch.uint8)),
            Tensor("o_tmp_gm", torch.zeros(ws["o_tmp"], dtype=torch.uint8)),
            Tensor("go_gm", torch.zeros(ws["go"], dtype=torch.uint8)),
            Tensor("o_core_tmp_gm", torch.zeros(ws["o_core_tmp"], dtype=torch.uint8)),
            Tensor("l_gm", torch.zeros(ws["l"], dtype=torch.uint8)),
            Tensor("gm_k16", torch.zeros(ws["k16"], dtype=torch.uint8)),
            Tensor("gm_v16", torch.zeros(ws["v16"], dtype=torch.uint8)),
            Tensor("tiling", tiling),
            Tensor("null", torch.zeros(1, dtype=torch.uint8)),
            Scalar("effective_block_dim", ctypes.c_int64(effective_block_dim)),
        )

    def compute_golden(self, args, params):
        batch = params["batch"]
        num_heads = params["num_heads"]
        num_kv_heads = params["num_kv_heads"]
        head_dim = params["head_dim"]
        kv_seq = params["kv_seq"]
        block_size = params["block_size"]
        scale = 1.0 / math.sqrt(float(head_dim))
        context_lens = torch.tensor([kv_seq] * batch, dtype=torch.int32)
        args.out[:] = _compute_gqa_golden(
            args.query.reshape(batch, num_heads, head_dim),
            args.key_cache.reshape(-1, block_size, num_kv_heads, head_dim),
            args.value_cache.reshape(-1, block_size, num_kv_heads, head_dim),
            args.block_table,
            context_lens,
            scale,
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
