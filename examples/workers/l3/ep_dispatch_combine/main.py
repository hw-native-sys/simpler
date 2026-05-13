#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end 2-card moe_router + EP dispatch + moe_expert + combine demo.

A single orchestration runs four back-to-back stages over a shared HCCL window
scratch:

  moe_router kernels  FFN half-compress pre-mix (hc_pre) + RMSNorm + learned-
                      score gate + top-k + weight normalize. Produces x_norm,
                      indices, weights (plus post_ffn / comb_ffn for hc_post).
                      18 PyPTO-generated AIV kernels.
  dispatch.cpp        EP count exchange + 3-channel push (x BF16 / weight FP32 /
                      idx INT32) + per-channel stage-out + recv_count emission.
                      Now reads the *chip-produced* x_norm and indices, plus
                      host-packed w_padded / idx_padded.
  moe_expert kernels  DeepSeek-V4 decode MoE block — routed local experts
                      (per-tile A8 gate/up matmul → SwiGLU → routing-weight mul
                      → A8 requant → w2 matmul → recv_y) + shared expert
                      (x_local A8 → gate/up → SwiGLU → A8 → w2 → sh). 17
                      PyPTO-generated incore kernels (4 AIC matmuls + 13 AIV).
  combine.cpp         TPUT recv_y rows by recv_idx_out into routed_y_buf,
                      barrier, reduce_sum along TOPK -> routed_y FP32.

Dimensions mirror the ``DEMO`` decode config: D = hidden_size = 4096,
MOE_INTER = 4096, L = N_LOCAL_EXPERTS = 8, T = 16, TOPK = 2, R = RECV_MAX = 32,
HC_MULT = 4. INT8 expert weight banks + FP32 router fixtures are generated
randomly on host (shared across the two ranks). The host golden is
``golden_moe_router`` (ported from ``models/deepseek/v4/moe_router.py``) →
dispatch protocol replay → ``golden_moe_expert`` → combine reduce.

Run:

    python examples/workers/l3/ep_dispatch_combine/main.py -p a2a3 -d 0-1
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from simpler.task_interface import (  # noqa: E402
    ArgDirection,
    CallConfig,
    ChipBootstrapConfig,
    ChipBufferSpec,
    ChipCallable,
    ChipCommBootstrapConfig,
    ChipContext,
    ContinuousTensor,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker  # noqa: E402

from simpler_setup.elf_parser import extract_text_section  # noqa: E402
from simpler_setup.kernel_compiler import KernelCompiler  # noqa: E402
from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: E402
from simpler_setup.torch_interop import make_tensor_arg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = "tensormap_and_ringbuffer"

# Demo dimensions — mirror the ``DEMO`` decode config and the constants at the
# top of dispatch.cpp / combine.cpp.
N_RANKS = 2
B = 16  # DECODE_BATCH
S = 1  # DECODE_SEQ
T = B * S  # tokens per rank
TOPK = 2
D = 4096  # hidden_size
L = 8  # N_LOCAL_EXPERTS per rank
N_EXPERTS = L  # global experts in moe_expert's view (its JIT was compiled with EP_WORLD_SIZE=1)
R = 32  # RECV_MAX
MOE_INTER = 4096
SWIGLU_LIMIT = 0.0
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
W_PAD = 8
IDX_PAD = 8
E_GLOBAL = N_RANKS * L  # global EP expert count
N_ROUTES = T * TOPK

# Router constants — must match the ``DEMO`` config the router was JIT'd against.
HC_MULT = 4
HC_DIM = HC_MULT * D  # 16384
MIX_HC = (2 + HC_MULT) * HC_MULT  # 24
HC_SINKHORN_ITER = 20
HC_EPS = 1e-6
NORM_EPS = 1e-6
VOCAB = 129280  # vocab_size (tid2eid is unused at LAYER_ID >= N_HASH_LAYERS; still must allocate)
ROUTE_SCALE = 1.0
LAYER_ID = 1
N_HASH_LAYERS = 0  # LAYER_ID >= N_HASH_LAYERS -> learned-score path

# Window region byte sizes — mirror k*Bytes / kOff* in dispatch.cpp / combine.cpp.
PUB_COUNTS_BYTES = N_RANKS * N_RANKS * L * 4
SIGNAL_BYTES = 64
RECV_X_BYTES = L * R * D * 2
RECV_W_BYTES = L * R * W_PAD * 4
RECV_IDX_BYTES = L * R * IDX_PAD * 4
ROUTED_Y_BUF_BYTES = T * TOPK * D * 2
SCRATCH_NBYTES = (
    PUB_COUNTS_BYTES + SIGNAL_BYTES + RECV_X_BYTES + RECV_W_BYTES + RECV_IDX_BYTES
    + SIGNAL_BYTES + ROUTED_Y_BUF_BYTES + SIGNAL_BYTES
)

# Kernel set: 18 router (0..17), dispatch (18), 17 moe_expert (19..35), combine (36).
# func_id matches rt_submit_*_task calls in ep_dispatch_combine_orch.cpp.
KERNELS: list[tuple[int, str, str, str]] = [
    (0, "cast_x", "kernels/aiv/cast_x.cpp", "aiv"),
    (1, "rms", "kernels/aiv/rms.cpp", "aiv"),
    (2, "linear", "kernels/aiv/linear.cpp", "aiv"),
    (3, "split_pre_post", "kernels/aiv/split_pre_post.cpp", "aiv"),
    (4, "split_pre_post_0", "kernels/aiv/split_pre_post_0.cpp", "aiv"),
    (5, "split_pre_post_1", "kernels/aiv/split_pre_post_1.cpp", "aiv"),
    (6, "split_pre_post_2", "kernels/aiv/split_pre_post_2.cpp", "aiv"),
    (7, "comb_sinkhorn", "kernels/aiv/comb_sinkhorn.cpp", "aiv"),
    (8, "write_post", "kernels/aiv/write_post.cpp", "aiv"),
    (9, "mix_x", "kernels/aiv/mix_x.cpp", "aiv"),
    (10, "ffn_norm_rms", "kernels/aiv/ffn_norm_rms.cpp", "aiv"),
    (11, "ffn_norm_apply", "kernels/aiv/ffn_norm_apply.cpp", "aiv"),
    (12, "gate_dot", "kernels/aiv/gate_dot.cpp", "aiv"),
    (13, "gate_dot_0", "kernels/aiv/gate_dot_0.cpp", "aiv"),
    (14, "route_sort_top2", "kernels/aiv/route_sort_top2.cpp", "aiv"),
    (15, "route_extract_top2", "kernels/aiv/route_extract_top2.cpp", "aiv"),
    (16, "route_normalize_weights", "kernels/aiv/route_normalize_weights.cpp", "aiv"),
    (17, "write_route_outputs", "kernels/aiv/write_route_outputs.cpp", "aiv"),
    (18, "dispatch", "kernels/aiv/dispatch.cpp", "aiv"),
    (19, "x_local_q", "kernels/aiv/x_local_q.cpp", "aiv"),
    (20, "recv_x_q", "kernels/aiv/recv_x_q.cpp", "aiv"),
    (21, "exp_gate_up_matmul", "kernels/aic/exp_gate_up_matmul.cpp", "aic"),
    (22, "exp_gate_up_dequant", "kernels/aiv/exp_gate_up_dequant.cpp", "aiv"),
    (23, "exp_swiglu", "kernels/aiv/exp_swiglu.cpp", "aiv"),
    (24, "exp_swiglu_mask", "kernels/aiv/exp_swiglu_mask.cpp", "aiv"),
    (25, "exp_h_q", "kernels/aiv/exp_h_q.cpp", "aiv"),
    (26, "exp_w2_matmul", "kernels/aic/exp_w2_matmul.cpp", "aic"),
    (27, "exp_w2_dequant", "kernels/aiv/exp_w2_dequant.cpp", "aiv"),
    (28, "exp_recv_y_write", "kernels/aiv/exp_recv_y_write.cpp", "aiv"),
    (29, "sh_gate_up_matmul", "kernels/aic/sh_gate_up_matmul.cpp", "aic"),
    (30, "sh_gate_up_dequant", "kernels/aiv/sh_gate_up_dequant.cpp", "aiv"),
    (31, "sh_swiglu", "kernels/aiv/sh_swiglu.cpp", "aiv"),
    (32, "sh_h_q", "kernels/aiv/sh_h_q.cpp", "aiv"),
    (33, "sh_w2_matmul", "kernels/aic/sh_w2_matmul.cpp", "aic"),
    (34, "sh_w2_dequant", "kernels/aiv/sh_w2_dequant.cpp", "aiv"),
    (35, "sh_write", "kernels/aiv/sh_write.cpp", "aiv"),
    (36, "combine", "kernels/aiv/combine.cpp", "aiv"),
]


def parse_device_range(spec: str) -> list[int]:
    if "," in spec:
        ids = [int(x) for x in spec.split(",")]
    elif "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        ids = list(range(lo, hi + 1))
    else:
        ids = [int(spec)]
    if len(ids) != N_RANKS:
        raise ValueError(f"ep_dispatch_combine needs exactly {N_RANKS} devices, got {ids}")
    return ids


def build_chip_callable(platform: str, pto_isa_commit: str | None) -> ChipCallable:
    """Compile the 18 router + dispatch + 17 moe_expert + combine kernels and
    the merged C++ orchestration into a single ChipCallable. ccec runs in a
    thread pool because the AIC matmuls alone take real time."""
    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root(commit=pto_isa_commit, clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(RUNTIME)
    is_sim = platform.endswith("sim")
    # dispatch.cpp uses `dcci(...)` (via aicore/aicore.h) to invalidate the
    # D-cache before its scalar read of the router-written `indices`. aicore.h
    # transitively includes "inner_kernel.h" which lives under the platform-
    # specific aicore subdir, so plumb that in.
    arch = "a2a3" if platform.startswith("a2a3") else "a5"
    inner_kernel_subdir = "sim" if is_sim else "onboard"
    kernel_include_dirs = list(include_dirs) + [
        str(kc.project_root / "src" / "common"),
        str(kc.project_root / "src" / arch / "platform" / inner_kernel_subdir / "aicore"),
    ]

    def compile_one(rel_src: str, core_type: str) -> bytes:
        b = kc.compile_incore(
            source_path=os.path.join(HERE, rel_src),
            core_type=core_type,
            pto_isa_root=pto_isa_root,
            extra_include_dirs=kernel_include_dirs,
        )
        if not is_sim:
            b = extract_text_section(b)
        return b

    with ThreadPoolExecutor(max_workers=min(8, len(KERNELS))) as ex:
        futs = {fid: ex.submit(compile_one, src, ct) for (fid, _name, src, ct) in KERNELS}
        bins = {fid: f.result() for fid, f in futs.items()}

    orch_bytes = kc.compile_orchestration(
        runtime_name=RUNTIME,
        source_path=os.path.join(HERE, "kernels/orchestration/ep_dispatch_combine_orch.cpp"),
    )

    # dispatch / combine keep explicit per-child sigs (they're hand-written
    # comm kernels). PyPTO-generated incore kernels get an empty signature like
    # the other generated examples (dependency tracking lives in the orch's
    # add_input / add_output / add_inout calls).
    sig_dispatch = [
        ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.IN,
        ArgDirection.OUT, ArgDirection.OUT, ArgDirection.OUT, ArgDirection.OUT,
        ArgDirection.INOUT,
    ]
    sig_combine = [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT]
    children: list[tuple[int, CoreCallable]] = []
    for fid, name, _src, _ct in KERNELS:
        sig = sig_dispatch if name == "dispatch" else sig_combine if name == "combine" else []
        children.append((fid, CoreCallable.build(signature=sig, binary=bins[fid])))

    # Orchestration arg view: 24 INs (x_hc..shared_w2_scale), 12 OUTPUT_EXISTING
    # (x_norm..routed_y), 1 INOUT scratch — 37 tensors + 2 scalars.
    sig_orch = [ArgDirection.IN] * 24 + [ArgDirection.OUT] * 12 + [ArgDirection.INOUT]

    return ChipCallable.build(
        signature=sig_orch,
        func_name="ep_dispatch_combine_orchestration",
        config_name="ep_dispatch_combine_orchestration_config",
        binary=orch_bytes,
        children=children,
    )


# --------------------------------------------------------------------------- #
# Host golden: moe_router (hc_pre + RMSNorm + learned-score gate + top-k)
# Ported from models/deepseek/v4/{hc_pre,moe_router}.py to keep this example
# self-contained (the pypto-lib model code isn't installed alongside the wheel).
# --------------------------------------------------------------------------- #
def _golden_hc_pre(x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base):
    """Half-compress pre-mix: produces (x_mixed BF16 [B,S,D], post_ffn FP32
    [B,S,HC_MULT], comb_ffn FP32 [B,S,HC_MULT,HC_MULT])."""
    x = x_hc.float()
    hc_fn = hc_ffn_fn.float()
    hc_scale = hc_ffn_scale.float()
    hc_base = hc_ffn_base.float()

    x_flat = x.flatten(2).reshape(T, HC_DIM)  # [T, HC_MULT*D]
    sq_sum = (x_flat * x_flat).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum / HC_DIM + NORM_EPS)
    mixes = (x_flat @ hc_fn.T) * rsqrt  # [T, MIX_HC]
    mixes = mixes.reshape(B, S, MIX_HC)

    pre = torch.sigmoid(mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post_t = 2 * torch.sigmoid(
        mixes[..., HC_MULT : HC_MULT * 2] * hc_scale[1] + hc_base[HC_MULT : HC_MULT * 2]
    )
    comb_t = (mixes[..., HC_MULT * 2 :] * hc_scale[2] + hc_base[HC_MULT * 2 :]).view(B, S, HC_MULT, HC_MULT)
    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    y = torch.zeros(B, S, D, dtype=torch.float32)
    for h in range(HC_MULT):
        y += x[:, :, h, :] * pre[:, :, h : h + 1]
    return y.to(torch.bfloat16), post_t, comb_t


def golden_moe_router(x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_w, gate_w, gate_bias):
    """Router golden — returns x_norm [T,D] BF16, indices [T,TOPK] INT32,
    weights [T,TOPK] FP32, post_ffn [B,S,HC_MULT] FP32, comb_ffn [B,S,HC_MULT,HC_MULT] FP32."""
    x_mixed, post_ffn, comb_ffn = _golden_hc_pre(x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)

    # FFN RMSNorm (returns BF16 to match the chip's cast-to-BF16 store).
    norm_w_f = norm_w.float()
    x_f = x_mixed.float()
    var = x_f.square().mean(-1, keepdim=True)
    x_n = x_f * torch.rsqrt(var + NORM_EPS)
    x_norm = (norm_w_f * x_n).to(torch.bfloat16).view(T, D)

    # Learned routing scores + top-k.
    scores = F.softplus(x_norm.float() @ gate_w.float().T).sqrt()  # [T, N_EXPERTS]
    biased = scores + gate_bias.float()
    indices = biased.topk(TOPK, dim=-1).indices.to(torch.int32)  # [T, TOPK]
    weights = scores.gather(1, indices.long())  # [T, TOPK] FP32
    weights = weights / weights.sum(dim=-1, keepdim=True) * ROUTE_SCALE

    # NOTE: the router writes `indices` modulo N_EXPERTS = L (its JIT was built
    # with EP_WORLD_SIZE=1 so it only addresses its own 8 local experts). The
    # dispatch protocol below treats them as *global* IDs in [0, E_GLOBAL), so
    # rebroadcast across ranks by mixing in the rank id at host fixture time.
    return x_norm, indices, weights.float(), post_ffn, comb_ffn


def build_router_inputs(seed: int):
    """Per-rank x_hc / input_ids plus shared FFN-norm + gate + hc_pre fixtures."""
    gen = torch.Generator().manual_seed(seed)
    x_hcs = [(torch.randn(B, S, HC_MULT, D, generator=gen) * 0.1).to(torch.bfloat16).share_memory_()
             for _ in range(N_RANKS)]
    hc_ffn_fn = (torch.randn(MIX_HC, HC_DIM, generator=gen) / HC_DIM**0.5).contiguous().share_memory_()
    hc_ffn_scale = (torch.ones(3, dtype=torch.float32) * 0.5).contiguous().share_memory_()
    hc_ffn_base = torch.zeros(MIX_HC, dtype=torch.float32).contiguous().share_memory_()
    norm_w = torch.ones(D, dtype=torch.float32).contiguous().share_memory_()
    # gate_w shaped [N_EXPERTS, D] — N_EXPERTS = L = 8 (the moe_expert JIT was
    # compiled with EP_WORLD_SIZE=1, so the router only addresses local experts).
    gate_w = (torch.randn(L, D, generator=gen) / D**0.5).contiguous().share_memory_()
    gate_bias = torch.zeros(L, dtype=torch.float32).contiguous().share_memory_()
    # tid2eid / input_ids are unused at LAYER_ID >= N_HASH_LAYERS=0 but must
    # still be passed (the orch binds the slots).
    tid2eid = torch.randint(0, L, (VOCAB, TOPK), generator=gen, dtype=torch.int32).contiguous().share_memory_()
    input_ids_list = [torch.randint(0, VOCAB, (B, S), generator=gen, dtype=torch.int64).contiguous().share_memory_()
                      for _ in range(N_RANKS)]
    return {
        "x_hcs": x_hcs,
        "hc_ffn_fn": hc_ffn_fn,
        "hc_ffn_scale": hc_ffn_scale,
        "hc_ffn_base": hc_ffn_base,
        "norm_w": norm_w,
        "gate_w": gate_w,
        "gate_bias": gate_bias,
        "tid2eid": tid2eid,
        "input_ids_list": input_ids_list,
    }


# --------------------------------------------------------------------------- #
# Routing / dispatch host model
# --------------------------------------------------------------------------- #
def _route_dst(src_rank: int, k: int) -> int:
    """EP routing policy mirrored from dispatch.cpp: route slot k from rank
    src_rank goes to peer (src_rank + k) % N_RANKS. The router's chip-produced
    `indices` are local expert IDs in [0, L); the rank component is layered on
    here so the demo spreads tokens across both ranks."""
    return (src_rank + k) % N_RANKS


def compute_dispatch_golden(x_norms, indices_local, weights):
    """Replay the dispatch protocol on host. ``indices_local`` is per-rank
    [N_RANKS, T, TOPK] of local expert IDs in [0, L); destination rank is
    derived from (src, k) per the EP policy above."""
    expected_recv_x = [torch.zeros(L, R, D, dtype=torch.bfloat16) for _ in range(N_RANKS)]
    expected_recv_w = [torch.zeros(L, R, dtype=torch.float32) for _ in range(N_RANKS)]
    expected_recv_idx = [torch.zeros(L, R, dtype=torch.int32) for _ in range(N_RANKS)]
    expected_count = [torch.zeros(L, dtype=torch.int32) for _ in range(N_RANKS)]
    route_dest = [[[None] * TOPK for _ in range(T)] for _ in range(N_RANKS)]

    send_counts = torch.zeros(N_RANKS, N_RANKS, L, dtype=torch.int32)
    for src in range(N_RANKS):
        for t in range(T):
            for k in range(TOPK):
                loc_e = int(indices_local[src][t, k].item())
                send_counts[src, _route_dst(src, k), loc_e] += 1

    for dst in range(N_RANKS):
        slot_offset = torch.zeros(N_RANKS, L, dtype=torch.int32)
        running = torch.zeros(L, dtype=torch.int32)
        for src in range(N_RANKS):
            slot_offset[src] = running.clone()
            running = running + send_counts[src, dst]

        for src in range(N_RANKS):
            cursor = torch.zeros(L, dtype=torch.int32)
            for t in range(T):
                for k in range(TOPK):
                    if _route_dst(src, k) != dst:
                        continue
                    loc_e = int(indices_local[src][t, k].item())
                    slot = int(slot_offset[src, loc_e].item() + cursor[loc_e].item())
                    cursor[loc_e] += 1
                    expected_recv_x[dst][loc_e, slot, :] = x_norms[src][t, :]
                    expected_recv_w[dst][loc_e, slot] = weights[src][t, k]
                    expected_recv_idx[dst][loc_e, slot] = t * TOPK + k
                    route_dest[src][t][k] = (dst, loc_e, slot)

        for e in range(L):
            expected_count[dst][e] = int(running[e].item())

    return expected_recv_x, expected_recv_w, expected_recv_idx, expected_count, route_dest


def pack_weights_padded(weights_row: torch.Tensor) -> torch.Tensor:
    """[T*TOPK, W_PAD] FP32 where row r = (weight_value, 0, …, 0)."""
    out = torch.zeros(N_ROUTES, W_PAD, dtype=torch.float32)
    for t in range(T):
        for k in range(TOPK):
            out[t * TOPK + k, 0] = weights_row[t, k]
    return out


def pack_idx_padded() -> torch.Tensor:
    """[T*TOPK, IDX_PAD] INT32 where row r = (r, 0, …, 0)."""
    out = torch.zeros(N_ROUTES, IDX_PAD, dtype=torch.int32)
    for t in range(T):
        for k in range(TOPK):
            out[t * TOPK + k, 0] = t * TOPK + k
    return out


# --------------------------------------------------------------------------- #
# moe_expert host golden (ported from models/deepseek/v4/moe_expert.py)
# --------------------------------------------------------------------------- #
def _round_half_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def _int8_quant_per_row(x: torch.Tensor):
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    out_i8 = _round_half_away_from_zero(rows * scale_quant).to(torch.int32).to(torch.float16).to(torch.int8)
    return out_i8.reshape_as(x), (1.0 / scale_quant).reshape(*x.shape[:-1], 1)


def _quant_w_per_channel(w: torch.Tensor):
    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    w_i8 = (_round_half_away_from_zero(w.float() * scale_quant.unsqueeze(-1))
            .to(torch.int32).to(torch.float16).to(torch.int8))
    return w_i8, (1.0 / scale_quant).float()


def build_expert_weights(seed: int):
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}

    def _store(name: str, w_bf16: torch.Tensor) -> None:
        w_i8, w_s = _quant_w_per_channel(w_bf16)
        out[name] = w_i8.contiguous().share_memory_()
        out[name + "_scale"] = w_s.contiguous().share_memory_()

    _store("expert_w1", (torch.randn(L, MOE_INTER, D, generator=gen) / D**0.5).to(torch.bfloat16))
    _store("expert_w3", (torch.randn(L, MOE_INTER, D, generator=gen) / D**0.5).to(torch.bfloat16))
    _store("expert_w2", (torch.randn(L, D, MOE_INTER, generator=gen) / MOE_INTER**0.5).to(torch.bfloat16))
    _store("shared_w1", (torch.randn(MOE_INTER, D, generator=gen) / D**0.5).to(torch.bfloat16))
    _store("shared_w3", (torch.randn(MOE_INTER, D, generator=gen) / D**0.5).to(torch.bfloat16))
    _store("shared_w2", (torch.randn(D, MOE_INTER, generator=gen) / MOE_INTER**0.5).to(torch.bfloat16))
    return out


def _dequant_w(w_i8: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    return w_i8.to(torch.float32) * w_scale.unsqueeze(-1)


def golden_moe_expert(recv_x, recv_weights, recv_count, x_local, w):
    """Torch reference for one rank's moe_expert call (see moe_expert.py)."""
    recv_x = recv_x.float()
    recv_weights = recv_weights.float()
    x_local = x_local.float()
    w1 = _dequant_w(w["expert_w1"], w["expert_w1_scale"].float())
    w3 = _dequant_w(w["expert_w3"], w["expert_w3_scale"].float())
    w2 = _dequant_w(w["expert_w2"], w["expert_w2_scale"].float())
    sw1 = _dequant_w(w["shared_w1"], w["shared_w1_scale"].float())
    sw3 = _dequant_w(w["shared_w3"], w["shared_w3_scale"].float())
    sw2 = _dequant_w(w["shared_w2"], w["shared_w2_scale"].float())

    x_local_i8, x_local_sd = _int8_quant_per_row(x_local)
    x_local_q = x_local_i8.float() * x_local_sd

    recv_y = torch.zeros(L, R, D, dtype=torch.float32)
    for e in range(L):
        n_rows = int(recv_count[e].item())
        if n_rows == 0:
            continue
        x_sub = recv_x[e, :n_rows, :]
        w_sub = recv_weights[e, :n_rows]

        x_sub_i8, x_sub_sd = _int8_quant_per_row(x_sub)
        x_sub_q = x_sub_i8.float() * x_sub_sd

        gate = x_sub_q @ w1[e].T
        up = x_sub_q @ w3[e].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        h = h * w_sub.unsqueeze(-1)
        h_i8, h_sd = _int8_quant_per_row(h)
        h = h_i8.float() * h_sd
        recv_y[e, :n_rows, :] = h @ w2[e].T

    sh_gate = x_local_q @ sw1.T
    sh_up = x_local_q @ sw3.T
    if SWIGLU_LIMIT > 0:
        sh_gate = sh_gate.clamp(max=SWIGLU_LIMIT)
        sh_up = sh_up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    sh_h = F.silu(sh_gate) * sh_up
    sh_h_i8, sh_h_sd = _int8_quant_per_row(sh_h)
    sh_h = sh_h_i8.float() * sh_h_sd
    sh = sh_h @ sw2.T

    return recv_y.to(torch.bfloat16), sh.to(torch.bfloat16)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _verify_router_outputs(
    nranks, x_norm_goldens, indices_goldens, weights_goldens,
    x_norm_outs, indices_outs, weights_outs,
) -> bool:
    ok = True
    for r in range(nranks):
        got_xn = x_norm_outs[r].float()
        exp_xn = x_norm_goldens[r].float()
        d_xn = (got_xn - exp_xn).abs()
        print(f"[ep_dispatch] chip {r}: router x_norm max|diff|={float(d_xn.max()):.3e} (BF16 tol 1e-2)")
        if not torch.allclose(got_xn, exp_xn, rtol=1e-2, atol=1e-2):
            ok = False
            print(f"[ep_dispatch] chip {r}: x_norm mismatch")

        # indices: rerank locally so chip[T,TOPK]'s local IDs in [0,L) align
        # with the golden (the host extended them by rank for dispatch; the
        # chip writes raw local IDs).
        got_idx = indices_outs[r]
        exp_idx_local = indices_goldens[r]  # already local IDs (pre-rerank)
        # Compare the (sorted) set of (index, weight) pairs per row to absorb
        # the tie-break order differences moe_router.py's topk_pair_compare
        # handles (sort32 vs torch.topk).
        got_pairs = sorted(zip(got_idx[r_t].tolist(), weights_outs[r][r_t].tolist())
                           for r_t in range(0))  # placeholder; we'll do it per row below
        idx_mismatches = 0
        for t in range(T):
            got_set = sorted((int(got_idx[t, k].item()), round(float(weights_outs[r][t, k].item()), 4)) for k in range(TOPK))
            exp_set = sorted((int(exp_idx_local[t, k].item()), round(float(weights_goldens[r][t, k].item()), 4)) for k in range(TOPK))
            # match expert set; weights within fp32 tol
            if [s[0] for s in got_set] != [s[0] for s in exp_set]:
                idx_mismatches += 1
                if idx_mismatches <= 3:
                    print(f"[ep_dispatch] chip {r} token {t}: indices got={[s[0] for s in got_set]} exp={[s[0] for s in exp_set]}")
        if idx_mismatches > 0:
            ok = False

        got_w = weights_outs[r]
        exp_w = weights_goldens[r]
        d_w = (got_w - exp_w).abs()
        print(f"[ep_dispatch] chip {r}: router weights max|diff|={float(d_w.max()):.3e}")
        if not torch.allclose(got_w, exp_w, rtol=1e-3, atol=1e-3):
            ok = False
            print(f"[ep_dispatch] chip {r}: weights mismatch")
    return ok


def _verify_recv_outputs(
    nranks, expected_count, expected_recv_x, expected_recv_w, expected_recv_idx,
    recv_count_outs, recv_x_outs, recv_w_outs, recv_idx_outs,
) -> bool:
    """dispatch outputs vs the protocol replay. recv_x now compares with BF16
    tolerance — its source x_norm goes through router rounding too."""
    ok = True
    for r in range(nranks):
        cnt = expected_count[r]
        print(f"[ep_dispatch] chip {r}: expected counts per expert = {cnt.tolist()}")
        got_count = recv_count_outs[r].squeeze(-1)
        if (got_count - cnt).abs().max().item() != 0:
            ok = False
            print(f"[ep_dispatch] chip {r}: recv_count mismatch got={got_count.tolist()} expected={cnt.tolist()}")
        for e in range(L):
            n = int(cnt[e].item())
            if n == 0:
                continue
            got_x = recv_x_outs[r][e, :n, :].float()
            exp_x = expected_recv_x[r][e, :n, :].float()
            x_diff = (got_x - exp_x).abs().max().item()
            w_diff = (recv_w_outs[r][e, :n] - expected_recv_w[r][e, :n]).abs().max().item()
            idx_diff = (recv_idx_outs[r][e, :n] - expected_recv_idx[r][e, :n]).abs().max().item()
            if (x_diff > 5e-2) or (w_diff > 1e-3) or (idx_diff != 0):
                ok = False
                print(f"[ep_dispatch] chip {r} expert {e}: cnt={n} x_diff={x_diff:.3e} w_diff={w_diff:.3e} idx_diff={idx_diff}")
    return ok


def _verify_expert_outputs(nranks, recv_y_goldens, sh_goldens, expected_count, recv_y_outs, sh_outs) -> bool:
    ok = True
    # Loose tolerance: chip x_norm differs from golden by up to 1 BF16 ulp
    # (~1.5e-2 in this fixture), then the INT8 per-row quant amax shifts,
    # compounded across the gate/up matmul → SwiGLU → A8 requant → w2 chain.
    rtol, atol = 1e-2, 5e-2
    for r in range(nranks):
        for e in range(L):
            n = int(expected_count[r][e].item())
            if n == 0:
                continue
            got = recv_y_outs[r][e, :n, :].float()
            exp = recv_y_goldens[r][e, :n, :].float()
            if not torch.allclose(got, exp, rtol=rtol, atol=atol):
                ok = False
                diff = (got - exp).abs()
                print(f"[ep_dispatch] chip {r} expert {e}: recv_y mismatch n={n} max|diff|={float(diff.max()):.3e}")
        got_sh, exp_sh = sh_outs[r].float(), sh_goldens[r].float()
        d_sh = (got_sh - exp_sh).abs()
        print(f"[ep_dispatch] chip {r}: sh max|diff|={float(d_sh.max()):.3e}")
        if not torch.allclose(got_sh, exp_sh, rtol=rtol, atol=atol):
            ok = False
            print(f"[ep_dispatch] chip {r}: sh mismatch")
    return ok


def _verify_routed_y(nranks, route_dest, recv_y_goldens, routed_y_outs) -> bool:
    ok = True
    # routed_y is the FP32 sum of TOPK BF16 recv_y rows; each row carries the
    # same kind of compounded noise we tolerate in _verify_expert_outputs, so
    # propagate that bound (TOPK * 5e-2 plus a small FP32 reorder buffer).
    atol = TOPK * 5e-2 + 1e-2
    rtol = 1e-2
    for me in range(nranks):
        expected = torch.zeros(T, D, dtype=torch.float32)
        for t in range(T):
            for k in range(TOPK):
                dst, loc_e, slot = route_dest[me][t][k]
                expected[t, :] += recv_y_goldens[dst][loc_e, slot, :].float()
        got = routed_y_outs[me]
        diff = (got - expected).abs()
        print(f"[ep_dispatch] chip {me}: routed_y max|diff|={float(diff.max()):.3e}")
        if not torch.allclose(got, expected, rtol=rtol, atol=atol):
            ok = False
            print(f"[ep_dispatch] chip {me}: routed_y mismatch (rtol={rtol}, atol={atol})")
    return ok


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(
    device_ids: list[int],
    platform: str = "a2a3",
    pto_isa_commit: str | None = None,
    build: bool = False,
    seed: int = 20260513,
) -> int:
    nranks = len(device_ids)
    assert nranks == N_RANKS

    window_size = max(SCRATCH_NBYTES, 128 * 1024)
    rootinfo_path = f"/tmp/pto_ep_dispatch_rootinfo_{os.getpid()}.bin"
    try:
        os.unlink(rootinfo_path)
    except FileNotFoundError:
        pass

    print(f"[ep_dispatch] platform={platform} devices={device_ids} nranks={nranks} seed={seed}")

    # ---- host fixtures: router inputs (per-rank x_hc / input_ids + shared FFN/gate banks) ----
    R_in = build_router_inputs(seed)
    # ---- moe_expert weight banks (shared across both ranks) ----
    weight_banks = build_expert_weights(seed=seed)

    # ---- host golden: run the router golden per rank to get x_norm / indices / weights ----
    print("[ep_dispatch] computing host golden (router -> dispatch replay -> moe_expert -> combine)...")
    x_norm_goldens: list[torch.Tensor] = []
    indices_goldens: list[torch.Tensor] = []  # local IDs in [0, L)
    weights_goldens: list[torch.Tensor] = []
    post_ffn_goldens: list[torch.Tensor] = []
    comb_ffn_goldens: list[torch.Tensor] = []
    for r in range(nranks):
        xn, idx_local, w, pf, cf = golden_moe_router(
            R_in["x_hcs"][r], R_in["hc_ffn_fn"], R_in["hc_ffn_scale"], R_in["hc_ffn_base"],
            R_in["norm_w"], R_in["gate_w"], R_in["gate_bias"],
        )
        x_norm_goldens.append(xn)
        indices_goldens.append(idx_local)
        weights_goldens.append(w)
        post_ffn_goldens.append(pf)
        comb_ffn_goldens.append(cf)

    # The router's `indices` are local IDs in [0, L); destination rank is
    # derived from (src_rank, k) per the EP routing policy that dispatch.cpp
    # and compute_dispatch_golden share. No host-side rebroadcast needed.
    (expected_recv_x, expected_recv_w, expected_recv_idx, expected_count, route_dest) = \
        compute_dispatch_golden(x_norm_goldens, indices_goldens, weights_goldens)

    # ---- moe_expert per-rank golden (recv_y, sh) ----
    recv_y_goldens, sh_goldens = [], []
    for r in range(nranks):
        ry, shg = golden_moe_expert(
            expected_recv_x[r], expected_recv_w[r], expected_count[r], x_norm_goldens[r], weight_banks,
        )
        recv_y_goldens.append(ry)
        sh_goldens.append(shg)

    # ---- pack dispatch host inputs (w_padded / idx_padded) from the GOLDEN weights/indices ----
    # The chip's router produces near-identical weights modulo fp32 rounding; we
    # feed the host-known versions through dispatch so recv_w / recv_idx stay
    # deterministic.
    w_padded_list = [pack_weights_padded(weights_goldens[r]).share_memory_() for r in range(nranks)]
    idx_padded_list = [pack_idx_padded().share_memory_() for _ in range(nranks)]
    recv_count_host_list = [expected_count[r].reshape(L, 1).clone().contiguous().share_memory_()
                            for r in range(nranks)]

    # ---- chip output / cross-stage tensors (zero-init, OUTPUT_EXISTING) ----
    x_norm_outs = [torch.zeros(T, D, dtype=torch.bfloat16).share_memory_() for _ in range(nranks)]
    indices_outs = [torch.zeros(T, TOPK, dtype=torch.int32).share_memory_() for _ in range(nranks)]
    weights_outs = [torch.zeros(T, TOPK, dtype=torch.float32).share_memory_() for _ in range(nranks)]
    post_ffn_outs = [torch.zeros(B, S, HC_MULT, dtype=torch.float32).share_memory_() for _ in range(nranks)]
    comb_ffn_outs = [torch.zeros(B, S, HC_MULT, HC_MULT, dtype=torch.float32).share_memory_() for _ in range(nranks)]
    recv_x_outs = [torch.zeros(L, R, D, dtype=torch.bfloat16).share_memory_() for _ in range(nranks)]
    recv_w_outs = [torch.zeros(L, R, dtype=torch.float32).share_memory_() for _ in range(nranks)]
    recv_idx_outs = [torch.zeros(L, R, dtype=torch.int32).share_memory_() for _ in range(nranks)]
    recv_count_outs = [torch.zeros(L, 1, dtype=torch.int32).share_memory_() for _ in range(nranks)]
    recv_y_outs = [torch.zeros(L, R, D, dtype=torch.bfloat16).share_memory_() for _ in range(nranks)]
    sh_outs = [torch.zeros(T, D, dtype=torch.bfloat16).share_memory_() for _ in range(nranks)]
    routed_y_outs = [torch.zeros(T, D, dtype=torch.float32).share_memory_() for _ in range(nranks)]

    cfgs = [
        ChipBootstrapConfig(
            comm=ChipCommBootstrapConfig(
                rank=rank, nranks=nranks, rootinfo_path=rootinfo_path, window_size=window_size,
            ),
            buffers=[ChipBufferSpec(name="scratch", dtype="float32", count=SCRATCH_NBYTES // 4, nbytes=SCRATCH_NBYTES)],
        )
        for rank in range(nranks)
    ]

    print(f"[ep_dispatch] compiling orchestration + {len(KERNELS)} kernels for {platform}...")
    chip_callable = build_chip_callable(platform, pto_isa_commit)

    worker = Worker(
        level=3, platform=platform, runtime=RUNTIME, device_ids=device_ids,
        num_sub_workers=0, chip_bootstrap_configs=cfgs, build=build,
    )
    chip_cid = worker.register(chip_callable)

    cfg = CallConfig()
    cfg.block_dim = 24
    cfg.aicpu_thread_num = 4
    # Swimlane / L2 perf trace — each chip writes <output_prefix>/l2_perf_records.json,
    # so per-chip dirs (chip-0 / chip-1) are required to avoid the second rank
    # overwriting the first. We also emit a func_names.json mapping
    # func_id -> kernel name so swimlane_converter can label tasks instead of
    # showing the default ``func_xx`` placeholders.
    swimlane_base = None
    if os.environ.get("EP_SWIMLANE", "") == "1":
        swimlane_base = os.environ.get("EP_SWIMLANE_DIR", os.path.join(HERE, "outputs", "swimlane"))
        os.makedirs(swimlane_base, exist_ok=True)
        with open(os.path.join(swimlane_base, "func_names.json"), "w") as f:
            import json as _json
            _json.dump({
                "orchestrator_name": "ep_dispatch_combine_orchestration",
                "callable_id_to_name": {str(fid): name for (fid, name, _, _) in KERNELS},
            }, f, indent=2)
        print(f"[ep_dispatch] swimlane enabled, base={swimlane_base}")

    try:
        print("[ep_dispatch] init worker (forks chip children + bootstraps HCCL)...")
        worker.init()
        contexts: list[ChipContext] = worker.chip_contexts
        assert len(contexts) == nranks
        for i, ctx in enumerate(contexts):
            print(
                f"[ep_dispatch] chip {i}: device={ctx.device_id} rank={ctx.rank}/{ctx.nranks} "
                f"window=[0x{ctx.local_window_base:x} +{ctx.actual_window_size}B] scratch=0x{ctx.buffer_ptrs['scratch']:x}"
            )

        def orch_fn(orch, _args, _cfg):
            for i, ctx in enumerate(contexts):
                a = TaskArgs()
                # 9 router host inputs (0..8)
                a.add_tensor(make_tensor_arg(R_in["x_hcs"][i]), TensorArgType.INPUT)
                a.add_tensor(make_tensor_arg(R_in["hc_ffn_fn"]), TensorArgType.INPUT)
                a.add_tensor(make_tensor_arg(R_in["hc_ffn_scale"]), TensorArgType.INPUT)
                a.add_tensor(make_tensor_arg(R_in["hc_ffn_base"]), TensorArgType.INPUT)
                a.add_tensor(make_tensor_arg(R_in["norm_w"]), TensorArgType.INPUT)
                a.add_tensor(make_tensor_arg(R_in["gate_w"]), TensorArgType.INPUT)
                a.add_tensor(make_tensor_arg(R_in["gate_bias"]), TensorArgType.INPUT)
                a.add_tensor(make_tensor_arg(R_in["tid2eid"]), TensorArgType.INPUT)
                a.add_tensor(make_tensor_arg(R_in["input_ids_list"][i]), TensorArgType.INPUT)
                # 2 dispatch host inputs (9..10)
                a.add_tensor(make_tensor_arg(w_padded_list[i]), TensorArgType.INPUT)
                a.add_tensor(make_tensor_arg(idx_padded_list[i]), TensorArgType.INPUT)
                # 1 host-known recv count (11)
                a.add_tensor(make_tensor_arg(recv_count_host_list[i]), TensorArgType.INPUT)
                # 12 moe_expert weight tensors (12..23)
                for name in ("expert_w1", "expert_w1_scale", "expert_w3", "expert_w3_scale",
                             "expert_w2", "expert_w2_scale", "shared_w1", "shared_w1_scale",
                             "shared_w3", "shared_w3_scale", "shared_w2", "shared_w2_scale"):
                    a.add_tensor(make_tensor_arg(weight_banks[name]), TensorArgType.INPUT)
                # 12 chip OUTPUT_EXISTING tensors (24..35)
                a.add_tensor(make_tensor_arg(x_norm_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(indices_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(weights_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(post_ffn_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(comb_ffn_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(recv_x_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(recv_w_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(recv_idx_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(recv_count_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(recv_y_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(sh_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(routed_y_outs[i]), TensorArgType.OUTPUT_EXISTING)
                # scratch (36)
                a.add_tensor(
                    ContinuousTensor.make(
                        data=ctx.buffer_ptrs["scratch"], shapes=(SCRATCH_NBYTES // 4,),
                        dtype=DataType.FLOAT32, child_memory=True,
                    ),
                    TensorArgType.INOUT,
                )
                a.add_scalar(ctx.nranks)
                a.add_scalar(ctx.device_ctx)
                # Per-chip cfg so each rank's swimlane lands in its own subdir.
                cfg_i = CallConfig()
                cfg_i.block_dim = cfg.block_dim
                cfg_i.aicpu_thread_num = cfg.aicpu_thread_num
                if swimlane_base is not None:
                    cfg_i.enable_l2_swimlane = True
                    cfg_i.output_prefix = os.path.join(swimlane_base, f"chip-{i}")
                    os.makedirs(cfg_i.output_prefix, exist_ok=True)
                orch.submit_next_level(chip_cid, a, cfg_i, worker=i)

        print("[ep_dispatch] running 2-chip router + dispatch + moe_expert + combine DAG...")
        worker.run(orch_fn, args=None, config=cfg)

        ok = _verify_router_outputs(
            nranks, x_norm_goldens, indices_goldens, weights_goldens,
            x_norm_outs, indices_outs, weights_outs,
        )
        ok = _verify_recv_outputs(
            nranks, expected_count, expected_recv_x, expected_recv_w, expected_recv_idx,
            recv_count_outs, recv_x_outs, recv_w_outs, recv_idx_outs,
        ) and ok
        ok = _verify_expert_outputs(nranks, recv_y_goldens, sh_goldens, expected_count, recv_y_outs, sh_outs) and ok
        ok = _verify_routed_y(nranks, route_dest, recv_y_goldens, routed_y_outs) and ok

        if not ok:
            print("[ep_dispatch] golden check FAILED")
            return 1
        print("[ep_dispatch] all ranks matched golden ✅")
        return 0
    finally:
        worker.close()
        try:
            os.unlink(rootinfo_path)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--device", default="0-1", help="Device range, e.g. '0-1'. Two chips required.")
    parser.add_argument("-p", "--platform", default="a2a3", help="Platform backend.")
    parser.add_argument("--build", action="store_true", help="Rebuild runtime from source instead of using cached libs.")
    parser.add_argument("--pto-isa-commit", default=None, help="Optional PTO ISA commit/tag to fetch before compiling.")
    parser.add_argument("--seed", type=int, default=20260513, help="Seed for the random input fixture.")
    cli = parser.parse_args()
    return run(
        parse_device_range(cli.device), platform=cli.platform, pto_isa_commit=cli.pto_isa_commit,
        build=cli.build, seed=cli.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
