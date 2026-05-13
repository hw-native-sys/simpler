#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end 2-card EP dispatch + moe_expert + combine demo.

A single orchestration runs three back-to-back stages over a shared HCCL window
scratch:

  dispatch.cpp        count exchange + 3-channel push (x BF16 / weight FP32 /
                      idx INT32) + per-channel stage-out + recv_count emission
  moe_expert kernels  the production DeepSeek-V4 decode MoE block (routed local
                      experts: per-tile A8 gate/up matmul → dequant → SwiGLU →
                      routing-weight mul → A8 requant → w2 matmul → recv_y; plus
                      the shared expert: x_local A8 → gate/up → SwiGLU → A8 → w2
                      → sh). 17 PyPTO-generated incore kernels (4 AIC matmuls +
                      13 AIV) wired by the transplanted moe_expert orchestration.
  combine.cpp         TPUT recv_y rows by recv_idx_out into routed_y_buf
                      (relies on HCCL window zero-init), barrier, reduce_sum
                      along TOPK -> routed_y FP32

Dimensions mirror the moe_expert ``DEMO`` decode config: D = hidden_size = 4096,
MOE_INTER = 4096, L = N_LOCAL_EXPERTS = 8, T = decode tokens per rank = 16,
TOPK = 2, R = RECV_MAX = 32, INT8 weight banks generated like
``moe_expert.py::build_tensor_specs`` (shared across the two ranks). The host
golden is the dispatch protocol replay → ``golden_moe_expert`` (ported from
``models/deepseek/v4/moe_expert.py``) → combine reduce.

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

# Demo dimensions — mirror the moe_expert ``DEMO`` decode config and the
# constants at the top of dispatch.cpp / combine.cpp.
N_RANKS = 2
T = 16  # decode tokens per rank (DECODE_BATCH * DECODE_SEQ)
TOPK = 2
D = 4096  # hidden_size
L = 8  # N_LOCAL_EXPERTS per rank (n_routed_experts, EP_WORLD_SIZE=1)
R = 32  # RECV_MAX (per-(local-expert) receive upper bound)
MOE_INTER = 4096  # moe_intermediate_size
SWIGLU_LIMIT = 0.0
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
W_PAD = 8  # weight tile width — minimum vector tile (1x8 FP32 = 32 B)
IDX_PAD = 8  # idx tile width   — minimum vector tile (1x8 INT32 = 32 B)
E_GLOBAL = N_RANKS * L
N_ROUTES = T * TOPK

# Window region byte sizes — mirror k*Bytes / kOff* in dispatch.cpp / combine.cpp.
PUB_COUNTS_BYTES = N_RANKS * N_RANKS * L * 4  # N*N*L INT32
SIGNAL_BYTES = 64  # padded slot per signal area
RECV_X_BYTES = L * R * D * 2  # 2 MiB (BF16)
RECV_W_BYTES = L * R * W_PAD * 4  # 8  KB (FP32; weight at slot 0)
RECV_IDX_BYTES = L * R * IDX_PAD * 4  # 8  KB (INT32; r at slot 0)
ROUTED_Y_BUF_BYTES = T * TOPK * D * 2  # 256 KB (BF16; combine push dest)
SCRATCH_NBYTES = (
    PUB_COUNTS_BYTES
    + SIGNAL_BYTES  # count_done_sig
    + RECV_X_BYTES
    + RECV_W_BYTES
    + RECV_IDX_BYTES
    + SIGNAL_BYTES  # data_done_sig
    + ROUTED_Y_BUF_BYTES  # combine push destination
    + SIGNAL_BYTES  # combine_done_sig
)

# (func_id, name, source rel-path, core_type) — func_id 0 = dispatch, 1..17 =
# the moe_expert kernels (same order as the generated kernel_config.py, +1),
# 18 = combine. Matches the rt_submit_*_task ids in the merged orchestration.
KERNELS: list[tuple[int, str, str, str]] = [
    (0, "dispatch", "kernels/aiv/dispatch.cpp", "aiv"),
    (1, "x_local_q", "kernels/aiv/x_local_q.cpp", "aiv"),
    (2, "recv_x_q", "kernels/aiv/recv_x_q.cpp", "aiv"),
    (3, "exp_gate_up_matmul", "kernels/aic/exp_gate_up_matmul.cpp", "aic"),
    (4, "exp_gate_up_dequant", "kernels/aiv/exp_gate_up_dequant.cpp", "aiv"),
    (5, "exp_swiglu", "kernels/aiv/exp_swiglu.cpp", "aiv"),
    (6, "exp_swiglu_mask", "kernels/aiv/exp_swiglu_mask.cpp", "aiv"),
    (7, "exp_h_q", "kernels/aiv/exp_h_q.cpp", "aiv"),
    (8, "exp_w2_matmul", "kernels/aic/exp_w2_matmul.cpp", "aic"),
    (9, "exp_w2_dequant", "kernels/aiv/exp_w2_dequant.cpp", "aiv"),
    (10, "exp_recv_y_write", "kernels/aiv/exp_recv_y_write.cpp", "aiv"),
    (11, "sh_gate_up_matmul", "kernels/aic/sh_gate_up_matmul.cpp", "aic"),
    (12, "sh_gate_up_dequant", "kernels/aiv/sh_gate_up_dequant.cpp", "aiv"),
    (13, "sh_swiglu", "kernels/aiv/sh_swiglu.cpp", "aiv"),
    (14, "sh_h_q", "kernels/aiv/sh_h_q.cpp", "aiv"),
    (15, "sh_w2_matmul", "kernels/aic/sh_w2_matmul.cpp", "aic"),
    (16, "sh_w2_dequant", "kernels/aiv/sh_w2_dequant.cpp", "aiv"),
    (17, "sh_write", "kernels/aiv/sh_write.cpp", "aiv"),
    (18, "combine", "kernels/aiv/combine.cpp", "aiv"),
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
    """Compile dispatch + the 17 moe_expert incore kernels + combine and the
    merged C++ orchestration into a single ChipCallable.

    ccec invocations run in a thread pool — 19 kernels (several full-size AIC
    matmuls at D=MOE_INTER=4096) compile far too slowly one at a time.
    """
    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root(commit=pto_isa_commit, clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(RUNTIME)
    kernel_include_dirs = list(include_dirs) + [str(kc.project_root / "src" / "common")]
    is_sim = platform.endswith("sim")

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

    # Per-child tensor footprints matching the orch's Arg packing. The 17
    # moe_expert children are PyPTO-generated incore kernels — they get an empty
    # signature like the other generated examples (dependency tracking happens
    # via the orchestration's add_input / add_output / add_inout calls).
    sig_dispatch = [
        ArgDirection.IN,  # indices
        ArgDirection.IN,  # x_norm
        ArgDirection.IN,  # w_padded
        ArgDirection.IN,  # idx_padded
        ArgDirection.OUT,  # recv_x_out
        ArgDirection.OUT,  # recv_w_out
        ArgDirection.OUT,  # recv_idx_out
        ArgDirection.OUT,  # recv_count_out
        ArgDirection.INOUT,  # scratch
    ]
    sig_combine = [
        ArgDirection.IN,  # recv_y (reused as INPUT)
        ArgDirection.IN,  # recv_idx_out (reused as INPUT)
        ArgDirection.OUT,  # routed_y
        ArgDirection.INOUT,  # scratch
    ]
    children: list[tuple[int, CoreCallable]] = []
    for fid, name, _src, _ct in KERNELS:
        sig = sig_dispatch if name == "dispatch" else sig_combine if name == "combine" else []
        children.append((fid, CoreCallable.build(signature=sig, binary=bins[fid])))

    # Orchestration arg view: 4 dispatch INs + recv_count_host IN + 6 expert
    # weight tensors + 6 shared weight tensors (17 INs), then 7 OUTPUT_EXISTING
    # (recv_x_out / recv_w_out / recv_idx_out / recv_count_out / recv_y / sh /
    # routed_y), then the INOUT scratch window — 25 tensors + 2 scalars.
    sig_orch = [ArgDirection.IN] * 17 + [ArgDirection.OUT] * 7 + [ArgDirection.INOUT]

    return ChipCallable.build(
        signature=sig_orch,
        func_name="ep_dispatch_combine_orchestration",
        config_name="ep_dispatch_combine_orchestration_config",
        binary=orch_bytes,
        children=children,
    )


# --------------------------------------------------------------------------- #
# Routing / dispatch host model
# --------------------------------------------------------------------------- #
def generate_routing_indices(seed: int) -> torch.Tensor:
    """Generate ``indices[N_RANKS][T, TOPK]`` so no expert exceeds RECV_MAX.

    Each (t, k) is a global expert id in [0, E_GLOBAL). Top-k entries within a
    single token are forced unique. Reseed if any per-expert receive count would
    overflow R.
    """
    rng = torch.Generator().manual_seed(seed)
    while True:
        indices = torch.zeros(N_RANKS, T, TOPK, dtype=torch.int32)
        for r in range(N_RANKS):
            for t in range(T):
                perm = torch.randperm(E_GLOBAL, generator=rng)[:TOPK]
                indices[r, t, :] = perm.to(torch.int32)

        per_expert = torch.zeros(N_RANKS, L, dtype=torch.int32)
        for r in range(N_RANKS):
            for t in range(T):
                for k in range(TOPK):
                    eid = int(indices[r, t, k].item())
                    per_expert[eid // L, eid % L] += 1
        if int(per_expert.max().item()) <= R:
            return indices
        seed += 1
        rng.manual_seed(seed)


def compute_dispatch_golden(
    x_norms: list[torch.Tensor],  # [N_RANKS] of [T, D] BF16
    indices: torch.Tensor,  # [N_RANKS, T, TOPK] INT32
    weights: torch.Tensor,  # [N_RANKS, T, TOPK] FP32
):
    """Replay the dispatch protocol on host.

    Returns, per rank:
      expected_recv_x[L, R, D]   BF16  (x payload)
      expected_recv_w[L, R]      FP32  (weight payload)
      expected_recv_idx[L, R]    INT32 (r = t*TOPK+k for each delivered row)
      expected_count[L]          INT32
    plus ``route_dest[src][t][k] = (dst, loc_e, slot)`` — where on which rank's
    receive area route (src, t, k) landed, so the combine golden can pick the
    matching recv_y row back out.
    """
    expected_recv_x = [torch.zeros(L, R, D, dtype=torch.bfloat16) for _ in range(N_RANKS)]
    expected_recv_w = [torch.zeros(L, R, dtype=torch.float32) for _ in range(N_RANKS)]
    expected_recv_idx = [torch.zeros(L, R, dtype=torch.int32) for _ in range(N_RANKS)]
    expected_count = [torch.zeros(L, dtype=torch.int32) for _ in range(N_RANKS)]
    route_dest = [[[None] * TOPK for _ in range(T)] for _ in range(N_RANKS)]

    send_counts = torch.zeros(N_RANKS, N_RANKS, L, dtype=torch.int32)
    for src in range(N_RANKS):
        for t in range(T):
            for k in range(TOPK):
                eid = int(indices[src, t, k].item())
                send_counts[src, eid // L, eid % L] += 1

    for dst in range(N_RANKS):
        # Per-destination slot_offset[src][e] = sum_{s < src} send_counts[s, dst, e].
        slot_offset = torch.zeros(N_RANKS, L, dtype=torch.int32)
        running = torch.zeros(L, dtype=torch.int32)
        for src in range(N_RANKS):
            slot_offset[src] = running.clone()
            running = running + send_counts[src, dst]

        for src in range(N_RANKS):
            cursor = torch.zeros(L, dtype=torch.int32)
            for t in range(T):
                for k in range(TOPK):
                    eid = int(indices[src, t, k].item())
                    if eid // L != dst:
                        continue
                    loc_e = eid % L
                    slot = int(slot_offset[src, loc_e].item() + cursor[loc_e].item())
                    cursor[loc_e] += 1
                    expected_recv_x[dst][loc_e, slot, :] = x_norms[src][t, :]
                    expected_recv_w[dst][loc_e, slot] = weights[src, t, k]
                    expected_recv_idx[dst][loc_e, slot] = t * TOPK + k
                    route_dest[src][t][k] = (dst, loc_e, slot)

        for e in range(L):
            expected_count[dst][e] = int(running[e].item())

    return expected_recv_x, expected_recv_w, expected_recv_idx, expected_count, route_dest


def pack_weights_padded(weights_row: torch.Tensor) -> torch.Tensor:
    """Build [N_ROUTES, W_PAD] FP32 where row r = (weight_value, 0, …, 0)."""
    out = torch.zeros(N_ROUTES, W_PAD, dtype=torch.float32)
    for t in range(T):
        for k in range(TOPK):
            out[t * TOPK + k, 0] = weights_row[t, k]
    return out


def pack_idx_padded() -> torch.Tensor:
    """Build [N_ROUTES, IDX_PAD] INT32 where row r = (r, 0, …, 0)."""
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
    """Per-row (per-token) INT8 symmetric quant — mirrors the kernels' *_q stages."""
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    out_i8 = _round_half_away_from_zero(rows * scale_quant).to(torch.int32).to(torch.float16).to(torch.int8)
    return out_i8.reshape_as(x), (1.0 / scale_quant).reshape(*x.shape[:-1], 1)


def _quant_w_per_channel(w: torch.Tensor):
    """Per-output-channel INT8 quant on the last axis (w shaped [..., N, K])."""
    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    w_i8 = (
        _round_half_away_from_zero(w.float() * scale_quant.unsqueeze(-1)).to(torch.int32).to(torch.float16).to(torch.int8)
    )
    return w_i8, (1.0 / scale_quant).float()


def build_expert_weights(seed: int):
    """Random INT8 weight banks shared across both ranks (one bank of L local
    experts + a shared expert), generated like moe_expert.py::build_tensor_specs.

    Returns a dict of {name: tensor} (each ``share_memory_()``'d).
    """
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}

    def _quant_and_store(name: str, w_bf16: torch.Tensor) -> None:
        w_i8, w_s = _quant_w_per_channel(w_bf16)
        out[name] = w_i8.contiguous().share_memory_()
        out[name + "_scale"] = w_s.contiguous().share_memory_()

    _quant_and_store("expert_w1", (torch.randn(L, MOE_INTER, D, generator=gen) / D**0.5).to(torch.bfloat16))
    _quant_and_store("expert_w3", (torch.randn(L, MOE_INTER, D, generator=gen) / D**0.5).to(torch.bfloat16))
    _quant_and_store("expert_w2", (torch.randn(L, D, MOE_INTER, generator=gen) / MOE_INTER**0.5).to(torch.bfloat16))
    _quant_and_store("shared_w1", (torch.randn(MOE_INTER, D, generator=gen) / D**0.5).to(torch.bfloat16))
    _quant_and_store("shared_w3", (torch.randn(MOE_INTER, D, generator=gen) / D**0.5).to(torch.bfloat16))
    _quant_and_store("shared_w2", (torch.randn(D, MOE_INTER, generator=gen) / MOE_INTER**0.5).to(torch.bfloat16))
    return out


def _dequant_w(w_i8: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    return w_i8.to(torch.float32) * w_scale.unsqueeze(-1)


def golden_moe_expert(
    recv_x: torch.Tensor,  # [L, R, D]  BF16  (= expected_recv_x[rank])
    recv_weights: torch.Tensor,  # [L, R]     FP32  (= expected_recv_w[rank])
    recv_count: torch.Tensor,  # [L]        INT32 (= expected_count[rank])
    x_local: torch.Tensor,  # [T, D]     BF16  (= x_norms[rank])
    w: dict[str, torch.Tensor],  # the INT8 weight banks
):
    """Torch reference for one rank's moe_expert call. Mirrors every A8 quant
    round-trip the kernels do. Returns (recv_y[L, R, D] BF16, sh[T, D] BF16).
    ``recv_y[e, count[e]:, :]`` stays zero (kernels mask the dirty tail rows)."""
    recv_x = recv_x.float()
    recv_weights = recv_weights.float()
    x_local = x_local.float()
    w1 = _dequant_w(w["expert_w1"], w["expert_w1_scale"].float())
    w3 = _dequant_w(w["expert_w3"], w["expert_w3_scale"].float())
    w2 = _dequant_w(w["expert_w2"], w["expert_w2_scale"].float())
    sw1 = _dequant_w(w["shared_w1"], w["shared_w1_scale"].float())
    sw3 = _dequant_w(w["shared_w3"], w["shared_w3_scale"].float())
    sw2 = _dequant_w(w["shared_w2"], w["shared_w2_scale"].float())

    # Mirror the A8 round-trip on x_local (shared-expert input).
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
def _verify_recv_outputs(
    nranks, expected_count, expected_recv_x, expected_recv_w, expected_recv_idx,
    recv_count_outs, recv_x_outs, recv_w_outs, recv_idx_outs,
) -> bool:
    """dispatch outputs vs the protocol replay — bit-exact (BF16 verbatim copies)."""
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
            x_diff = (recv_x_outs[r][e, :n, :].float() - expected_recv_x[r][e, :n, :].float()).abs().max().item()
            w_diff = (recv_w_outs[r][e, :n] - expected_recv_w[r][e, :n]).abs().max().item()
            idx_diff = (recv_idx_outs[r][e, :n] - expected_recv_idx[r][e, :n]).abs().max().item()
            if x_diff > 0 or w_diff > 1e-5 or idx_diff != 0:
                ok = False
                print(f"[ep_dispatch] chip {r} expert {e}: cnt={n} x_diff={x_diff:.3e} w_diff={w_diff:.3e} idx_diff={idx_diff}")
    return ok


def _verify_expert_outputs(nranks, recv_y_goldens, sh_goldens, expected_count, recv_y_outs, sh_outs) -> bool:
    """moe_expert outputs vs golden_moe_expert. INT8-matmul vs FP32-reference =>
    not bit-exact; use the same tolerances moe_expert.py asserts."""
    ok = True
    rtol, atol = 1e-2, 5e-3
    for r in range(nranks):
        for e in range(L):
            n = int(expected_count[r][e].item())
            if n == 0:
                continue
            got = recv_y_outs[r][e, :n, :].float()
            exp = recv_y_goldens[r][e, :n, :].float()
            diff = (got - exp).abs()
            denom = exp.abs().clamp_min(1e-9)
            if not torch.allclose(got, exp, rtol=rtol, atol=atol):
                ok = False
                print(f"[ep_dispatch] chip {r} expert {e}: recv_y mismatch n={n} "
                      f"max|diff|={float(diff.max()):.3e} max_rel={float((diff / denom).max()):.3e}")
        got_sh, exp_sh = sh_outs[r].float(), sh_goldens[r].float()
        diff_sh = (got_sh - exp_sh).abs()
        print(f"[ep_dispatch] chip {r}: sh max|diff|={float(diff_sh.max()):.3e} "
              f"max_rel={float((diff_sh / exp_sh.abs().clamp_min(1e-9)).max()):.3e}")
        if not torch.allclose(got_sh, exp_sh, rtol=rtol, atol=atol):
            ok = False
            print(f"[ep_dispatch] chip {r}: sh mismatch (rtol={rtol}, atol={atol})")
    return ok


def _verify_routed_y(nranks, route_dest, recv_y_goldens, routed_y_outs) -> bool:
    """combine output vs sum_k recv_y_golden[holder][loc_e, slot, :] (FP32 accumulate).

    Uses the *golden* recv_y so the only modelled error is the INT8/FP32 matmul
    gap that already shows up in _verify_expert_outputs; combine itself is exact.
    """
    ok = True
    atol = TOPK * 5e-3 + 1e-3
    rtol = 1e-2
    for me in range(nranks):
        expected = torch.zeros(T, D, dtype=torch.float32)
        for t in range(T):
            for k in range(TOPK):
                dst, loc_e, slot = route_dest[me][t][k]
                expected[t, :] += recv_y_goldens[dst][loc_e, slot, :].float()
        got = routed_y_outs[me]
        diff = (got - expected).abs()
        print(f"[ep_dispatch] chip {me}: routed_y max|diff|={float(diff.max()):.3e} "
              f"max_rel={float((diff / expected.abs().clamp_min(1e-9)).max()):.3e}")
        if not torch.allclose(got, expected, rtol=rtol, atol=atol):
            ok = False
            print(f"[ep_dispatch] chip {me}: routed_y mismatch (rtol={rtol}, atol={atol})")
            per_token = diff.max(dim=1).values
            for t in range(T):
                if per_token[t] > atol:
                    print(f"  token {t}: got[0]={float(got[t, 0]):.4f} expected[0]={float(expected[t, 0]):.4f}")
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

    # Small random BF16 activations (like moe_expert.py's fixtures) — dispatch
    # copies BF16 verbatim, so the recv_x check stays bit-exact, and the MoE
    # matmul outputs stay in a sane magnitude range for the tolerance check.
    gen = torch.Generator().manual_seed(seed)
    x_norms = [(torch.randn(T, D, generator=gen) * 0.05).to(torch.bfloat16).share_memory_() for _ in range(nranks)]
    weights = torch.tensor(
        [[[(r + 1) * 0.01 + t * 0.1 + k * 0.001 for k in range(TOPK)] for t in range(T)] for r in range(nranks)],
        dtype=torch.float32,
    )

    indices = generate_routing_indices(seed=seed)
    print(f"[ep_dispatch] indices shape={tuple(indices.shape)} (rank,t,k -> global expert id)")

    indices_per_rank = [indices[r].clone().contiguous().share_memory_() for r in range(nranks)]
    w_padded_list = [pack_weights_padded(weights[r]).share_memory_() for r in range(nranks)]
    idx_padded_list = [pack_idx_padded().share_memory_() for _ in range(nranks)]

    # dispatch outputs (also moe_expert inputs).
    recv_x_outs = [torch.zeros(L, R, D, dtype=torch.bfloat16).share_memory_() for _ in range(nranks)]
    recv_w_outs = [torch.zeros(L, R, dtype=torch.float32).share_memory_() for _ in range(nranks)]
    recv_idx_outs = [torch.zeros(L, R, dtype=torch.int32).share_memory_() for _ in range(nranks)]
    recv_count_outs = [torch.zeros(L, 1, dtype=torch.int32).share_memory_() for _ in range(nranks)]
    # moe_expert outputs.
    recv_y_outs = [torch.zeros(L, R, D, dtype=torch.bfloat16).share_memory_() for _ in range(nranks)]
    sh_outs = [torch.zeros(T, D, dtype=torch.bfloat16).share_memory_() for _ in range(nranks)]
    # combine output.
    routed_y_outs = [torch.zeros(T, D, dtype=torch.float32).share_memory_() for _ in range(nranks)]

    print("[ep_dispatch] computing host golden (dispatch replay -> moe_expert -> combine)...")
    (expected_recv_x, expected_recv_w, expected_recv_idx, expected_count, route_dest) = compute_dispatch_golden(
        x_norms, indices, weights
    )
    weight_banks = build_expert_weights(seed=seed)
    # recv_count the orchestration reads at build time (host knows it; equals
    # what dispatch will emit into recv_count_out).
    recv_count_host_list = [expected_count[r].reshape(L, 1).clone().contiguous().share_memory_() for r in range(nranks)]
    recv_y_goldens, sh_goldens = [], []
    for r in range(nranks):
        ry, shg = golden_moe_expert(
            expected_recv_x[r], expected_recv_w[r], expected_count[r], x_norms[r], weight_banks
        )
        recv_y_goldens.append(ry)
        sh_goldens.append(shg)

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
    cfg.aicpu_thread_num = 4  # tensormap_and_ringbuffer w/ AIC tasks: 3 schedulers + 1 orchestrator

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
                a.add_tensor(make_tensor_arg(indices_per_rank[i]), TensorArgType.INPUT)
                a.add_tensor(make_tensor_arg(x_norms[i]), TensorArgType.INPUT)
                a.add_tensor(make_tensor_arg(w_padded_list[i]), TensorArgType.INPUT)
                a.add_tensor(make_tensor_arg(idx_padded_list[i]), TensorArgType.INPUT)
                a.add_tensor(make_tensor_arg(recv_count_host_list[i]), TensorArgType.INPUT)
                for name in ("expert_w1", "expert_w1_scale", "expert_w3", "expert_w3_scale", "expert_w2", "expert_w2_scale",
                             "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale", "shared_w2", "shared_w2_scale"):
                    a.add_tensor(make_tensor_arg(weight_banks[name]), TensorArgType.INPUT)
                a.add_tensor(make_tensor_arg(recv_x_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(recv_w_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(recv_idx_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(recv_count_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(recv_y_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(sh_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(make_tensor_arg(routed_y_outs[i]), TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(
                    ContinuousTensor.make(
                        data=ctx.buffer_ptrs["scratch"], shapes=(SCRATCH_NBYTES // 4,),
                        dtype=DataType.FLOAT32, child_memory=True,
                    ),
                    TensorArgType.INOUT,
                )
                a.add_scalar(ctx.nranks)
                a.add_scalar(ctx.device_ctx)
                orch.submit_next_level(chip_cid, a, cfg, worker=i)

        print("[ep_dispatch] running 2-chip dispatch + moe_expert + combine DAG...")
        worker.run(orch_fn, args=None, config=cfg)

        ok = _verify_recv_outputs(
            nranks, expected_count, expected_recv_x, expected_recv_w, expected_recv_idx,
            recv_count_outs, recv_x_outs, recv_w_outs, recv_idx_outs,
        )
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
    parser.add_argument("-p", "--platform", default="a2a3", help="Platform backend, e.g. a2a3 or a2a3sim.")
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
