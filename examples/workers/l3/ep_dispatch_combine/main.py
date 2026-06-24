#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end 2-card EP dispatch + local_expert + combine.

Runs the real DeepSeek-V4 FLASH MoE shapes (pypto-lib/models/deepseek/v4):
T=128, TOPK=6, D=4096, per-rank L=16 local experts, R=192 receive cap. The
only deviation from the production deployment is EP=2 here vs EP=16 there —
each rank keeps the same 16-expert load, so only the global expert count
differs (32 vs 256).

A single orchestration runs three child AIV kernels back-to-back over a
shared HCCL window scratch:

  dispatch.cpp      count exchange + 3-channel push + per-channel stage-out
  local_expert.cpp  recv_y[e, s, :] = recv_x[e, s, :] * recv_w[e, s]
                    (placeholder for the production moe_expert)
  combine.cpp       TPUT recv_y rows by recv_idx_out into
                    routed_y_buf[t, k, :] (relies on HCCL window zero-init,
                    no per-call clear), barrier, reduce_sum along
                    TOPK -> routed_y FP32

Each rank drives the dispatch kernel through these phases (0..4):

  histogram     scalar histogram + (dst, loc_e)-sorted route table from indices
  publish       publish full send_counts table to peers via TNOTIFY(AtomicAdd)
                + count_done barrier
  prefix_sum    local prefix sums over global pub_counts (no comm)
  payload_push  for each route: TPUT three independent payload tiles —
                  x      [BF16, 1xD]        x_norm[t, :]
                  weight [FP32, 1xW_PAD]    w_padded[r, :]   = [weight, 0, …, 0]
                  idx    [INT32, 1xIDX_PAD] idx_padded[r, :] = [r, 0, …, 0]
                                            where r = t * TOPK + k
                to peer's recv_x[loc_e][slot, :] / recv_w[…] / recv_idx[…]
                + data_done barrier
  stage_out     stage out recv_x / recv_w / recv_idx windows -> host-backed outputs

Type/shape contract:
  - ``x_norm`` and ``recv_x_out`` are **BF16**. The dispatch x channel is a
    pure copy, so ``recv_x_out`` is compared BF16-vs-BF16 (bit-exact) against
    the host golden regardless of magnitude — no ``≤ 256`` assumption needed at
    this scale (D=4096 pushes values well past BF16's exact-integer range).
  - Weight uses a 1xW_PAD=8 FP32 tile per route (the minimum vector tile
    granularity = 32 B = one MTE burst). The host pre-packs each row as
    [weight, 0, 0, …, 0]; receiver writes recv_w[loc_e][slot, :W_PAD]
    and the kernel TROWSUM-compacts to a [L, R] FP32 host output.
  - Idx uses the same minimum-tile rationale: 1xIDX_PAD=8 INT32 per
    route, actual r=t*TOPK+k at slot [0]; TROWSUM-compacted to
    [L, R] INT32 host output. Combine reads it to address
    routed_y_buf[t, k, :] without a host-built origin_map.
  - ``recv_count_out`` is [L, 1] INT32 emitted by dispatch's prefix_sum
    phase.

Run:

    python examples/workers/l3/ep_dispatch_combine/main.py -p a2a3sim -d 0-1

"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from simpler.task_interface import (  # noqa: E402
    ArgDirection,
    CallConfig,
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    DataType,
    TaskArgs,
    Tensor,
    TensorArgType,
)
from simpler.worker import Worker  # noqa: E402

from simpler_setup.elf_parser import extract_text_section  # noqa: E402
from simpler_setup.kernel_compiler import KernelCompiler  # noqa: E402
from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: E402
from simpler_setup.torch_interop import make_tensor_arg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Real DeepSeek-V4 FLASH MoE shapes (pypto-lib/models/deepseek/v4) — must
# mirror constants at the top of the kernel. T = DECODE_BATCH*DECODE_SEQ = 128,
# TOPK = num_experts_per_tok = 6, D = hidden_size = 4096, L = N_LOCAL = 16
# experts/rank, R = RECV_MAX = 192. Production runs EP=16; here EP=2 keeps the
# same per-rank load, so only the global expert count differs (32 vs 256).
N_RANKS = 2
T = 128
TOPK = 6
D = 4096
L = 16  # N_LOCAL_EXPERTS per rank
R = 192  # RECV_MAX (single-expert receive upper bound)
W_PAD = 8  # weight tile width — minimum vector tile (1x8 FP32 = 32 B)
IDX_PAD = 8  # idx tile width   — minimum vector tile (1x8 INT32 = 32 B)
E_GLOBAL = N_RANKS * L  # 32 routed experts
N_ROUTES = T * TOPK  # 768

# Window region byte sizes — mirror k*Bytes / kOff* in the kernels.
PUB_COUNTS_BYTES = N_RANKS * N_RANKS * L * 4  # N*N*L INT32
SIGNAL_BYTES = 64  # padded slot per signal area
RECV_X_BYTES = L * R * D * 2  # 24 MB (BF16)
RECV_W_BYTES = L * R * W_PAD * 4  # 384 KB (FP32; weight at slot 0)
RECV_IDX_BYTES = L * R * IDX_PAD * 4  # 384 KB (INT32; r at slot 0)
ROUTED_Y_BUF_BYTES = T * TOPK * D * 2  # 6 MB (BF16; combine push dest)
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
    """Compile the dispatch / local_expert / combine AIV kernels + their
    shared C++ orchestration shim into a single ChipCallable with three
    child callables.
    """
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(commit=pto_isa_commit, clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(runtime)
    kernel_include_dirs = list(include_dirs) + [str(kc.project_root / "src" / "common")]

    def compile_aiv(name: str) -> bytes:
        b = kc.compile_incore(
            source_path=os.path.join(HERE, "kernels/aiv", name),
            core_type="aiv",
            pto_isa_root=pto_isa_root,
            extra_include_dirs=kernel_include_dirs,
        )
        if not platform.endswith("sim"):
            b = extract_text_section(b)
        return b

    dispatch_bin = compile_aiv("dispatch.cpp")
    local_expert_bin = compile_aiv("local_expert.cpp")
    combine_bin = compile_aiv("combine.cpp")

    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/ep_dispatch_combine_orch.cpp"),
    )

    # Per-child signatures — each kernel sees only the args it actually
    # consumes (matching the orch's `Arg` packing for that submit).
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
    sig_local_expert = [
        ArgDirection.IN,  # recv_x_out (reused as INPUT)
        ArgDirection.IN,  # recv_w_out (reused as INPUT)
        ArgDirection.IN,  # recv_count_out (reused as INPUT)
        ArgDirection.OUT,  # recv_y
    ]
    sig_combine = [
        ArgDirection.IN,  # recv_y (reused as INPUT)
        ArgDirection.IN,  # recv_idx_out (reused as INPUT)
        ArgDirection.OUT,  # routed_y
        ArgDirection.INOUT,  # scratch
    ]

    # The orch's view is the union of every child's tensor footprint.
    sig_orch = [
        ArgDirection.IN,  # indices
        ArgDirection.IN,  # x_norm
        ArgDirection.IN,  # w_padded
        ArgDirection.IN,  # idx_padded
        ArgDirection.OUT,  # recv_x_out
        ArgDirection.OUT,  # recv_w_out
        ArgDirection.OUT,  # recv_idx_out
        ArgDirection.OUT,  # recv_count_out
        ArgDirection.OUT,  # recv_y
        ArgDirection.OUT,  # routed_y
        ArgDirection.INOUT,  # scratch
    ]

    return ChipCallable.build(
        signature=sig_orch,
        func_name="ep_dispatch_combine_orchestration",
        config_name="ep_dispatch_combine_orchestration_config",
        binary=orch_bytes,
        children=[
            (0, CoreCallable.build(signature=sig_dispatch, binary=dispatch_bin)),
            (1, CoreCallable.build(signature=sig_local_expert, binary=local_expert_bin)),
            (2, CoreCallable.build(signature=sig_combine, binary=combine_bin)),
        ],
    )


def generate_routing_indices(seed: int) -> torch.Tensor:
    """Generate `indices[N_RANKS][T, TOPK]` so no expert exceeds RECV_MAX.

    Each (t, k) is a global expert id in [0, E_GLOBAL). Top-k entries within
    a single token are forced unique. Reseed if any per-expert receive count
    would overflow R.
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
                    dst = eid // L
                    loc_e = eid % L
                    per_expert[dst, loc_e] += 1
        if int(per_expert.max().item()) <= R:
            return indices
        seed += 1
        rng.manual_seed(seed)


def compute_golden(
    x_norms: list[torch.Tensor],  # [N_RANKS] of [T, D] BF16
    indices: torch.Tensor,  # [N_RANKS, T, TOPK] INT32
    weights: torch.Tensor,  # [N_RANKS, T, TOPK] FP32
):
    """Replay dispatch protocol on host -> per-rank
    expected_recv_x[L, R, D]   BF16  (x payload)
    expected_recv_w[L, R]      FP32  (weight payload)
    expected_recv_idx[L, R]    INT32 (r = t*TOPK+k for each delivered row)
    expected_count[L]          INT32
    """
    expected_recv_x = [torch.zeros(L, R, D, dtype=torch.bfloat16) for _ in range(N_RANKS)]
    expected_recv_w = [torch.zeros(L, R, dtype=torch.float32) for _ in range(N_RANKS)]
    expected_recv_idx = [torch.zeros(L, R, dtype=torch.int32) for _ in range(N_RANKS)]
    expected_count = [torch.zeros(L, dtype=torch.int32) for _ in range(N_RANKS)]

    send_counts = torch.zeros(N_RANKS, N_RANKS, L, dtype=torch.int32)
    for src in range(N_RANKS):
        for t in range(T):
            for k in range(TOPK):
                eid = int(indices[src, t, k].item())
                dst = eid // L
                loc_e = eid % L
                send_counts[src, dst, loc_e] += 1

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

        for e in range(L):
            expected_count[dst][e] = int(running[e].item())

    return expected_recv_x, expected_recv_w, expected_recv_idx, expected_count


def pack_weights_padded(weights_row: torch.Tensor) -> torch.Tensor:
    """Build [N_ROUTES, W_PAD] FP32 where row r = (weight_value, 0, …, 0).

    The kernel TPUTs row r as a 1xW_PAD tile to the receiver's
    recv_w[loc_e][slot, :], so the actual weight ends up at recv_w[..., 0].
    Slots [1, W_PAD) are zero — bandwidth waste vs. a true [L, R] FP32
    output, but W_PAD=8 is the minimum vector tile size in PTO ISA.
    """
    out = torch.zeros(N_ROUTES, W_PAD, dtype=torch.float32)
    for t in range(T):
        for k in range(TOPK):
            r = t * TOPK + k
            out[r, 0] = weights_row[t, k]
    return out


def pack_idx_padded() -> torch.Tensor:
    """Build [N_ROUTES, IDX_PAD] INT32 where row r = (r, 0, …, 0).

    Identical layout for every rank — `r = t*TOPK + k` is an intrinsic
    label, not rank-specific. Receiver picks slot [0] in the combine
    kernel to address routed_y_buf[t, k, :].
    """
    out = torch.zeros(N_ROUTES, IDX_PAD, dtype=torch.int32)
    for t in range(T):
        for k in range(TOPK):
            r = t * TOPK + k
            out[r, 0] = r
    return out


def _verify_recv_outputs(
    nranks: int,
    expected_count: list[torch.Tensor],
    expected_recv_x: list[torch.Tensor],
    expected_recv_w: list[torch.Tensor],
    expected_recv_idx: list[torch.Tensor],
    recv_count_outs: list[torch.Tensor],
    recv_x_outs: list[torch.Tensor],
    recv_w_outs: list[torch.Tensor],
    recv_idx_outs: list[torch.Tensor],
) -> bool:
    """Compare dispatch outputs against the host golden, per rank and per expert."""
    ok = True
    for r in range(nranks):
        cnt = expected_count[r]
        print(f"[ep_dispatch] chip {r}: expected counts per expert = {cnt.tolist()}")
        # recv_count_out is [L, 1] INT32 per the protocol.
        got_count = recv_count_outs[r].squeeze(-1)
        if (got_count - cnt).abs().max().item() != 0:
            ok = False
            print(f"[ep_dispatch] chip {r}: recv_count mismatch got={got_count.tolist()} expected={cnt.tolist()}")
        for e in range(L):
            n = int(cnt[e].item())
            if n == 0:
                continue
            # Cast BF16 → FP32 for diff math; recv_x is a pure copy so got/exp
            # are the same BF16 bits — comparison is bit-exact at any magnitude.
            got_x = recv_x_outs[r][e, :n, :].to(torch.float32)
            exp_x = expected_recv_x[r][e, :n, :].to(torch.float32)
            got_w = recv_w_outs[r][e, :n]
            exp_w = expected_recv_w[r][e, :n]
            got_idx = recv_idx_outs[r][e, :n]
            exp_idx = expected_recv_idx[r][e, :n]
            x_diff = (got_x - exp_x).abs().max().item()
            w_diff = (got_w - exp_w).abs().max().item()
            idx_diff = (got_idx - exp_idx).abs().max().item()
            if x_diff > 0 or w_diff > 1e-5 or idx_diff != 0:
                ok = False
                print(
                    f"[ep_dispatch] chip {r} expert {e}: cnt={n} "
                    f"x_diff={x_diff:.3e} w_diff={w_diff:.3e} idx_diff={idx_diff}"
                )
                if x_diff > 0:
                    for s in range(min(n, 3)):
                        print(f"  slot {s}: got x[0]={float(got_x[s, 0])} expected={float(exp_x[s, 0])}")
                if idx_diff != 0:
                    for s in range(min(n, 3)):
                        print(f"  slot {s}: got idx={int(got_idx[s])} expected={int(exp_idx[s])}")
    return ok


def _bf16_ulp_report(
    tag: str,
    r: int,
    got: torch.Tensor,  # [rows, D] FP32
    expected: torch.Tensor,  # [rows, D] FP32
    rtol: float,
    atol: float,
) -> bool:
    """Shared ULP-tolerant comparison + structural diagnostics.

    A single BF16 cast can differ from torch's round-to-nearest-even by one ULP
    on an exact tie, so we allow `atol + rtol*|expected|`. Anything larger is a
    real fault — the diagnostics expose its shape: which rows, which d-range,
    and the got/expected ratio (a dropped data chunk shows ratio → 0).
    """
    elem_diff = (got - expected).abs()
    elem_tol = atol + rtol * expected.abs()
    bad = elem_diff > elem_tol
    n_bad = int(bad.sum().item())
    max_diff = elem_diff.max().item()
    rel = max_diff / (expected.abs().max().item() + 1e-9)
    print(f"[ep_dispatch] chip {r}: {tag} max|diff|={max_diff:.3e} (rel={rel:.3e}) bad={n_bad}/{got.numel()}")
    if n_bad == 0:
        return True
    rows, dloc = got.shape
    n_bad_rows = int(bad.any(dim=1).sum().item())
    bad_cols = bad.any(dim=0).nonzero().flatten()
    col_lo = int(bad_cols.min().item())
    col_hi = int(bad_cols.max().item())
    safe = expected.clone()
    safe[safe == 0] = float("nan")
    ratios = (got / safe)[bad]
    ratios = ratios[~torch.isnan(ratios)]
    print(f"  {tag}: bad rows={n_bad_rows}/{rows} | bad-d span=[{col_lo},{col_hi}] of {dloc}")
    if ratios.numel() > 0:
        qs = torch.quantile(ratios, torch.tensor([0.05, 0.5, 0.95]))
        print(
            f"  {tag}: got/expected over bad elems — min={ratios.min():.4f} median={qs[1]:.4f} max={ratios.max():.4f}"
        )
    return False


def _verify_routed_y(
    nranks: int,
    x_norms: list[torch.Tensor],
    weights: torch.Tensor,
    routed_y_outs: list[torch.Tensor],
) -> bool:
    """Post-reduce check: routed_y[t, :] == sum_k bf16(x_norms[t]*weights[t,k]) (fp32 accum).

    Mirrors the kernel's TOPK reduce (fp32 accumulator over TOPK bf16 terms).
    Each term carries the same <=1 BF16 ULP cast diff as recv_y; summing TOPK in
    fp32 keeps the *relative* error ~1-2 ULP (the sum magnitude grows with it),
    so rtol = 2 ULP holds for a correct reduce. The rich diagnostics flag any
    structural error — a dropped term shows as ratio (TOPK-1)/TOPK, a high-d
    tail loss as a bad-d span at high d.
    """
    rtol = 2.0**-6  # 2 BF16 ULPs (1 ULP = 2**-7); fp32 accumulation is exact
    atol = 2.0**-6
    ok = True
    for r in range(nranks):
        expected = torch.zeros(T, D, dtype=torch.float32)
        for t in range(T):
            for k in range(TOPK):
                term = weights[r, t, k] * x_norms[r][t, :].to(torch.float32)
                expected[t, :] += term.to(torch.bfloat16).to(torch.float32)
        got = routed_y_outs[r].to(torch.float32)
        ok = _bf16_ulp_report("routed_y", r, got, expected, rtol, atol) and ok
    return ok


def run(
    device_ids: list[int],
    platform: str = "a2a3",
    pto_isa_commit: str | None = None,
) -> int:
    """Core logic — callable from CLI and pytest."""
    nranks = len(device_ids)
    assert nranks == N_RANKS

    window_size = max(SCRATCH_NBYTES, 128 * 1024)

    print(f"[ep_dispatch] platform={platform} devices={device_ids} nranks={nranks}")

    # x_norm[r, t, d] = r*100 + t*10 + d. At this scale (D=4096) values exceed
    # BF16's exact-integer range, but recv_x is a pure copy so the host compares
    # the BF16 tensor against itself bit-for-bit — magnitude is irrelevant. The
    # recv_y / routed_y checks tolerate the BF16 cast(s) via a ULP bound.
    x_norms = [
        torch.tensor(
            [[r * 100 + t * 10 + d for d in range(D)] for t in range(T)],
            dtype=torch.bfloat16,
        ).share_memory_()
        for r in range(nranks)
    ]
    weights = torch.tensor(
        [[[(r + 1) * 0.01 + t * 0.1 + k * 0.001 for k in range(TOPK)] for t in range(T)] for r in range(nranks)],
        dtype=torch.float32,
    )

    indices = generate_routing_indices(seed=20260510)
    print(f"[ep_dispatch] indices shape={tuple(indices.shape)} (rank,t,k -> global expert id)")

    indices_per_rank = [indices[r].clone().contiguous().share_memory_() for r in range(nranks)]
    w_padded_list = [pack_weights_padded(weights[r]).share_memory_() for r in range(nranks)]
    # idx_padded is rank-independent (r = t*TOPK + k is intrinsic), but each
    # rank gets its own shared-memory copy so the framework can plumb it as a
    # per-rank input tensor.
    idx_padded_list = [pack_idx_padded().share_memory_() for _ in range(nranks)]

    # Outputs — recv_x_out is BF16 (matches kernel TPUT/TSTORE element type);
    # recv_w_out / recv_idx_out are compacted to [L, R] inside the kernel.
    # recv_count_out is [L, 1] INT32 — dispatch's prefix_sum phase fills it
    # from pub_counts so local_expert can iterate `recv_count[e]` rows per expert.
    recv_x_outs = [torch.zeros(L, R, D, dtype=torch.bfloat16).share_memory_() for _ in range(nranks)]
    recv_w_outs = [torch.zeros(L, R, dtype=torch.float32).share_memory_() for _ in range(nranks)]
    recv_idx_outs = [torch.zeros(L, R, dtype=torch.int32).share_memory_() for _ in range(nranks)]
    recv_count_outs = [torch.zeros(L, 1, dtype=torch.int32).share_memory_() for _ in range(nranks)]
    # Cross-kernel host-backed tensors:
    #   recv_y    [L, R, D]  BF16 — local_expert output; feeds combine as its
    #                                OUTPUT_EXISTING input.
    #   routed_y  [T, D]     FP32 — combine output: the TOPK reduce sum, written
    #                                straight from the FP32 accumulator.
    recv_y_outs = [torch.zeros(L, R, D, dtype=torch.bfloat16).share_memory_() for _ in range(nranks)]
    routed_y_outs = [torch.zeros(T, D, dtype=torch.float32).share_memory_() for _ in range(nranks)]

    print("[ep_dispatch] computing host golden...")
    expected_recv_x, expected_recv_w, expected_recv_idx, expected_count = compute_golden(x_norms, indices, weights)

    print("[ep_dispatch] compiling kernels...")

    chip_callable = build_chip_callable(platform, pto_isa_commit)

    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
    )
    chip_handle = worker.register(chip_callable)

    try:
        print("[ep_dispatch] init worker (forks chip children; base comm is lazy)...")
        worker.init()

        def orch_fn(orch, _args, cfg):
            with orch.allocate_domain(
                name="default",
                workers=list(range(nranks)),
                window_size=window_size,
                buffers=[
                    CommBufferSpec(
                        name="scratch",
                        dtype="float32",
                        count=SCRATCH_NBYTES // 4,
                        nbytes=SCRATCH_NBYTES,
                    )
                ],
            ) as handle:
                for i in range(nranks):
                    domain = handle[i]
                    print(
                        f"[ep_dispatch] chip {i}: rank={domain.domain_rank}/{domain.domain_size} "
                        f"window=[0x{domain.local_window_base:x} +{domain.actual_window_size}B] "
                        f"scratch=0x{domain.buffer_ptrs['scratch']:x}"
                    )
                    chip_args = TaskArgs()
                    chip_args.add_tensor(make_tensor_arg(indices_per_rank[i]), TensorArgType.INPUT)
                    chip_args.add_tensor(make_tensor_arg(x_norms[i]), TensorArgType.INPUT)
                    chip_args.add_tensor(make_tensor_arg(w_padded_list[i]), TensorArgType.INPUT)
                    chip_args.add_tensor(make_tensor_arg(idx_padded_list[i]), TensorArgType.INPUT)
                    chip_args.add_tensor(make_tensor_arg(recv_x_outs[i]), TensorArgType.OUTPUT_EXISTING)
                    chip_args.add_tensor(make_tensor_arg(recv_w_outs[i]), TensorArgType.OUTPUT_EXISTING)
                    chip_args.add_tensor(make_tensor_arg(recv_idx_outs[i]), TensorArgType.OUTPUT_EXISTING)
                    chip_args.add_tensor(make_tensor_arg(recv_count_outs[i]), TensorArgType.OUTPUT_EXISTING)
                    chip_args.add_tensor(make_tensor_arg(recv_y_outs[i]), TensorArgType.OUTPUT_EXISTING)
                    chip_args.add_tensor(make_tensor_arg(routed_y_outs[i]), TensorArgType.OUTPUT_EXISTING)
                    chip_args.add_tensor(
                        Tensor.make(
                            data=domain.buffer_ptrs["scratch"],
                            shapes=(SCRATCH_NBYTES // 4,),
                            dtype=DataType.FLOAT32,
                            child_memory=True,
                        ),
                        TensorArgType.INOUT,
                    )
                    chip_args.add_scalar(domain.domain_size)
                    chip_args.add_scalar(domain.device_ctx)
                    orch.submit_next_level(chip_handle, chip_args, cfg, worker=i)

        print("[ep_dispatch] running 2-chip dispatch DAG...")
        worker.run(orch_fn, args=None, config=CallConfig())

        if platform.endswith("sim"):
            # Sim keeps intermediate child outputs (recv_y) device-local when they
            # feed later child tasks, so only the final routed_y is host-visible.
            ok = _verify_routed_y(nranks, x_norms, weights, routed_y_outs)
        else:
            ok = _verify_recv_outputs(
                nranks,
                expected_count,
                expected_recv_x,
                expected_recv_w,
                expected_recv_idx,
                recv_count_outs,
                recv_x_outs,
                recv_w_outs,
                recv_idx_outs,
            )
            ok = _verify_routed_y(nranks, x_norms, weights, routed_y_outs) and ok

        if not ok:
            print("[ep_dispatch] golden check FAILED")
            return 1
        print("[ep_dispatch] all ranks matched golden ✅")
        return 0
    finally:
        worker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--device", default="0-1", help="Device range, e.g. '0-1'. Two chips required.")
    parser.add_argument("-p", "--platform", default="a2a3", help="Platform backend, e.g. a2a3 or a2a3sim.")
    parser.add_argument("--pto-isa-commit", default=None, help="Optional PTO ISA commit/tag to fetch before compiling.")
    cli = parser.parse_args()

    return run(parse_device_range(cli.device), platform=cli.platform, pto_isa_commit=cli.pto_isa_commit)


if __name__ == "__main__":
    sys.exit(main())
