"""
Usage:
  python bench_pa_standalone.py [--warmup W] [--iters N] [--bf16] [--device ID]
"""
from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys

import torch
import torch_npu  # noqa: F401

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from pa_tiling import make_pa_nd_decode_tiling, workspace_sizes
from test_pa_accuracy import load_lib, _launch, empty_buf, as_ptr


# ── flop / byte model ─────────────────────────────────────────────────────────

def gqa_decode_matmul_flops(b: int, nq: int, q_seq: int, kv_seq: int, d: int) -> float:
    """2 * B * nq * q_seq * kv_seq * d  (QK + PV, each a batch-matmul)."""
    return 2 * b * nq * q_seq * kv_seq * d * 2  # QK + PV = ×2


def gqa_tensor_bytes_bsh(b: int, q_seq: int, kv_seq: int,
                          nq: int, nkv: int, d: int, elem: int) -> float:
    """Logical Q + K + V + O (reads Q,K,V; writes O)."""
    return b * elem * (nq * q_seq * d + 2 * nkv * kv_seq * d + nq * q_seq * d)


# ── NPU event timer ───────────────────────────────────────────────────────────

def benchmark_with_events(fn, warmup_iters: int = 5, benchmark_iters: int = 20) -> float:
    """Return mean time in ms over benchmark_iters, measured with NPU events."""
    for _ in range(warmup_iters):
        fn()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end   = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(benchmark_iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / benchmark_iters


# ── pack helpers ──────────────────────────────────────────────────────────────

def pack_kv_to_paged(k_dense, v_dense, nkv, head_dim, block_size):
    b, L, _ = k_dense.shape
    nb = L // block_size
    k_page = (k_dense.view(b, L, nkv, head_dim)
               .view(b, nb, block_size, nkv, head_dim)
               .reshape(b * nb, block_size, nkv, head_dim)
               .contiguous())
    v_page = (v_dense.view(b, L, nkv, head_dim)
               .view(b, nb, block_size, nkv, head_dim)
               .reshape(b * nb, block_size, nkv, head_dim)
               .contiguous())
    dev = k_dense.device
    bt = (torch.arange(nb, dtype=torch.int32, device=dev)
          .unsqueeze(0).expand(b, -1).clone()
          + torch.arange(b, dtype=torch.int32, device=dev).unsqueeze(1) * nb)
    return k_page, v_page, bt


# ── custom kernel runner ──────────────────────────────────────────────────────

class CustomPARunner:
    """Stateful runner for the standalone paged-attention kernel.

    Allocates workspace and tiling once, reuses for every call.
    """
    def __init__(self, lib, q, k_page, v_page, bt, ctx_lens,
                 nq, nkv, head_dim, scale, block_dim, device, dtype):
        self.lib = lib
        self.q = q
        self.k_page = k_page
        self.v_page = v_page
        self.bt = bt
        self.ctx_lens = ctx_lens
        self.block_dim = block_dim
        self.device = device
        self.dtype = dtype

        b = q.shape[0]
        num_blocks = k_page.shape[0]
        block_size = k_page.shape[1]
        max_blocks = bt.shape[1]

        self.tiling, self.eff_bd = make_pa_nd_decode_tiling(
            batch=b,
            kv_seq_lens=ctx_lens.tolist(),
            num_heads=nq,
            kv_heads=nkv,
            head_dim=head_dim,
            head_dim_v=head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            max_blocks_per_query=max_blocks,
            scale=scale,
            block_dim=block_dim,
            device=device,
            dtype=dtype,
        )
        ws = workspace_sizes(b, nq, head_dim, head_dim, block_dim)
        self.s_gm    = torch.zeros(ws["s"],         dtype=torch.uint8, device=device)
        self.p_gm    = torch.zeros(ws["p"],         dtype=torch.uint8, device=device)
        self.o_tmp   = torch.zeros(ws["o_tmp"],     dtype=torch.uint8, device=device)
        self.go_gm   = torch.zeros(ws["go"],        dtype=torch.uint8, device=device)
        self.o_core  = torch.zeros(ws["o_core_tmp"],dtype=torch.uint8, device=device)
        self.l_gm    = torch.zeros(ws["l"],         dtype=torch.uint8, device=device)
        self.k16     = torch.zeros(ws["k16"],       dtype=torch.uint8, device=device)
        self.v16     = torch.zeros(ws["v16"],       dtype=torch.uint8, device=device)
        self.o       = torch.empty(b, nq, head_dim, dtype=dtype, device=device)
        self.null    = empty_buf(device)
        torch.npu.synchronize()

    def __call__(self):
        stream = torch.npu.current_stream()._as_parameter_
        _launch(
            self.lib, self.eff_bd, stream,
            self.q, self.k_page, self.v_page, self.bt, self.null,
            self.o,
            self.s_gm, self.p_gm, self.o_tmp, self.go_gm,
            self.o_core, self.l_gm, self.k16, self.v16,
            self.tiling,
        )
        return self.o


# ── per-case benchmark ────────────────────────────────────────────────────────

def bench_case(lib, name, b, nq, nkv, d, s_kv, bs, dtype, device,
               block_dim, warmup, iters):
    scale = 1.0 / math.sqrt(float(d))
    elem  = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    flops  = gqa_decode_matmul_flops(b, nq, 1, s_kv, d)
    nbytes = gqa_tensor_bytes_bsh(b, 1, s_kv, nq, nkv, d, elem)

    q       = torch.randn(b, nq, d, dtype=dtype, device=device)
    k_dense = torch.randn(b, s_kv, nkv * d, dtype=dtype, device=device)
    v_dense = torch.randn(b, s_kv, nkv * d, dtype=dtype, device=device)
    k_page, v_page, bt = pack_kv_to_paged(k_dense, v_dense, nkv, d, bs)
    ctx_lens = torch.tensor([s_kv] * b, dtype=torch.int32, device="cpu")
    out_ref = torch.empty(b, nq, d, dtype=dtype, device=device)

    def forward_atb_v2():
        torch_npu.atb._npu_paged_attention_v2(
            q, k_page, bt, ctx_lens,
            value_cache=v_page, mask=None,
            num_kv_heads=nkv, num_heads=nq,
            scale_value=scale, mask_type=0, out=out_ref,
        )
        return out_ref

    runner = CustomPARunner(lib, q, k_page, v_page, bt, ctx_lens,
                             nq, nkv, d, scale, block_dim, device, dtype)

    ms_custom = benchmark_with_events(runner,        warmup_iters=warmup, benchmark_iters=iters)
    ms_atb_v2 = benchmark_with_events(forward_atb_v2, warmup_iters=warmup, benchmark_iters=iters)

    def fmt(label, ms):
        t_s  = ms * 1e-3
        tflops = flops / t_s / 1e12
        gibs   = (nbytes / t_s) / (1024 ** 3)
        ai     = flops / nbytes if nbytes else 0.0
        return (f"{name} | {label:28s} | {ms:7.4f} ms |"
                f" {tflops:7.4f} TFLOP/s | {gibs:7.4f} GiB/s | AI={ai:.4f} F/B")

    print(fmt("standalone (pa_lib.so)", ms_custom))
    print(fmt("ATB v2 (_npu_paged_attention_v2)", ms_atb_v2))
    speedup = ms_atb_v2 / ms_custom if ms_custom > 0 else float("inf")
    print(f"{name} | speedup standalone/ATB_v2: {speedup:.3f}x")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

# Same shapes as atb_pa/README.md (ifa-gpa suite), block_size=128
DEFAULT_CASES = [
    # (name, batch, nq, nkv, head_dim, kv_seq, block_size)
    ("Qwen3-0.6B b1 h16/kv8 kv2048",  1,  16,  8, 128, 2048, 128),
    ("Qwen3-1.7B b1 h16/kv8 kv4096",  1,  16,  8, 128, 4096, 128),
    ("Qwen3-4B   b1 h32/kv8 kv2048",  1,  32,  8, 128, 2048, 128),
    ("Qwen3-8B   b1 h32/kv8 kv4096",  1,  32,  8, 128, 4096, 128),
    ("Qwen3-8B   b1 h32/kv8 kv8192",  1,  32,  8, 128, 8192, 128),
    ("Qwen3-14B  b1 h40/kv8 kv2048",  1,  40,  8, 128, 2048, 128),
    ("Qwen3-32B  b1 h64/kv8 kv2048",  1,  64,  8, 128, 2048, 128),
    ("MHA        b1 h32/kv32 kv2048", 1,  32, 32, 128, 2048, 128),
    ("Qwen3-8B   b4 h32/kv8 kv2048",  4,  32,  8, 128, 2048, 128),
    ("Qwen3-8B   b8 h32/kv8 kv2048",  8,  32,  8, 128, 2048, 128),
    ("Qwen3-8B  b16 h32/kv8 kv2048", 16,  32,  8, 128, 2048, 128),
    ("Qwen3-8B  b32 h32/kv8 kv2048", 32,  32,  8, 128, 2048, 128),
    ("Qwen3-8B  b64 h32/kv8 kv2048", 64,  32,  8, 128, 2048, 128),
]


def main():
    parser = argparse.ArgumentParser(
        description="Standalone PA kernel benchmark vs ATB v2.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters",  type=int, default=20)
    parser.add_argument("--bf16",   action="store_true")
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    npu_id = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    if args.device is not None:
        npu_id = args.device
    device = f"npu:{npu_id}"
    torch.npu.set_device(device)
    block_dim = int(getattr(torch.npu.get_device_properties(device),
                            "cube_core_num", 24))
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    lib_path = os.path.join(here, "pa_lib.so")
    lib = load_lib(lib_path)

    print(f"Device: {device}  cube_cores={block_dim}")
    print(f"dtype={dtype}  warmup={args.warmup}  iters={args.iters}")
    print(f"Standalone lib: {lib_path}")
    print(
        "case | API | ms | TFLOP/s | GiB/s | AI (F/B)\n"
        "(FLOPs = QK+PV matmuls; Bytes = logical Q+K+V+O)\n---"
    )
    for case in DEFAULT_CASES:
        name, b, nq, nkv, d, s_kv, bs = case
        try:
            bench_case(lib, name, b, nq, nkv, d, s_kv, bs,
                       dtype, device, block_dim, args.warmup, args.iters)
        except Exception as e:
            print(f"{name}: FAILED — {e}")


if __name__ == "__main__":
    main()
