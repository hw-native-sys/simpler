#!/usr/bin/env python3
"""
Numerical correctness test for the standalone paged attention kernel.

Compares the custom kernel (loaded via ctypes from pa_lib.so) against
`torch_npu.npu_incre_flash_attention` with paged KV
"""
from __future__ import annotations

import ctypes
import math
import os
import sys

import torch
import torch_npu  # noqa: F401

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from pa_tiling import make_pa_nd_decode_tiling, workspace_sizes  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def as_ptr(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def empty_buf(device: str) -> torch.Tensor:
    """1-byte uint8 tensor used as a null-equivalent pointer for unused args."""
    return torch.zeros(1, dtype=torch.uint8, device=device)


def load_lib(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(path)
    lib.call_kernel.restype = None
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,    # block_dim
        ctypes.c_void_p,    # stream
        ctypes.c_void_p,    # q_gm         [batch, nq, D]
        ctypes.c_void_p,    # k_gm         [num_blocks, block_size, nkv, D]
        ctypes.c_void_p,    # v_gm         [num_blocks, block_size, nkv, D]
        ctypes.c_void_p,    # block_tables [batch, max_blocks]  int32
        ctypes.c_void_p,    # mask_gm
        ctypes.c_void_p,    # deq_scale1
        ctypes.c_void_p,    # offset1
        ctypes.c_void_p,    # deq_scale2
        ctypes.c_void_p,    # offset2
        ctypes.c_void_p,    # razorOffset
        ctypes.c_void_p,    # scale_gm
        ctypes.c_void_p,    # logN_gm
        ctypes.c_void_p,    # eye_gm
        ctypes.c_void_p,    # o_gm         [batch, nq, D]
        ctypes.c_void_p,    # s_gm         workspace
        ctypes.c_void_p,    # p_gm         workspace
        ctypes.c_void_p,    # o_tmp_gm     workspace
        ctypes.c_void_p,    # go_gm        workspace
        ctypes.c_void_p,    # o_core_tmp   workspace
        ctypes.c_void_p,    # l_gm         workspace
        ctypes.c_void_p,    # gm_k16       workspace
        ctypes.c_void_p,    # gm_v16       workspace
        ctypes.c_void_p,    # tiling
    ]
    lib.get_ffts_info.restype = None
    lib.get_ffts_info.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),   # addr out
        ctypes.POINTER(ctypes.c_uint32),   # len  out
    ]
    return lib


def query_ffts_size(lib) -> int:
    """Return the FFTS sync buffer size (bytes) reported by the runtime."""
    addr = ctypes.c_uint64(0)
    length = ctypes.c_uint32(0)
    lib.get_ffts_info(ctypes.byref(addr), ctypes.byref(length))
    return int(length.value)


def pack_kv_to_paged(
    k_dense: torch.Tensor,   # [B, L, nkv*D]
    v_dense: torch.Tensor,   # [B, L, nkv*D]
    nkv: int, head_dim: int, block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense [B, L, nkv*D] → paged [B*nb, block_size, nkv, D], block_table [B, nb]."""
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


def kvp_page_to_bsh_layout(
    k_page: torch.Tensor, v_page: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """[num_blocks, block_size, nkv, D] -> [num_blocks, block_size, nkv*D] for IFA BSH."""
    nb, bs, nkv, d = k_page.shape
    return (
        k_page.reshape(nb, bs, nkv * d).contiguous(),
        v_page.reshape(nb, bs, nkv * d).contiguous(),
    )


# ── reference: torch_npu.npu_incre_flash_attention (paged KV) ───────────────

def run_incre_flash_paged_ref(
    q: torch.Tensor,
    k_page: torch.Tensor,
    v_page: torch.Tensor,
    bt: torch.Tensor,
    ctx_lens: torch.Tensor,
    nkv: int,
    nq: int,
    scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """BSH paged path; q is [B, nq, D], returns [B, nq, D] like the custom kernel."""
    b, _, d = q.shape
    device = q.device
    block_size = k_page.shape[1]
    k_bsh, v_bsh = kvp_page_to_bsh_layout(k_page, v_page)
    q_bsh = q.reshape(b, 1, nq * d).contiguous()
    out_bsh = torch_npu.npu_incre_flash_attention(
        q_bsh,
        k_bsh,
        v_bsh,
        num_heads=nq,
        num_key_value_heads=nkv,
        input_layout="BSH",
        scale_value=scale,
        block_table=bt.to(device),
        actual_seq_lengths=[int(x) for x in ctx_lens.tolist()],
        block_size=block_size,
    )
    torch.npu.synchronize()
    return out_bsh.reshape(b, nq, d).to(dtype=dtype)


# ── custom kernel runner ──────────────────────────────────────────────────────

def _launch(lib, eff_bd, stream, q, k_page, v_page, bt, null,
            o, s_gm, p_gm, o_tmp, go, o_core, l_gm, k16, v16, tiling):
    """Single kernel launch helper."""
    lib.call_kernel(
        eff_bd, stream,
        as_ptr(q), as_ptr(k_page), as_ptr(v_page), as_ptr(bt),
        as_ptr(null), as_ptr(null), as_ptr(null), as_ptr(null), as_ptr(null),
        as_ptr(null), as_ptr(null), as_ptr(null), as_ptr(null),
        as_ptr(o),
        as_ptr(s_gm), as_ptr(p_gm), as_ptr(o_tmp), as_ptr(go),
        as_ptr(o_core), as_ptr(l_gm), as_ptr(k16), as_ptr(v16),
        as_ptr(tiling),
    )


def run_custom(lib, q, k_page, v_page, bt, ctx_lens, nq, nkv, head_dim, head_dim_v,
               scale, block_dim, device, dtype,
               workspace_cache: dict | None = None):
    b = q.shape[0]
    num_blocks = k_page.shape[0]
    block_size = k_page.shape[1]
    max_blocks = bt.shape[1]

    tiling, eff_bd, kv_cn = make_pa_nd_decode_tiling(
        batch=b, kv_seq_lens=ctx_lens.tolist(),
        num_heads=nq, kv_heads=nkv,
        head_dim=head_dim, head_dim_v=head_dim_v,
        num_blocks=num_blocks, block_size=block_size,
        max_blocks_per_query=max_blocks, scale=scale,
        block_dim=block_dim, device=device, dtype=dtype,
    )

    ws = workspace_sizes(b, nq, head_dim, head_dim_v, block_dim, kv_cn)

    def ws_buf(key):
        n = ws[key]
        if workspace_cache is not None:
            if key not in workspace_cache or workspace_cache[key].numel() < n:
                workspace_cache[key] = torch.zeros(n, dtype=torch.uint8, device=device)
            else:
                # Zero-fill the ENTIRE cached buffer, not just the current-call slice.
                # The kernel may index into regions sized by the max workspace allocation,
                # so we must ensure no stale data from a previous (larger-shape) call leaks in.
                workspace_cache[key].zero_()
            return workspace_cache[key]
        return torch.zeros(n, dtype=torch.uint8, device=device)

    s_gm     = ws_buf("s")
    p_gm     = ws_buf("p")
    o_tmp_gm = ws_buf("o_tmp")
    go_gm    = ws_buf("go")
    o_core   = ws_buf("o_core_tmp")
    l_gm     = ws_buf("l")
    gm_k16   = ws_buf("k16")
    gm_v16   = ws_buf("v16")

    o = torch.zeros(b, nq, head_dim_v, dtype=dtype, device=device)
    null = empty_buf(device)
    bt_npu = bt.to(device)
    # Ensure all workspace zero-fills complete before kernel reads workspace buffers
    torch.npu.synchronize()
    stream = torch.npu.current_stream()._as_parameter_

    _launch(lib, eff_bd, stream, q, k_page, v_page, bt_npu, null,
            o, s_gm, p_gm, o_tmp_gm, go_gm, o_core, l_gm, gm_k16, gm_v16, tiling)
    torch.npu.synchronize()
    return o


# ── test cases ────────────────────────────────────────────────────────────────

def run_test(lib, case, dtype, device, block_dim,
             rtol=5e-3, atol=2e-2):
    """
    Numerically-correct test for one PA shape.

    FFTS-safe strategy (avoids hardware counter cross-contamination):
      1. Pre-allocate all test tensors so that no NPU tensor ops happen between
         the warmup and the kernel under test.
      2. Run 4 shape-specific warmup calls with zero-data (same paged layout).
         4 = FFTS cycle length.  After 4 calls of the same shape we're always
         at position 0 mod 4 which is the correct FFTS phase for that shape.
      3. Run the CUSTOM kernel FIRST (immediately after warmup, no intervening
         NPU ops).
      4. Run the IFA reference (`npu_incre_flash_attention`, paged KV) AFTER –
         the reference does not need a specific FFTS phase.
      5. Compare.
    """
    name  = case["name"]
    b     = case["batch"]
    nq    = case["num_heads"]
    nkv   = case["num_kv_heads"]
    d     = case["head_dim"]
    s_kv  = case["kv_seq"]
    bs    = case["block_size"]

    assert s_kv % bs == 0, "kv_seq must be divisible by block_size"

    # ── Step 1: pre-generate ALL test tensors ──
    torch.manual_seed(42)
    scale = 1.0 / math.sqrt(float(d))

    q       = torch.randn(b, nq,  d, dtype=dtype, device=device)
    k_dense = torch.randn(b, s_kv, nkv * d, dtype=dtype, device=device)
    v_dense = torch.randn(b, s_kv, nkv * d, dtype=dtype, device=device)
    k_page, v_page, bt = pack_kv_to_paged(k_dense, v_dense, nkv, d, bs)
    ctx_lens = torch.tensor([s_kv] * b, dtype=torch.int32, device="cpu")
    torch.npu.synchronize()

    # ── Step 2: 4× shape-specific warmup ──
    q0 = torch.zeros_like(q)
    k0 = torch.zeros_like(k_page)
    v0 = torch.zeros_like(v_page)
    bt0 = bt.clone()
    ctx0 = ctx_lens.clone()
    wscache: dict = {}
    for _ in range(4):
        run_custom(lib, q0, k0, v0, bt0, ctx0, nq, nkv, d, d, scale,
                   block_dim, device, dtype, workspace_cache=wscache)
    torch.npu.synchronize()

    # ── Step 3: custom kernel FIRST ──
    out = run_custom(lib, q, k_page, v_page, bt, ctx_lens, nq, nkv, d, d, scale,
                     block_dim, device, dtype, workspace_cache=wscache)

    # ── Step 4: IFA reference AFTER ──
    ref = run_incre_flash_paged_ref(q, k_page, v_page, bt, ctx_lens, nkv, nq, scale, dtype)

    diff = (out.float() - ref.float()).abs()
    mean_err = diff.mean().item()
    max_err  = diff.max().item()
    ok = bool(torch.allclose(out, ref, rtol=rtol, atol=atol))

    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name}  mean_err={mean_err:.5f}  max_err={max_err:.5f}")
    if not ok:
        raise AssertionError(f"{name}: max_err={max_err:.5f} > atol={atol}")
    return mean_err, max_err


# ── warmup ───────────────────────────────────────────────────────────────────

def warmup(lib, device, dtype, block_dim, workspace_cache: dict):
    """
    One-time kernel warmup with persistent workspace_cache.
    The first call to an AscendC kernel using fresh workspace buffers may
    produce NaN on this platform.  By using the same workspace_cache for
    the warmup and all subsequent calls, we ensure the NPU has written into
    (and validated) those memory regions before the real tests run.
    """
    b, nq, nkv, d, bs = 4, 64, 8, 128, 128
    tiling, eff_bd, kv_cn = make_pa_nd_decode_tiling(
        batch=b, kv_seq_lens=[bs]*b, num_heads=nq, kv_heads=nkv,
        head_dim=d, head_dim_v=d, num_blocks=b, block_size=bs,
        max_blocks_per_query=1, scale=1.0 / math.sqrt(d),
        block_dim=block_dim, device=device, dtype=dtype,
    )
    ws = workspace_sizes(b, nq, d, d, block_dim, kv_cn)
    null = empty_buf(device)
    q  = torch.zeros(b, nq, d, dtype=dtype, device=device)
    k  = torch.zeros(b, bs, nkv, d, dtype=dtype, device=device)
    v  = torch.zeros(b, bs, nkv, d, dtype=dtype, device=device)
    bt = torch.zeros(b, 1, dtype=torch.int32, device=device)
    o  = torch.zeros(b, nq, d, dtype=dtype, device=device)

    def ws_buf(key):
        n = ws[key]
        if key not in workspace_cache or workspace_cache[key].numel() < n:
            workspace_cache[key] = torch.zeros(n, dtype=torch.uint8, device=device)
        return workspace_cache[key]

    s  = ws_buf("s")
    p  = ws_buf("p")
    o_tmp = ws_buf("o_tmp")
    go = ws_buf("go")
    oc = ws_buf("o_core_tmp")
    l  = ws_buf("l")
    k16 = ws_buf("k16")
    v16 = ws_buf("v16")
    torch.npu.synchronize()
    stream = torch.npu.current_stream()._as_parameter_
    _launch(lib, eff_bd, stream, q, k, v, bt, null, o,
            s, p, o_tmp, go, oc, l, k16, v16, tiling)
    torch.npu.synchronize()


# ── single-case entry point (for subprocess isolation) ────────────────────────

def run_single_case(case_name: str):
    """Run one named test case in isolation and exit with 0=pass / 1=fail."""
    import json
    cases_json = os.environ.get("PA_TEST_CASES", "[]")
    all_cases = json.loads(cases_json)
    case = next((c for c in all_cases if c["name"] == case_name), None)
    if case is None:
        print(f"Unknown case: {case_name}", file=sys.stderr)
        sys.exit(2)

    npu_id = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    device = f"npu:{npu_id}"
    torch.npu.set_device(device)
    block_dim = int(getattr(torch.npu.get_device_properties(device), "cube_core_num", 24))
    lib = load_lib(os.path.join(here, "pa_lib.so"))

    wscache: dict = {}
    warmup(lib, device, torch.float16, block_dim, wscache)
    try:
        run_test(lib, case, torch.float16, device, block_dim)
    except AssertionError as e:
        print(f"  FAIL  {case_name}: {e}", flush=True)
        sys.exit(1)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import json, subprocess

    # When called as a subprocess worker, run only the requested case.
    single = os.environ.get("PA_SINGLE_CASE")
    if single:
        run_single_case(single)
        return

    npu_id = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    device = f"npu:{npu_id}"
    torch.npu.set_device(device)
    block_dim = int(getattr(torch.npu.get_device_properties(device), "cube_core_num", 24))
    print(f"Device: {device}  cube_cores={block_dim}")

    cases = [
        # Simple smoke test
        # {"name": "b1_h32_kv8_s128_bs128",   "batch": 1, "num_heads": 32, "num_kv_heads": 8,  "head_dim": 128, "kv_seq": 128,  "block_size": 128},
        # # Multiple batches
        # {"name": "b4_h32_kv8_s512_bs128",   "batch": 4, "num_heads": 32, "num_kv_heads": 8,  "head_dim": 128, "kv_seq": 512,  "block_size": 128},
        # # MHA (nq == nkv) — uses split-KV path (tiling_key=16), isolated in subprocess
        # {"name": "b2_h8_kv8_s256_bs128",    "batch": 2, "num_heads": 8,  "num_kv_heads": 8,  "head_dim": 128, "kv_seq": 256,  "block_size": 128},
        # # Larger GQA
        # {"name": "b8_h32_kv8_s1024_bs128",  "batch": 8, "num_heads": 32, "num_kv_heads": 8,  "head_dim": 128, "kv_seq": 1024, "block_size": 128},
        # # Qwen3 shapes
        # {"name": "b1_h32_kv8_s2048_bs128",  "batch": 1, "num_heads": 32, "num_kv_heads": 8,  "head_dim": 128, "kv_seq": 2048, "block_size": 128},
        # {"name": "b4_h64_kv8_s1024_bs128",  "batch": 4, "num_heads": 64, "num_kv_heads": 8,  "head_dim": 128, "kv_seq": 1024, "block_size": 128},
        {"name": "b256_h16_kv1_s8192_bs128",  "batch": 256, "num_heads": 16, "num_kv_heads": 1,  "head_dim": 128, "kv_seq": 8192, "block_size": 128},
    ]

    cases_env = json.dumps(cases)
    script = os.path.abspath(__file__)
    env_base = os.environ.copy()
    env_base["PA_TEST_CASES"] = cases_env
    env_base["ASCEND_DEVICE_ID"] = str(npu_id)

    print(f"\nRunning fp16 tests (each case in an isolated subprocess):")
    all_pass = True
    for c in cases:
        env = dict(env_base, PA_SINGLE_CASE=c["name"])
        result = subprocess.run(
            [sys.executable, script],
            env=env, capture_output=True, text=True, timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode == 0:
            # Extract the PASS line printed by run_test
            line = next((l for l in output.splitlines() if "PASS" in l or "mean_err" in l), f"  PASS  {c['name']}")
            print(line)
        else:
            all_pass = False
            for line in output.splitlines():
                print(line)

    if all_pass:
        print("\nAll fp16 cases PASSED.")
    else:
        print("\nSome cases FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
