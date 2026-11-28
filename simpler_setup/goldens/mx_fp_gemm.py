# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Golden helpers for a5 ``mx_fp_gemm`` ST / MX FP8/FP4 UT decode.

MXFP8 matmul golden follows pto-isa ``tmatmul_mx/gen_data.py`` and pypto2
``test_matmul_mx.py``: per-block E8M0 scales (factor 32 along K), ZZ/NN GM packing
for ``TMATMUL_MX``, then FP32 matmul of dequantized A/B.

Pure torch (no numpy) — CI torch CPU wheels do not pull numpy.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def decode_e4m3fn(byte: int) -> float:
    """Decode one IEEE-style E4M3FN byte (bias=7, no Inf; exp=15 → NaN)."""
    b = int(byte) & 0xFF
    sign = (b >> 7) & 1
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    if exp == 0:
        val = 0.0 if mant == 0 else (mant / 8.0) * (2.0 ** (1 - 7))
    elif exp == 0xF:
        return math.nan
    else:
        val = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
    return -val if sign else val


def decode_e2m1(nibble: int) -> float:
    """Decode one E2M1 nibble (OCP MXFP4; bias=1)."""
    n = int(nibble) & 0xF
    sign = (n >> 3) & 1
    exp = (n >> 1) & 0x3
    mant = n & 0x1
    if exp == 0:
        val = 0.0 if mant == 0 else 0.5
    else:
        val = (1.0 + 0.5 * mant) * (2.0 ** (exp - 1))
    return -val if sign else val


def decode_e8m0(byte: int) -> float:
    """Decode one E8M0 shared-exponent byte as ``2**(x - 127)`` (OCP MX scale)."""
    return math.ldexp(1.0, (int(byte) & 0xFF) - 127)


def fp8_e4m3fn_to_float32(t: torch.Tensor) -> torch.Tensor:
    """Bit-exact E4M3FN → float32 (matches ``t.to(torch.float32)`` for finite values)."""
    bits = t.contiguous().view(torch.uint8).reshape(-1)
    out = torch.empty(bits.numel(), dtype=torch.float32)
    for i, b in enumerate(bits.tolist()):
        out[i] = decode_e4m3fn(b)
    return out.view(t.shape)


def fp4_e2m1_x2_to_float32(t: torch.Tensor) -> torch.Tensor:
    """Unpack ``float4_e2m1fn_x2`` → float32 with last dim expanded ×2 (lo nibble first)."""
    bits = t.contiguous().view(torch.uint8)
    flat = bits.reshape(-1)
    out = torch.empty(flat.numel() * 2, dtype=torch.float32)
    for i, b in enumerate(flat.tolist()):
        out[2 * i] = decode_e2m1(b & 0xF)
        out[2 * i + 1] = decode_e2m1((b >> 4) & 0xF)
    shape = list(t.shape)
    shape[-1] = shape[-1] * 2
    return out.view(*shape)


def convert_x1_scale_format(x1_mx_gm: torch.Tensor, block_size: int = 16, c0_size_mx: int = 2) -> torch.Tensor:
    """Pack A-side E8M0 scales to MX_A_ZZ GM layout (pto-isa gen_data)."""
    m, k = x1_mx_gm.shape
    pad_m = (block_size - m % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_m > 0 or pad_k > 0:
        # F.pad pads last dim first: (left, right, top, bottom)
        padded = F.pad(x1_mx_gm, (0, pad_k, 0, pad_m), value=0)
    else:
        padded = x1_mx_gm
    m_padded = m + pad_m
    k_padded = k + pad_k
    x1_scale_gm = padded.reshape(m_padded // block_size, block_size, k_padded // c0_size_mx, c0_size_mx)
    x1_scale_gm = x1_scale_gm.permute(0, 2, 1, 3)
    return x1_scale_gm.reshape(
        x1_scale_gm.shape[0] * x1_scale_gm.shape[1],
        x1_scale_gm.shape[2] * x1_scale_gm.shape[3],
    )


def convert_x2_scale_format(x2_mx_gm: torch.Tensor, block_size: int = 16, c0_size_mx: int = 2) -> torch.Tensor:
    """Pack B-side E8M0 scales to MX_B_NN GM layout (pto-isa gen_data)."""
    k, n = x2_mx_gm.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_n > 0 or pad_k > 0:
        padded = F.pad(x2_mx_gm, (0, pad_n, 0, pad_k), value=0)
    else:
        padded = x2_mx_gm
    k_padded, n_padded = padded.shape
    x2_scale_gm = padded.reshape(k_padded // c0_size_mx, c0_size_mx, n_padded // 16, 16).permute(2, 0, 3, 1)
    return x2_scale_gm.reshape(
        x2_scale_gm.shape[1] * x2_scale_gm.shape[3],
        x2_scale_gm.shape[0] * x2_scale_gm.shape[2],
    )


def _u8_e8m0_to_torch(buf: torch.Tensor, logical_shape: tuple[int, int]) -> torch.Tensor:
    """View ZZ/NN-packed uint8 E8M0 bytes as torch.float8_e8m0fnu with logical shape."""
    flat = buf.contiguous().reshape(-1).to(torch.uint8)
    assert flat.numel() == logical_shape[0] * logical_shape[1]
    return flat.clone().view(torch.float8_e8m0fnu).reshape(logical_shape)


def mx_fp8_matmul_golden(
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    a_scale_u8: torch.Tensor,
    b_scale_u8: torch.Tensor,
) -> torch.Tensor:
    """Dequant A/B with block E8M0 scales (factor 32 along K), then FP32 matmul.

    ``a_scale_u8`` is logical ``[M, ceil(K/32)]``, ``b_scale_u8`` is ``[ceil(K/32), N]``
    (before ZZ/NN packing). Matches pto-isa ``gen_golden_data`` for MXFP8.
    """
    _m, k = a_fp8.shape
    k2, _n = b_fp8.shape
    assert k == k2
    a_mx = torch.pow(2.0, a_scale_u8.to(torch.float64) - 127)  # [M, KMX]
    b_mx = torch.pow(2.0, b_scale_u8.to(torch.float64) - 127)  # [KMX, N]
    k_idx = torch.arange(k) // 32
    a_scaled = a_fp8.to(torch.float64) * a_mx[:, k_idx]
    b_scaled = b_fp8.to(torch.float64) * b_mx[k_idx, :]
    return torch.matmul(a_scaled, b_scaled).to(torch.float32)


def make_mx_fp8_case(m: int = 128, k: int = 64, n: int = 64, seed: int = 19):
    """Build host-prequant MXFP8 tensors + golden for ``TMATMUL_MX`` ST.

    Returns ``(a, a_s, b, b_s, c, expected)`` where ``a_s``/``b_s`` are ZZ/NN-packed
    ``float8_e8m0fnu`` with logical shapes ``[M, K/32]`` / ``[K/32, N]``.
    """
    if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e8m0fnu"):
        raise RuntimeError("torch.float8_e4m3fn / float8_e8m0fnu required")
    assert k % 64 == 0, "K must be multiple of 64 for single-tile MX sample"
    kmx = k // 32
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randint(-10, 10, (m, k), dtype=torch.int64, generator=generator).to(torch.float32).to(torch.float8_e4m3fn)
    b = torch.randint(-10, 10, (k, n), dtype=torch.int64, generator=generator).to(torch.float32).to(torch.float8_e4m3fn)
    a_scale_u8 = torch.randint(127, 130, (m, kmx), dtype=torch.int64, generator=generator).to(torch.uint8)
    b_scale_u8 = torch.randint(127, 130, (kmx, n), dtype=torch.int64, generator=generator).to(torch.uint8)
    expected = mx_fp8_matmul_golden(a, b, a_scale_u8, b_scale_u8)
    a_s = _u8_e8m0_to_torch(convert_x1_scale_format(a_scale_u8), (m, kmx))
    b_s = _u8_e8m0_to_torch(convert_x2_scale_format(b_scale_u8), (kmx, n))
    c = torch.zeros((m, n), dtype=torch.float32)
    return a, a_s, b, b_s, c, expected


def pack_two_fp4(nibble_matrix: torch.Tensor) -> torch.Tensor:
    """Pack adjacent E2M1 nibble codes along the last dim (pto-isa ``pack_two_fp4``).

    Even index → lo nibble, odd → hi nibble. ``nibble_matrix`` is uint8 ``[R, C]``
    with C even; returns uint8 ``[R, C/2]``.
    """
    rows, cols = nibble_matrix.shape
    assert cols % 2 == 0
    flat = nibble_matrix.contiguous().to(torch.uint8).reshape(-1)
    high = flat[::2] & 0x0F
    low = (flat[1::2] & 0x0F) << 4
    return (low | high).reshape(rows, cols // 2)


def _fp4_logical_to_torch_x2(logical_u8: torch.Tensor) -> torch.Tensor:
    """``[R, C]`` nibble codes → torch ``float4_e2m1fn_x2`` with shape ``[R, C/2]``."""
    packed = pack_two_fp4(logical_u8).contiguous()
    return packed.view(torch.float4_e2m1fn_x2)


def mx_fp4_matmul_golden(
    a_fp4_x2: torch.Tensor,
    b_fp4_x2: torch.Tensor,
    a_scale_u8: torch.Tensor,
    b_scale_u8: torch.Tensor,
) -> torch.Tensor:
    """Dequant packed MXFP4 A/B (pto-isa last-dim packing) with E8M0 scales, then FP32 matmul.

    ``a_fp4_x2`` is ``[M, K/2]``, ``b_fp4_x2`` is ``[K, N/2]`` (pack along last dim).
    Scales are logical ``[M, K/32]`` / ``[K/32, N]``.
    """
    a_f = fp4_e2m1_x2_to_float32(a_fp4_x2).to(torch.float64)  # [M, K]
    b_f = fp4_e2m1_x2_to_float32(b_fp4_x2).to(torch.float64)  # [K, N]
    _m, k = a_f.shape
    k2, _n = b_f.shape
    assert k == k2
    a_mx = torch.pow(2.0, a_scale_u8.to(torch.float64) - 127)
    b_mx = torch.pow(2.0, b_scale_u8.to(torch.float64) - 127)
    k_idx = torch.arange(k) // 32
    a_scaled = a_f * a_mx[:, k_idx]
    b_scaled = b_f * b_mx[k_idx, :]
    return torch.matmul(a_scaled, b_scaled).to(torch.float32)


def make_mx_fp4_case(m: int = 128, k: int = 64, n: int = 64, seed: int = 19):
    """Build host-prequant MXFP4 tensors + golden for ``TMATMUL_MX`` ST.

    A/B follow pto-isa packing: logical ``[M,K]``/``[K,N]`` E2M1 → torch
    ``float4_e2m1fn_x2`` of shapes ``[M, K/2]`` / ``[K, N/2]``.
    """
    if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "float8_e8m0fnu"):
        raise RuntimeError("torch.float4_e2m1fn_x2 / float8_e8m0fnu required")
    assert k % 64 == 0 and n % 64 == 0, "K/N must be multiples of 64 for MXFP4 sample"
    kmx = k // 32
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Logical nibble codes in a small finite set (matches pto-isa e2m1 range).
    a_logical = torch.randint(0, 8, (m, k), dtype=torch.int64, generator=generator).to(torch.uint8)
    b_logical = torch.randint(0, 8, (k, n), dtype=torch.int64, generator=generator).to(torch.uint8)
    a = _fp4_logical_to_torch_x2(a_logical)
    b = _fp4_logical_to_torch_x2(b_logical)
    a_scale_u8 = torch.randint(127, 130, (m, kmx), dtype=torch.int64, generator=generator).to(torch.uint8)
    b_scale_u8 = torch.randint(127, 130, (kmx, n), dtype=torch.int64, generator=generator).to(torch.uint8)
    expected = mx_fp4_matmul_golden(a, b, a_scale_u8, b_scale_u8)
    a_s = _u8_e8m0_to_torch(convert_x1_scale_format(a_scale_u8), (m, kmx))
    b_s = _u8_e8m0_to_torch(convert_x2_scale_format(b_scale_u8), (kmx, n))
    c = torch.zeros((m, n), dtype=torch.float32)
    return a, a_s, b, b_s, c, expected


def fp8_matmul_golden(a_fp8: torch.Tensor, b_fp8: torch.Tensor) -> torch.Tensor:
    """``(A_fp8 → f32) @ (B_fp8 → f32)`` (no MX scales; UT helper)."""
    return torch.matmul(a_fp8.to(torch.float32), b_fp8.to(torch.float32))


def fp4_matmul_golden(a_fp4_x2: torch.Tensor, b_fp4_x2: torch.Tensor) -> torch.Tensor:
    """Unpack packed FP4 A/B along the contracting dim, then matmul in float32.

    ``a_fp4_x2`` has shape ``[M, K_pack]`` (each element = 2 E2M1 along K),
    ``b_fp4_x2`` has shape ``[K_pack, N]`` (each element = 2 E2M1 along K).
    Logical GEMM is ``[M, 2*K_pack] @ [2*K_pack, N]``.
    """
    a = fp4_e2m1_x2_to_float32(a_fp4_x2)  # [M, 2*K_pack]
    b_bits = b_fp4_x2.contiguous().view(torch.uint8)
    k_pack, n = b_bits.shape
    b = torch.empty(k_pack * 2, n, dtype=torch.float32)
    for kp in range(k_pack):
        for j in range(n):
            byte = int(b_bits[kp, j].item())
            b[2 * kp, j] = decode_e2m1(byte & 0xF)
            b[2 * kp + 1, j] = decode_e2m1((byte >> 4) & 0xF)
    return torch.matmul(a, b)
