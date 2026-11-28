# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Unit tests: MX FP8/FP4 decode + small-matmul numerical correctness (host)."""

import math

import pytest
import torch

from simpler_setup.goldens.mx_fp_gemm import (
    convert_x1_scale_format,
    convert_x2_scale_format,
    decode_e2m1,
    decode_e8m0,
    fp4_e2m1_x2_to_float32,
    fp4_matmul_golden,
    fp8_e4m3fn_to_float32,
    fp8_matmul_golden,
    make_mx_fp4_case,
    make_mx_fp8_case,
    mx_fp4_matmul_golden,
    mx_fp8_matmul_golden,
    pack_two_fp4,
)
from simpler_setup.torch_interop import make_chip_tensor_arg


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch.float8_e4m3fn required")
class TestFp8E4m3Numeric:
    def test_decode_matches_torch_cast(self):
        torch.manual_seed(0)
        src = torch.randn(64)
        fp8 = src.to(torch.float8_e4m3fn)
        ref = fp8.to(torch.float32)
        got = fp8_e4m3fn_to_float32(fp8)
        finite = torch.isfinite(ref) & torch.isfinite(got)
        assert torch.allclose(ref[finite], got[finite], rtol=0, atol=0)
        assert torch.equal(torch.isnan(ref), torch.isnan(got))

    def test_all_bytes_match_torch(self):
        bits = torch.arange(256, dtype=torch.uint8)
        fp8 = bits.view(torch.float8_e4m3fn)
        torch.testing.assert_close(fp8_e4m3fn_to_float32(fp8), fp8.float(), rtol=0, atol=0, equal_nan=True)

    def test_matmul_matches_torch(self):
        torch.manual_seed(1)
        a = torch.randn(8, 8).to(torch.float8_e4m3fn)
        b = torch.randn(8, 8).to(torch.float8_e4m3fn)
        golden = fp8_matmul_golden(a, b)
        ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        assert torch.allclose(golden, ref, rtol=0, atol=0)

    def test_make_chip_tensor_arg_preserves_bytes_for_matmul(self):
        torch.manual_seed(2)
        a = torch.randn(4, 4).to(torch.float8_e4m3fn).contiguous()
        b = torch.randn(4, 4).to(torch.float8_e4m3fn).contiguous()
        arg_a = make_chip_tensor_arg(a)
        arg_b = make_chip_tensor_arg(b)
        assert arg_a.nbytes() == a.nbytes
        assert arg_b.nbytes() == b.nbytes
        a_round = a.view(torch.uint8).clone().view(torch.float8_e4m3fn).reshape(a.shape)
        b_round = b.view(torch.uint8).clone().view(torch.float8_e4m3fn).reshape(b.shape)
        assert torch.allclose(fp8_matmul_golden(a, b), fp8_matmul_golden(a_round, b_round), rtol=0, atol=0)


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="torch.float8_e8m0fnu required (torch>=2.7)")
class TestFp8E8m0Numeric:
    def test_decode_scale_pow2(self):
        # 127 → 2^0 = 1; 128 → 2; 126 → 0.5
        assert decode_e8m0(127) == 1.0
        assert decode_e8m0(128) == 2.0
        assert decode_e8m0(126) == 0.5
        assert math.isnan(decode_e8m0(255))

    def test_scale_mul_matches_torch(self):
        data = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float32)
        scale_bytes = torch.tensor([127, 128, 126, 129], dtype=torch.uint8)
        scales = torch.tensor([decode_e8m0(int(b)) for b in scale_bytes.tolist()])
        # Mirror via torch E8M0 when available: 2**(x-127)
        e8 = scale_bytes.view(torch.float8_e8m0fnu)
        # torch may not implement mul for e8m0; use explicit decode.
        out = data * scales
        assert out.tolist() == pytest.approx([1.0, -4.0, 0.25, 12.0])
        assert e8.numel() == 4

    def test_zz_nn_pack_preserves_nbytes(self):
        m, kmx, n = 128, 2, 64
        a_s = torch.randint(127, 130, (m, kmx), dtype=torch.int64).to(torch.uint8)
        b_s = torch.randint(127, 130, (kmx, n), dtype=torch.int64).to(torch.uint8)
        assert convert_x1_scale_format(a_s).numel() == a_s.numel()
        assert convert_x2_scale_format(b_s).numel() == b_s.numel()


@pytest.mark.skipif(
    not (hasattr(torch, "float8_e4m3fn") and hasattr(torch, "float8_e8m0fnu")),
    reason="torch float8_e4m3fn + float8_e8m0fnu required",
)
class TestMxFp8MatmulGolden:
    def test_nan_scale_propagates(self):
        a, _, b, _, _, _ = make_mx_fp8_case()
        a_scale = torch.full((128, 2), 127, dtype=torch.uint8)
        b_scale = torch.full((2, 64), 127, dtype=torch.uint8)
        a_scale[0, 0] = 255
        output = mx_fp8_matmul_golden(a, b, a_scale, b_scale)
        assert output[0].isnan().all()
        assert output[1:].isfinite().all()

    def test_make_case_shapes_and_finite(self):
        a, a_s, b, b_s, c, expected = make_mx_fp8_case(128, 64, 64)
        assert a.shape == (128, 64) and a.dtype == torch.float8_e4m3fn
        assert b.shape == (64, 64) and b.dtype == torch.float8_e4m3fn
        assert a_s.shape == (128, 2) and a_s.dtype == torch.float8_e8m0fnu
        assert b_s.shape == (2, 64) and b_s.dtype == torch.float8_e8m0fnu
        assert c.shape == expected.shape == (128, 64)
        assert torch.isfinite(expected).all()

    def test_unit_scales_match_plain_fp8_matmul(self):
        torch.manual_seed(0)
        a = torch.randn(16, 64).clamp(-4, 4).to(torch.float8_e4m3fn)
        b = torch.randn(64, 32).clamp(-4, 4).to(torch.float8_e4m3fn)
        a_s = torch.full((16, 2), 127, dtype=torch.uint8)  # 2^0
        b_s = torch.full((2, 32), 127, dtype=torch.uint8)
        got = mx_fp8_matmul_golden(a, b, a_s, b_s)
        ref = fp8_matmul_golden(a, b)
        assert torch.allclose(got, ref, rtol=0, atol=0)

    def test_make_chip_tensor_arg_nbytes(self):
        a, a_s, b, b_s, c, _ = make_mx_fp8_case(128, 64, 64)
        assert make_chip_tensor_arg(a).nbytes() == a.nbytes
        assert make_chip_tensor_arg(a_s).nbytes() == a_s.nbytes
        assert make_chip_tensor_arg(b).nbytes() == b.nbytes
        assert make_chip_tensor_arg(b_s).nbytes() == b_s.nbytes
        assert make_chip_tensor_arg(c).nbytes() == c.nbytes


@pytest.mark.skipif(
    not (hasattr(torch, "float4_e2m1fn_x2") and hasattr(torch, "float8_e8m0fnu")),
    reason="torch float4_e2m1fn_x2 + float8_e8m0fnu required",
)
class TestMxFp4MatmulGolden:
    def test_nan_scale_propagates(self):
        a, _, b, _, _, _ = make_mx_fp4_case()
        a_scale = torch.full((128, 2), 127, dtype=torch.uint8)
        b_scale = torch.full((2, 64), 127, dtype=torch.uint8)
        a_scale[0, 0] = 255
        output = mx_fp4_matmul_golden(a, b, a_scale, b_scale)
        assert output[0].isnan().all()
        assert output[1:].isfinite().all()

    def test_pack_two_fp4_lo_hi(self):
        # [0.5=0b0001, 1.0=0b0010] → byte 0x21
        packed = pack_two_fp4(torch.tensor([[1, 2]], dtype=torch.uint8))
        assert packed.shape == (1, 1)
        assert int(packed[0, 0].item()) == 0x21

    def test_make_case_shapes_and_finite(self):
        a, a_s, b, b_s, c, expected = make_mx_fp4_case(128, 64, 64)
        assert a.shape == (128, 32) and a.dtype == torch.float4_e2m1fn_x2
        assert b.shape == (64, 32) and b.dtype == torch.float4_e2m1fn_x2
        assert a_s.shape == (128, 2) and a_s.dtype == torch.float8_e8m0fnu
        assert b_s.shape == (2, 64) and b_s.dtype == torch.float8_e8m0fnu
        assert c.shape == expected.shape == (128, 64)
        assert torch.isfinite(expected).all()

    def test_unit_scales_match_unpacked_matmul(self):
        a_logical = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.uint8)  # M=2,K=4
        b_logical = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 0]], dtype=torch.uint8)  # K=4,N=2
        a = pack_two_fp4(a_logical).contiguous().view(torch.float4_e2m1fn_x2)
        b = pack_two_fp4(b_logical).contiguous().view(torch.float4_e2m1fn_x2)
        a_s = torch.full((2, 1), 127, dtype=torch.uint8)
        b_s = torch.full((1, 2), 127, dtype=torch.uint8)
        got = mx_fp4_matmul_golden(a, b, a_s, b_s)
        a_f = fp4_e2m1_x2_to_float32(a)
        b_f = fp4_e2m1_x2_to_float32(b)
        ref = torch.matmul(a_f, b_f)
        assert torch.allclose(got, ref, rtol=0, atol=0)


@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="torch.float4_e2m1fn_x2 required")
class TestFp4E2m1Numeric:
    def test_known_nibbles(self):
        # lo=0b0001 (0.5), hi=0b0010 (1.0) → byte 0x21
        t = torch.empty(1, dtype=torch.float4_e2m1fn_x2)
        t.view(torch.uint8)[0] = 0x21
        got = fp4_e2m1_x2_to_float32(t)
        assert got.tolist() == pytest.approx([0.5, 1.0])

    def test_decode_table_finite(self):
        for n in range(16):
            assert math.isfinite(decode_e2m1(n))

    def test_matmul_unpack_consistency(self):
        torch.manual_seed(3)
        m, k_pack, n = 4, 4, 4  # logical K = 8
        a = torch.empty(m, k_pack, dtype=torch.float4_e2m1fn_x2)
        b = torch.empty(k_pack, n, dtype=torch.float4_e2m1fn_x2)
        a.view(torch.uint8).reshape(-1)[:] = torch.randint(0, 256, (m * k_pack,), dtype=torch.uint8)
        b.view(torch.uint8).reshape(-1)[:] = torch.randint(0, 256, (k_pack * n,), dtype=torch.uint8)
        golden = fp4_matmul_golden(a, b)
        assert golden.shape == (m, n)
        assert torch.isfinite(golden).all()
        # Re-run after byte clone (simulates host↔device payload preserve).
        a2 = a.view(torch.uint8).clone().view(torch.float4_e2m1fn_x2).reshape(a.shape)
        b2 = b.view(torch.uint8).clone().view(torch.float4_e2m1fn_x2).reshape(b.shape)
        assert torch.allclose(golden, fp4_matmul_golden(a2, b2), rtol=0, atol=0)

    def test_make_chip_tensor_arg_nbytes(self):
        t = torch.empty(2, 4, dtype=torch.float4_e2m1fn_x2)
        t.view(torch.uint8)[:] = 0x13
        arg = make_chip_tensor_arg(t)
        assert arg.nbytes() == 8
