#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pinned MoE input fixtures; Torch is required only while preparing a bundle."""

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial

import torch


@dataclass
class TensorSpec:
    name: str
    shape: list[int]
    dtype: torch.dtype
    init_value: Callable[[], torch.Tensor] | None = None
    resident: str | None = None

    def create_tensor(self):
        return (
            torch.zeros(self.shape, dtype=self.dtype)
            if self.init_value is None
            else self.init_value().to(dtype=self.dtype)
        )


@dataclass
class ScalarSpec:
    name: str
    dtype: torch.dtype
    value: int
    compile_runtime: bool = False


BALANCED_ROUTING = False
D = 5120
EP_SIZE = 2
FP8_E4M3_MAX = 448.0
HC_DIM = 20480
HC_MULT = 4
MIX_HC = 24
MOE_INTER = 2304
MOE_TOKENS = 80
MX_GROUP = 32
MX_K_TILE = 256
MX_MM_INTER_TILE = 256
MX_PACKED_LANE_COLS = 128
MX_W1_PACKED_ROWS = 46080
MX_W2_D_OUT_TILE = 256
MX_W2_PACKED_ROWS = 46080
MX_W3_PACKED_ROWS = 46080
N_EXPERTS = 128
N_LOCAL_EXPERTS = 64
ROUTED_DEQUANT_STD = {"w1": 0.0247, "w2": 0.0244, "w3": 0.0246}
TOPK = 6
HC_EPS = 1e-06
PARAM_NAMES = [
    "x_hc",
    "pre_mix",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_weight",
    "gate_weight",
    "correction_bias",
    "routed_w1",
    "routed_w1_scale",
    "routed_w2",
    "routed_w2_scale",
    "routed_w3",
    "routed_w3_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w2",
    "shared_w2_scale",
    "shared_w3",
    "shared_w3_scale",
    "next_pre_mix",
    "x_mixed",
    "x_next",
    "num_tokens",
    "moe_epoch",
    "force_topk",
]
FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)


def decode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 storage codes into FP32 powers of two."""
    codes = scale.contiguous().view(torch.uint8)
    return torch.exp2(codes.to(torch.float32) - 127.0)


def encode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Encode positive FP32 powers of two as UE8M0 storage codes."""
    exponent = torch.round(torch.log2(scale.float())).clamp(-127, 128)
    return (exponent + 127).to(torch.uint8)


def pack_mx_b_scale(scale: torch.Tensor) -> torch.Tensor:
    """Pack logical ``[..., K/32, N]`` scales into the MX_B_NN physical order."""
    *leading, k_groups, output_dim = scale.shape
    if k_groups % 2 or output_dim % 16:
        raise ValueError("MX_B_NN requires an even K-group count and an output dimension divisible by 16")
    packed = scale.reshape(*leading, k_groups // 2, 2, output_dim // 16, 16)
    leading_axes = list(range(len(leading)))
    packed = packed.permute(*leading_axes, len(leading) + 2, len(leading), len(leading) + 3, len(leading) + 1)
    return packed.contiguous().reshape(*leading, k_groups, output_dim)


def _nearest_fp4_indices(values: torch.Tensor, *, round_to_even: bool = False) -> torch.Tensor:
    table = FP4_VALUES[:8].to(values.device)
    magnitude = values.abs().unsqueeze(-1)
    if round_to_even:
        order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=values.device)
        selected = (magnitude - table[order]).abs().argmin(dim=-1)
        index = order[selected].to(torch.uint8)
    else:
        boundaries = (table[:-1] + table[1:]) * 0.5
        absolute = magnitude.squeeze(-1)
        if torch.any(absolute > table[-1]):
            index = (magnitude - table).abs().argmin(dim=-1).to(torch.uint8)
        else:
            index = torch.bucketize(absolute, boundaries, out_int32=True).to(torch.uint8)
            index.masked_fill_(~torch.isfinite(absolute), 0)
    return index | (torch.signbit(values).to(torch.uint8) << 3)


def _pack_fp4(indices: torch.Tensor) -> torch.Tensor:
    if indices.shape[-1] % 2:
        raise ValueError("packed FP4 requires an even logical last dimension")
    pairs = indices.unflatten(-1, (-1, 2))
    return pairs[..., 0] | (pairs[..., 1] << 4)


def quantize_mxfp4_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize output-major ``[..., N, K]`` weights to checkpoint MXFP4 carriers."""
    if weight.shape[-1] % MX_GROUP:
        raise ValueError("MXFP4 weights require K divisible by 32")
    grouped = weight.float().unflatten(-1, (-1, MX_GROUP))
    amax = grouped.abs().amax(dim=-1)
    exponent = torch.ceil(torch.log2((amax / 6.0).clamp_min(2.0**-127)))
    scale = torch.exp2(exponent.clamp(-127, 128))
    normalized = (grouped / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    payload = _pack_fp4(_nearest_fp4_indices(normalized).flatten(-2))
    return payload, encode_e8m0(scale)


def _ocp_e8m0_codes_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """Compute OCP shared-exponent codes for activations and random weight fixtures."""
    _, exponent = torch.frexp(amax.float())
    codes = exponent.to(torch.int32) - 1 - 8 + 127
    codes = codes.clamp(0, 255)
    return torch.where(amax == 0, torch.zeros_like(codes), codes).to(torch.uint8)


def generate_mxfp8_weight_kn(out: int, inn: int, dequant_std: float, *, chan_cv: float = 0.5, seed: int = 0):
    """Generate MXFP8 weights in the device K-N layout with packed E8M0 scales."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(out, inn, generator=g) * torch.exp(chan_cv * torch.randn(out, 1, generator=g))
    groups = raw.reshape(out, inn // MX_GROUP, MX_GROUP)
    codes = _ocp_e8m0_codes_from_amax(groups.abs().amax(-1))
    scale = decode_e8m0(codes)
    quantized = (groups / scale.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    data_kn = quantized.reshape(out, inn).transpose(0, 1).contiguous()
    codes_kn = codes.transpose(0, 1).contiguous()
    decoded = data_kn.float() * decode_e8m0(codes_kn).repeat_interleave(MX_GROUP, dim=0)
    gain = dequant_std / decoded.float().std().clamp_min(1e-8)
    shift = int(torch.round(torch.log2(gain)).item())
    codes_kn = (codes_kn.to(torch.int32) + shift).clamp(0, 255).to(torch.uint8)
    scale_e8m0 = pack_mx_b_scale(codes_kn).contiguous()
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is not None:
        scale_e8m0 = scale_e8m0.view(e8m0)
    return data_kn, scale_e8m0


def _pack_mxfp4_nibbles_kn_tiles(nibbles_kn: torch.Tensor, k_tile: int, n_tile: int) -> torch.Tensor:
    """Pack a logical [K, N] nibble matrix into Cube lane rows."""
    k, n = nibbles_kn.shape
    if k % k_tile or n % n_tile or k_tile % 2 or n_tile % 2:
        raise ValueError("MXFP4 tile dimensions must divide K/N and be even")
    k_blocks = k // k_tile
    n_blocks = n // n_tile
    blocked = nibbles_kn.reshape(k_blocks, k_tile, n_blocks, n_tile)
    blocked = blocked.permute(2, 0, 1, 3)
    lanes = blocked.reshape(n_blocks, k_blocks, 2, k_tile // 2, n_tile)
    low = lanes[..., 0::2] & 0x0F
    high = lanes[..., 1::2] & 0x0F
    packed = low | (high << 4)
    lane_bytes = k_tile * n_tile // 4
    return packed.contiguous().reshape(n_blocks * k_blocks * 2, lane_bytes)


def pack_mxfp4_weight_tiles(packed_weight_fp4: torch.Tensor, k_tile: int, n_tile: int) -> torch.Tensor:
    """Reorder checkpoint [N, K/2] bytes into Cube FP4 tiles."""
    payload = packed_weight_fp4.contiguous().view(torch.uint8)
    *lead, n, half_k = payload.shape
    k = 2 * half_k
    matrices = payload.reshape(-1, n, half_k)
    packed_rows = k * n // n_tile
    packed_cols = n_tile // 2
    result = torch.empty([matrices.shape[0], packed_rows, packed_cols], dtype=torch.uint8)
    for index, matrix in enumerate(matrices):
        nibbles = torch.empty([n, k], dtype=torch.uint8)
        nibbles[:, 0::2] = matrix & 0x0F
        nibbles[:, 1::2] = matrix >> 4
        result[index] = _pack_mxfp4_nibbles_kn_tiles(nibbles.t(), k_tile, n_tile).reshape(packed_rows, packed_cols)
    return result.reshape(*lead, packed_rows, packed_cols)


def gen_routed_mxfp4_weights(n_experts, dequant_std, seed_base=0):
    """Generate checkpoint-style packed MXFP4 routed weights and scales."""

    w1_weights = torch.empty((n_experts, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS), dtype=torch.uint8)
    w3_weights = torch.empty_like(w1_weights)
    w2_weights = torch.empty((n_experts, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS), dtype=torch.uint8)
    w1_scales = torch.empty((n_experts, D // MX_GROUP, MOE_INTER), dtype=torch.uint8)
    w3_scales = torch.empty_like(w1_scales)
    w2_scales = torch.empty((n_experts, MOE_INTER // MX_GROUP, D), dtype=torch.uint8)

    def generate_expert(e):
        seed = seed_base + e * 3
        generator = torch.Generator().manual_seed(seed)
        w1, w1s = quantize_mxfp4_weight(torch.randn(MOE_INTER, D, generator=generator) * dequant_std["w1"])
        w3, w3s = quantize_mxfp4_weight(torch.randn(MOE_INTER, D, generator=generator) * dequant_std["w3"])
        w2, w2s = quantize_mxfp4_weight(torch.randn(D, MOE_INTER, generator=generator) * dequant_std["w2"])
        w1_tiles = pack_mxfp4_weight_tiles(w1, MX_K_TILE, MX_MM_INTER_TILE)
        w1_weights[e] = w1_tiles.reshape(MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS)
        w3_tiles = pack_mxfp4_weight_tiles(w3, MX_K_TILE, MX_MM_INTER_TILE)
        w3_weights[e] = w3_tiles.reshape(MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS)
        w2_tiles = pack_mxfp4_weight_tiles(w2, MX_K_TILE, MX_W2_D_OUT_TILE)
        w2_weights[e] = w2_tiles.reshape(MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS)
        w1_scales[e] = w1s.transpose(0, 1).contiguous().view(torch.uint8)
        w3_scales[e] = w3s.transpose(0, 1).contiguous().view(torch.uint8)
        w2_scales[e] = w2s.transpose(0, 1).contiguous().view(torch.uint8)

    with ThreadPoolExecutor(max_workers=min(4, n_experts)) as pool:
        list(pool.map(generate_expert, range(n_experts)))

    def pack_expert_scales(scales):
        packed = pack_mx_b_scale(scales.flatten(0, 1))
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        return packed.view(e8m0) if e8m0 is not None else packed

    return (
        w1_weights,
        pack_expert_scales(w1_scales),
        w3_weights,
        pack_expert_scales(w3_scales),
        w2_weights,
        pack_expert_scales(w2_scales),
    )


def _normalise_num_tokens(value):
    """Return validated local-token counts for every EP source rank."""

    counts = torch.as_tensor(value, dtype=torch.int32).reshape(-1)
    if counts.numel() == 1:
        counts = counts.repeat(EP_SIZE)
    if counts.numel() != EP_SIZE:
        raise ValueError(f"num_tokens must be a scalar or contain {EP_SIZE} counts, got shape {tuple(counts.shape)}")
    if bool((counts < 0).any()) or bool((counts > MOE_TOKENS).any()):
        raise ValueError(f"num_tokens values must be in [0, {MOE_TOKENS}], got {counts.tolist()}")
    if BALANCED_ROUTING:
        if not bool((counts == counts[0]).all()) or int(counts[0]) * TOPK % EP_SIZE:
            raise ValueError("balanced routing requires equal token counts and per-rank routes divisible by EP")
    return counts.contiguous()


def _route_bias():
    """Build correction biases for TopK expert ids spread across ranks."""

    selected = torch.arange(TOPK, dtype=torch.int64) * max(1, N_EXPERTS // TOPK)
    selected = torch.remainder(selected, N_EXPERTS)
    bias = torch.full((N_EXPERTS,), -1.0, dtype=torch.float32)
    bias[selected] = 4.0
    return bias


def build_moe_specs(num_tokens=MOE_TOKENS, *, initialize=True):
    """Build HC, expert weights, EP outputs, and runtime specifications."""

    counts = _normalise_num_tokens(num_tokens)
    torch.manual_seed(41)
    _unused_x = (torch.randn(EP_SIZE, MOE_TOKENS, D) * 0.25).to(torch.bfloat16)
    norm_weight = torch.ones(EP_SIZE, D, dtype=torch.bfloat16)
    gate_weight = torch.randn(N_EXPERTS, D) / D**0.5
    gate_weight = gate_weight.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
    correction_bias = _route_bias().unsqueeze(0).expand(EP_SIZE, -1).contiguous()

    shapes = {
        "norm_weight": [EP_SIZE, D],
        "gate_weight": [EP_SIZE, N_EXPERTS, D],
        "correction_bias": [EP_SIZE, N_EXPERTS],
        "routed_w1": [EP_SIZE, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
        "routed_w1_scale": [EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        "routed_w2": [EP_SIZE, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
        "routed_w2_scale": [EP_SIZE, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
        "routed_w3": [EP_SIZE, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
        "routed_w3_scale": [EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        "shared_w1": [EP_SIZE, D, MOE_INTER],
        "shared_w1_scale": [EP_SIZE, D // MX_GROUP, MOE_INTER],
        "shared_w2": [EP_SIZE, MOE_INTER, D],
        "shared_w2_scale": [EP_SIZE, MOE_INTER // MX_GROUP, D],
        "shared_w3": [EP_SIZE, D, MOE_INTER],
        "shared_w3_scale": [EP_SIZE, D // MX_GROUP, MOE_INTER],
        "x_hc": [EP_SIZE, MOE_TOKENS, HC_MULT, D],
        "pre_mix": [EP_SIZE, MOE_TOKENS, HC_MULT],
        "hc_ffn_fn": [EP_SIZE, MIX_HC, HC_DIM],
        "hc_ffn_scale": [EP_SIZE, 3],
        "hc_ffn_base": [EP_SIZE, MIX_HC],
        "next_pre_mix": [EP_SIZE, MOE_TOKENS, HC_MULT],
        "x_mixed": [EP_SIZE, MOE_TOKENS, D],
        "x_next": [EP_SIZE, MOE_TOKENS, HC_MULT, D],
        "num_tokens": [EP_SIZE],
        "force_topk": [EP_SIZE, MOE_TOKENS, TOPK],
    }
    values = {"norm_weight": norm_weight, "gate_weight": gate_weight, "correction_bias": correction_bias}
    if initialize:
        shards = [
            gen_routed_mxfp4_weights(N_LOCAL_EXPERTS, ROUTED_DEQUANT_STD, seed_base=rank * N_LOCAL_EXPERTS * 3)
            for rank in range(EP_SIZE)
        ]
        routed_names = ("routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale", "routed_w2", "routed_w2_scale")
        for index, name in enumerate(routed_names):
            values[name] = torch.stack([shard[index] for shard in shards])
        for name, out_dim, in_dim, std, chan_cv, seed in (
            ("shared_w1", MOE_INTER, D, 1.71e-2, 0.50, 101),
            ("shared_w3", MOE_INTER, D, 1.70e-2, 0.50, 102),
            ("shared_w2", D, MOE_INTER, 1.68e-2, 0.33, 103),
        ):
            weight, scale = generate_mxfp8_weight_kn(out_dim, in_dim, std, chan_cv=chan_cv, seed=seed)
            values[name] = weight.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
            values[name + "_scale"] = scale.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()

    generator = torch.Generator().manual_seed(3)
    magnitudes = (0.5 + torch.arange(MOE_TOKENS) % 4).reshape(1, -1, 1, 1)
    x_hc = torch.randn(EP_SIZE, MOE_TOKENS, HC_MULT, D, generator=generator) * magnitudes
    values["x_hc"] = x_hc.bfloat16().float()
    values["pre_mix"] = torch.sigmoid(torch.randn(EP_SIZE, MOE_TOKENS, HC_MULT, generator=generator)) + HC_EPS
    hc_ffn_fn = torch.randn(MIX_HC, HC_DIM, generator=generator) / math.sqrt(HC_DIM)
    hc_ffn_scale = torch.randn(3, generator=generator)
    hc_ffn_base = torch.randn(MIX_HC, generator=generator)
    values["hc_ffn_fn"] = hc_ffn_fn.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
    values["hc_ffn_scale"] = hc_ffn_scale.unsqueeze(0).expand(EP_SIZE, -1).contiguous()
    values["hc_ffn_base"] = hc_ffn_base.unsqueeze(0).expand(EP_SIZE, -1).contiguous()
    values["num_tokens"] = counts
    force_topk = torch.zeros(EP_SIZE, MOE_TOKENS, TOPK, dtype=torch.int32)
    for rank in range(EP_SIZE):
        active = int(counts[rank])
        expanded = active * TOPK
        slots = torch.arange(expanded, dtype=torch.int64)
        offsets = slots + expanded * rank + rank
        dst = offsets % EP_SIZE
        local_expert = offsets // EP_SIZE % N_LOCAL_EXPERTS
        force_topk[rank, :active] = (dst * N_LOCAL_EXPERTS + local_expert).reshape(active, TOPK)
    values["force_topk"] = force_topk

    fp8 = torch.float8_e4m3fn
    e8m0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    nonresident = {"x_hc", "pre_mix", "next_pre_mix", "x_mixed", "x_next", "num_tokens"}
    specs = []
    for name in PARAM_NAMES:
        if name == "moe_epoch":
            specs.append(ScalarSpec(name, torch.int32, 1, compile_runtime=True))
            continue
        dtype = torch.float32
        if name in ("norm_weight", "x_mixed", "gate_weight"):
            dtype = torch.bfloat16
        elif name in {"num_tokens", "force_topk"}:
            dtype = torch.int32
        elif name.startswith("routed_"):
            dtype = e8m0 if name.endswith("_scale") else torch.uint8
        elif name.startswith("shared_"):
            dtype = e8m0 if name.endswith("_scale") else fp8
        value = values.get(name)
        init = None if value is None else partial(torch.clone, value)
        resident = None if name in nonresident else "stacked"
        specs.append(TensorSpec(name, shapes[name], dtype, init_value=init, resident=resident))
    return specs
