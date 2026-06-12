# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import ctypes
import struct

from ..kernel_analyzer import read_arg_indices
from ..models import KernelShape, KernelSpec, SceneCaseContext, TraceArg, TraceScalarArg, TraceTensorArg


def resolve_builtin_args(context: SceneCaseContext, kernel: KernelSpec) -> tuple[TraceArg, ...]:
    args = _paged_attention_recipe(context, kernel)
    read_indices = read_arg_indices(kernel.source_path)
    missing = sorted(index for index in read_indices if index not in {arg.index for arg in args})
    if missing:
        raise ValueError(f"Argument recipe for {kernel.name} does not cover args indices: {missing}")
    return args


def _paged_attention_recipe(context: SceneCaseContext, kernel: KernelSpec) -> tuple[TraceArg, ...]:
    module_path = context.module_dir.as_posix()
    if "spmd_multiblock_mix" in module_path:
        total_cl = sum(int(blocks) * 3 for blocks, _ in ((2, 0), (8, 6), (12, 30), (24, 66), (48, 138)))
        return (
            TraceTensorArg(0, "output", "FLOAT32", (total_cl * 16,), role="inout"),
            TraceScalarArg(1, "base_cl", "UINT64", 0),
        )
    if "paged_attention" not in module_path:
        raise ValueError("No built-in insight trace recipe for this test module; pass --arg-spec")
    params = context.case.get("params", {})
    q_tile = 16
    block_size = int(params["block_size"])
    head_dim = int(params["head_dim"])
    scale = _scalar_value(context, "scale", default=1.0)

    if kernel.shape == KernelShape.SPMD_MIX:
        batch = int(params["batch"])
        num_heads = int(params["num_heads"])
        kv_head_num = int(params["kv_head_num"])
        context_len = int(params["context_len"])
        max_model_len = int(params["max_model_len"])
        max_num_blocks_per_req = max_model_len // block_size
        q_tile = 16 if num_heads <= 16 else 64
        q_loop = (num_heads + q_tile - 1) // q_tile
        total_logical_blocks = batch * q_loop
        total_blocks = batch * ((context_len + block_size - 1) // block_size)
        return (
            TraceTensorArg(0, "query", "BFLOAT16", (batch, num_heads, head_dim)),
            TraceTensorArg(1, "key_cache", "BFLOAT16", (total_blocks, block_size, kv_head_num, head_dim)),
            TraceTensorArg(2, "value_cache", "BFLOAT16", (total_blocks, block_size, kv_head_num, head_dim)),
            TraceTensorArg(3, "block_table", "INT32", (batch, max_num_blocks_per_req)),
            TraceTensorArg(4, "context_lens", "INT32", (batch,)),
            TraceTensorArg(5, "out", "FLOAT32", (batch, num_heads, head_dim)),
            TraceTensorArg(6, "sij_fifo", "FLOAT32", (1,)),
            TraceTensorArg(7, "pij_fifo", "BFLOAT16", (1,)),
            TraceTensorArg(8, "oi_fifo", "FLOAT32", (1,)),
            TraceScalarArg(9, "scale_value", "FLOAT32_BITS", _f32_bits(float(scale)), "bits"),
            TraceScalarArg(10, "num_heads", "UINT64", num_heads),
            TraceScalarArg(11, "head_dim", "UINT64", head_dim),
            TraceScalarArg(12, "block_size", "UINT64", block_size),
            TraceScalarArg(13, "max_num_blocks_per_req", "UINT64", max_num_blocks_per_req),
            TraceScalarArg(14, "q_loop", "UINT64", q_loop),
            TraceScalarArg(15, "total_logical_blocks", "UINT64", total_logical_blocks),
            TraceScalarArg(16, "q_tile", "UINT64", q_tile),
        )

    recipes: dict[str, tuple[TraceArg, ...]] = {
        "QK": (
            TraceTensorArg(0, "qi", "BFLOAT16", (q_tile, head_dim)),
            TraceTensorArg(1, "kj", "BFLOAT16", (block_size, head_dim)),
            TraceTensorArg(2, "sij", "FLOAT32", (q_tile, block_size)),
            TraceScalarArg(4, "head_dim", "UINT64", head_dim),
            TraceScalarArg(5, "block_size", "UINT64", block_size),
        ),
        "SF": (
            TraceTensorArg(0, "sij", "FLOAT32", (q_tile, block_size)),
            TraceTensorArg(1, "pij", "BFLOAT16", (q_tile, block_size)),
            TraceTensorArg(2, "mij", "FLOAT32", (q_tile,)),
            TraceTensorArg(3, "lij", "FLOAT32", (q_tile,)),
            TraceScalarArg(4, "scale", "FLOAT32_BITS", _f32_bits(float(scale)), "bits"),
        ),
        "PV": (
            TraceTensorArg(0, "pij", "BFLOAT16", (q_tile, block_size)),
            TraceTensorArg(1, "vj", "BFLOAT16", (block_size, head_dim)),
            TraceTensorArg(2, "oi_new", "FLOAT32", (q_tile, head_dim)),
            TraceScalarArg(4, "block_size", "UINT64", block_size),
            TraceScalarArg(5, "head_dim", "UINT64", head_dim),
        ),
        "UP": (
            TraceTensorArg(0, "mij", "FLOAT32", (q_tile,)),
            TraceTensorArg(1, "lij", "FLOAT32", (q_tile,)),
            TraceTensorArg(2, "oi_new", "FLOAT32", (q_tile, head_dim)),
            TraceTensorArg(3, "mi", "FLOAT32", (q_tile,)),
            TraceTensorArg(4, "li", "FLOAT32", (q_tile,)),
            TraceTensorArg(5, "oi", "FLOAT32", (q_tile, head_dim)),
            TraceTensorArg(6, "dst", "FLOAT32", (q_tile, head_dim)),
            TraceScalarArg(7, "is_first", "UINT64", 1),
            TraceScalarArg(8, "is_last", "UINT64", 1),
            TraceScalarArg(10, "head_dim", "UINT64", head_dim),
        ),
    }
    if kernel.name not in recipes:
        raise ValueError(f"No paged_attention recipe for kernel {kernel.name}")
    return recipes[kernel.name]


def _scalar_value(context: SceneCaseContext, name: str, default):
    try:
        builder = context.test_class().generate_args(context.case.get("params", {}))
    except (TypeError, ValueError, KeyError):
        return default
    for spec in getattr(builder, "specs", []):
        if getattr(spec, "name", None) != name:
            continue
        value = spec.value
        if isinstance(value, ctypes._SimpleCData):
            return value.value
        return value
    return default


def _f32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", value))[0]
