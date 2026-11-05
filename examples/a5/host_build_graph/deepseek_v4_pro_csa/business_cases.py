# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Metamorphic CSA fixtures: rename physical pages without changing the math."""

import torch

# Every consumer of a physical page must use the same old-to-new bijection.
POOL_LAYOUTS = {
    "main_state": (("compress_state",), "compress_state_block_table", "state_slot_mapping", 4),
    "inner_state": (("inner_compress_state",), "inner_compress_state_block_table", "inner_state_slot_mapping", 4),
    "kv": (("kv_cache",), None, "ori_slot_mapping", 128),
    "cmp": (("cmp_kv",), "cmp_block_table", "cmp_slot_mapping", 128),
    "idx": (("idx_kv_cache", "idx_kv_scale"), "idx_block_table", "idx_slot_mapping", 128),
}
VARIANTS = [
    ("baseline", {}),
    *[(f"{group}_reverse", {"pages": {group: "reverse"}}) for group in POOL_LAYOUTS],
    ("all_rotate", {"pages": dict.fromkeys(POOL_LAYOUTS, "rotate")}),
    ("all_scramble", {"pages": dict.fromkeys(POOL_LAYOUTS, "scramble")}),
    ("all_reverse", {"pages": dict.fromkeys(POOL_LAYOUTS, "reverse")}),
    ("idx_rotate", {"pages": {"idx": "rotate"}}),
    ("scale_write_poison", {"poison_scale": True}),
    (
        "all_pages_scale_poison",
        {
            "pages": dict.fromkeys(POOL_LAYOUTS, "scramble"),
            "poison_scale": True,
        },
    ),
]


def page_order(count, style):
    if count <= 0:
        raise ValueError("page count must be positive")
    if style == "reverse":
        return torch.arange(count - 1, -1, -1)
    if style == "rotate":
        return torch.arange(count).roll(1)
    if style == "scramble":
        return torch.randperm(count, generator=torch.Generator().manual_seed(73 + count))
    raise ValueError(f"unknown page permutation: {style}")


def take_rows(value, order):
    # Byte indexing also supports FP8 payloads without rounding/conversion.
    rows = value.contiguous().reshape(value.shape[0], -1).view(torch.uint8)
    return rows.index_select(0, order).view(value.dtype).reshape(len(order), *value.shape[1:])


def remap_indices(value, old_to_new, block_size=1):
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError("page/slot indices must be int32 or int64")
    if (value < -1).any() or (value >= len(old_to_new) * block_size).any():
        raise ValueError("page/slot index outside the pool; only -1 is a sentinel")
    result = value.clone()
    valid = value >= 0
    selected = value[valid].long()
    result[valid] = (old_to_new[selected // block_size] * block_size + selected % block_size).to(value.dtype)
    return result


def validate_params(params):
    # Requests 0 and 2 alias state page 2; swapping requests changes read-after-write order.
    if set(params) - {"pages", "poison_scale", "resident"}:
        raise ValueError("unsupported CSA transformation; batch permutation is not equivalent for this snapshot")


def transform_inputs(values, params):
    validate_params(params)
    result = dict(values)
    for group, style in params.get("pages", {}).items():
        pools, table, slots, block_size = POOL_LAYOUTS[group]
        count = values[pools[0]].shape[0]
        order = page_order(count, style)
        inverse = torch.argsort(order)
        for name in pools:
            if values[name].shape[0] != count:
                raise ValueError("paired cache payload and scale must share their page count")
            result[name] = take_rows(values[name], order)
        if table:
            result[table] = remap_indices(values[table], inverse)
        result[slots] = remap_indices(values[slots], inverse, block_size)
        if group == "kv":
            result["window_swa_indices"] = remap_indices(values["window_swa_indices"], inverse, block_size)
    if params.get("poison_scale", False):
        # kv_and_cache_write overwrites these rows before score reads the scale pool.
        scale = result["idx_kv_scale"].clone()
        slots = result["idx_slot_mapping"]
        scale.reshape(scale.shape[0] * scale.shape[1], -1)[slots[slots >= 0]] = 2.0
        result["idx_kv_scale"] = scale
    return result


def transform_outputs(values, params):
    validate_params(params)
    result = dict(values)
    for group, style in params.get("pages", {}).items():
        for name in POOL_LAYOUTS[group][0]:
            result[name] = take_rows(values[name], page_order(values[name].shape[0], style))
    return result
