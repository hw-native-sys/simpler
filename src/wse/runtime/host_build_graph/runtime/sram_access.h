/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#pragma once
#include "common/hardware_spec.h"
#include "tensor.h"
#include <cstdint>

namespace wse {
struct SramAccess {
    uint64_t regions_offset;
    uint64_t allocations_offset;
    uint32_t region_count;
    uint32_t allocation_count;
};
struct SramView {
    uint64_t address = 0;
    uint64_t bytes = 0;
    uint32_t owner = 0;
};
inline bool tensor_span(const Tensor &tensor, uint64_t &address, uint64_t &bytes) noexcept {
    const auto width = get_element_size(tensor.dtype);
    if (!width || !tensor.ndims || tensor.ndims > MAX_TENSOR_DIMS || !tensor.buffer.addr ||
        tensor.buffer.size > UINT64_MAX - tensor.buffer.addr)
        return false;
    uint64_t last = tensor.start_offset;
    for (uint32_t i = 0; i < tensor.ndims; ++i) {
        if (!tensor.shapes[i] || !tensor.strides[i] || tensor.shapes[i] - 1 > (UINT64_MAX - last) / tensor.strides[i])
            return false;
        last += static_cast<uint64_t>(tensor.shapes[i] - 1) * tensor.strides[i];
    }
    if (last >= tensor.buffer.size / width) return false;
    address = tensor.buffer.addr + tensor.start_offset * width;
    bytes = (last - tensor.start_offset + 1) * width;
    return true;
}
inline bool resolve_sram(
    uint64_t base, const SramAccess &access, uint64_t address, uint64_t bytes, SramView &out,
    uint32_t expected_owner = UINT32_MAX
) noexcept {
    out = {};
    if (!base || !bytes || bytes > UINT64_MAX - address) return false;
    const auto *allocations = reinterpret_cast<const SramAllocation *>(base + access.allocations_offset);
    const auto *regions = reinterpret_cast<const SramRegion *>(base + access.regions_offset);
    for (uint32_t i = 0; i < access.allocation_count; ++i) {
        const auto &a = allocations[i];
        if (address < a.base || address - a.base > a.bytes || bytes > a.bytes - (address - a.base)) continue;
        if (a.owner >= access.region_count || (expected_owner != UINT32_MAX && expected_owner != a.owner)) return false;
        const auto &r = regions[a.owner];
        if (r.reserve >= r.capacity || a.base < r.base || a.base - r.base > r.capacity - r.reserve ||
            a.bytes > r.capacity - r.reserve - (a.base - r.base))
            return false;
        out = {address, bytes, a.owner};
        return true;
    }
    return false;
}
static_assert(std::is_standard_layout_v<SramAccess> && std::is_trivially_copyable_v<SramAccess>);
}  // namespace wse
