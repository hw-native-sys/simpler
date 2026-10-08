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

#include "host_build_graph/host_tensor_access.h"

#include <cstring>

bool HostTensorAccessor::add(const ChipTensor &tensor, ArgDirection direction) {
    if (tensor.address_space != AddressSpace::HOST || tensor.transfer != TensorTransfer::NONE || tensor.ndims == 0 ||
        tensor.ndims > MAX_TENSOR_DIMS || tensor.dtype >= DataType::DATA_TYPE_NUM ||
        (direction != ArgDirection::IN && direction != ArgDirection::OUT && direction != ArgDirection::INOUT)) {
        return false;
    }
    const uint64_t elem = get_element_size(tensor.dtype);
    if (tensor.start_offset > tensor.buffer.size / elem) return false;
    for (uint32_t d = 0; d < tensor.ndims; ++d) {
        if (tensor.shapes[d] == 0) return true;
    }
    uint64_t extent = 1;
    for (uint32_t d = 0; d < tensor.ndims; ++d) {
        if (tensor.strides[d] == 0) return false;
        const uint64_t span = static_cast<uint64_t>(tensor.shapes[d] - 1) * tensor.strides[d];
        if (span > UINT64_MAX - extent) return false;
        extent += span;
    }
    if (extent > tensor.buffer.size / elem - tensor.start_offset || tensor.buffer.addr == 0 ||
        tensor.buffer.size > UINT64_MAX - tensor.buffer.addr)
        return false;
    const uint64_t base = tensor.buffer.addr + tensor.start_offset * elem;
    const uint64_t size = extent * elem;
    for (const Region &other : regions_) {
        if (base < other.base + other.size && other.base < base + size &&
            (direction != ArgDirection::IN || other.direction != ArgDirection::IN))
            return false;
    }
    regions_.push_back({base, size, direction});
    return true;
}

const HostTensorAccessor::Region *HostTensorAccessor::find(uint64_t addr, uint64_t bytes) const {
    for (const Region &region : regions_) {
        if (addr >= region.base && addr - region.base <= region.size && bytes <= region.size - (addr - region.base))
            return &region;
    }
    return nullptr;
}

bool HostTensorAccessor::read(uint64_t addr, void *dst, uint64_t bytes) const {
    const Region *region = find(addr, bytes);
    if (region == nullptr || region->direction == ArgDirection::OUT) return false;
    std::memcpy(dst, reinterpret_cast<const void *>(addr), bytes);
    return true;
}

bool HostTensorAccessor::write(uint64_t addr, const void *src, uint64_t bytes) const {
    const Region *region = find(addr, bytes);
    if (region == nullptr || region->direction == ArgDirection::IN) return false;
    std::memcpy(reinterpret_cast<void *>(addr), src, bytes);
    return true;
}

bool host_tensor_read(HostTensorAccessor *accessor, uint64_t addr, void *dst, uint64_t bytes) {
    return accessor != nullptr && accessor->read(addr, dst, bytes);
}

bool host_tensor_write(HostTensorAccessor *accessor, uint64_t addr, const void *src, uint64_t bytes) {
    return accessor != nullptr && accessor->write(addr, src, bytes);
}
