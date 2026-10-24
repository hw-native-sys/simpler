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

#include "host_build_graph/runtime_types.h"

namespace hbg {

inline bool is_supported_external_tensor_layout(const simpler::hbg::TensorData &tensor) noexcept {
    if (tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS || tensor.strides[tensor.ndims - 1] != 1) return false;
    for (uint32_t i = tensor.ndims - 1; i > 0; --i) {
        const uint64_t inner = tensor.strides[i];
        const uint64_t shape = tensor.shapes[i];
        if (shape == 0 || inner > UINT64_MAX / shape || tensor.strides[i - 1] < inner * shape) return false;
    }
    return tensor.shapes[tensor.ndims - 1] != 0;
}

inline bool is_valid_graph_tensor_descriptor(const simpler::hbg::TensorData &tensor) noexcept {
    if (tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS || get_element_size(tensor.dtype) == 0) return false;
    uint64_t extent = 1;
    for (uint32_t i = 0; i < tensor.ndims; ++i) {
        if (tensor.shapes[i] == 0 || tensor.strides[i] == 0) return false;
        const uint64_t part = uint64_t{tensor.shapes[i] - 1} * tensor.strides[i];
        if (part > UINT64_MAX - extent) return false;
        extent += part;
    }
    if (extent != tensor.extent_elem_cache || extent > UINT64_MAX - tensor.start_offset ||
        extent + tensor.start_offset > UINT64_MAX / get_element_size(tensor.dtype))
        return false;
    if (tensor.owner_task_id.space() == TaskId::Space::PARAM) return tensor.buffer.addr == 0;
    if ((extent + tensor.start_offset) * get_element_size(tensor.dtype) > tensor.buffer.size) return false;
    if (tensor.buffer.addr >= HEAP_VIRTUAL_BASE) return true;
    return tensor.address_space == AddressSpace::DEVICE && is_supported_external_tensor_layout(tensor);
}

inline bool
is_valid_graph_tensor_binding(const simpler::hbg::TensorData &tensor, uint64_t heap, uint64_t bytes) noexcept {
    if (!is_valid_graph_tensor_descriptor(tensor)) return false;
    if (tensor.buffer.addr >= heap && tensor.buffer.addr - heap <= bytes)
        return tensor.buffer.size <= bytes - (tensor.buffer.addr - heap);
    return tensor.buffer.addr < HEAP_VIRTUAL_BASE && tensor.address_space == AddressSpace::DEVICE;
}

}  // namespace hbg
