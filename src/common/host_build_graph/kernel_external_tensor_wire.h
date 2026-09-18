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

#include <cstdint>

#include "host_build_graph/graph_execution.h"
#include "host_build_graph/runtime_types.h"

namespace hbg {

inline bool supported_kernel_graph_stride_family(const GraphTensor &tensor) noexcept {
    if (tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS || tensor.strides[tensor.ndims - 1] != 1) return false;
    for (uint32_t i = tensor.ndims - 1; i > 0; --i) {
        const uint64_t inner = tensor.strides[i];
        const uint64_t shape = tensor.shapes[i];
        if (shape != 0 && inner > UINT64_MAX / shape) return false;
        if (tensor.strides[i - 1] < inner * shape) return false;
    }
    return true;
}

// Host builds use virtual heap addresses. Restored task payloads use the
// registered heap window; Definitions retain their virtual internal addresses.
inline bool valid_kernel_graph_tensor(const GraphTensor &tensor) noexcept {
    if (!graph_tensor_wire_valid(tensor)) return false;
    if (tensor.buffer_addr >= HEAP_VIRTUAL_BASE) return true;
    return tensor.address_space == static_cast<uint8_t>(AddressSpace::DEVICE) &&
           supported_kernel_graph_stride_family(tensor);
}

inline bool valid_restored_kernel_graph_tensor(const GraphTensor &tensor, uint64_t heap, uint64_t bytes) noexcept {
    if (!graph_tensor_wire_valid(tensor)) return false;
    if (tensor.buffer_addr >= heap && tensor.buffer_addr - heap <= bytes)
        return tensor.buffer_size <= bytes - (tensor.buffer_addr - heap);
    return tensor.buffer_addr < HEAP_VIRTUAL_BASE &&
           tensor.address_space == static_cast<uint8_t>(AddressSpace::DEVICE) &&
           supported_kernel_graph_stride_family(tensor);
}

inline bool
valid_restored_kernel_graph_tensor(const simpler::hbg::Tensor &tensor, uint64_t heap, uint64_t bytes) noexcept {
    return valid_restored_kernel_graph_tensor(graph_tensor_pack(tensor), heap, bytes);
}

inline bool valid_kernel_graph_tensor(const simpler::hbg::Tensor &tensor) noexcept {
    return valid_kernel_graph_tensor(graph_tensor_pack(tensor));
}

}  // namespace hbg
