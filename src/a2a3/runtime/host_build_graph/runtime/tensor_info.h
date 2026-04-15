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

#ifndef SRC_A2A3_RUNTIME_HOST_BUILD_GRAPH_RUNTIME_TENSOR_INFO_H_
#define SRC_A2A3_RUNTIME_HOST_BUILD_GRAPH_RUNTIME_TENSOR_INFO_H_

#include <cstdint>

#include "common/platform_config.h"
#include "data_type.h"
#include "tensor_arg.h"

// =============================================================================
// Profiling Configuration
// =============================================================================

#ifndef PTO2_PROFILING
#define PTO2_PROFILING 1
#endif

struct TensorInfo {
    DataType dtype;
    uint8_t ndims;
    uint16_t reserved;
    uint32_t shapes[PLATFORM_DUMP_MAX_DIMS];
    uint32_t raw_shapes[PLATFORM_DUMP_MAX_DIMS];
    uint32_t offsets[PLATFORM_DUMP_MAX_DIMS];
};

static_assert(sizeof(TensorInfo) == 64, "TensorInfo must stay compact");

struct TensorAllocationInfo {
    uint64_t base_addr;
    uint64_t size_bytes;

    bool contains(uint64_t addr) const { return addr >= base_addr && addr < base_addr + size_bytes; }
};

static_assert(sizeof(TensorAllocationInfo) == 16, "TensorAllocationInfo must stay compact");

inline TensorInfo make_tensor_info(
    DataType dtype, uint32_t ndims, const uint32_t *shapes, const uint32_t *raw_shapes = nullptr,
    const uint32_t *offsets = nullptr
) {
    TensorInfo info = {};
    info.dtype = dtype;
    info.ndims = static_cast<uint8_t>(ndims);
    for (uint32_t i = 0; i < ndims && i < PLATFORM_DUMP_MAX_DIMS; i++) {
        info.shapes[i] = shapes[i];
        info.raw_shapes[i] = (raw_shapes != nullptr) ? raw_shapes[i] : shapes[i];
        info.offsets[i] = (offsets != nullptr) ? offsets[i] : 0;
    }
    return info;
}

inline TensorInfo make_tensor_info_from_tensor_arg(const ContinuousTensor &tensor) {
    return make_tensor_info(tensor.dtype, tensor.ndims, tensor.shapes);
}

#endif  // SRC_A2A3_RUNTIME_HOST_BUILD_GRAPH_RUNTIME_TENSOR_INFO_H_
