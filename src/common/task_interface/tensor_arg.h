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
/**
 * ContinuousTensor - Compact tensor descriptor for orchestration arguments
 *
 * 40 bytes, trivially copyable, suitable for DMA and device-side access.
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include "data_type.h"  // NOLINT(build/include_subdir)

constexpr int CONTINUOUS_TENSOR_MAX_DIMS = 5;

struct ContinuousTensor {
    uint64_t data;                                // Host/device memory address
    uint32_t shapes[CONTINUOUS_TENSOR_MAX_DIMS];  // Shape per dim (element count)
    uint32_t ndims;                               // Number of dimensions (1..5)
    DataType dtype;                               // DataType : uint8_t

    uint64_t nbytes() const {
        uint64_t total = 1;
        for (uint32_t i = 0; i < ndims; i++) total *= shapes[i];
        return total * get_element_size(dtype);
    }

    template <typename T>
    T* data_as() const {
        return reinterpret_cast<T*>(static_cast<uintptr_t>(data));
    }
};

static_assert(
    std::is_trivially_copyable<ContinuousTensor>::value, "ContinuousTensor must be trivially copyable for DMA");
static_assert(
    sizeof(ContinuousTensor) == 40, "ContinuousTensor size must be exactly 40B (33B fields + 7B tail padding)");

/**
 * TensorArgType - Distinguishes inputs, outputs, and in-place updates
 */
enum class TensorArgType : int32_t {
    INPUT = 0,            // Read-only input buffer
    OUTPUT = 1,           // Write-only output buffer (runtime allocates)
    INOUT = 2,            // Read-then-write: modifier for downstream
    OUTPUT_EXISTING = 3,  // Write-only existing tensor: skips OverlapMap lookup, depends on creator
    NO_DEP = 4,           // No-dependency existing tensor: skips OverlapMap lookup, no publish
};
