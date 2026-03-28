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
 * TaskArg - Tagged union for orchestration function arguments
 *
 * Each TaskArg carries either a Tensor (ptr/shape/ndims/dtype) or a Scalar
 * (uint64_t value). Host side builds a TaskArg[] array which is copied to
 * device; AICPU reads fields directly.
 *
 * This struct is trivially copyable (required for DMA) and fixed at 48 bytes.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "data_type.h"  // NOLINT(build/include_subdir)

constexpr int TASK_ARG_MAX_DIMS = 5;

enum class TaskArgKind : uint32_t {
    TENSOR = 0,
    SCALAR = 1,
};

struct TaskArg {
    TaskArgKind kind;  // 4B: TENSOR or SCALAR

    union {
        struct {                                 // --- Tensor metadata ---
            uint64_t data;                       // Host/device memory address
            uint32_t shapes[TASK_ARG_MAX_DIMS];  // Shape per dim (element count)
            uint32_t ndims;                      // Number of dimensions (1..5)
            DataType dtype;                      // DataType : uint32_t
        } tensor;                                // 36B

        uint64_t scalar;  // --- Scalar value ---  8B
    };

    // Compute total bytes for this tensor from shape x element_size
    uint64_t nbytes() const {
        uint64_t total = 1;
        for (uint32_t i = 0; i < tensor.ndims; i++) total *= tensor.shapes[i];
        return total * get_element_size(tensor.dtype);
    }

    // Get raw pointer to tensor data
    template <typename T>
    T* data() const {
        return reinterpret_cast<T*>(static_cast<uintptr_t>(tensor.data));
    }

    // Reinterpret scalar bits as target type (delegates to from_u64)
    template <typename T>
    T value_as() const {
        return from_u64<T>(scalar);
    }
};

static_assert(std::is_trivially_copyable<TaskArg>::value, "TaskArg must be trivially copyable for DMA");
static_assert(sizeof(TaskArg) == 48, "TaskArg size must be exactly 48B for stable ABI");
