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

#include <stdint.h>
#include "data_type.h"

enum class KernelErrorCode : int32_t {
    INVALID_OUTPUT = 105,
    INDEX_OUT_OF_RANGE = 106,
    NON_FINITE_VALUE = 107,
};

PTO_DEVICE_FUNC constexpr bool is_kernel_error(int32_t code) {
    return code == static_cast<int32_t>(KernelErrorCode::INVALID_OUTPUT) ||
           code == static_cast<int32_t>(KernelErrorCode::INDEX_OUT_OF_RANGE) ||
           code == static_cast<int32_t>(KernelErrorCode::NON_FINITE_VALUE);
}

// Bounds are signed and upper is exclusive. The producer supplies the argument
// index and the data value, which cannot be inferred from the tensor metadata.
struct KernelErrorDetail {
    int64_t value;
    int64_t lower;
    int64_t upper;
    int32_t arg_index;
};

namespace kernel_validation {
// Only the lane's return word carries this bit. Published runtime/stream
// status always uses the original code. A code-only return never reads the
// detail left in the slab by a previous invocation.
inline constexpr int32_t detail_bit = 1 << 30;

PTO_DEVICE_FUNC constexpr bool has_detail(int32_t raw) {
    return (raw & detail_bit) != 0 && is_kernel_error(raw ^ detail_bit);
}

PTO_DEVICE_FUNC constexpr int32_t code(int32_t raw) { return has_detail(raw) ? raw ^ detail_bit : raw; }

// Fits the existing padding of the shared headers. Code is cleared at run
// initialization. The scheduler that wins the first-error latch fills this
// record, and the device copies it to the run's epoch-owned terminal result.
struct Failure {
    uint64_t task;
    int64_t value;
    int64_t lower;
    int64_t upper;
    int32_t func;
    int32_t arg_index;
    int32_t block;
    int32_t code;
};
static_assert(sizeof(Failure) == 48);

PTO_DEVICE_FUNC inline void capture(
    Failure &record, int32_t raw, uint64_t task, int32_t func, int32_t block, const volatile KernelErrorDetail &detail
) {
    const bool present = has_detail(raw);
    record.task = task;
    record.value = present ? detail.value : 0;
    record.lower = present ? detail.lower : 0;
    record.upper = present ? detail.upper : 0;
    record.func = func;
    record.arg_index = present ? detail.arg_index : -1;
    record.block = block;
    record.code = code(raw);
}
}  // namespace kernel_validation
