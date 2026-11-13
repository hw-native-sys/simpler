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

#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

#include "intrinsic.h"
#include "scheduler/scheduler_memory.h"
#include "tensor.h"

// The atomic word and writeback metadata occupy different 64B cache lines.
// Never write back a stale copy of the line holding the atomic visit count.
struct alignas(128) ExecutionRecord {
    volatile uint64_t visits;
    uint64_t atomic_padding[7];
    int64_t block_plus_one;
    int64_t block_num;
    int64_t slot_plus_one;
    int64_t task_plus_one;
    int64_t metadata_padding[4];
};

static_assert(sizeof(ExecutionRecord) == 128);
static_assert(offsetof(ExecutionRecord, block_plus_one) == 64);

static inline __aicore__ __gm__ int64_t *tensor_data(int64_t tensor_arg) {
    auto *tensor = reinterpret_cast<__gm__ Tensor *>(tensor_arg);
    return reinterpret_cast<__gm__ int64_t *>(tensor->buffer.addr) + tensor->start_offset;
}
