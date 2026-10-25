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
#include "inner_kernel.h"
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

static inline __aicore__ void record_execution(__gm__ int64_t *args, int32_t slot) {
    auto *records = reinterpret_cast<__gm__ ExecutionRecord *>(tensor_data(args[0]));
    const auto expected_blocks = static_cast<int32_t>(args[1]);
    const auto mask = static_cast<uint32_t>(args[2]);
    const auto task = static_cast<int32_t>(args[3]);
    // Inspect pointer presence only; read context values through the public accessors below.
    // A missing context must fail the host assertion instead of crashing the kernel.
    if (args[SPMD_LOCAL_CONTEXT_INDEX] == 0 || args[SPMD_GLOBAL_CONTEXT_INDEX] == 0) {
        scheduler_gm_fetch_add(records[expected_blocks * 3].visits, 1);
        return;
    }
    const auto block = get_block_idx(args);
    const auto blocks = get_block_num(args);
    if (block < 0 || block >= expected_blocks || blocks != expected_blocks || !(mask & (1U << slot))) {
        scheduler_gm_fetch_add(records[expected_blocks * 3].atomic_padding[0], 1);
        return;
    }
    if (args[5] != 0) {
        auto &arrived = records[expected_blocks * 3].atomic_padding[1];
        scheduler_gm_fetch_add(arrived, 1);
        const uint64_t expected = static_cast<uint64_t>(blocks) * __builtin_popcount(mask);
        const uint64_t started = get_sys_cnt_aicore();
        while (args[5] != 2 || block != 0) {
            if (scheduler_gm_fetch_add(arrived, 0) == expected) break;
            if (get_sys_cnt_aicore() - started > PLATFORM_PROF_SYS_CNT_FREQ * 5) {
                scheduler_gm_fetch_add(records[expected_blocks * 3].atomic_padding[2], 1);
                break;
            }
        }
    }
    // Delay before the final publication, exposing consumers released on claim exhaustion.
    if (block + 1 == blocks) {
        for (volatile uint32_t delay = 0; delay < static_cast<uint32_t>(args[4]); ++delay) {}
    }
    auto *record = &records[block * 3 + slot];
    scheduler_gm_fetch_add(record->visits, 1);
    record->block_plus_one = block + 1;
    record->block_num = blocks;
    record->slot_plus_one = slot + 1;
    record->task_plus_one = task + 1;
    scheduler_publish_cache_line(&record->block_plus_one);
}
