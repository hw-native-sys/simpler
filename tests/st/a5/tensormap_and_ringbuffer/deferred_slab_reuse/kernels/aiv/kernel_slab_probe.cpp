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

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>
#include "async_kernel_api.h"
#include "tensor.h"

struct alignas(CHIP_ALIGN_SIZE) SlabRecord {
    int32_t sequence;
    int32_t initial_count;
    int32_t initial_error;
    int32_t requested;
    int32_t registered;
    int32_t published_count;
    int32_t published_error;
    int32_t previous_checksum;
    int32_t checksum;
    int32_t address_low;
    int32_t address_high;
    int32_t valid;
    int32_t capacity;
    int32_t aligned;
    int32_t line_bytes;
    int32_t magic;
};
static_assert(CHIP_ALIGN_SIZE == 64 && sizeof(SlabRecord) == CHIP_ALIGN_SIZE);

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *counter_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *record_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *counters = reinterpret_cast<__gm__ int32_t *>(counter_tensor->buffer.addr) + counter_tensor->start_offset;
    auto *records = reinterpret_cast<__gm__ SlabRecord *>(
        reinterpret_cast<__gm__ int32_t *>(record_tensor->buffer.addr) + record_tensor->start_offset
    );
    const int32_t sequence = static_cast<int32_t>(args[2]);
    const int32_t requested = static_cast<int32_t>(args[3]);
    auto *row = records + sequence;
    AsyncCtx ctx = get_async_ctx(args);
    const bool valid = async_ctx_is_deferred(ctx) && ctx.completion_count != nullptr &&
                       ctx.completion_error_code != nullptr && ctx.completion_entries != nullptr;
    const uint32_t initial_count = valid ? *ctx.completion_count : UINT32_MAX;
    const int32_t initial_error = valid ? *ctx.completion_error_code : -1;
    const bool aligned = (reinterpret_cast<uint64_t>(counters) % CHIP_ALIGN_SIZE) == 0 &&
                         (reinterpret_cast<uint64_t>(row) % CHIP_ALIGN_SIZE) == 0;
    int32_t registered = 0;
    if (valid && aligned && initial_count == 0 && initial_error == 0 &&
        ctx.completion_capacity == MAX_COMPLETIONS_PER_TASK) {
        for (int32_t i = 0; i < requested; ++i) {
            CompletionToken token{
                reinterpret_cast<uint64_t>(counters + i * 16), static_cast<uint32_t>(i + 1), COMPLETION_ENGINE_SDMA,
                COMPLETION_TYPE_COUNTER, 0
            };
            registered += register_completion_condition(ctx, token) ? 1 : 0;
        }
    }
    // Ordinary tasks leave the slab untouched. Nonzero counts use canonical publication.
    if (requested != 0) pto2::detail::defer_flush(ctx);
    // FIN is emitted after this kernel returns, so the scheduler cannot reset this snapshot.
    const uint32_t published_count = valid ? *ctx.completion_count : UINT32_MAX;
    const int32_t published_error = valid ? *ctx.completion_error_code : -1;
    int32_t previous_checksum = 0;
    if (sequence != 0) {
        auto *previous = records + sequence - 1;
#if defined(__CPU_SIM)
        dsb(0);
#else
        dcci(reinterpret_cast<__gm__ int32_t *>(previous), SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
#endif
        pipe_barrier(PIPE_ALL);
        previous_checksum = reinterpret_cast<volatile __gm__ SlabRecord *>(previous)->checksum;
    }
    const uint64_t address = reinterpret_cast<uint64_t>(ctx.completion_count);
    row->sequence = sequence;
    row->initial_count = static_cast<int32_t>(initial_count);
    row->initial_error = initial_error;
    row->requested = requested;
    row->registered = registered;
    row->published_count = static_cast<int32_t>(published_count);
    row->published_error = published_error;
    row->previous_checksum = previous_checksum;
    row->checksum = previous_checksum + sequence + 1 + requested + registered;
    row->address_low = static_cast<int32_t>(address);
    row->address_high = static_cast<int32_t>(address >> 32);
    row->valid = valid ? 1 : 0;
    row->capacity = static_cast<int32_t>(ctx.completion_capacity);
    row->aligned = aligned ? 1 : 0;
    row->line_bytes = CHIP_ALIGN_SIZE;
    row->magic = 0x534C4142;
    pto2::detail::defer_flush_range(row, sizeof(SlabRecord));
#if defined(__CPU_SIM)
    dsb(0);
#else
    dsb(DSB_DDR);
#endif
    pipe_barrier(PIPE_ALL);
}
