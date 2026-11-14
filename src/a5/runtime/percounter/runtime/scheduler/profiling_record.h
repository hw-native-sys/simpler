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
#include "scheduler/profiling_types.h"
#if defined(__CPU_SIM)
#include <pthread.h>
#include "aicpu/device_time.h"
#endif

namespace simpler::percounter {
namespace {
inline __aicore__ uint64_t percounter_scalar_clock() {
#if defined(__CCE_AICORE__)
    uint64_t value;
    asm volatile("MOV %0, SYS_CNT\n" : "=&l"(value) : : "memory");
    return value;
#elif defined(__CPU_SIM)
    return sys_cnt_now_ticks();
#else
    return 0;
#endif
}

inline __aicore__ void percounter_scalar_raw_barrier() {
#if defined(__CCE_AICORE__)
    dsb((mem_dsb_t)0);
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

// Each core owns a cache-line-aligned partition and local cursors. Records
// occupy consecutive 8-byte words, sharing each 64-byte cache line.
// No other core reads or writes this partition before finish publishes it.
struct PercounterScalarTrace {
    __gm__ uint64_t *records{nullptr};
    __gm__ PercounterProfileBuffer *buffer{nullptr};
    uint32_t storage_words{0};
    uint32_t depth{0};
    uint32_t dropped{0};
    uint64_t function_min_cycles{0};
    uint64_t short_intervals{0};
    uint64_t poll_loops{0};
    uint64_t window_start{0};
    uint64_t write_cycles{0};

    inline __aicore__ __attribute__((noinline)) void write(const SchedulerScalarRecord &record) {
        if (record.end_time < record.start_time) {
            if (dropped != UINT32_MAX) ++dropped;
            return;
        }
        const uint64_t offset = record.start_time - window_start;
        const uint64_t duration = record.end_time - record.start_time;
        const bool compact = record.task_ref == -1 && record.count == 1 && offset <= UINT32_MAX &&
                             duration <= UINT16_MAX && (record.kind & UINT32_C(0xff80ff00)) == 0;
        const uint32_t needed = compact ? 1 : 5;
        if (storage_words > PERCOUNTER_SCALAR_WORD_CAPACITY - needed) {
            if (dropped != UINT32_MAX) ++dropped;
            return;
        }
        auto *words = records + storage_words;
        if (compact) {
            words[0] =
                offset | (duration << 32) | (uint64_t{record.kind & 0xffU} << 48) | (uint64_t{record.kind >> 16} << 56);
        } else {
            words[0] = PERCOUNTER_SCALAR_EXTENDED | record.kind;
            words[1] = record.start_time;
            words[2] = record.end_time;
            words[3] = static_cast<uint64_t>(record.task_ref);
            words[4] = record.count;
        }
        storage_words += needed;
    }

    inline __aicore__ void clean_records(uint32_t first, uint64_t *clean_end = nullptr) {
        auto *base = reinterpret_cast<__gm__ uint8_t *>(records);
        const uint64_t end = uint64_t{storage_words} * sizeof(uint64_t);
        for (uint64_t offset = (uint64_t{first} * sizeof(uint64_t)) & ~UINT64_C(63); offset < end; offset += 64)
#if defined(__CCE_AICORE__)
            dcci(base + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
            (void)base;
#endif
        if (clean_end != nullptr) *clean_end = percounter_scalar_clock();
        percounter_scalar_raw_barrier();
    }

    inline __aicore__ __attribute__((noinline)) void append(
        ProfilePhase kind, uint64_t begin, uint64_t end, int64_t task_ref = -1, uint32_t record_depth = 0,
        uint32_t calls = 1
    ) {
        if (buffer == nullptr) return;
        write(
            {begin, end, task_ref, static_cast<uint32_t>(kind) | ((record_depth == 0 ? depth + 1 : record_depth) << 16),
             calls}
        );
    }

    inline __aicore__ __attribute__((always_inline)) void
    append_memory(ProfilePhase kind, uint64_t begin, uint64_t end, uint32_t calls = 1) {
        const uint64_t offset = begin - window_start;
        const uint64_t duration = end - begin;
        if (calls == 1 && offset <= UINT32_MAX && duration <= UINT16_MAX && depth < 127 &&
            storage_words < PERCOUNTER_SCALAR_WORD_CAPACITY) {
            records[storage_words++] =
                offset | (duration << 32) | (static_cast<uint64_t>(kind) << 48) | (uint64_t{depth + 1} << 56);
            return;
        }
        write({begin, end, -1, static_cast<uint32_t>(kind) | ((depth + 1) << 16), calls});
    }

    inline __aicore__ void end_iteration(bool progress) {
        if (buffer != nullptr && !progress) ++poll_loops;
    }

    // Only semantic function spans are duration-filtered. Primitive memory
    // operations, backoff and recorder intervals always use append directly.
    inline __aicore__ void
    append_function(ProfilePhase kind, uint64_t begin, uint64_t end, int64_t task_ref = -1, uint32_t record_depth = 0) {
        if (buffer == nullptr) return;
        if (end >= begin && end - begin < function_min_cycles) {
            ++short_intervals;
            return;
        }
        append(kind, begin, end, task_ref, record_depth);
    }

    inline __aicore__ __attribute__((noinline)) void finish(uint64_t begin) {
        if (buffer == nullptr) return;
        // Evicted lines are already in GM; one whole-cache clean covers every remaining dirty trace line.
#if defined(__CCE_AICORE__)
        dcci(records, ENTIRE_DATA_CACHE, CACHELINE_OUT);
#endif
        const uint64_t clean_end = percounter_scalar_clock();
        percounter_scalar_raw_barrier();
        const uint64_t end = percounter_scalar_clock();
        write_cycles += end - begin;
        const uint32_t first = storage_words;
        write({begin, end, -1, static_cast<uint32_t>(ProfilePhase::TraceFinish) | (1U << 16), 1});
        write({begin, clean_end, -1, static_cast<uint32_t>(ProfilePhase::TraceFinishCleanEntire) | (2U << 16), 1});
        write({clean_end, end, -1, static_cast<uint32_t>(ProfilePhase::TraceFinishBarrier) | (2U << 16), 1});
        write({window_start, end, -1, static_cast<uint32_t>(ProfilePhase::Window), 1});
        // Publication of the terminal observer rows lies outside their window.
        clean_records(first);
        buffer->scalar_dropped = dropped;
        buffer->scalar_write_cycles = write_cycles;
        buffer->scalar_poll_loops = poll_loops;
        buffer->scalar_short_intervals = short_intervals;
        buffer->scalar_function_min_cycles = function_min_cycles;
        buffer->scalar_window_start = window_start;
        buffer->scalar_storage_words = storage_words;
        auto *header = reinterpret_cast<__gm__ uint8_t *>(buffer);
        const uint64_t bytes = sizeof(PercounterProfileBuffer);
        for (uint64_t offset = 0; offset < bytes; offset += 64)
#if defined(__CCE_AICORE__)
            dcci(header + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
            (void)header;
#endif
        percounter_scalar_raw_barrier();
    }
};

template <typename = void>
struct ProfileCompilation {
    static constexpr bool enabled = false;
};
}  // namespace
}  // namespace simpler::percounter
