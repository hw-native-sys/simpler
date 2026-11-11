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

#include "scheduler_types.h"
#include "scheduler_scalar_policy.h"
#include "host_build_graph/sched_phase_kind.h"

#if defined(__CPU_SIM)
#include "aicpu/device_time.h"
#endif

static_assert(static_cast<uint32_t>(SchedPhaseKind::ScalarTraceFinishBarrier) < SCHEDULER_SCALAR_STORAGE_EXTENDED);

inline __aicore__ uint64_t scheduler_scalar_clock() {
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

inline __aicore__ void scheduler_scalar_raw_barrier() {
#if defined(__CCE_AICORE__)
    dsb((mem_dsb_t)0);
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

// Each core owns a cache-line-aligned partition and local cursors. Records
// occupy consecutive 16-byte storage slots, sharing each 64-byte cache line.
// No other core reads or writes this partition before finish publishes it.
struct SchedulerScalarTrace {
    __gm__ SchedulerActivityBuffer *buffer{nullptr};
    __gm__ SchedulerScalarStorageSlot *records{nullptr};
    uint64_t window_start{0};
    uint64_t write_cycles{0};
    uint64_t poll_loops{0};
    uint64_t function_min_cycles{0};
    uint64_t short_intervals{0};
    uint32_t committed{0};
    uint32_t storage_slots{0};
    uint32_t dropped{0};
    uint32_t depth{0};

    inline __aicore__ __attribute__((always_inline)) void write(const SchedulerScalarRecord &record) {
        if (record.end_time < record.start_time || committed >= SCHEDULER_SCALAR_RECORD_CAPACITY ||
            (record.kind & SCHEDULER_SCALAR_STORAGE_EXTENDED) != 0) {
            if (dropped != UINT32_MAX) ++dropped;
            return;
        }
        const uint64_t duration = record.end_time - record.start_time;
        const bool compact = record.task_ref == -1 && record.count == 1 && duration <= UINT32_MAX;
        auto *slots = records;
#if defined(__CCE_AICORE__)
        // A hint for this core's disjoint trace partition. Four slots fill a
        // line; sixteen-line lookahead overlaps the intervening record stores.
        // The terminal clean/barrier publishes the complete partition.
        if ((storage_slots & 3U) == 0 && storage_slots + 64 < 2 * SCHEDULER_SCALAR_RECORD_CAPACITY)
            dc_preload(
                reinterpret_cast<__gm__ uint64_t *>(records),
                static_cast<int64_t>(uint64_t{storage_slots + 64} * sizeof(SchedulerScalarStorageSlot))
            );
        auto *words = reinterpret_cast<__gm__ uint64_t *>(&slots[storage_slots]);
        words[0] = record.start_time;
        words[1] = static_cast<uint64_t>(record.kind | (compact ? 0 : SCHEDULER_SCALAR_STORAGE_EXTENDED)) |
                   (uint64_t{compact ? static_cast<uint32_t>(duration) : record.count} << 32);
        if (!compact) {
            words[2] = record.end_time;
            words[3] = static_cast<uint64_t>(record.task_ref);
        }
#else
        slots[storage_slots] = {
            record.start_time, record.kind | (compact ? 0 : SCHEDULER_SCALAR_STORAGE_EXTENDED),
            compact ? static_cast<uint32_t>(duration) : record.count
        };
        if (!compact)
            slots[storage_slots + 1] = {
                record.end_time, static_cast<uint32_t>(record.task_ref),
                static_cast<uint32_t>(static_cast<uint64_t>(record.task_ref) >> 32)
            };
#endif
        storage_slots += compact ? 1 : 2;
        ++committed;
    }

    inline __aicore__ void clean_records(uint32_t first, uint64_t *clean_end = nullptr) {
        auto *base = reinterpret_cast<__gm__ uint8_t *>(records);
        const uint64_t end = uint64_t{storage_slots} * sizeof(SchedulerScalarStorageSlot);
        for (uint64_t offset = (uint64_t{first} * sizeof(SchedulerScalarStorageSlot)) & ~UINT64_C(63); offset < end;
             offset += 64)
#if defined(__CCE_AICORE__)
            dcci(base + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
            (void)base;
#endif
        if (clean_end != nullptr) *clean_end = scheduler_scalar_clock();
        scheduler_scalar_raw_barrier();
    }

    inline __aicore__ __attribute__((noinline)) void append(
        SchedPhaseKind kind, uint64_t begin, uint64_t end, int64_t task_ref = -1, uint32_t record_depth = 0,
        uint32_t calls = 1
    ) {
        if (buffer == nullptr) return;
        write(
            {begin, end, task_ref, static_cast<uint32_t>(kind) | ((record_depth == 0 ? depth + 1 : record_depth) << 16),
             calls}
        );
    }

    inline __aicore__ __attribute__((always_inline)) void
    append_memory(SchedPhaseKind kind, uint64_t begin, uint64_t end, uint32_t calls = 1) {
        if (buffer == nullptr) return;
        write({begin, end, -1, static_cast<uint32_t>(kind) | ((depth + 1) << 16), calls});
    }

    inline __aicore__ void end_iteration(bool progress) {
        if (buffer != nullptr && !progress) ++poll_loops;
    }

    // Only semantic function spans are duration-filtered. Primitive memory
    // operations, backoff and recorder intervals always use append directly.
    inline __aicore__ void append_function(
        SchedPhaseKind kind, uint64_t begin, uint64_t end, int64_t task_ref = -1, uint32_t record_depth = 0
    ) {
        if (buffer == nullptr) return;
        if (end >= begin && end - begin < function_min_cycles) {
            ++short_intervals;
            return;
        }
        append(kind, begin, end, task_ref, record_depth);
    }

    inline __aicore__ __attribute__((noinline)) void finish(uint64_t begin) {
        if (buffer == nullptr) return;
        uint64_t clean_end = 0;
        clean_records(0, &clean_end);
        const uint64_t end = scheduler_scalar_clock();
        write_cycles += end - begin;
        const uint32_t first = storage_slots;
        write({begin, end, -1, static_cast<uint32_t>(SchedPhaseKind::ScalarTraceFinish) | (1U << 16), 1});
        write(
            {begin, clean_end, -1, static_cast<uint32_t>(SchedPhaseKind::ScalarTraceFinishClean) | (2U << 16),
             (first + 3) / 4}
        );
        write({clean_end, end, -1, static_cast<uint32_t>(SchedPhaseKind::ScalarTraceFinishBarrier) | (2U << 16), 1});
        write({window_start, end, -1, static_cast<uint32_t>(SchedPhaseKind::ScalarWindow), 1});
        // Publication of the terminal observer rows lies outside their window.
        clean_records(first);
        buffer->scalar_committed = committed;
        buffer->scalar_dropped = dropped;
        buffer->scalar_write_cycles = write_cycles;
        buffer->scalar_poll_loops = poll_loops;
        buffer->scalar_short_intervals = short_intervals;
        buffer->scalar_function_min_cycles = function_min_cycles;
        buffer->scalar_storage_format = 1;
        buffer->scalar_storage_slots = storage_slots;
        auto *header = reinterpret_cast<__gm__ uint8_t *>(&buffer->scalar_committed);
        const uint64_t bytes = sizeof(SchedulerActivityBuffer) - offsetof(SchedulerActivityBuffer, scalar_committed);
        for (uint64_t offset = 0; offset < bytes; offset += 64)
#if defined(__CCE_AICORE__)
            dcci(header + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
            (void)header;
#endif
        scheduler_scalar_raw_barrier();
    }
};

// The slot belongs to this physical core (or one CPU test thread). Binding is
// confined to the resident scheduler lifetime and never survives its return.
static constexpr bool kSchedulerScalarInstrumentation = SchedulerScalarCompilation<>::enabled;
#if defined(__CCE_AICORE__)
[[block_local]] static SchedulerScalarTrace *scheduler_scalar_active = nullptr;
// Primitive wrappers own the slot address calculation and load its current
// binding; no observer pointer survives initialization or teardown.
static inline __aicore__ __attribute__((always_inline)) auto scheduler_scalar_slot() {
    auto *slot = &scheduler_scalar_active;
    // The register identity preserves the core-local address through folding.
    asm("MOV %0, %0" : "+l"(slot));
    return slot;
}
static inline __aicore__ SchedulerScalarTrace *scheduler_scalar_current() {
    if constexpr (kSchedulerScalarInstrumentation) return *scheduler_scalar_slot();
    return nullptr;
}
static inline __aicore__ void scheduler_scalar_bind(SchedulerScalarTrace *trace) {
    if constexpr (kSchedulerScalarInstrumentation) *scheduler_scalar_slot() = trace;
}
#elif defined(__CPU_SIM)
#include <pthread.h>

struct SchedulerScalarThreadKey {
    pthread_key_t key{};
    bool valid{pthread_key_create(&key, nullptr) == 0};
    ~SchedulerScalarThreadKey() {
        if (valid) pthread_key_delete(key);
    }
};
inline SchedulerScalarThreadKey &scheduler_scalar_thread_key() {
    static SchedulerScalarThreadKey key;
    return key;
}
inline SchedulerScalarTrace *scheduler_scalar_current() {
    auto &key = scheduler_scalar_thread_key();
    return key.valid ? static_cast<SchedulerScalarTrace *>(pthread_getspecific(key.key)) : nullptr;
}
inline void scheduler_scalar_bind(SchedulerScalarTrace *trace) {
    auto &key = scheduler_scalar_thread_key();
    if (key.valid) (void)pthread_setspecific(key.key, trace);
}
#else
inline SchedulerScalarTrace *scheduler_scalar_current() { return nullptr; }
inline void scheduler_scalar_bind(SchedulerScalarTrace *) {}
#endif

namespace {

class SchedulerScalarBinding {
public:
    inline __aicore__ SchedulerScalarBinding() = default;
    inline __aicore__ void attach(SchedulerScalarTrace *trace) {
        scheduler_scalar_bind(trace != nullptr && trace->buffer != nullptr ? trace : nullptr);
    }
    inline __aicore__ ~SchedulerScalarBinding() { scheduler_scalar_bind(nullptr); }
    SchedulerScalarBinding(const SchedulerScalarBinding &) = delete;
    SchedulerScalarBinding &operator=(const SchedulerScalarBinding &) = delete;
};

}  // namespace

// Observer helpers have internal linkage because AIC and AIV objects share
// the final ELF. Clocks and record submission stay inside the bounded
// primitive wrapper; scheduler functions call that wrapper as one unit.
template <typename Operation>
static inline __aicore__ __attribute__((always_inline)) void
scheduler_scalar_observed_issue(SchedulerScalarTrace *trace, SchedPhaseKind kind, uint32_t calls, Operation operation) {
    const uint64_t begin = scheduler_scalar_clock();
    operation();
    const uint64_t end = scheduler_scalar_clock();
    trace->append_memory(kind, begin, end, calls);
}

template <typename Operation>
static inline __aicore__ __attribute__((always_inline)) auto
scheduler_scalar_observed_result(SchedulerScalarTrace *trace, SchedPhaseKind kind, Operation operation) {
    const uint64_t begin = scheduler_scalar_clock();
    auto result = operation();
    uint64_t end;
#if defined(__CCE_AICORE__)
    auto dependency = static_cast<uint64_t>(result);
    asm volatile("MOV %0, %0\nMOV %1, SYS_CNT\n" : "+l"(dependency), "=&l"(end) : : "memory");
#else
    end = scheduler_scalar_clock();
#endif
    trace->append_memory(kind, begin, end);
    return result;
}

// Each callback contains exactly one primitive and cannot nest observation.
// Keep the enable check inside this boundary to bound scheduler optimization
// work without splitting the timed primitive across another function call.
template <typename Operation>
static __aicore__ __attribute__((noinline)) void
scheduler_scalar_memory_issue_runtime(SchedPhaseKind kind, uint32_t calls, Operation operation) {
    auto *trace = scheduler_scalar_current();
    if (trace == nullptr) operation();
    else scheduler_scalar_observed_issue(trace, kind, calls, operation);
}

template <typename Operation>
static __aicore__ __attribute__((noinline)) auto
scheduler_scalar_memory_result_runtime(SchedPhaseKind kind, Operation operation) {
    auto *trace = scheduler_scalar_current();
    if (trace == nullptr) return operation();
    return scheduler_scalar_observed_result(trace, kind, operation);
}

template <typename Operation>
static inline __aicore__ __attribute__((always_inline)) void
scheduler_scalar_memory_issue(SchedPhaseKind kind, uint32_t calls, Operation operation) {
    if constexpr (kSchedulerScalarInstrumentation) scheduler_scalar_memory_issue_runtime(kind, calls, operation);
    else operation();
}

template <typename Operation>
static inline __aicore__ __attribute__((always_inline)) auto
scheduler_scalar_memory_result(SchedPhaseKind kind, Operation operation) {
    if constexpr (kSchedulerScalarInstrumentation) return scheduler_scalar_memory_result_runtime(kind, operation);
    else return operation();
}
