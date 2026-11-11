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

#include "common/args_dump.h"
#include "scheduler_memory.h"
#include "scheduler_types.h"

enum class SchedulerArgsDumpStatus : uint64_t { Idle = 0, WouldBlock = 1, InProgress = 2, Sealed = 3, Aborted = 4 };
enum class SchedulerArgsDumpArrival : uint8_t { Waiting, Publisher, Ready, Invalid };

struct alignas(64) SchedulerArgsDumpHeader {
    uint64_t run_epoch;
    uint64_t task_count;
    uint64_t selected_count;
    uint64_t worker_count;
    uint64_t task_indices_offset;
    uint64_t tasks_offset;
    uint64_t requests_offset;
    uint64_t acknowledgements_offset;
    uint64_t observed_ticks_offset;
    uint64_t pending_workers_offset;
    uint64_t total_size;
    uint64_t dump_level;
    uint64_t initialized_count;
    uint64_t reserved[3];
};

// The mutable cache line is accessed only by GM atomics after publication.
// Argument metadata occupies separate immutable lines; AICPU initializes its
// output mask before READY and never writes a participant's state line.
struct alignas(64) SchedulerArgsDumpTask {
    volatile uint64_t before_state;
    volatile uint64_t returned_participants;
    volatile uint64_t publisher_before;
    volatile uint64_t publisher_after;
    volatile uint64_t aborted;
    volatile uint64_t after_sealed;
    uint64_t task_ref;
    uint64_t state_reserved;

    TaskId task_id;
    uint64_t tensors_address;
    uint64_t scalars_address;
    uint64_t argument_region_base;
    uint64_t argument_region_bytes;
    uint64_t output_tensor_mask;
    int32_t kernel_id[3];
    uint16_t tensor_count;
    uint16_t scalar_count;
    ArgsDumpTaskMetadata dump_metadata;
    uint64_t active_mask;
    uint64_t expected_participants;
    uint64_t reserved[2];
};

// A selected output snapshot cannot precede deferred completion. The existing
// async context check refuses the request before a kernel can submit work.
inline constexpr uint32_t SCHEDULER_ARGS_DUMP_ASYNC_OUTPUT_REFUSED = 1;

inline __aicore__ void
scheduler_args_dump_limit_async_output(__gm__ const SchedulerArgsDumpTask *task, __gm__ GlobalContext &context) {
    if (task != nullptr && task->output_tensor_mask != 0 && context.async_present == 1)
        context.async_output_capture_refused = SCHEDULER_ARGS_DUMP_ASYNC_OUTPUT_REFUSED;
}

struct alignas(64) SchedulerArgsDumpRequest {
    uint64_t run_epoch;
    uint64_t task_id;
    uint64_t stage;
    uint64_t publisher;
    uint64_t tick;
    uint64_t arg_index;
    uint64_t byte_cursor;
    uint64_t reserved;
};

struct alignas(64) SchedulerArgsDumpAck {
    uint64_t run_epoch;
    uint64_t task_id;
    uint64_t stage;
    uint64_t publisher;
    uint64_t tick;
    uint64_t arg_index;
    uint64_t byte_cursor;
    uint64_t status;
};

// Phase identity survives chunk requests on the publisher's channel.
inline __aicore__ bool scheduler_args_dump_ack_matches_phase(
    const SchedulerArgsDumpAck &ack, uint64_t epoch, uint64_t task_ref, ArgsDumpStage stage, uint64_t publisher
) {
    return ack.run_epoch == epoch && ack.task_id == task_ref && ack.stage == static_cast<uint64_t>(stage) &&
           ack.publisher == publisher;
}

// The task and boundary identify one publisher in a run; the echoed request
// tick orders chunks on that publisher's channel.
inline __aicore__ uint64_t
scheduler_args_dump_ack_commit(uint64_t task_ref, uint64_t stage, SchedulerArgsDumpStatus status) {
    const uint64_t state = static_cast<uint64_t>(status);
    if (task_ref >= INT32_MAX || stage > static_cast<uint64_t>(ArgsDumpStage::AFTER_COMPLETION) ||
        state < static_cast<uint64_t>(SchedulerArgsDumpStatus::WouldBlock) ||
        state > static_cast<uint64_t>(SchedulerArgsDumpStatus::Aborted))
        return 0;
    return ((task_ref + 1) << 32) | (stage << 8) | state;
}

inline __aicore__ SchedulerArgsDumpStatus scheduler_args_dump_ack_status(const SchedulerArgsDumpAck &ack) {
    const auto status = static_cast<SchedulerArgsDumpStatus>(ack.status & UINT64_C(0xff));
    return ack.tick != 0 && ack.publisher != 0 && ack.publisher <= SCHEDULER_WORKER_CAPACITY && ack.status != 0 &&
                   ack.status == scheduler_args_dump_ack_commit(ack.task_id, ack.stage, status) ?
               status :
               SchedulerArgsDumpStatus::Idle;
}

namespace {

struct SchedulerArgsDumpAckReader {
    inline __aicore__ uint64_t load(__gm__ uint64_t &word) const { return scheduler_gm_query(word); }
    inline __aicore__ void barrier() const { scheduler_cache_barrier(); }
};

}  // namespace

// ld_dev does not implement C++ acquire ordering. Explicit DSBs separate the
// probes and body loads. The final tick probe follows the second commit probe
// so a same-state chunk update cannot hide behind a reused commit value.
template <typename Reader = SchedulerArgsDumpAckReader>
static inline __aicore__ bool scheduler_args_dump_observe_ack(
    __gm__ SchedulerArgsDumpAck *ack, SchedulerArgsDumpAck *observed, Reader reader = Reader{}
) {
    *observed = {};
    const uint64_t commit = reader.load(ack->status);
    const uint64_t tick = reader.load(ack->tick);
    reader.barrier();
    SchedulerArgsDumpAck next{};
    next.run_epoch = reader.load(ack->run_epoch);
    next.task_id = reader.load(ack->task_id);
    next.stage = reader.load(ack->stage);
    next.publisher = reader.load(ack->publisher);
    next.arg_index = reader.load(ack->arg_index);
    next.byte_cursor = reader.load(ack->byte_cursor);
    reader.barrier();
    const uint64_t last_commit = reader.load(ack->status);
    reader.barrier();
    const uint64_t last_tick = reader.load(ack->tick);
    reader.barrier();
    next.status = commit;
    next.tick = tick;
    if (commit != last_commit || tick != last_tick ||
        scheduler_args_dump_ack_status(next) == SchedulerArgsDumpStatus::Idle)
        return false;
    *observed = next;
    return true;
}

static_assert(sizeof(SchedulerArgsDumpHeader) == 128);
static_assert(sizeof(SchedulerArgsDumpTask) == 192);
static_assert(offsetof(SchedulerArgsDumpTask, task_id) == 64);
static_assert(sizeof(SchedulerArgsDumpRequest) == 64);
static_assert(sizeof(SchedulerArgsDumpAck) == 64);
static_assert(
    std::is_trivially_copyable_v<SchedulerArgsDumpHeader> && std::is_standard_layout_v<SchedulerArgsDumpHeader>
);
static_assert(std::is_trivially_copyable_v<SchedulerArgsDumpTask> && std::is_standard_layout_v<SchedulerArgsDumpTask>);
static_assert(
    std::is_trivially_copyable_v<SchedulerArgsDumpRequest> && std::is_standard_layout_v<SchedulerArgsDumpRequest>
);
static_assert(std::is_trivially_copyable_v<SchedulerArgsDumpAck> && std::is_standard_layout_v<SchedulerArgsDumpAck>);

#if !defined(__CCE_AICORE__)
struct SchedulerArgsDumpLayout {
    uint64_t task_count{0};
    uint64_t selected_count{0};
    uint64_t worker_count{0};
    uint64_t task_indices_offset{0};
    uint64_t tasks_offset{0};
    uint64_t requests_offset{0};
    uint64_t acknowledgements_offset{0};
    uint64_t observed_ticks_offset{0};
    uint64_t pending_workers_offset{0};
    uint64_t total_size{0};
};

inline bool scheduler_args_dump_reserve_array(
    uint64_t *cursor, uint64_t count, uint64_t element_size, uint64_t alignment, uint64_t *offset
) {
    uint64_t bytes = 0;
    return scheduler_layout_checked_mul(count, element_size, &bytes) &&
           scheduler_layout_reserve(cursor, bytes, alignment, offset);
}

inline bool scheduler_args_dump_plan_layout(
    uint64_t task_count, uint64_t selected_count, uint64_t worker_count, bool enabled, SchedulerArgsDumpLayout *layout
) {
    if (layout == nullptr || selected_count > task_count || task_count > INT32_MAX ||
        worker_count > SCHEDULER_WORKER_CAPACITY)
        return false;
    SchedulerArgsDumpLayout next{};
    if (!enabled || selected_count == 0) {
        *layout = next;
        return true;
    }
    if (worker_count == 0) return false;
    next.task_count = task_count;
    next.selected_count = selected_count;
    next.worker_count = worker_count;
    uint64_t cursor = sizeof(SchedulerArgsDumpHeader);
    if (!scheduler_args_dump_reserve_array(
            &cursor, task_count, sizeof(uint32_t), alignof(uint32_t), &next.task_indices_offset
        ) ||
        !scheduler_args_dump_reserve_array(
            &cursor, selected_count, sizeof(SchedulerArgsDumpTask), alignof(SchedulerArgsDumpTask), &next.tasks_offset
        ) ||
        !scheduler_args_dump_reserve_array(
            &cursor, worker_count, sizeof(SchedulerArgsDumpRequest), alignof(SchedulerArgsDumpRequest),
            &next.requests_offset
        ) ||
        !scheduler_args_dump_reserve_array(
            &cursor, worker_count, sizeof(SchedulerArgsDumpAck), alignof(SchedulerArgsDumpAck),
            &next.acknowledgements_offset
        ) ||
        !scheduler_args_dump_reserve_array(
            &cursor, worker_count, sizeof(uint64_t), alignof(uint64_t), &next.observed_ticks_offset
        ) ||
        !scheduler_args_dump_reserve_array(
            &cursor, worker_count, sizeof(uint32_t), alignof(uint32_t), &next.pending_workers_offset
        ) ||
        !scheduler_layout_checked_align(cursor, alignof(SchedulerArgsDumpHeader), &next.total_size))
        return false;
    *layout = next;
    return true;
}

// Initialization and selection run before publication; no participant can yet
// update the phase words.
inline bool scheduler_args_dump_init_image(
    void *base, uint64_t capacity, const SchedulerArgsDumpLayout &layout, uint64_t run_epoch, DumpArgsLevel level
) {
    if (base == nullptr || layout.total_size == 0 || layout.total_size > capacity ||
        reinterpret_cast<uintptr_t>(base) % alignof(SchedulerArgsDumpHeader) != 0)
        return false;
    auto *bytes = static_cast<uint8_t *>(base);
    for (uint64_t offset = 0; offset < layout.total_size; ++offset)
        bytes[offset] = 0;
    auto *header = static_cast<SchedulerArgsDumpHeader *>(base);
    header->run_epoch = run_epoch;
    header->task_count = layout.task_count;
    header->selected_count = layout.selected_count;
    header->worker_count = layout.worker_count;
    header->task_indices_offset = layout.task_indices_offset;
    header->tasks_offset = layout.tasks_offset;
    header->requests_offset = layout.requests_offset;
    header->acknowledgements_offset = layout.acknowledgements_offset;
    header->observed_ticks_offset = layout.observed_ticks_offset;
    header->pending_workers_offset = layout.pending_workers_offset;
    header->total_size = layout.total_size;
    header->dump_level = static_cast<uint64_t>(level);
    auto *indices = reinterpret_cast<uint32_t *>(bytes + layout.task_indices_offset);
    for (uint64_t task = 0; task < layout.task_count; ++task)
        indices[task] = UINT32_MAX;
    return true;
}

inline bool scheduler_args_dump_select_task(void *base, const SchedulerArgsDumpTask &task) {
    if (base == nullptr || task.active_mask == 0 || (task.active_mask & ~UINT64_C(7)) != 0 ||
        task.expected_participants == 0 || task.expected_participants > INT16_MAX)
        return false;
    auto *header = static_cast<SchedulerArgsDumpHeader *>(base);
    if (task.task_ref >= header->task_count || header->initialized_count >= header->selected_count) return false;
    auto *bytes = static_cast<uint8_t *>(base);
    auto *indices = reinterpret_cast<uint32_t *>(bytes + header->task_indices_offset);
    if (indices[task.task_ref] != UINT32_MAX) return false;
    const uint64_t index = header->initialized_count++;
    auto *tasks = reinterpret_cast<SchedulerArgsDumpTask *>(bytes + header->tasks_offset);
    tasks[index] = task;
    indices[task.task_ref] = static_cast<uint32_t>(index);
    return true;
}

#endif

inline __aicore__ __gm__ SchedulerArgsDumpTask *scheduler_args_dump_task(__gm__ void *base, uint64_t task_id) {
    if (base == nullptr) return nullptr;
    auto *header = static_cast<__gm__ SchedulerArgsDumpHeader *>(base);
    if (task_id >= header->task_count) return nullptr;
    auto *bytes = static_cast<__gm__ uint8_t *>(base);
    auto *indices = reinterpret_cast<__gm__ uint32_t *>(bytes + header->task_indices_offset);
    const uint32_t index = indices[task_id];
    if (index >= header->selected_count) return nullptr;
    auto *task = reinterpret_cast<__gm__ SchedulerArgsDumpTask *>(bytes + header->tasks_offset) + index;
    return task->task_ref == task_id ? task : nullptr;
}

inline __aicore__ uint64_t scheduler_args_dump_publisher(uint32_t worker) {
    return worker >= SCHEDULER_WORKER_CAPACITY ? 0 : worker + 1U;
}

// These phase words are published by AICore atomics. On A5, ld_dev can
// observe stale atomic results even after DSB; confirmation uses an atomic
// read. AICPU only reads its invalidated copy and never mutates this line.
static inline __aicore__ uint64_t scheduler_args_dump_load(__gm__ volatile uint64_t &word) {
#if defined(__CCE_AICORE__)
    return scheduler_gm_fetch_add(word, UINT64_C(0));
#else
    return scheduler_gm_query(word);
#endif
}

static inline __aicore__ bool scheduler_args_dump_task_valid(__gm__ SchedulerArgsDumpTask *task) {
    return task != nullptr && task->active_mask != 0 && (task->active_mask & ~UINT64_C(7)) == 0 &&
           task->expected_participants != 0 && task->expected_participants <= INT16_MAX &&
           scheduler_args_dump_load(task->aborted) == 0;
}

// BEFORE does not await future blocks: a task may occupy more than one wave.
static inline __aicore__ SchedulerArgsDumpArrival scheduler_args_dump_enter(__gm__ SchedulerArgsDumpTask *task) {
    if (!scheduler_args_dump_task_valid(task)) return SchedulerArgsDumpArrival::Invalid;
    const uint64_t prior = scheduler_gm_compare_exchange(task->before_state, UINT64_C(0), UINT64_C(1));
    if (prior == 0) return SchedulerArgsDumpArrival::Publisher;
    if (prior == 1) return SchedulerArgsDumpArrival::Waiting;
    return prior == 2 ? SchedulerArgsDumpArrival::Ready : SchedulerArgsDumpArrival::Invalid;
}

static inline __aicore__ bool scheduler_args_dump_can_execute(__gm__ SchedulerArgsDumpTask *task) {
    if (!scheduler_args_dump_task_valid(task) || scheduler_args_dump_load(task->before_state) != 2) return false;
    scheduler_cache_barrier();
    return true;
}

// Every caller publishes its output cache writes before this RMW. Earlier
// blocks retire normally; the last one holds the task's final completion.
template <typename Publish>
static inline __aicore__ SchedulerArgsDumpArrival
scheduler_args_dump_return(__gm__ SchedulerArgsDumpTask *task, Publish publish) {
    if (!scheduler_args_dump_can_execute(task)) return SchedulerArgsDumpArrival::Invalid;
    if (!publish()) {
        scheduler_gm_store(task->aborted, UINT64_C(1));
        return SchedulerArgsDumpArrival::Invalid;
    }
    scheduler_cache_barrier();
    const uint64_t prior = scheduler_gm_fetch_add(task->returned_participants, UINT64_C(1));
    scheduler_cache_barrier();
    if (prior >= task->expected_participants) {
        scheduler_gm_store(task->aborted, UINT64_C(1));
        return SchedulerArgsDumpArrival::Invalid;
    }
    return prior + 1 == task->expected_participants ? SchedulerArgsDumpArrival::Publisher :
                                                      SchedulerArgsDumpArrival::Waiting;
}

static inline __aicore__ bool scheduler_args_dump_seal(
    __gm__ SchedulerArgsDumpTask *task, ArgsDumpStage stage, const SchedulerArgsDumpAck &ack, uint64_t epoch,
    uint64_t request_tick
) {
    if (!scheduler_args_dump_task_valid(task) ||
        (stage != ArgsDumpStage::BEFORE_DISPATCH && stage != ArgsDumpStage::AFTER_COMPLETION))
        return false;
    const bool before = stage == ArgsDumpStage::BEFORE_DISPATCH;
    const uint64_t publisher = scheduler_args_dump_load(before ? task->publisher_before : task->publisher_after);
    if (publisher == 0 || !scheduler_args_dump_ack_matches_phase(ack, epoch, task->task_ref, stage, publisher) ||
        ack.tick != request_tick || scheduler_args_dump_ack_status(ack) != SchedulerArgsDumpStatus::Sealed ||
        (before ? scheduler_args_dump_load(task->before_state) != 1 :
                  !scheduler_args_dump_can_execute(task) ||
                      scheduler_args_dump_load(task->returned_participants) != task->expected_participants))
        return false;
    scheduler_cache_barrier();
    scheduler_gm_store(before ? task->before_state : task->after_sealed, before ? UINT64_C(2) : UINT64_C(1));
    return true;
}

static inline __aicore__ bool scheduler_args_dump_can_complete(__gm__ SchedulerArgsDumpTask *task) {
    return scheduler_args_dump_can_execute(task) &&
           scheduler_args_dump_load(task->returned_participants) == task->expected_participants &&
           scheduler_args_dump_load(task->after_sealed) == 1;
}
