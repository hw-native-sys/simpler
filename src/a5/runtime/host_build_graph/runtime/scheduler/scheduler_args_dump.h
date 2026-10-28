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
enum class SchedulerArgsDumpArrival : uint8_t { Waiting, Publisher, Invalid };

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

struct alignas(64) SchedulerArgsDumpTask {
    volatile uint64_t before_masks;
    volatile uint64_t after_masks;
    volatile uint64_t publisher_before;
    volatile uint64_t publisher_after;
    volatile uint64_t aborted;
    uint64_t task_id;
    uint64_t output_tensor_mask;
    uint64_t active_mask;
};

struct alignas(64) SchedulerArgsDumpRequest {
    uint64_t run_epoch;
    uint64_t task_id;
    uint64_t stage;
    uint64_t generation;
    uint64_t tick;
    uint64_t arg_index;
    uint64_t byte_cursor;
    uint64_t reserved;
};

struct alignas(64) SchedulerArgsDumpAck {
    uint64_t run_epoch;
    uint64_t task_id;
    uint64_t stage;
    uint64_t generation;
    uint64_t tick;
    uint64_t arg_index;
    uint64_t byte_cursor;
    uint64_t status;
};

// The commit binds the state to one dispatch generation and boundary. BEFORE
// and AFTER may share a generation; the echoed request tick orders chunks.
inline __aicore__ uint64_t
scheduler_args_dump_ack_commit(uint64_t generation, uint64_t stage, SchedulerArgsDumpStatus status) {
    const uint64_t state = static_cast<uint64_t>(status);
    if (generation == 0 || generation > UINT32_MAX || stage > static_cast<uint64_t>(ArgsDumpStage::AFTER_COMPLETION) ||
        state < static_cast<uint64_t>(SchedulerArgsDumpStatus::WouldBlock) ||
        state > static_cast<uint64_t>(SchedulerArgsDumpStatus::Aborted))
        return 0;
    return (generation << 32) | (stage << 8) | state;
}

inline __aicore__ SchedulerArgsDumpStatus scheduler_args_dump_ack_status(const SchedulerArgsDumpAck &ack) {
    const auto status = static_cast<SchedulerArgsDumpStatus>(ack.status & UINT64_C(0xff));
    return ack.tick != 0 && ack.status != 0 &&
                   ack.status == scheduler_args_dump_ack_commit(ack.generation, ack.stage, status) ?
               status :
               SchedulerArgsDumpStatus::Idle;
}

struct SchedulerArgsDumpAckReader {
    inline __aicore__ uint64_t load(__gm__ uint64_t &word) const { return scheduler_gm_query(word); }
    inline __aicore__ void barrier() const { scheduler_cache_barrier(); }
};

// ld_dev does not implement C++ acquire ordering. Explicit DSBs separate the
// probes and body loads. The final tick probe follows the second commit probe
// so a same-state chunk update cannot hide behind a reused commit value.
template <typename Reader = SchedulerArgsDumpAckReader>
inline __aicore__ bool scheduler_args_dump_observe_ack(
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
    next.generation = reader.load(ack->generation);
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
static_assert(sizeof(SchedulerArgsDumpTask) == 64);
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

// The compact SM remains bounded by its original size. The diagnostic image is
// a separate tail in the same charged arena and is never a compact/rebase input.
inline bool scheduler_args_dump_plan_arena(
    uint64_t sm_offset, uint64_t sm_bytes, const SchedulerArgsDumpLayout &layout, uint64_t *dump_offset,
    uint64_t *arena_bytes
) {
    if (dump_offset == nullptr || arena_bytes == nullptr || sm_offset > UINT64_MAX - sm_bytes) return false;
    const uint64_t sm_end = sm_offset + sm_bytes;
    if (layout.total_size == 0) {
        *dump_offset = 0;
        *arena_bytes = sm_end;
        return true;
    }
    uint64_t offset = 0;
    if (!scheduler_layout_checked_align(sm_end, alignof(SchedulerArgsDumpHeader), &offset) ||
        offset > UINT64_MAX - layout.total_size)
        return false;
    *dump_offset = offset;
    *arena_bytes = offset + layout.total_size;
    return true;
}

// Initialization and selection run before publication; no participant can yet
// update the phase words that share the immutable task metadata's cache line.
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

inline bool scheduler_args_dump_select_task(void *base, uint64_t task_id, uint64_t active_mask) {
    if (base == nullptr || active_mask == 0 || (active_mask & ~UINT64_C(7)) != 0) return false;
    auto *header = static_cast<SchedulerArgsDumpHeader *>(base);
    if (task_id >= header->task_count || header->initialized_count >= header->selected_count) return false;
    auto *bytes = static_cast<uint8_t *>(base);
    auto *indices = reinterpret_cast<uint32_t *>(bytes + header->task_indices_offset);
    if (indices[task_id] != UINT32_MAX) return false;
    const uint64_t index = header->initialized_count++;
    auto *tasks = reinterpret_cast<SchedulerArgsDumpTask *>(bytes + header->tasks_offset);
    tasks[index].task_id = task_id;
    tasks[index].active_mask = active_mask;
    indices[task_id] = static_cast<uint32_t>(index);
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
    return task->task_id == task_id ? task : nullptr;
}

inline __aicore__ uint64_t scheduler_args_dump_publisher(uint32_t worker, uint32_t generation) {
    return generation == 0 || worker == UINT32_MAX ? 0 : (static_cast<uint64_t>(generation) << 32) | (worker + 1U);
}

inline __aicore__ bool scheduler_args_dump_stage_valid(ArgsDumpStage stage) {
    return stage == ArgsDumpStage::BEFORE_DISPATCH || stage == ArgsDumpStage::AFTER_COMPLETION;
}

inline __aicore__ __gm__ volatile uint64_t &
scheduler_args_dump_masks(__gm__ SchedulerArgsDumpTask *task, ArgsDumpStage stage) {
    return stage == ArgsDumpStage::BEFORE_DISPATCH ? task->before_masks : task->after_masks;
}

inline __aicore__ SchedulerArgsDumpArrival
scheduler_args_dump_arrive(__gm__ SchedulerArgsDumpTask *task, ArgsDumpStage stage, SubtaskSlot subtask) {
    const auto slot = static_cast<uint32_t>(subtask);
    if (task == nullptr || !scheduler_args_dump_stage_valid(stage) || slot >= SUBTASK_SLOT_COUNT ||
        task->active_mask == 0 || (task->active_mask & ~UINT64_C(7)) != 0 ||
        (task->active_mask & (UINT64_C(1) << slot)) == 0 || scheduler_gm_query(task->aborted) != 0)
        return SchedulerArgsDumpArrival::Invalid;
    const uint64_t bit = UINT64_C(1) << slot;
    const uint64_t previous = scheduler_gm_fetch_or(scheduler_args_dump_masks(task, stage), bit);
    if ((previous & bit) != 0) return SchedulerArgsDumpArrival::Invalid;
    return ((previous | bit) & UINT64_C(7)) == task->active_mask ? SchedulerArgsDumpArrival::Publisher :
                                                                   SchedulerArgsDumpArrival::Waiting;
}

inline __aicore__ bool scheduler_args_dump_depart(
    __gm__ SchedulerArgsDumpTask *task, ArgsDumpStage stage, SubtaskSlot subtask, const SchedulerArgsDumpAck &ack,
    uint64_t run_epoch
) {
    const auto slot = static_cast<uint32_t>(subtask);
    if (task == nullptr || !scheduler_args_dump_stage_valid(stage) || slot >= SUBTASK_SLOT_COUNT ||
        (task->active_mask & (UINT64_C(1) << slot)) == 0 || scheduler_gm_query(task->aborted) != 0)
        return false;
    const uint64_t publisher =
        scheduler_gm_query(stage == ArgsDumpStage::BEFORE_DISPATCH ? task->publisher_before : task->publisher_after);
    if ((scheduler_gm_query(scheduler_args_dump_masks(task, stage)) & UINT64_C(7)) != task->active_mask ||
        publisher == 0 || scheduler_args_dump_ack_status(ack) != SchedulerArgsDumpStatus::Sealed ||
        ack.run_epoch != run_epoch || ack.task_id != task->task_id || ack.stage != static_cast<uint64_t>(stage) ||
        ack.generation != publisher >> 32)
        return false;
    const uint64_t bit = UINT64_C(1) << (slot + 8U);
    return (scheduler_gm_fetch_or(scheduler_args_dump_masks(task, stage), bit) & bit) == 0;
}

inline __aicore__ bool scheduler_args_dump_phase_consumed(__gm__ SchedulerArgsDumpTask *task, ArgsDumpStage stage) {
    return task != nullptr && scheduler_args_dump_stage_valid(stage) && task->active_mask != 0 &&
           scheduler_gm_query(task->aborted) == 0 &&
           ((scheduler_gm_query(scheduler_args_dump_masks(task, stage)) >> 8) & UINT64_C(7)) == task->active_mask;
}

inline __aicore__ bool scheduler_args_dump_can_execute(__gm__ SchedulerArgsDumpTask *task) {
    return scheduler_args_dump_phase_consumed(task, ArgsDumpStage::BEFORE_DISPATCH);
}

inline __aicore__ bool scheduler_args_dump_can_complete(__gm__ SchedulerArgsDumpTask *task) {
    return scheduler_args_dump_phase_consumed(task, ArgsDumpStage::BEFORE_DISPATCH) &&
           scheduler_args_dump_phase_consumed(task, ArgsDumpStage::AFTER_COMPLETION);
}

enum class SchedulerArgsDumpPoll { Waiting, Consumed, Aborted, Invalid };

inline __aicore__ SchedulerArgsDumpPoll scheduler_args_dump_poll_phase(
    __gm__ SchedulerArgsDumpTask *task, ArgsDumpStage stage, SubtaskSlot subtask, const SchedulerArgsDumpAck &ack,
    uint64_t run_epoch, bool *departed
) {
    if (task == nullptr || departed == nullptr || !scheduler_args_dump_stage_valid(stage))
        return SchedulerArgsDumpPoll::Invalid;
    const bool aborted = scheduler_gm_query(task->aborted) != 0;
    const uint64_t publisher =
        scheduler_gm_query(stage == ArgsDumpStage::BEFORE_DISPATCH ? task->publisher_before : task->publisher_after);
    const bool same_phase = publisher != 0 && ack.run_epoch == run_epoch && ack.task_id == task->task_id &&
                            ack.stage == static_cast<uint64_t>(stage) && ack.generation == publisher >> 32;
    const auto status = scheduler_args_dump_ack_status(ack);
    if (same_phase && status == SchedulerArgsDumpStatus::Aborted) {
        (void)scheduler_gm_compare_exchange(task->aborted, UINT64_C(0), UINT64_C(1));
        return SchedulerArgsDumpPoll::Aborted;
    }
    if (aborted) return SchedulerArgsDumpPoll::Waiting;
    // Departure is a task-local proof. The publisher's worker can reuse its
    // acknowledgement line after the last participant sets its departure.
    if (*departed && scheduler_args_dump_phase_consumed(task, stage)) return SchedulerArgsDumpPoll::Consumed;
    if (!same_phase) return SchedulerArgsDumpPoll::Waiting;
    if (!*departed && status == SchedulerArgsDumpStatus::Sealed) {
        *departed = scheduler_args_dump_depart(task, stage, subtask, ack, run_epoch);
        if (!*departed) return SchedulerArgsDumpPoll::Invalid;
    }
    return *departed && scheduler_args_dump_phase_consumed(task, stage) ? SchedulerArgsDumpPoll::Consumed :
                                                                          SchedulerArgsDumpPoll::Waiting;
}
