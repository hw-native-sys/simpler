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

#include "async_condition_writer.h"
#include "scheduler_async_layout.h"
#include "scheduler_graph.h"
#include "scheduler_completion_counter.h"

struct SchedulerAsyncLocalState {
    __gm__ uint8_t *base{nullptr};
    __gm__ const SchedulerAsyncLayout *layout{nullptr};
    __gm__ SchedulerTaskControl *controls{nullptr};
    uint64_t bytes{0};
    uint64_t completion_counters_offset{0};
    uint64_t task_metadata_offset{0};
    uint64_t protocol_layout_offset{0};
    uint32_t worker_index{0};
    uint32_t pending_group{pto2::resident::ASYNC_INVALID_INDEX};
    uint32_t queued_head{pto2::resident::ASYNC_INVALID_INDEX};
    uint32_t queued_tail{pto2::resident::ASYNC_INVALID_INDEX};
    uint32_t condition_cursor{0};
    uint64_t pending_mask{0};
    // A validated singleton COUNTER retained after the first pending observation.
    uint64_t retry_counter_address{0};
    uint32_t retry_counter_threshold{0};
    int64_t failed_task_ref{-1};
    int32_t error_code{0};
    bool work_activated{false};
};

enum class SchedulerAsyncCompletionResult : uint8_t { NONE, WAITING, FINISH_OWNER, ERROR };

// FINISH_OWNER is produced only by the join's closing or final-settlement RMW.
// Its proof is consumed in the same scheduler call, before any business kernel.
struct SchedulerAsyncCompletion {
    SchedulerAsyncCompletionResult result{SchedulerAsyncCompletionResult::NONE};
    int64_t task_ref{-1};
    uint64_t run_epoch{0};
    uint64_t joined{0};
};

inline __aicore__ bool scheduler_async_service_due(const SchedulerAsyncLocalState &local) {
    return local.pending_group != pto2::resident::ASYNC_INVALID_INDEX || local.error_code != 0;
}

inline __aicore__ SchedulerGraphResult scheduler_async_error_result(const SchedulerAsyncLocalState *local) {
    if (local == nullptr) return SchedulerGraphResult::INVALID_ARGUMENTS;
    switch (local->error_code) {
    case SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW:
        return SchedulerGraphResult::ASYNC_WAIT_OVERFLOW;
    case SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED:
        return SchedulerGraphResult::ASYNC_REGISTRATION_FAILED;
    case SIMPLER_ERROR_ASYNC_COMPLETION_INVALID:
        return SchedulerGraphResult::ASYNC_COMPLETION_INVALID;
    default:
        return SchedulerGraphResult::INVALID_ARGUMENTS;
    }
}

namespace scheduler_async_detail {

using namespace pto2::resident;

inline __aicore__ bool
fail(SchedulerAsyncLocalState &local, int64_t task_ref, int32_t code = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID) {
    if (local.error_code == 0) {
        local.failed_task_ref = task_ref;
        local.error_code = code == 0 ? SIMPLER_ERROR_ASYNC_COMPLETION_INVALID : code;
    }
    return false;
}

inline __aicore__ bool span(uint64_t bytes, uint64_t offset, uint64_t count, uint64_t stride, uint64_t alignment) {
    return offset % alignment == 0 && offset <= bytes && count <= (bytes - offset) / stride;
}

static inline __aicore__ void observe(__gm__ const void *address, uint64_t bytes) {
    if (bytes == 0) return;
    const uint64_t first = reinterpret_cast<uint64_t>(address) & ~UINT64_C(63);
    const uint64_t end = reinterpret_cast<uint64_t>(address) + bytes;
    for (uint64_t line = first; line < end; line += 64)
        scheduler_invalidate_cache_line(reinterpret_cast<__gm__ void *>(line));
    scheduler_cache_barrier();
}

static inline __aicore__ void writeback(__gm__ void *address, uint64_t bytes) {
    // Publication spans start/end on isolated lines and exclude atomic words.
    for (uint64_t offset = 0; offset < bytes; offset += 64)
        scheduler_writeback_cache_line(reinterpret_cast<__gm__ uint8_t *>(address) + offset);
}

static inline __aicore__ void publish_activity(SchedulerAsyncLocalState &local) {
    if (local.work_activated) return;
    auto *activity = scheduler_state_at<SchedulerAsyncActivity>(local.base, local.layout->activity_offset);
    // Host initializes zero before the run. Every device writer stores one;
    // this isolated line has no cached writes or RMW, and never returns to zero.
#if defined(__CCE_AICORE__)
    st_dev(UINT64_C(1), const_cast<__gm__ uint64_t *>(&activity->activated), 0);
    scheduler_ddr_barrier();
#else
    __atomic_store_n(&activity->activated, UINT64_C(1), __ATOMIC_RELEASE);
#endif
    local.work_activated = true;
}

inline __aicore__ bool task_index(SchedulerAsyncLocalState &local, int64_t task_ref, uint32_t &index) {
    index = ASYNC_INVALID_INDEX;
    if (local.error_code != 0 || local.layout == nullptr || task_ref < 0 ||
        static_cast<uint64_t>(task_ref) >= local.layout->task_count)
        return fail(local, task_ref);
    // Run-entry acquisition covers these Host-initialized, immutable tables.
    auto *map = scheduler_state_at<uint32_t>(local.base, local.layout->task_map_offset);
    index = map[task_ref];
    if (index == ASYNC_INVALID_INDEX) return true;
    if (index >= local.layout->async_task_count) return fail(local, task_ref);
    auto *tasks = scheduler_state_at<SchedulerAsyncTaskData>(local.base, local.layout->task_data_offset);
    return tasks[index].task_ref == task_ref || fail(local, task_ref);
}

static inline __aicore__ SchedulerAsyncCompletion completion_error(SchedulerAsyncLocalState &local, int64_t task_ref) {
    fail(local, task_ref);
    return {SchedulerAsyncCompletionResult::ERROR, task_ref};
}

static inline __aicore__ SchedulerAsyncCompletion settle_group(SchedulerAsyncLocalState &local, int64_t task_ref) {
    uint32_t index = ASYNC_INVALID_INDEX;
    if (!task_index(local, task_ref, index) || index == ASYNC_INVALID_INDEX) return completion_error(local, task_ref);
    auto *joins = scheduler_state_at<SchedulerAsyncTaskJoin>(local.base, local.layout->joins_offset);
    scheduler_cache_barrier();
    const auto settled = scheduler_async_pending::complete_group(joins[index]);
    if (settled.result == scheduler_async_pending::Result::FINISH_OWNER)
        return {SchedulerAsyncCompletionResult::FINISH_OWNER, task_ref, local.layout->run_epoch, settled.previous - 1};
    if (settled.result == scheduler_async_pending::Result::PENDING)
        return {SchedulerAsyncCompletionResult::WAITING, task_ref};
    return completion_error(local, task_ref);
}

// The sink watchdog samples pending_mask from GM, including for the group kept in the worker.
inline __aicore__ void
publish_condition_progress(SchedulerAsyncLocalState &local, uint32_t index, uint64_t pending, uint32_t cursor) {
    auto &row = scheduler_state_at<SchedulerAsyncGroupProgress>(local.base, local.layout->group_progress_offset)[index];
    row.pending_mask = pending;
    row.condition_cursor = cursor;
    scheduler_publish_cache_line(&row);
}

// An execution has a unique Host-assigned group. Only its registering worker
// polls it. Keep the current group in the worker, spilling extra groups to
// their own cache lines. Rotating after a bounded probe keeps later groups live.
inline __aicore__ void enqueue(SchedulerAsyncLocalState &local, uint32_t group, uint64_t mask, uint32_t cursor) {
    if (local.pending_group == ASYNC_INVALID_INDEX) {
        local.pending_group = group;
        local.pending_mask = mask;
        local.condition_cursor = cursor;
        return;
    }
    local.retry_counter_address = 0;
    auto *rows = scheduler_state_at<SchedulerAsyncGroupProgress>(local.base, local.layout->group_progress_offset);
    auto &row = rows[group];
    row.pending_mask = mask;
    row.condition_cursor = cursor;
    row.next_group = ASYNC_INVALID_INDEX;
    scheduler_writeback_cache_line(&row);
    if (local.queued_tail == ASYNC_INVALID_INDEX) {
        local.queued_head = group;
    } else {
        rows[local.queued_tail].next_group = group;
        scheduler_writeback_cache_line(&rows[local.queued_tail]);
    }
    scheduler_cache_barrier();
    local.queued_tail = group;
}

inline __aicore__ void advance(SchedulerAsyncLocalState &local) {
    local.retry_counter_address = 0;
    local.pending_group = local.queued_head;
    if (local.queued_head == ASYNC_INVALID_INDEX) return;
    const auto *rows = scheduler_state_at<SchedulerAsyncGroupProgress>(local.base, local.layout->group_progress_offset);
    const auto &row = rows[local.queued_head];
    local.pending_mask = row.pending_mask;
    local.condition_cursor = row.condition_cursor;
    local.queued_head = row.next_group;
    if (local.queued_head == ASYNC_INVALID_INDEX) local.queued_tail = ASYNC_INVALID_INDEX;
}

}  // namespace scheduler_async_detail

static inline __aicore__ bool scheduler_async_work_activated(SchedulerAsyncLocalState *local) {
    if (local == nullptr || local->layout == nullptr) return false;
    if (local->work_activated) return true;
    auto *activity = scheduler_state_at<SchedulerAsyncActivity>(local->base, local->layout->activity_offset);
    local->work_activated = scheduler_gm_query(activity->activated) != 0;
    return local->work_activated;
}

static inline __aicore__ bool scheduler_initialize_async(
    __gm__ void *base, uint64_t bytes, uint64_t offset, uint64_t epoch, uint64_t tasks, uint64_t controls_offset,
    SchedulerAsyncLocalState &local
) {
    using namespace scheduler_async_detail;
    local = SchedulerAsyncLocalState{};
    if (offset == 0) return true;
    if (base == nullptr || reinterpret_cast<uintptr_t>(base) % pto2::resident::ASYNC_RECORD_ALIGNMENT != 0 ||
        bytes > UINTPTR_MAX - reinterpret_cast<uintptr_t>(base) ||
        !span(bytes, offset, 1, sizeof(SchedulerAsyncLayout), pto2::resident::ASYNC_RECORD_ALIGNMENT) ||
        !span(bytes, controls_offset, tasks, sizeof(SchedulerTaskControl), alignof(SchedulerTaskControl)))
        return fail(local, -1);
    auto *layout = scheduler_state_at<SchedulerAsyncLayout>(base, offset);
    observe(layout, sizeof(*layout));
    if (epoch == 0 || layout->run_epoch != epoch || layout->image_bytes != bytes || layout->task_count != tasks ||
        layout->async_task_count == 0 || layout->async_task_count > tasks ||
        layout->group_count < layout->async_task_count ||
        !span(
            bytes, layout->activity_offset, 1, sizeof(SchedulerAsyncActivity), pto2::resident::ASYNC_RECORD_ALIGNMENT
        ) ||
        (layout->sdma_guard_count != 0 &&
         !span(
             bytes, layout->sdma_guards_offset, layout->sdma_guard_count,
             sizeof(pto2::resident::AsyncSdmaSubmissionGuard), pto2::resident::ASYNC_RECORD_ALIGNMENT
         )) ||
        !span(bytes, layout->task_map_offset, tasks, sizeof(uint32_t), alignof(uint32_t)) ||
        !span(
            bytes, layout->task_data_offset, layout->async_task_count, sizeof(SchedulerAsyncTaskData),
            alignof(SchedulerAsyncTaskData)
        ) ||
        !span(
            bytes, layout->groups_offset, layout->group_count, sizeof(AsyncGroupData),
            pto2::resident::ASYNC_RECORD_ALIGNMENT
        ) ||
        !span(
            bytes, layout->group_progress_offset, layout->group_count, sizeof(SchedulerAsyncGroupProgress),
            pto2::resident::ASYNC_RECORD_ALIGNMENT
        ) ||
        !span(
            bytes, layout->joins_offset, layout->async_task_count, sizeof(SchedulerAsyncTaskJoin),
            pto2::resident::ASYNC_RECORD_ALIGNMENT
        ))
        return fail(local, -1);
    local.base = static_cast<__gm__ uint8_t *>(base);
    local.bytes = bytes;
    local.layout = layout;
    local.controls = scheduler_state_at<SchedulerTaskControl>(base, controls_offset);
    return true;
}

// group_index is a valid index obtained from map_execution in this same run.
// The layout and base remain fixed; kernels receive a GM context, not this
// private state. Reconstructing here need not retain a full writer across a call.
static inline __aicore__ pto2::resident::AsyncWriter
scheduler_async_writer_for_group(const SchedulerAsyncLocalState &local, uint32_t group_index) {
    const uint64_t offset =
        local.layout->groups_offset + static_cast<uint64_t>(group_index) * sizeof(pto2::resident::AsyncGroupData);
    return {local.base, local.bytes, local.layout->run_epoch, offset, group_index};
}

static inline __aicore__ bool scheduler_async_begin_prebound_execution(
    SchedulerAsyncLocalState *local, int64_t task_ref, __gm__ GlobalContext &global, pto2::resident::AsyncWriter &writer
) {
    SchedulerScalarScope scalar_detail(SchedPhaseKind::ScalarAsyncBeginPrebound, task_ref);
    using namespace scheduler_async_detail;
    writer = AsyncWriter{};
    const uint32_t index = global.async_group_index;
    if (local == nullptr || index == ASYNC_INVALID_INDEX) {
        global.async_present = 0;
        return true;
    }
    if (local->layout == nullptr || index >= local->layout->group_count ||
        global.async_image_address != reinterpret_cast<uint64_t>(local->base) ||
        global.async_image_bytes != local->bytes || global.async_run_epoch != local->layout->run_epoch ||
        global.async_group_offset != local->layout->groups_offset + uint64_t{index} * sizeof(AsyncGroupData)) {
        global.async_present = 0;
        return fail(*local, task_ref);
    }
    writer = scheduler_async_writer_for_group(*local, index);
    global.async_output_capture_refused = 0;
    global.async_present = 1;
    return true;
}

// The first nonempty registration holds one task-wide async marker until FINISH.
// Its own physical contribution is outstanding, so the gate cannot reach zero.
// Validate before CAS: an unchecked overflowing addition could manufacture a
// syntactically valid zero gate from corrupted input.
inline __aicore__ bool scheduler_async_credit_available(uint64_t observed) {
    return scheduler_completion_valid(observed) && scheduler_completion_remaining(observed) != 0 &&
           scheduler_completion_debt(observed) == scheduler_completion_remaining(observed);
}

// The caller validates the task identity and supplies a reliable gate snapshot.
// Only the successful CAS grants a credit.
static inline __aicore__ bool
scheduler_async_register_credit_observed(SchedulerAsyncLocalState &local, int64_t task_ref, uint64_t observed) {
    auto &gate = local.controls[task_ref].completion_state;
    for (;;) {
        if (!scheduler_async_credit_available(observed)) return scheduler_async_detail::fail(local, task_ref);
        const uint64_t previous = scheduler_gm_compare_exchange(gate, observed, observed + 1);
        if (previous == observed) return true;
        observed = previous;
    }
}

static inline __aicore__ bool
scheduler_async_register_completion_credit(SchedulerAsyncLocalState &local, int64_t task_ref) {
    if (local.controls == nullptr || task_ref < 0 || static_cast<uint64_t>(task_ref) >= local.layout->task_count)
        return scheduler_async_detail::fail(local, task_ref);
    return scheduler_async_register_credit_observed(
        local, task_ref, scheduler_gm_query(local.controls[task_ref].completion_state)
    );
}

// BoundExecution uses the private group identity retained by the resident executor.
// Run entry bounds its Host-built group array, and map_execution selects its
// immutable identity/layout before the kernel. The kernel can mutate only
// registration state through AsyncCtx; it never receives this private writer.
template <bool BoundExecution = false>
static inline __aicore__ bool scheduler_async_seal_execution(
    SchedulerAsyncLocalState *local, int64_t task_ref, uint32_t block, uint32_t lane,
    const pto2::resident::AsyncWriter &writer, bool *has_deferred = nullptr
) {
    SchedulerScalarScope scalar_detail(SchedPhaseKind::ScalarAsyncSealExecution, task_ref);
    if (has_deferred != nullptr) *has_deferred = false;
    using namespace scheduler_async_detail;
    if (local == nullptr) return true;
    if (writer.image == nullptr) return true;
    __gm__ AsyncGroupData *data;
    if constexpr (BoundExecution) {
        data = reinterpret_cast<__gm__ AsyncGroupData *>(writer.image + writer.group_offset);
    } else {
        if (writer.image != local->base || writer.image_bytes != local->bytes ||
            writer.run_epoch != local->layout->run_epoch || writer.group_index >= local->layout->group_count ||
            writer.group_offset !=
                local->layout->groups_offset + static_cast<uint64_t>(writer.group_index) * sizeof(AsyncGroupData))
            return fail(*local, task_ref, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
        data = pto2::resident::detail::group(writer);
        if (data == nullptr || data->task_ref != task_ref || data->block_index != block || data->subtask_slot != lane)
            return fail(*local, task_ref, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    }
    const bool unpublished = data->state == AsyncPublicationState::OPEN;
    const uint32_t count = unpublished ? data->count : 0;
    if (!unpublished || !pto2::resident::detail::seal_open_conditions<BoundExecution>(writer, *data, count)) {
        return fail(
            *local, task_ref, data->error_code != 0 ? data->error_code : SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED
        );
    }
    // Registration state shares the first line with immutable identity/offsets.
    // The second line is Host-owned; publish line one and [0, count) entries.
    writeback(data, offsetof(AsyncGroupData, sdma_guards_offset));
    if (count == 0) {
        // An empty group has no CHECK reader. The resident return's mandatory
        // pre-contribution DSB publishes this header before the completion RMW;
        // standalone callers publish it here instead.
        if constexpr (!BoundExecution) scheduler_cache_barrier();
        return true;
    }
    writeback(writer.image + data->entries_offset, static_cast<uint64_t>(count) * sizeof(DeferredCompletionEntry));
    scheduler_cache_barrier();
    if (!SCHEDULER_REQUIRES_CNT_NOTIFY &&
        scheduler_completion_counter_at(local->base, local->completion_counters_offset, task_ref) == nullptr) {
        uint32_t task_index_value = ASYNC_INVALID_INDEX;
        if (!task_index(*local, task_ref, task_index_value) || task_index_value == ASYNC_INVALID_INDEX)
            return fail(*local, task_ref);
        auto *joins = scheduler_state_at<SchedulerAsyncTaskJoin>(local->base, local->layout->joins_offset);
        uint64_t registered_before = 0;
        if (scheduler_async_pending::register_group(joins[task_index_value], &registered_before) !=
            scheduler_async_pending::Result::PENDING)
            return fail(*local, task_ref);
        // The first registrant cannot release its physical contribution before
        // installing the marker. Other groups can progress without waiting for it.
        if (scheduler_async_pending::registered(registered_before) == 0 &&
            !scheduler_async_register_completion_credit(*local, task_ref))
            return false;
    }
    const uint64_t pending = scheduler_async_condition_mask(count);
    if (count > 1 && local->pending_group == ASYNC_INVALID_INDEX)
        publish_condition_progress(*local, writer.group_index, pending, 0);
    enqueue(*local, writer.group_index, pending, 0);
    // Consumers still use the run-wide flag when acquiring asynchronous inputs.
    publish_activity(*local);
    if (has_deferred != nullptr) *has_deferred = true;
    return true;
}

// The last physical FAA supplies a stable gate after every execution sealed
// its conditions. The close RMW supplies the join observation and ownership.
static inline __aicore__ SchedulerAsyncCompletion
scheduler_async_close_computation(SchedulerAsyncLocalState *local, int64_t task_ref, uint64_t gate) {
    SchedulerScalarScope scalar_detail(SchedPhaseKind::ScalarAsyncCloseComputation, task_ref);
    using namespace scheduler_async_detail;
    if (local == nullptr) return {};
    uint32_t index = ASYNC_INVALID_INDEX;
    if (!task_index(*local, task_ref, index)) return completion_error(*local, task_ref);
    if (index == ASYNC_INVALID_INDEX) return {};
    auto *joins = scheduler_state_at<SchedulerAsyncTaskJoin>(local->base, local->layout->joins_offset);
    if (!scheduler_completion_valid(gate) || scheduler_completion_remaining(gate) != 0)
        return completion_error(*local, task_ref);
    // A synchronous checked call has no close RMW. Keep its validation out of
    // the deferred path, whose previous join word comes from the required RMW.
    if (scheduler_completion_debt(gate) == 0) {
        scheduler_cache_barrier();
        const uint64_t joined = scheduler_gm_query(joins[index].word);
        if (!scheduler_async_pending::valid(joined) || scheduler_async_pending::closed(joined) ||
            scheduler_async_pending::registered(joined) != 0)
            return completion_error(*local, task_ref);
        return {};
    }
    scheduler_cache_barrier();
    const auto closed = scheduler_async_pending::close_computation(joins[index]);
    if ((closed.result != scheduler_async_pending::Result::FINISH_OWNER &&
         closed.result != scheduler_async_pending::Result::PENDING) ||
        scheduler_completion_debt(gate) != 1 || scheduler_async_pending::registered(closed.previous) == 0 ||
        scheduler_async_pending::registered(closed.previous) > scheduler_completion_total(gate))
        return completion_error(*local, task_ref);
    if (closed.result == scheduler_async_pending::Result::FINISH_OWNER)
        return {
            SchedulerAsyncCompletionResult::FINISH_OWNER, task_ref, local->layout->run_epoch,
            closed.previous | scheduler_async_pending::COMPUTATION_CLOSED
        };
    if (closed.result == scheduler_async_pending::Result::PENDING)
        return {SchedulerAsyncCompletionResult::WAITING, task_ref};
    return completion_error(*local, task_ref);
}

namespace {

static inline __aicore__ bool scheduler_async_counter_reached(uint64_t address, uint32_t threshold) {
    auto *counter = reinterpret_cast<__gm__ uint32_t *>(address);
    const uint32_t value = scheduler_scalar_memory_result(SchedPhaseKind::ScalarBypassLoad, [counter]() __aicore__ {
#if defined(__CCE_AICORE__)
        return static_cast<uint32_t>(__builtin_cce_ld_dev(counter, 0));
#else
        return __atomic_load_n(counter, __ATOMIC_ACQUIRE);
#endif
    });
    // A negative INT32 counter is below every supported threshold.
    return value <= INT32_MAX && value >= threshold;
}

struct SchedulerAsyncCompletionObserver {
    inline __aicore__ CompletionPollResult operator()(const CompletionToken &token, uint32_t observer_mask) const {
        if (observer_mask == 0 || (observer_mask & ~3U) != 0 || !pto2::resident::detail::token_valid(token))
            return {CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
        if (token.completion_type == COMPLETION_TYPE_CCU) {
            auto *event = reinterpret_cast<__gm__ uint64_t *>(token.addr);
            const auto load = [event]() __aicore__ {
                return scheduler_scalar_memory_result(SchedPhaseKind::ScalarBypassLoad, [event]() __aicore__ {
#if defined(__CCE_AICORE__)
                    const auto value = static_cast<uint64_t>(__builtin_cce_ld_dev(event, 0));
                    scheduler_ddr_barrier();
                    return value;
#else
                    return __atomic_load_n(event, __ATOMIC_ACQUIRE);
#endif
                });
            };
            if ((load() & token.backend_cookie) != token.backend_cookie) return {CompletionPollState::PENDING, 0};
#if defined(__CCE_AICORE__)
            st_dev(UINT64_C(0), event, 0);
            scheduler_ddr_barrier();
#else
            __atomic_store_n(event, UINT64_C(0), __ATOMIC_RELEASE);
#endif
            return load() == 0 ?
                       CompletionPollResult{CompletionPollState::READY, 0} :
                       CompletionPollResult{CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
        }
        if (token.completion_type == COMPLETION_TYPE_SDMA_EVENT_RECORD) {
            auto *record = reinterpret_cast<__gm__ uint64_t *>(token.addr);
            const uint64_t completed =
                scheduler_scalar_memory_result(SchedPhaseKind::ScalarBypassLoad, [record]() __aicore__ {
#if defined(__CCE_AICORE__)
                    return static_cast<uint64_t>(__builtin_cce_ld_dev(record, 0));
#else
                return __atomic_load_n(record, __ATOMIC_ACQUIRE);
#endif
                });
            return {completed >= token.backend_cookie ? CompletionPollState::READY : CompletionPollState::PENDING, 0};
        }
        return {
            scheduler_async_counter_reached(token.addr, token.expected_value) ? CompletionPollState::READY :
                                                                                CompletionPollState::PENDING,
            0
        };
    }
};

}  // namespace

template <typename Observer>
static inline __aicore__ bool scheduler_async_check_step(
    SchedulerAsyncLocalState &local, uint32_t observer_mask, uint32_t budget, const Observer &observer,
    bool &made_progress, SchedulerAsyncCompletion &completion, bool retry_reached = false
) {
    using namespace scheduler_async_detail;
    made_progress = false;
    completion = {};
    if (local.error_code != 0 || local.layout == nullptr || budget == 0 || observer_mask == 0 ||
        (observer_mask & ~3U) != 0)
        return fail(local, -1);
    const uint32_t index = local.pending_group;
    if (index == ASYNC_INVALID_INDEX) return true;
    if (index >= local.layout->group_count) return fail(local, -1);
    const uint64_t offset = local.layout->groups_offset + static_cast<uint64_t>(index) * sizeof(AsyncGroupData);
    if (!span(local.bytes, offset, 1, sizeof(AsyncGroupData), pto2::resident::ASYNC_RECORD_ALIGNMENT))
        return fail(local, -1);
    auto *data = scheduler_state_at<AsyncGroupData>(local.base, offset);
    const uint32_t count = data->count;
    const int64_t task_ref = data->task_ref;
    AsyncWriter writer{local.base, local.bytes, local.layout->run_epoch, offset, index};
    if (task_ref < 0 || static_cast<uint64_t>(task_ref) >= local.layout->task_count ||
        pto2::resident::detail::group(writer) == nullptr || data->state != AsyncPublicationState::SEALED ||
        data->error_code != 0 || count == 0 || count > data->capacity ||
        !pto2::resident::detail::layout_valid(writer, *data))
        return fail(local, task_ref);
    uint64_t pending = local.pending_mask;
    uint32_t cursor = local.condition_cursor;
    if (pending == 0 || (pending & ~scheduler_async_condition_mask(count)) != 0 || cursor >= 64)
        return fail(local, task_ref);
    auto *entries = scheduler_state_at<DeferredCompletionEntry>(local.base, data->entries_offset);
    uint64_t visited = 0;
    for (uint32_t used = 0; used < budget && (pending & ~visited) != 0; ++used) {
        uint64_t candidates = pending & ~visited;
        const uint64_t later = candidates & (UINT64_MAX << cursor);
        if (later != 0) candidates = later;
        const uint32_t condition = static_cast<uint32_t>(__builtin_ctzll(candidates));
        visited |= UINT64_C(1) << condition;
        cursor = (condition + 1) % 64;
        const auto &entry = entries[condition];
        const CompletionToken token{
            entry.addr, entry.expected_value, entry.engine, entry.completion_type, entry.backend_cookie
        };
        if (entry._pad != 0 || !pto2::resident::detail::token_valid(token)) return fail(local, task_ref);
        if (retry_reached &&
            (count != 1 || token.completion_type != COMPLETION_TYPE_COUNTER ||
             token.addr != local.retry_counter_address || token.expected_value != local.retry_counter_threshold))
            return fail(local, task_ref);
        const CompletionPollResult result =
            retry_reached ? CompletionPollResult{CompletionPollState::READY, 0} : observer(token, observer_mask);
        if (result.state == CompletionPollState::FAILED || result.error_code != 0)
            return fail(local, task_ref, result.error_code);
        if (result.state == CompletionPollState::READY) {
            pending &= ~(UINT64_C(1) << condition);
            made_progress = true;
        } else if (result.state != CompletionPollState::PENDING) {
            return fail(local, task_ref);
        } else if (count == 1 && token.completion_type == COMPLETION_TYPE_COUNTER &&
                   local.queued_head == ASYNC_INVALID_INDEX) {
            local.retry_counter_address = token.addr;
            local.retry_counter_threshold = token.expected_value;
        }
    }
    // Rotation writes the pending row in enqueue. Polling alone only moves the local cursor.
    if (made_progress && count > 1 && (pending == 0 || local.queued_head == ASYNC_INVALID_INDEX))
        publish_condition_progress(local, index, pending, cursor);
    if (pending != 0 && local.queued_head == ASYNC_INVALID_INDEX) {
        local.pending_mask = pending;
        local.condition_cursor = cursor;
        return true;
    }
    advance(local);
    if (pending != 0) {
        enqueue(local, index, pending, cursor);
        return true;
    }
    const auto *counter = scheduler_execution_counter_at(local.base, local.completion_counters_offset, task_ref);
    if (scheduler_execution_uses_counter(counter)) {
        if (!scheduler_counter_publish_condition(
                local.base, *counter, task_ref, data->block_index, data->subtask_slot, local.task_metadata_offset,
                local.protocol_layout_offset, local.worker_index, true
            ))
            return fail(local, task_ref);
        completion = {SchedulerAsyncCompletionResult::WAITING, task_ref};
        return true;
    }
    completion = settle_group(local, task_ref);
    return completion.result != SchedulerAsyncCompletionResult::ERROR;
}
