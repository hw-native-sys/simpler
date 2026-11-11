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
#include "scheduler_async_directory.h"
#include "scheduler_async_layout.h"
#include "scheduler_graph.h"

struct SchedulerAsyncLocalState {
    __gm__ uint8_t *base{nullptr};
    __gm__ const SchedulerAsyncLayout *layout{nullptr};
    uint64_t bytes{0};
    uint32_t check_cursor{0};
    uint32_t finish_cursor{0};
    int64_t failed_task_ref{-1};
    int32_t error_code{0};
    uint32_t business_returns{0};
    bool finish_first{false};
    bool work_activated{false};
    uint8_t idle_probe_countdown{0};
};

static_assert(sizeof(SchedulerAsyncLocalState) == 56);

inline constexpr uint8_t SCHEDULER_ASYNC_IDLE_PROBE_INTERVAL = 16;

// Cold idle cores still discover remote work within a bounded number of service
// opportunities. Business returns, active work and local errors are never delayed.
inline __aicore__ bool scheduler_async_service_due(SchedulerAsyncLocalState &local) {
    if (local.work_activated || local.error_code != 0) {
        local.idle_probe_countdown = 0;
        return true;
    }
    if (local.business_returns != 0 || local.idle_probe_countdown == 0) {
        local.idle_probe_countdown = SCHEDULER_ASYNC_IDLE_PROBE_INTERVAL - 1;
        return true;
    }
    --local.idle_probe_countdown;
    return false;
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

static inline __aicore__ void publish(__gm__ void *address, uint64_t bytes) {
    writeback(address, bytes);
    scheduler_cache_barrier();
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

static inline __aicore__ bool activate_finish(SchedulerAsyncLocalState &local, uint32_t index) {
    auto *leases = scheduler_state_at<SchedulerAsyncLease>(local.base, local.layout->finish_leases_offset);
    auto *directory =
        scheduler_state_at<SchedulerAsyncDirectoryWord>(local.base, local.layout->finish_directory_offset);
    scheduler_gm_store(leases[index].word, scheduler_async_lease_encode(1, SchedulerAsyncPhase::DORMANT));
    publish_activity(local);
    scheduler_cache_barrier();
    if (scheduler_async_lease_publish(leases[index]).result != SchedulerAsyncLeaseResult::SUCCESS ||
        scheduler_async_directory_publish(directory[index / 64].bits, index % 64) !=
            SchedulerAsyncDirectoryResult::SUCCESS)
        return fail(local, -1);
    return true;
}

static inline __aicore__ bool settle_group(SchedulerAsyncLocalState &local, int64_t task_ref) {
    uint32_t index = ASYNC_INVALID_INDEX;
    if (!task_index(local, task_ref, index) || index == ASYNC_INVALID_INDEX) return fail(local, task_ref);
    auto *joins = scheduler_state_at<SchedulerAsyncTaskJoin>(local.base, local.layout->joins_offset);
    scheduler_cache_barrier();
    const auto settled = scheduler_async_pending::complete_group(joins[index]);
    if (settled.result == scheduler_async_pending::Result::FINISH_OWNER) return activate_finish(local, index);
    return settled.result == scheduler_async_pending::Result::PENDING || fail(local, task_ref);
}

// A directory scan visits one word. Cursors advance past a claimed record so
// a low pending bit cannot starve a later condition in the same word.
static inline __aicore__ bool claim(
    SchedulerAsyncLocalState &local, uint64_t directory_offset, uint64_t lease_offset, uint32_t count, uint32_t &cursor,
    uint32_t &index, uint64_t &owned
) {
    index = ASYNC_INVALID_INDEX;
    owned = 0;
    if (count == 0 || cursor >= count) return fail(local, -1);
    auto *words = scheduler_state_at<SchedulerAsyncDirectoryWord>(local.base, directory_offset);
    const uint32_t word = cursor / 64;
    const uint32_t bit = cursor % 64;
    const uint64_t mask = scheduler_gm_query(words[word].bits) & (UINT64_MAX << bit);
    const uint64_t next_word = static_cast<uint64_t>(word + 1) * 64;
    cursor = next_word < count ? static_cast<uint32_t>(next_word) : 0;
    if (mask == 0) return true;
    const uint32_t found = word * 64 + static_cast<uint32_t>(__builtin_ctzll(mask));
    if (found >= count) return fail(local, -1);
    cursor = found + 1 == count ? 0 : found + 1;
    auto *leases = scheduler_state_at<SchedulerAsyncLease>(local.base, lease_offset);
    const auto result = scheduler_async_lease_try_claim(leases[found], scheduler_async_lease_load(leases[found]));
    if (result.result == SchedulerAsyncLeaseResult::NOT_RUNNABLE ||
        result.result == SchedulerAsyncLeaseResult::CONTENDED)
        return true;
    if (result.result != SchedulerAsyncLeaseResult::SUCCESS) return fail(local, -1);
    scheduler_cache_barrier();
    index = found;
    owned = result.word;
    return true;
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
    __gm__ void *base, uint64_t bytes, uint64_t offset, uint64_t epoch, uint64_t tasks, SchedulerAsyncLocalState &local
) {
    using namespace scheduler_async_detail;
    local = SchedulerAsyncLocalState{};
    if (offset == 0) return true;
    if (base == nullptr || reinterpret_cast<uintptr_t>(base) % pto2::resident::ASYNC_RECORD_ALIGNMENT != 0 ||
        bytes > UINTPTR_MAX - reinterpret_cast<uintptr_t>(base) ||
        !span(bytes, offset, 1, sizeof(SchedulerAsyncLayout), pto2::resident::ASYNC_RECORD_ALIGNMENT))
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
            bytes, layout->group_leases_offset, layout->group_count, sizeof(SchedulerAsyncLease),
            pto2::resident::ASYNC_RECORD_ALIGNMENT
        ) ||
        !span(
            bytes, layout->group_progress_offset, layout->group_count, sizeof(SchedulerAsyncGroupProgress),
            pto2::resident::ASYNC_RECORD_ALIGNMENT
        ) ||
        !span(
            bytes, layout->joins_offset, layout->async_task_count, sizeof(SchedulerAsyncTaskJoin),
            pto2::resident::ASYNC_RECORD_ALIGNMENT
        ) ||
        !span(
            bytes, layout->finish_leases_offset, layout->async_task_count, sizeof(SchedulerAsyncLease),
            pto2::resident::ASYNC_RECORD_ALIGNMENT
        ) ||
        !span(
            bytes, layout->check_directory_offset, (static_cast<uint64_t>(layout->group_count) + 63) / 64,
            pto2::resident::ASYNC_RECORD_ALIGNMENT, pto2::resident::ASYNC_RECORD_ALIGNMENT
        ) ||
        !span(
            bytes, layout->finish_directory_offset, (static_cast<uint64_t>(layout->async_task_count) + 63) / 64,
            pto2::resident::ASYNC_RECORD_ALIGNMENT, pto2::resident::ASYNC_RECORD_ALIGNMENT
        ))
        return fail(local, -1);
    local.base = static_cast<__gm__ uint8_t *>(base);
    local.bytes = bytes;
    local.layout = layout;
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

inline __aicore__ bool scheduler_async_map_execution(
    SchedulerAsyncLocalState &local, int64_t task_ref, uint32_t block, uint32_t lane,
    pto2::resident::AsyncWriter &writer
) {
    using namespace scheduler_async_detail;
    writer = AsyncWriter{};
    uint32_t index = ASYNC_INVALID_INDEX;
    if (!task_index(local, task_ref, index)) return false;
    if (index == ASYNC_INVALID_INDEX) return true;
    auto *tasks = scheduler_state_at<SchedulerAsyncTaskData>(local.base, local.layout->task_data_offset);
    const auto &task = tasks[index];
    const uint8_t mask = *reinterpret_cast<__gm__ const uint8_t *>(&task.async_mask);
    if (lane >= 3 || mask == 0 || (mask & ~UINT8_C(7)) != 0 || block >= task.block_count) return fail(local, task_ref);
    if ((mask & (1U << lane)) == 0) return true;
    const uint64_t wide_group_index = static_cast<uint64_t>(task.group_base) +
                                      block * static_cast<uint32_t>(__builtin_popcount(mask)) +
                                      static_cast<uint32_t>(__builtin_popcount(mask & ((1U << lane) - 1)));
    if (wide_group_index >= local.layout->group_count) return fail(local, task_ref);
    const uint32_t group_index = static_cast<uint32_t>(wide_group_index);
    // Bootstrap bounds the contiguous, Host-initialized header array. No
    // header is read until the kernel requests AsyncCtx or returns to seal it.
    writer = scheduler_async_writer_for_group(local, group_index);
    return true;
}

// The caller has observed the existing computation join's TASK_LAST result.
// A pre-join zero observation cannot exclude another execution's registration.
static inline __aicore__ bool
scheduler_async_has_registered_groups(SchedulerAsyncLocalState *local, int64_t task_ref, bool &registered) {
    using namespace scheduler_async_detail;
    registered = false;
    if (local == nullptr) return true;
    uint32_t index = ASYNC_INVALID_INDEX;
    if (!task_index(*local, task_ref, index)) return false;
    if (index == ASYNC_INVALID_INDEX) return true;
    auto *joins = scheduler_state_at<SchedulerAsyncTaskJoin>(local->base, local->layout->joins_offset);
    const uint64_t word = scheduler_gm_fetch_add(joins[index].word, UINT64_C(0));
    if (!scheduler_async_pending::valid(word) || scheduler_async_pending::closed(word)) return fail(*local, task_ref);
    registered = scheduler_async_pending::registered(word) != 0;
    return true;
}

// The resident topology assigns a dense rank to each AIV independently of task blocks.
inline __aicore__ uint32_t scheduler_async_urma_sender(uint32_t lane, uint32_t type_rank) {
    return lane == 1 || lane == 2 ? type_rank : UINT32_MAX;
}

inline __aicore__ bool scheduler_async_begin_execution(
    SchedulerAsyncLocalState *local, int64_t task_ref, uint32_t block, uint32_t lane, __gm__ GlobalContext &global,
    pto2::resident::AsyncWriter &writer, uint32_t urma_sender_id = UINT32_MAX
) {
    using namespace scheduler_async_detail;
    writer = AsyncWriter{};
    global.urma_sender_id = urma_sender_id < UINT16_MAX ? static_cast<uint16_t>(urma_sender_id) : UINT16_MAX;
    if (local == nullptr) return true;
    global.async_present = 0;
    if (!scheduler_async_map_execution(*local, task_ref, block, lane, writer)) return false;
    if (writer.image == nullptr) return true;
    global.async_image_address = reinterpret_cast<uint64_t>(writer.image);
    global.async_image_bytes = writer.image_bytes;
    global.async_run_epoch = writer.run_epoch;
    global.async_group_offset = writer.group_offset;
    global.async_group_index = writer.group_index;
    global.async_output_capture_refused = 0;
    global.async_present = 1;
    return true;
}

// BoundExecution uses the private group identity retained by the resident executor.
// Run entry bounds its Host-built group array, and map_execution selects its
// immutable identity/layout before the kernel. The kernel can mutate only
// registration state through AsyncCtx; it never receives this private writer.
template <bool BoundExecution = false>
static inline __aicore__ bool scheduler_async_seal_execution(
    SchedulerAsyncLocalState *local, int64_t task_ref, uint32_t block, uint32_t lane,
    const pto2::resident::AsyncWriter &writer
) {
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
    auto *leases = scheduler_state_at<SchedulerAsyncLease>(local->base, local->layout->group_leases_offset);
    const bool unpublished = data->state == AsyncPublicationState::OPEN;
    const uint32_t count = unpublished ? data->count : 0;
    if (!unpublished || !pto2::resident::detail::seal_open_conditions<BoundExecution>(writer, *data, count)) {
        // No CHECK can own a group whose first seal failed. A duplicate seal
        // must not overwrite a lease that may already be RUNNABLE or OWNED.
        if (unpublished)
            scheduler_gm_store(
                leases[writer.group_index].word, scheduler_async_lease_encode(1, SchedulerAsyncPhase::FAILED)
            );
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
    uint32_t task_index_value = ASYNC_INVALID_INDEX;
    if (!task_index(*local, task_ref, task_index_value) || task_index_value == ASYNC_INVALID_INDEX)
        return fail(*local, task_ref);
    auto *joins = scheduler_state_at<SchedulerAsyncTaskJoin>(local->base, local->layout->joins_offset);
    if (scheduler_async_pending::register_group(joins[task_index_value]).result !=
        scheduler_async_pending::Result::PENDING)
        return fail(*local, task_ref);
    auto *progress = scheduler_state_at<SchedulerAsyncGroupProgress>(local->base, local->layout->group_progress_offset);
    progress[writer.group_index].pending_mask = scheduler_async_condition_mask(count);
    progress[writer.group_index].condition_cursor = 0;
    progress[writer.group_index].acquire_cursor = 0;
    publish(&progress[writer.group_index], sizeof(SchedulerAsyncGroupProgress));
    auto *directory =
        scheduler_state_at<SchedulerAsyncDirectoryWord>(local->base, local->layout->check_directory_offset);
    scheduler_gm_store(leases[writer.group_index].word, scheduler_async_lease_encode(1, SchedulerAsyncPhase::DORMANT));
    publish_activity(*local);
    if (scheduler_async_lease_publish(leases[writer.group_index]).result != SchedulerAsyncLeaseResult::SUCCESS ||
        scheduler_async_directory_publish(directory[writer.group_index / 64].bits, writer.group_index % 64) !=
            SchedulerAsyncDirectoryResult::SUCCESS)
        return fail(*local, task_ref);
    return true;
}

// All executions have sealed their condition registrations before computation closes.
static inline __aicore__ bool
scheduler_async_close_computation(SchedulerAsyncLocalState *local, int64_t task_ref, bool &deferred) {
    using namespace scheduler_async_detail;
    deferred = false;
    if (local == nullptr) return true;
    uint32_t index = ASYNC_INVALID_INDEX;
    if (!task_index(*local, task_ref, index)) return false;
    if (index == ASYNC_INVALID_INDEX) return true;
    auto *joins = scheduler_state_at<SchedulerAsyncTaskJoin>(local->base, local->layout->joins_offset);
    scheduler_cache_barrier();
    const auto closed = scheduler_async_pending::close_computation(joins[index]);
    if (closed.result == scheduler_async_pending::Result::SYNCHRONOUS) return true;
    deferred = true;
    if (closed.result == scheduler_async_pending::Result::FINISH_OWNER) return activate_finish(*local, index);
    return closed.result == scheduler_async_pending::Result::PENDING || fail(*local, task_ref);
}

namespace {

struct SchedulerAsyncCompletionObserver {
    inline __aicore__ CompletionPollResult operator()(const CompletionToken &token, uint32_t observer_mask) const {
        if (observer_mask == 0 || (observer_mask & ~3U) != 0 || token.engine == COMPLETION_ENGINE_URMA ||
            !pto2::resident::detail::token_valid(token))
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
        auto *counter = reinterpret_cast<__gm__ uint32_t *>(token.addr);
        const uint32_t value = scheduler_scalar_memory_result(SchedPhaseKind::ScalarBypassLoad, [counter]() __aicore__ {
#if defined(__CCE_AICORE__)
            return static_cast<uint32_t>(__builtin_cce_ld_dev(counter, 0));
#else
            return __atomic_load_n(counter, __ATOMIC_ACQUIRE);
#endif
        });
        // COUNTER is INT32. A negative sample is below every supported
        // nonnegative threshold; it is neither completion nor malformed data.
        if (value > INT32_MAX) return {CompletionPollState::PENDING, 0};
        return {value >= token.expected_value ? CompletionPollState::READY : CompletionPollState::PENDING, 0};
    }
};

}  // namespace

template <typename Observer>
static inline __aicore__ bool scheduler_async_check_step(
    SchedulerAsyncLocalState &local, uint32_t observer_mask, uint32_t budget, const Observer &observer,
    bool &made_progress
) {
    using namespace scheduler_async_detail;
    made_progress = false;
    if (local.error_code != 0 || local.layout == nullptr || budget == 0 || observer_mask == 0 ||
        (observer_mask & ~3U) != 0)
        return fail(local, -1);
    uint32_t index = ASYNC_INVALID_INDEX;
    uint64_t owned = 0;
    if (!claim(
            local, local.layout->check_directory_offset, local.layout->group_leases_offset, local.layout->group_count,
            local.check_cursor, index, owned
        ))
        return false;
    if (index == ASYNC_INVALID_INDEX) return true;
    auto *leases = scheduler_state_at<SchedulerAsyncLease>(local.base, local.layout->group_leases_offset);
    auto failed = [&](int64_t ref, int32_t code = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID) __aicore__ {
        scheduler_async_lease_fail(leases[index], owned);
        return fail(local, ref, code);
    };
    const uint64_t offset = local.layout->groups_offset + static_cast<uint64_t>(index) * sizeof(AsyncGroupData);
    if (!span(local.bytes, offset, 1, sizeof(AsyncGroupData), pto2::resident::ASYNC_RECORD_ALIGNMENT))
        return failed(-1);
    auto *data = scheduler_state_at<AsyncGroupData>(local.base, offset);
    observe(data, sizeof(*data));
    const int64_t task_ref = data->task_ref;
    AsyncWriter writer{local.base, local.bytes, local.layout->run_epoch, offset, index};
    if (task_ref < 0 || static_cast<uint64_t>(task_ref) >= local.layout->task_count ||
        pto2::resident::detail::group(writer) == nullptr || data->state != AsyncPublicationState::SEALED ||
        data->error_code != 0 || data->count == 0 || data->count > data->capacity ||
        !pto2::resident::detail::layout_valid(writer, *data))
        return failed(task_ref);
    auto *progress = scheduler_state_at<SchedulerAsyncGroupProgress>(local.base, local.layout->group_progress_offset);
    auto &current = progress[index];
    observe(&current, sizeof(current));
    uint64_t pending = current.pending_mask;
    uint32_t cursor = current.condition_cursor;
    if (pending == 0 || (pending & ~scheduler_async_condition_mask(data->count)) != 0 || cursor >= 64 ||
        current.acquire_cursor != 0)
        return failed(task_ref);
    auto *entries = scheduler_state_at<DeferredCompletionEntry>(local.base, data->entries_offset);
    uint64_t visited = 0;
    for (uint32_t used = 0; used < budget && (pending & ~visited) != 0; ++used) {
        uint64_t candidates = pending & ~visited;
        const uint64_t later = candidates & (UINT64_MAX << cursor);
        if (later != 0) candidates = later;
        const uint32_t condition = static_cast<uint32_t>(__builtin_ctzll(candidates));
        visited |= UINT64_C(1) << condition;
        cursor = (condition + 1) % 64;
        observe(&entries[condition], sizeof(DeferredCompletionEntry));
        const auto &entry = entries[condition];
        const CompletionToken token{
            entry.addr, entry.expected_value, entry.engine, entry.completion_type, entry.backend_cookie
        };
        if (entry._pad != 0 || !pto2::resident::detail::token_valid(token)) return failed(task_ref);
        const CompletionPollResult result = observer(token, observer_mask);
        if (result.state == CompletionPollState::FAILED || result.error_code != 0)
            return failed(task_ref, result.error_code);
        if (result.state == CompletionPollState::READY) {
            pending &= ~(UINT64_C(1) << condition);
            made_progress = true;
        } else if (result.state != CompletionPollState::PENDING) return failed(task_ref);
    }
    current.pending_mask = pending;
    current.condition_cursor = cursor;
    publish(&current, sizeof(current));
    if (pending != 0)
        return scheduler_async_lease_release(leases[index], owned).result == SchedulerAsyncLeaseResult::SUCCESS ||
               failed(task_ref);
    auto *directory = scheduler_state_at<SchedulerAsyncDirectoryWord>(local.base, local.layout->check_directory_offset);
    if (scheduler_async_lease_finish(leases[index], owned).result != SchedulerAsyncLeaseResult::SUCCESS ||
        scheduler_async_directory_clear(directory[index / 64].bits, index % 64) !=
            SchedulerAsyncDirectoryResult::SUCCESS)
        return fail(local, task_ref);
    return settle_group(local, task_ref);
}
