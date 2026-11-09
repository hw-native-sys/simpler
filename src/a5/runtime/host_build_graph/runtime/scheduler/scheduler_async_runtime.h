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

#include "scheduler_async_check.h"
#include "scheduler_async_discovery.h"
#include "scheduler_async_finish.h"
#include "async_completion_writer.h"
#include "scheduler_local_storage.h"
#include "host_build_graph/runtime_status.h"

// Local views only. No pointer or extra field enters the dispatch/wire ABI.
struct SchedulerAsyncRuntimeView {
    __gm__ uint8_t *base{nullptr};
    SchedulerAsyncLayout header{};
    inline __aicore__ const SchedulerAsyncLayout &layout() const { return header; }
    inline __aicore__ bool enabled() const { return base != nullptr; }
};

enum class SchedulerAsyncRuntimeServiceResult : uint8_t { IDLE, PROGRESS, ERROR };

static_assert(sizeof(AsyncCtxStorage) == 40 && sizeof(LocalContext) == 48);
static_assert(sizeof(DispatchPayload) == 512 && offsetof(DispatchPayload, args) == 64);

namespace scheduler_async_runtime_detail {
// Owner progress and immutable metadata use bypass accesses. Atomic-only lease,
// directory and join lines never pass through these helpers or cache writeback.
template <typename T>
inline __aicore__ void load_into(__gm__ const T *source, T *destination) {
    static_assert(std::is_trivially_copyable_v<T> && sizeof(T) % 8 == 0 && alignof(T) >= 8);
    for (uint32_t offset = 0; offset < sizeof(T); offset += 8) {
        uint64_t word = 0;
#if defined(__CCE_AICORE__)
        auto *address = reinterpret_cast<__gm__ uint64_t *>(
            const_cast<__gm__ uint8_t *>(reinterpret_cast<__gm__ const uint8_t *>(source) + offset)
        );
        word = static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0));
#else
        __builtin_memcpy(&word, reinterpret_cast<const uint8_t *>(source) + offset, 8);
#endif
        __builtin_memcpy(reinterpret_cast<uint8_t *>(destination) + offset, &word, 8);
    }
}

// Only the sealed registration prefix is needed; never load writer padding.
inline __aicore__ void scheduler_async_writer_snapshot(
    __gm__ const SchedulerAsyncGroupWriter *source, SchedulerAsyncWriterSnapshot *destination
) {
    load_into(reinterpret_cast<__gm__ const SchedulerAsyncWriterSnapshot *>(source), destination);
}

template <typename T>
inline __aicore__ T load(__gm__ const T *source) {
    static_assert(alignof(T) <= 8, "over-aligned local reads require SchedulerLocalStorage and load_into");
    T result{};
    load_into(source, &result);
    return result;
}

template <typename T>
inline __aicore__ void publish(__gm__ T *destination, const T &value) {
    static_assert(std::is_trivially_copyable_v<T> && sizeof(T) % 8 == 0 && alignof(T) >= 8);
    for (uint32_t offset = 0; offset < sizeof(T); offset += 8) {
        uint64_t word = 0;
        __builtin_memcpy(&word, reinterpret_cast<const uint8_t *>(&value) + offset, 8);
#if defined(__CCE_AICORE__)
        auto *address = reinterpret_cast<__gm__ uint64_t *>(reinterpret_cast<__gm__ uint8_t *>(destination) + offset);
        st_dev(word, address, 0);
#else
        __builtin_memcpy(reinterpret_cast<uint8_t *>(destination) + offset, &word, 8);
#endif
    }
    scheduler_cache_barrier();
}

template <typename T>
inline __aicore__ __gm__ T *at(const SchedulerAsyncRuntimeView &view, uint64_t offset, uint64_t index = 0) {
    return reinterpret_cast<__gm__ T *>(view.base + offset) + index;
}

inline __aicore__ bool
region(uint64_t bytes, uint64_t offset, uint64_t count, uint64_t stride, uint64_t alignment, uint64_t *end) {
    if (offset < *end || offset % alignment != 0 || offset > bytes || count > (bytes - offset) / stride) return false;
    *end = offset + count * stride;
    return true;
}

inline __aicore__ bool contained(uint64_t begin, uint64_t bytes, uint64_t offset, uint64_t count, uint64_t stride) {
    return offset >= begin && offset - begin <= bytes && count <= (bytes - (offset - begin)) / stride;
}

inline __aicore__ bool fail(
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run, int64_t task,
    const SchedulerAsyncRuntimeView *view = nullptr, uint64_t finish = SCHEDULER_ASYNC_INDEX_NONE,
    SchedulerGraphResult status = SchedulerGraphResult::INVALID_ARGUMENTS
) {
    // Claim the original cause before a peer can react to the poisoned join
    // and report a secondary generic failure.
    scheduler_record_error(run, task, status, &graph, local);
    if (view != nullptr && view->enabled() && finish < view->layout().finish_count)
        scheduler_async_join_abort(*at<SchedulerAsyncTaskJoin>(*view, view->layout().finish_join_offset, finish));
    return false;
}

inline __aicore__ SchedulerGraphResult writer_error_result(int32_t error) {
    switch (error) {
    case SIMPLER_ERROR_ASYNC_COMPLETION_INVALID:
        return SchedulerGraphResult::ASYNC_COMPLETION_INVALID;
    case SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW:
        return SchedulerGraphResult::ASYNC_WAIT_OVERFLOW;
    case SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED:
        return SchedulerGraphResult::ASYNC_REGISTRATION_FAILED;
    default:
        return SchedulerGraphResult::INVALID_ARGUMENTS;
    }
}

inline __aicore__ bool task_map(const SchedulerAsyncRuntimeView &view, int64_t task, SchedulerAsyncTaskMap *out) {
    if (!view.enabled() || task < 0 || static_cast<uint64_t>(task) >= view.layout().task_count) return false;
    *out = load(at<SchedulerAsyncTaskMap>(view, view.layout().task_map_offset, task));
    if (out->finish_index == SCHEDULER_ASYNC_INDEX_NONE)
        return out->group_base == SCHEDULER_ASYNC_INDEX_NONE && out->group_count == 0 && out->block_count == 0 &&
               out->groups_per_block == 0 && out->async_lane_mask == 0;
    const uint64_t groups = static_cast<uint64_t>(out->block_count) * out->groups_per_block;
    return out->finish_index < view.layout().finish_count && out->block_count > 0 &&
           out->block_count <= SCHEDULER_ASYNC_JOIN_LIMIT && out->async_lane_mask != 0 &&
           (out->async_lane_mask & ~7u) == 0 && out->groups_per_block == __builtin_popcount(out->async_lane_mask) &&
           groups == out->group_count && groups <= SCHEDULER_ASYNC_JOIN_LIMIT &&
           out->group_base < view.layout().group_count && groups <= view.layout().group_count - out->group_base;
}

inline __aicore__ bool
finish_identity(const SchedulerAsyncRuntimeView &view, uint64_t index, SchedulerAsyncFinishIdentity *out) {
    if (index >= view.layout().finish_count) return false;
    load_into(at<SchedulerAsyncFinishIdentity>(view, view.layout().finish_identity_offset, index), out);
    SchedulerAsyncTaskMap map{};
    return out->run_epoch == view.layout().run_epoch && task_map(view, out->task_ref, &map) &&
           map.finish_index == index && map.group_base == out->group_base && map.group_count == out->group_count;
}

inline __aicore__ bool group_identity_valid(
    const SchedulerAsyncRuntimeView &view, uint64_t index, int64_t task, const SchedulerAsyncTaskMap &map,
    const SchedulerAsyncGroupIdentity &identity
) {
    return identity.run_epoch == view.layout().run_epoch && identity.task_ref == task &&
           map.finish_index == identity.finish_index &&
           scheduler_async_group_index(map, identity.block_idx, identity.subtask_slot) == index &&
           identity.capacity > 0 && identity.capacity <= MAX_COMPLETIONS_PER_TASK &&
           identity.entries_offset % 128 == 0 &&
           contained(
               view.layout().entries_offset, view.layout().entries_bytes, identity.entries_offset, identity.capacity,
               sizeof(DeferredCompletionEntry)
           );
}

inline __aicore__ bool
group_identity(const SchedulerAsyncRuntimeView &view, uint64_t index, SchedulerAsyncGroupIdentity *out) {
    if (index >= view.layout().group_count) return false;
    load_into(at<SchedulerAsyncGroupIdentity>(view, view.layout().group_identity_offset, index), out);
    SchedulerAsyncTaskMap map{};
    return task_map(view, out->task_ref, &map) && group_identity_valid(view, index, out->task_ref, map, *out);
}

inline __aicore__ bool task_token(const SchedulerGraphAccess &graph, int64_t task, TaskId *out) {
    uint64_t value = 0;
    if (scheduler_node_identity(graph, task, &value) != SchedulerGraphResult::OK) return false;
    if ((value >> TaskId::SPACE_SHIFT) == static_cast<uint64_t>(TaskId::Space::SUB_TASK))
        *out =
            TaskId::make_sub_task((value >> TaskId::PARENT_SHIFT) & TaskId::PARENT_MASK, static_cast<int32_t>(value));
    else *out = TaskId::make_global(static_cast<int32_t>(value));
    return TaskId::to_uint64(*out) == value;
}

inline __aicore__ void clear_async_storage(__gm__ AsyncCtxStorage &ctx, volatile __gm__ int32_t *error) {
    ctx.completion_count = nullptr;
    ctx.completion_error_code = error;
    ctx.completion_entries = nullptr;
    ctx.completion_capacity = 0;
    TaskId::assign(ctx.task_token, TaskId::invalid());
}

inline __aicore__ bool activate_finish(
    const SchedulerAsyncRuntimeView &view, const SchedulerGraphAccess &graph, SchedulerLocalState *local,
    __gm__ SchedulerRunControl *run, uint64_t index
) {
    SchedulerAsyncFinishIdentity identity{};
    if (!finish_identity(view, index, &identity) || scheduler_read_error(run) != 0)
        return fail(graph, local, run, identity.task_ref, &view, index);
    auto &join = *at<SchedulerAsyncTaskJoin>(view, view.layout().finish_join_offset, index);
    if (!scheduler_async_join_complete(scheduler_async_join_load(join)))
        return fail(graph, local, run, identity.task_ref, &view, index);
    scheduler_cache_barrier();
    auto &lease = *at<SchedulerAsyncLease>(view, view.layout().finish_lease_offset, index);
    if (scheduler_async_lease_publish(lease).result != SchedulerAsyncLeaseResult::SUCCESS)
        return fail(graph, local, run, identity.task_ref, &view, index);
    scheduler_cache_barrier();
    auto &word = *at<SchedulerAsyncDirectoryWord>(view, view.layout().finish_directory_offset, index / 64);
    if (scheduler_async_directory_publish_once(word, index % 64) != SchedulerAsyncDirectoryResult::SUCCESS)
        return fail(graph, local, run, identity.task_ref, &view, index);
    return true;
}

inline __aicore__ SchedulerAsyncDirectoryView directory_view(const SchedulerAsyncRuntimeView &view, bool check) {
    return {
        at<SchedulerAsyncDirectoryWord>(
            view, check ? view.layout().check_directory_offset : view.layout().finish_directory_offset
        ),
        at<SchedulerAsyncLease>(view, check ? view.layout().group_lease_offset : view.layout().finish_lease_offset),
        check ? view.layout().group_count : view.layout().finish_count
    };
}
}  // namespace scheduler_async_runtime_detail

// Called after the ordinary scheduler bootstrap acquire. Layout checks are fixed
// in number; task/group identities are checked when indexed, not scanned per core.
inline __aicore__ bool scheduler_async_runtime_mount(
    const SchedulerGraphAccess &graph, __gm__ void *scheduler_base, SchedulerLocalState *local,
    __gm__ SchedulerRunControl *run, SchedulerAsyncRuntimeView *out
) {
    using namespace scheduler_async_runtime_detail;
    if (out == nullptr || scheduler_base == nullptr || local == nullptr || run == nullptr) return false;
    const uint64_t base_address = reinterpret_cast<uint64_t>(scheduler_base);
    const uint64_t bytes = graph.scheduler_state_bytes;
    uint64_t end = sizeof(SchedulerRunControl);
    if (base_address % 128 != 0 || graph.scheduler_state_address != base_address || bytes > UINT64_MAX - base_address ||
        !region(bytes, local->config.graph_summary_offset, 1, sizeof(SchedulerGraphSummary), 128, &end))
        return fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    const auto *summary = scheduler_state_at<SchedulerGraphSummary>(scheduler_base, local->config.graph_summary_offset);
    const uint64_t async_offset = load(&summary->async_layout_offset);
    if (async_offset == 0) {
        *out = {};
        return true;
    }
    const uint64_t run_epoch = load(&summary->run_epoch);
    if (run_epoch == 0 || !region(bytes, async_offset, 1, sizeof(SchedulerAsyncLayout), 128, &end))
        return fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    SchedulerAsyncRuntimeView next{};
    next.base = reinterpret_cast<__gm__ uint8_t *>(scheduler_base) + async_offset;
    const auto l = load(reinterpret_cast<__gm__ const SchedulerAsyncLayout *>(next.base));
    if (l.version != SCHEDULER_ASYNC_LAYOUT_VERSION || l.run_epoch != run_epoch ||
        l.total_bytes > bytes - async_offset || l.total_bytes % 128 != 0 || l.task_count != graph.task_count ||
        l.task_count > INT32_MAX || l.group_count == 0 || l.finish_count == 0 || l.finish_count > l.task_count ||
        l.finish_count > l.group_count || l.condition_capacity < l.group_count ||
        l.condition_capacity > l.entries_bytes / sizeof(DeferredCompletionEntry) ||
        l.check_directory_words != scheduler_async_directory_word_count(l.group_count) ||
        l.finish_directory_words != scheduler_async_directory_word_count(l.finish_count))
        return fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    end = sizeof(SchedulerAsyncLayout);
#define SCHEDULER_ASYNC_CHECK_REGION(field, count, type) region(l.total_bytes, l.field, count, sizeof(type), 128, &end)
    const bool valid =
        SCHEDULER_ASYNC_CHECK_REGION(work_signal_offset, 1, uint64_t) &&
        SCHEDULER_ASYNC_CHECK_REGION(task_map_offset, l.task_count, SchedulerAsyncTaskMap) &&
        SCHEDULER_ASYNC_CHECK_REGION(group_identity_offset, l.group_count, SchedulerAsyncGroupIdentity) &&
        SCHEDULER_ASYNC_CHECK_REGION(group_writer_offset, l.group_count, SchedulerAsyncGroupWriter) &&
        SCHEDULER_ASYNC_CHECK_REGION(group_lease_offset, l.group_count, SchedulerAsyncLease) &&
        SCHEDULER_ASYNC_CHECK_REGION(group_progress_offset, l.group_count, SchedulerAsyncGroupProgress) &&
        SCHEDULER_ASYNC_CHECK_REGION(finish_identity_offset, l.finish_count, SchedulerAsyncFinishIdentity) &&
        SCHEDULER_ASYNC_CHECK_REGION(finish_lease_offset, l.finish_count, SchedulerAsyncLease) &&
        SCHEDULER_ASYNC_CHECK_REGION(finish_join_offset, l.finish_count, SchedulerAsyncTaskJoin) &&
        SCHEDULER_ASYNC_CHECK_REGION(finish_progress_offset, l.finish_count, SchedulerAsyncFinishProgress) &&
        SCHEDULER_ASYNC_CHECK_REGION(check_directory_offset, l.check_directory_words, SchedulerAsyncDirectoryWord) &&
        SCHEDULER_ASYNC_CHECK_REGION(finish_directory_offset, l.finish_directory_words, SchedulerAsyncDirectoryWord) &&
        SCHEDULER_ASYNC_CHECK_REGION(entries_offset, l.entries_bytes, uint8_t);
#undef SCHEDULER_ASYNC_CHECK_REGION
    if (!valid || l.sdma_guard_count > ASYNC_SDMA_CHANNELS ||
        (l.sdma_guard_count == 0 ?
             l.sdma_guard_context_offset != 0 :
             !region(l.total_bytes, l.sdma_guard_context_offset, l.sdma_guard_count + 1, 128, 128, &end)))
        return fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    if (l.sdma_guard_count != 0) {
        const auto *context = at<AsyncSdmaGuardContext>(next, l.sdma_guard_context_offset);
        if (!async_sdma_guard_context_valid(context) || context->run_epoch != l.run_epoch ||
            context->guard_count != l.sdma_guard_count)
            return fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    }
    // Only immutable layout metadata belongs to this per-core snapshot. Writer,
    // lease, directory and completion values remain in their shared GM records.
    next.header = l;
    *out = next;
    return true;
}

inline __aicore__ bool scheduler_async_runtime_bind_payload(
    const SchedulerAsyncRuntimeView &view, const SchedulerGraphAccess &graph, SchedulerLocalState *local,
    __gm__ SchedulerRunControl *run, int64_t task, uint32_t block, uint32_t subtask, __gm__ DispatchPayload *payload,
    uint64_t *bound_index
) {
    using namespace scheduler_async_runtime_detail;
    if (bound_index == nullptr || payload == nullptr) return false;
    *bound_index = SCHEDULER_ASYNC_INDEX_NONE;
    auto &ctx = payload->local_context.async_ctx;
    if (!view.enabled()) {
        payload->async_context_address = 0;
        clear_async_storage(ctx, nullptr);
        return true;
    }
    if (task < 0 || static_cast<uint64_t>(task) >= view.layout().task_count || subtask >= 3)
        return fail(graph, local, run, task);
    auto &execution = payload->async_context;
    execution.region_address = reinterpret_cast<uint64_t>(view.base);
    execution.writer_state = 0;
    execution.task_ref = static_cast<uint32_t>(task);
    execution.subtask_slot = subtask;
    execution.error_code = 0;
    execution.magic = ASYNC_EXECUTION_CONTEXT_MAGIC;
    payload->async_context_address = reinterpret_cast<uint64_t>(&execution);
    SchedulerAsyncTaskMap map{};
    if (!task_map(view, task, &map)) return fail(graph, local, run, task);
    if (map.finish_index == SCHEDULER_ASYNC_INDEX_NONE) {
        clear_async_storage(ctx, &execution.error_code);
        return true;
    }
    if (block >= map.block_count || subtask >= 3) return fail(graph, local, run, task, &view, map.finish_index);
    const uint64_t index = scheduler_async_group_index(map, block, subtask);
    if (index == SCHEDULER_ASYNC_INDEX_NONE) {
        clear_async_storage(ctx, &execution.error_code);
        return true;
    }
    SchedulerAsyncGroupIdentity identity{};
    TaskId token = TaskId::invalid();
    load_into(at<SchedulerAsyncGroupIdentity>(view, view.layout().group_identity_offset, index), &identity);
    if (!group_identity_valid(view, index, task, map, identity) || !task_token(graph, task, &token))
        return fail(graph, local, run, task, &view, map.finish_index);
    // Host initializes one writer per physical execution. The admitted slot
    // generation binds it once; no CHECK can own it before this execution seals.
    auto *destination = at<SchedulerAsyncGroupWriter>(view, view.layout().group_writer_offset, index);
    execution.writer_state = reinterpret_cast<uint64_t>(&destination->reservation);
    ctx.completion_count = &destination->count;
    ctx.completion_error_code = &destination->error_code;
    ctx.completion_entries = at<DeferredCompletionEntry>(view, identity.entries_offset);
    ctx.completion_capacity = identity.capacity;
    TaskId::assign(ctx.task_token, token);
    *bound_index = index;
    return true;
}

// Runs only on the original executing core after kernel return and before its
// execution-release contribution. Validate token format and allowed types before
// publishing the sealed immutable entries to CHECK.
// Ordinary graph projections carry no scheduler backing address; it is passed separately.
inline __aicore__ bool scheduler_async_runtime_seal(
    const SchedulerAsyncRuntimeView &view, const SchedulerGraphAccess &graph, __gm__ void *scheduler_base,
    SchedulerLocalState *local, __gm__ SchedulerRunControl *run, int64_t task, uint32_t block, uint32_t subtask,
    uint64_t index, __gm__ DispatchPayload *payload, bool *registered
) {
    using namespace scheduler_async_runtime_detail;
    if (registered == nullptr || payload == nullptr) return false;
    *registered = false;
    if (!view.enabled()) return true;
    const auto &execution = payload->async_context;
    if (payload->async_context_address != reinterpret_cast<uint64_t>(&execution) ||
        execution.magic != ASYNC_EXECUTION_CONTEXT_MAGIC ||
        execution.region_address != reinterpret_cast<uint64_t>(view.base) || execution.task_ref != task ||
        execution.subtask_slot != subtask)
        return fail(graph, local, run, task);
    if (execution.error_code != 0)
        return fail(
            graph, local, run, task, nullptr, SCHEDULER_ASYNC_INDEX_NONE, writer_error_result(execution.error_code)
        );
    if (index == SCHEDULER_ASYNC_INDEX_NONE) return true;
    if (index >= view.layout().group_count) return fail(graph, local, run, task);
    auto *writer = at<SchedulerAsyncGroupWriter>(view, view.layout().group_writer_offset, index);
    const auto &ctx = payload->local_context.async_ctx;
    if (writer->count == 0 && execution.writer_state == reinterpret_cast<uint64_t>(&writer->reservation) &&
        ctx.completion_count == &writer->count && ctx.completion_error_code == &writer->error_code) {
        // Binding already validates immutable identity and token storage. An
        // empty return needs only the shared writer's seal/error/reservation
        // decision; it never reads tokens or publishes a CHECK lease.
        auto registration = AsyncCtx::from_storage(ctx);
        registration.completion_reservation = &writer->reservation;
        const bool valid = async_completion_seal(registration) && scheduler_read_error(run) == 0;
        // No CHECK can read an empty writer. Native retirement/exit barriers
        // drain this writeback before the run storage can be reused.
        if (valid) {
            scheduler_writeback_cache_line(writer);
            return true;
        }
        scheduler_publish_cache_line(writer);
        SchedulerAsyncTaskMap map{};
        if (!task_map(view, task, &map)) return fail(graph, local, run, task);
        auto &lease = *at<SchedulerAsyncLease>(view, view.layout().group_lease_offset, index);
        (void)scheduler_async_lease_try_transition(
            lease, scheduler_async_lease_word(1, SchedulerAsyncPhase::DORMANT), SchedulerAsyncPhase::FAILED
        );
        return fail(graph, local, run, task, &view, map.finish_index, writer_error_result(writer->error_code));
    }
    SchedulerAsyncTaskMap map{};
    if (!task_map(view, task, &map) || scheduler_async_group_index(map, block, subtask) != index)
        return fail(graph, local, run, task);
    SchedulerAsyncGroupIdentity identity{};
    TaskId token = TaskId::invalid();
    load_into(at<SchedulerAsyncGroupIdentity>(view, view.layout().group_identity_offset, index), &identity);
    if (!group_identity_valid(view, index, task, map, identity) || !task_token(graph, task, &token))
        return fail(graph, local, run, task, &view, map.finish_index);
    auto &lease = *at<SchedulerAsyncLease>(view, view.layout().group_lease_offset, index);
    if (scheduler_async_lease_load(lease) != scheduler_async_lease_word(1, SchedulerAsyncPhase::DORMANT) ||
        writer->reservation.sealed != 0)
        return fail(graph, local, run, task, &view, map.finish_index);
    if (execution.writer_state != reinterpret_cast<uint64_t>(&writer->reservation) ||
        ctx.completion_count != &writer->count || ctx.completion_error_code != &writer->error_code ||
        ctx.completion_entries != at<DeferredCompletionEntry>(view, identity.entries_offset) ||
        ctx.completion_capacity != identity.capacity || TaskId::to_uint64(ctx.task_token) != TaskId::to_uint64(token))
        return fail(graph, local, run, task, &view, map.finish_index);
    const uint32_t count = writer->count;
    auto registration = AsyncCtx::from_storage(ctx);
    registration.completion_reservation = &writer->reservation;
    bool valid = async_completion_seal(registration) && scheduler_read_error(run) == 0;
    // Failed registrations also retain their actual prefix for diagnostics.
    // Clamp corrupt counts to this execution's independent token allocation.
    const uint32_t published_count = count < identity.capacity ? count : identity.capacity;
    for (uint64_t offset = 0; offset < published_count * sizeof(DeferredCompletionEntry); offset += 64)
        scheduler_writeback_cache_line(view.base + identity.entries_offset + offset);
    scheduler_cache_barrier();
    if (valid) {
        SchedulerAsyncCheckView entries{at<DeferredCompletionEntry>(view, identity.entries_offset), identity.capacity};
        for (uint32_t condition = 0; condition < count; ++condition) {
            const auto entry = scheduler_async_check_detail::entry(entries, condition);
            if (!async_completion_format_valid(entry) || !scheduler_async_check_detail::entry_valid(entry)) {
                valid = false;
                break;
            }
        }
    }
    if (!valid && writer->error_code == 0)
        writer->error_code = count > identity.capacity ? SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW : SIMPLER_ERROR_INVALID_ARGS;
    scheduler_publish_cache_line(writer);
    SchedulerAsyncGroupProgress progress{};
    load_into(at<SchedulerAsyncGroupProgress>(view, view.layout().group_progress_offset, index), &progress);
    SchedulerAsyncWriterSnapshot snapshot{};
    scheduler_async_writer_snapshot(writer, &snapshot);
    const auto initialized = scheduler_async_check_initialize(identity, snapshot, &progress);
    publish(at<SchedulerAsyncGroupProgress>(view, view.layout().group_progress_offset, index), progress);
    if (!valid || initialized == SchedulerAsyncCheckResult::ERROR) {
        (void)scheduler_async_lease_try_transition(
            lease, scheduler_async_lease_word(1, SchedulerAsyncPhase::DORMANT), SchedulerAsyncPhase::FAILED
        );
        return fail(graph, local, run, task, &view, map.finish_index, writer_error_result(snapshot.error_code));
    }

    auto &join = *at<SchedulerAsyncTaskJoin>(view, view.layout().finish_join_offset, map.finish_index);
    if (scheduler_async_join_add_group(join).result != SchedulerAsyncJoinResult::PENDING)
        return fail(graph, local, run, task, &view, map.finish_index);
    // A monotonic search hint, never a completion proof. First publication
    // enables discovery; later registrations need no additional shared write.
    auto &work_signal = *at<uint64_t>(view, view.layout().work_signal_offset);
    if (scheduler_gm_query(work_signal) == 0) scheduler_gm_store(work_signal, UINT64_C(1));
    scheduler_cache_barrier();
    if (scheduler_async_lease_publish(lease).result != SchedulerAsyncLeaseResult::SUCCESS)
        return fail(graph, local, run, task, &view, map.finish_index);
    scheduler_cache_barrier();
    if (scheduler_async_directory_publish_once(
            *at<SchedulerAsyncDirectoryWord>(view, view.layout().check_directory_offset, index / 64), index % 64
        ) != SchedulerAsyncDirectoryResult::SUCCESS)
        return fail(graph, local, run, task, &view, map.finish_index);
    *registered = true;
    return true;
}

// Only native TASK_LAST calls this after the final slot/cohort retirement tail.
// The existing native counters already account for every physical execution.
inline __aicore__ bool scheduler_async_runtime_release_task(
    const SchedulerAsyncRuntimeView &view, const SchedulerGraphAccess &graph, __gm__ void *scheduler_base,
    SchedulerLocalState *local, __gm__ SchedulerRunControl *run, int64_t task, bool *finish_published
) {
    using namespace scheduler_async_runtime_detail;
    if (finish_published == nullptr) return false;
    *finish_published = false;
    if (!view.enabled()) return true;
    SchedulerAsyncTaskMap map{};
    if (!task_map(view, task, &map)) return fail(graph, local, run, task);
    if (map.finish_index == SCHEDULER_ASYNC_INDEX_NONE) return true;
    if (scheduler_read_error(run) != 0) return fail(graph, local, run, task, &view, map.finish_index);
    // Drain native retirement publications before contributing to the join.
    scheduler_cache_barrier();
    const auto result = scheduler_async_join_release_block(
        *at<SchedulerAsyncTaskJoin>(view, view.layout().finish_join_offset, map.finish_index)
    );
    if (result.result == SchedulerAsyncJoinResult::FINISH_OWNER) {
        // The native last-block completion owns TaskDone for an empty task.
        if (scheduler_async_join_registered(result.previous) == 0) return true;
        *finish_published = activate_finish(view, graph, local, run, map.finish_index);
        return *finish_published;
    }
    if (result.result != SchedulerAsyncJoinResult::PENDING)
        return fail(graph, local, run, task, &view, map.finish_index);
    return true;
}

// One claim per invocation; a CHECK that activates FINISH never services it in
// the same slice. All terminal/clear/contribution operations below are fixed
// tails with no budget checks or ownership handoff between their steps.
template <typename Backend, typename QueueOps = SchedulerCompletionQueueOps, typename Link = SchedulerResolveLink>
inline __aicore__ SchedulerAsyncRuntimeServiceResult scheduler_async_runtime_service(
    const SchedulerAsyncRuntimeView &view, const SchedulerGraphAccess &graph, __gm__ void *scheduler_base,
    SchedulerLocalState *local, __gm__ SchedulerRunControl *run, SchedulerAsyncDiscoveryState *discovery,
    SchedulerAsyncSliceBudget *budget, Backend &backend
) {
    using namespace scheduler_async_runtime_detail;
    if (!view.enabled()) return SchedulerAsyncRuntimeServiceResult::IDLE;
    if (discovery == nullptr || budget == nullptr) return SchedulerAsyncRuntimeServiceResult::ERROR;
    if (budget->work_units == 0) return SchedulerAsyncRuntimeServiceResult::IDLE;
    // No directory can contain work before the first actual registration.
    // A zero query can only postpone a racing publication; the producer drains
    // the signal before publishing its CHECK lease and returning from seal.
    if (scheduler_gm_query(*at<uint64_t>(view, view.layout().work_signal_offset)) == 0)
        return SchedulerAsyncRuntimeServiceResult::IDLE;
    // Empty service owns no work; the resident loop retains its error poll.
    // Active service must observe failure before taking any CHECK/FINISH lease.
    if (scheduler_read_error(run) != 0) return SchedulerAsyncRuntimeServiceResult::ERROR;
    SchedulerAsyncSliceBudget slice{
        budget->directory_words < 4 ? budget->directory_words : 4, budget->lease_probes < 2 ? budget->lease_probes : 2,
        budget->claim_attempts < 2 ? budget->claim_attempts : 2, budget->work_units < 8 ? budget->work_units : 8
    };
    const auto initial = slice;
    SchedulerAsyncClaim claim{};
    const auto found =
        scheduler_async_discover(directory_view(view, true), directory_view(view, false), discovery, &slice, &claim);
    budget->directory_words -= initial.directory_words - slice.directory_words;
    budget->lease_probes -= initial.lease_probes - slice.lease_probes;
    budget->claim_attempts -= initial.claim_attempts - slice.claim_attempts;
    if (found == SchedulerAsyncDiscoverResult::NONE) return SchedulerAsyncRuntimeServiceResult::IDLE;
    if (found != SchedulerAsyncDiscoverResult::CLAIMED) {
        fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
        return SchedulerAsyncRuntimeServiceResult::ERROR;
    }
    const bool check = claim.kind == SchedulerAsyncWorkKind::CHECK;
    auto &lease = *at<SchedulerAsyncLease>(
        view, check ? view.layout().group_lease_offset : view.layout().finish_lease_offset, claim.index
    );
    auto &directory = *at<SchedulerAsyncDirectoryWord>(
        view, check ? view.layout().check_directory_offset : view.layout().finish_directory_offset, claim.index / 64
    );
    uint64_t finish = claim.index;
    int64_t task = SCHEDULER_TASK_REF_INVALID;
    bool ok = true;
    bool advanced = false;
    if (check) {
        SchedulerAsyncGroupIdentity identity{};
        // Seal validated this immutable identity against the task map before
        // publishing the lease. CHECK does not repeat that map lookup per slice.
        load_into(at<SchedulerAsyncGroupIdentity>(view, view.layout().group_identity_offset, claim.index), &identity);
        ok = identity.run_epoch == view.layout().run_epoch && identity.finish_index < view.layout().finish_count;
        task = identity.task_ref;
        finish = identity.finish_index;
        if (ok) {
            SchedulerAsyncWriterSnapshot writer{};
            scheduler_async_writer_snapshot(
                at<SchedulerAsyncGroupWriter>(view, view.layout().group_writer_offset, claim.index), &writer
            );
            SchedulerAsyncGroupProgress progress{};
            load_into(
                at<SchedulerAsyncGroupProgress>(view, view.layout().group_progress_offset, claim.index), &progress
            );
            SchedulerAsyncCheckView conditions{
                at<DeferredCompletionEntry>(view, identity.entries_offset), identity.capacity
            };
            const uint64_t remaining = progress.remaining_mask;
            const auto result =
                scheduler_async_check_step(identity, writer, conditions, backend, &progress, &slice.work_units);
            advanced = progress.remaining_mask != remaining || result == SchedulerAsyncCheckResult::DONE;
            publish(at<SchedulerAsyncGroupProgress>(view, view.layout().group_progress_offset, claim.index), progress);
            ok = result != SchedulerAsyncCheckResult::ERROR && scheduler_read_error(run) == 0;
            if (ok && result == SchedulerAsyncCheckResult::PENDING) {
                ok = scheduler_async_lease_handoff(lease, claim.owned_word, SchedulerAsyncPhase::RUNNABLE).result ==
                     SchedulerAsyncLeaseResult::SUCCESS;
            } else if (ok) {
                ok = scheduler_async_lease_handoff(lease, claim.owned_word, SchedulerAsyncPhase::TERMINAL).result ==
                         SchedulerAsyncLeaseResult::SUCCESS &&
                     scheduler_async_directory_clear_once(directory, claim.index % 64) ==
                         SchedulerAsyncDirectoryResult::SUCCESS;
                if (ok) {
                    const auto settled = scheduler_async_join_settle_group(
                        *at<SchedulerAsyncTaskJoin>(view, view.layout().finish_join_offset, finish)
                    );
                    ok = settled.result == SchedulerAsyncJoinResult::PENDING ||
                         (settled.result == SchedulerAsyncJoinResult::FINISH_OWNER &&
                          activate_finish(view, graph, local, run, finish));
                }
            }
        }
    } else {
        SchedulerAsyncFinishIdentity identity{};
        // Activation validated the immutable task mapping before the FINISH lease.
        load_into(at<SchedulerAsyncFinishIdentity>(view, view.layout().finish_identity_offset, claim.index), &identity);
        ok = identity.run_epoch == view.layout().run_epoch;
        task = identity.task_ref;
        if (ok) {
            SchedulerAsyncFinishProgress progress{};
            load_into(
                at<SchedulerAsyncFinishProgress>(view, view.layout().finish_progress_offset, claim.index), &progress
            );
            auto &join = *at<SchedulerAsyncTaskJoin>(view, view.layout().finish_join_offset, claim.index);
            const auto &cursor = progress.completion;
            const auto phase = cursor.publication.phase;
            const auto route_phase = cursor.resolve.phase;
            const auto next_fanin = cursor.resolve.next_fanin;
            const auto visited = cursor.visited_count;
            const auto pending = cursor.publication.pending_count;
            const auto result = scheduler_async_finish_step<QueueOps, Link>(
                graph, scheduler_base, local, run, identity, &join, &progress, &slice.work_units
            );
            advanced = result == SchedulerCompletionResult::DONE || cursor.publication.phase != phase ||
                       cursor.resolve.phase != route_phase || cursor.resolve.next_fanin != next_fanin ||
                       cursor.visited_count != visited || cursor.publication.pending_count != pending;
            publish(
                at<SchedulerAsyncFinishProgress>(view, view.layout().finish_progress_offset, claim.index), progress
            );
            ok = result != SchedulerCompletionResult::ERROR && scheduler_read_error(run) == 0;
            if (ok && result == SchedulerCompletionResult::PENDING) {
                ok = scheduler_async_lease_handoff(lease, claim.owned_word, SchedulerAsyncPhase::RUNNABLE).result ==
                     SchedulerAsyncLeaseResult::SUCCESS;
            } else if (ok) {
                ok = scheduler_async_lease_handoff(lease, claim.owned_word, SchedulerAsyncPhase::TERMINAL).result ==
                         SchedulerAsyncLeaseResult::SUCCESS &&
                     scheduler_async_directory_clear_once(directory, claim.index % 64) ==
                         SchedulerAsyncDirectoryResult::SUCCESS &&
                     scheduler_async_finish_contribute(local, run, join, lease, directory, claim.index % 64, &progress);
                publish(
                    at<SchedulerAsyncFinishProgress>(view, view.layout().finish_progress_offset, claim.index), progress
                );
            }
        }
    }
    budget->work_units -= initial.work_units - slice.work_units;
    if (!ok) {
        // A failed owner leaves its record and activity evidence resident for
        // the existing run-error/quiescence path; it never contributes success.
        (void)scheduler_async_lease_handoff(lease, claim.owned_word, SchedulerAsyncPhase::FAILED);
        fail(graph, local, run, task, &view, finish);
        return SchedulerAsyncRuntimeServiceResult::ERROR;
    }
    // A probe or failed queue CAS spends a slice without resetting idle backoff.
    return advanced ? SchedulerAsyncRuntimeServiceResult::PROGRESS : SchedulerAsyncRuntimeServiceResult::IDLE;
}
