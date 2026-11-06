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

#include "scheduler_submission_bundle.h"
#include "backend/async_resource_binding.h"

// Local mounted view; all offsets remain in the retained async image.
struct SchedulerSubmissionExecutionView {
    SchedulerSubmissionView view{};
    uint64_t async_offset{0};
};

namespace scheduler_submission_execution_detail {
using scheduler_submission_detail::at;
using scheduler_submission_detail::load;
using scheduler_submission_detail::load_into;

inline __aicore__ void flush(__gm__ void *destination, uint64_t bytes) {
    for (uint64_t offset = 0; offset < bytes; offset += 64)
        scheduler_writeback_cache_line(reinterpret_cast<__gm__ uint8_t *>(destination) + offset);
    scheduler_cache_barrier();
}

// Context and provenance rows are single-writer metadata, disjoint from every
// atomic line. Covering each whole row also replaces a cached previous-run image.
template <typename T>
inline __aicore__ void cached_publish(__gm__ T *destination, const T &value) {
    static_assert(sizeof(T) % 64 == 0);
    auto *words = reinterpret_cast<__gm__ uint64_t *>(destination);
    for (uint32_t offset = 0; offset < sizeof(T); offset += 8) {
        uint64_t word = 0;
        __builtin_memcpy(&word, reinterpret_cast<const uint8_t *>(&value) + offset, 8);
        words[offset / 8] = word;
    }
    flush(destination, sizeof(T));
}

struct Frame {
    uint64_t execution_index{SCHEDULER_ASYNC_INDEX_NONE};
    SchedulerSubmissionExecution execution{};
    SchedulerAsyncGroupIdentity group{};
    __gm__ DispatchPayload *payload{nullptr};
    __gm__ AsyncSubmissionContext *context{nullptr};
    AsyncSubmissionContextHeader header{};
};

template <typename Backend>
inline __aicore__ bool frame(
    const SchedulerSubmissionExecutionView &submissions, const SchedulerAsyncExecutionState<Backend> &state,
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, uint32_t slot_index, Frame *out
) {
    if (out == nullptr || local == nullptr || slot_index >= SCHEDULER_PENDING_SLOT_COUNT) return false;
    scheduler_reset_local(out);
    const auto &slot = local->slots[slot_index];
    if (slot.state != SchedulerDispatchSlotState::READY || slot.generation == 0 ||
        !scheduler_submission_execution_index(
            submissions.view, slot.task_ref, slot.block_idx, slot.subtask_slot, &out->execution_index
        ))
        return false;
    out->payload =
        scheduler_state_at<DispatchPayload>(base, local->dispatch_payload_offset(local->config.self_lane, slot_index));
    if (out->execution_index == SCHEDULER_ASYNC_INDEX_NONE) return true;
    if (!state.view.enabled() || submissions.view.header.run_epoch != state.view.layout.run_epoch ||
        !scheduler_submission_detail::execution(submissions.view, out->execution_index, &out->execution))
        return false;
    const auto &execution = out->execution;
    auto &header = out->header;
    SchedulerSubmissionExecutionUseMap allowed{};
    if (!scheduler_submission_detail::use_map(submissions.view, out->execution_index, &allowed)) return false;
    const uint64_t use_capacity = state.view.layout.uses_bytes / sizeof(AsyncConditionContract);
    if (allowed.use_count != 0 &&
        (allowed.use_begin > use_capacity || allowed.use_count > use_capacity - allowed.use_begin))
        return false;
    header.use_begin = allowed.use_begin;
    header.use_count = allowed.use_count;
    header.run_epoch = state.view.layout.run_epoch;
    header.execution_index = out->execution_index;
    header.region_address = reinterpret_cast<uint64_t>(state.view.base);
    header.region_bytes = state.view.layout.total_bytes;
    header.submission_layout_offset = submissions.async_offset;
    header.dispatch_generation = slot.generation;
    header.frame_address = reinterpret_cast<uint64_t>(out->payload);
    header.block_idx = slot.block_idx;
    header.subtask_slot = slot.subtask_slot;
    header.completion_mode = execution.completion_mode;
    const auto metadata = scheduler_load_dispatch_metadata(base, local, slot.task_ref, 0);
    header.frame_sub_block_id = scheduler_task_is_mix(metadata.flags) ? (local->config.self_lane == 2 ? 1u : 0u) :
                                                                        (slot.subtask_slot == 2 ? 1u : 0u);
    if (out->payload->global_context.sub_block_id != header.frame_sub_block_id ||
        out->payload->local_context.block_idx != static_cast<int32_t>(slot.block_idx))
        return false;
    SchedulerAsyncTaskMap map{};
    if (!scheduler_async_runtime_detail::task_map(state.view, slot.task_ref, &map)) return false;
    const auto group_index = scheduler_async_group_index(map, slot.block_idx, slot.subtask_slot);
    const auto &ctx = out->payload->local_context.async_ctx;
    TaskId token = TaskId::invalid();
    if (execution.completion_mode == CoreAsyncMode::DEFERRED) {
        if (group_index == SCHEDULER_ASYNC_INDEX_NONE ||
            !scheduler_async_runtime_detail::group_identity(state.view, group_index, &out->group) ||
            out->group.capacity != execution.condition_capacity ||
            out->group.condition_types != execution.condition_types ||
            !scheduler_async_runtime_detail::task_token(graph, slot.task_ref, &token))
            return false;
        const auto *writer = scheduler_async_runtime_detail::at<SchedulerAsyncGroupWriter>(
            state.view, state.view.layout.group_writer_offset, group_index
        );
        if (ctx.completion_count != &writer->count || ctx.completion_error_code != &writer->error_code ||
            ctx.completion_entries !=
                scheduler_async_runtime_detail::at<DeferredCompletionEntry>(state.view, out->group.entries_offset) ||
            ctx.completion_capacity != execution.condition_capacity)
            return false;
        const uint64_t use_offset =
            allowed.use_count == 0 ?
                (out->group.binding_uses_offset == SCHEDULER_ASYNC_USES_DYNAMIC ? SCHEDULER_ASYNC_USES_DYNAMIC : 0) :
                state.view.layout.uses_offset + allowed.use_begin * sizeof(AsyncConditionContract);
        if (out->group.binding_use_count != allowed.use_count || out->group.binding_uses_offset != use_offset)
            return false;
    } else if (group_index != SCHEDULER_ASYNC_INDEX_NONE || ctx.completion_count != nullptr ||
               ctx.completion_error_code != nullptr || ctx.completion_entries != nullptr ||
               ctx.completion_capacity != 0) {
        return false;
    }
    if (TaskId::to_uint64(ctx.task_token) != TaskId::to_uint64(token)) return false;
    header.task_token = TaskId::to_uint64(token);
    header.provenance_capacity = execution.provenance_capacity;
    if (header.provenance_capacity != 0)
        header.provenance_offset = submissions.async_offset + submissions.view.header.provenance_offset +
                                   execution.provenance_begin * sizeof(SchedulerSubmissionProvenance);
    if (execution.grant_count != 0)
        out->context =
            at<AsyncSubmissionContext>(submissions.view, submissions.view.header.contexts_offset, out->execution_index);
    return true;
}

inline __aicore__ bool bound(const Frame &frame) {
    if (frame.context == nullptr) return frame.payload->async_submission_context == 0;
    if (frame.payload->async_submission_context != reinterpret_cast<uint64_t>(frame.context)) return false;
    SchedulerLocalStorage<AsyncSubmissionContextHeader> actual_storage;
    load_into(&frame.context->header, &actual_storage.get());
    const auto &actual = actual_storage.get();
    return async_submission_headers_equal(actual, frame.header);
}

inline __aicore__ bool bind(const Frame &frame) {
    if (frame.context == nullptr) return frame.payload->async_submission_context == 0;
    SchedulerLocalStorage<AsyncSubmissionContext> saved_storage;
    load_into(frame.context, &saved_storage.get());
    const auto &saved = saved_storage.get();
    if (saved.header.run_epoch != 0) return bound(frame);
    if (frame.payload->async_submission_context != 0 || saved.header.dispatch_generation != 0 ||
        saved.evidence.attempted != 0 || saved.evidence.committed != 0 || saved.evidence.error_code != 0 ||
        saved.evidence.flags != ASYNC_SUBMISSION_METADATA_PUBLISHED)
        return false;
    SchedulerLocalStorage<AsyncSubmissionContext> context_storage;
    auto &context = context_storage.get();
    context.header = frame.header;
    auto &evidence = context.evidence;
    evidence.run_epoch = frame.header.run_epoch;
    evidence.task_ref = frame.execution.task_ref;
    evidence.execution_index = frame.execution_index;
    evidence.dispatch_generation = frame.header.dispatch_generation;
    evidence.block_idx = frame.header.block_idx;
    evidence.subtask_slot = frame.header.subtask_slot;
    cached_publish(frame.context, context);
    frame.payload->async_submission_context = reinterpret_cast<uint64_t>(frame.context);
    scheduler_publish_cache_line(&frame.payload->async_submission_context);
    return true;
}

inline __aicore__ bool evidence_ready(const Frame &frame, const AsyncSubmissionEvidence &evidence) {
    if (evidence.version != ASYNC_SUBMISSION_VERSION || evidence.run_epoch != frame.header.run_epoch ||
        evidence.task_ref != frame.execution.task_ref || evidence.execution_index != frame.execution_index ||
        evidence.dispatch_generation != frame.header.dispatch_generation ||
        evidence.block_idx != frame.header.block_idx || evidence.subtask_slot != frame.header.subtask_slot ||
        evidence.error_code != 0 || evidence.attempted != evidence.committed ||
        evidence.flags != ASYNC_SUBMISSION_METADATA_PUBLISHED)
        return false;
    for (uint64_t value : evidence.reserved)
        if (value != 0) return false;
    return true;
}

struct SealPublication {
    const Frame *frame;
    inline __aicore__ bool
    after_seal(const SchedulerAsyncGroupIdentity &identity, const SchedulerAsyncGroupWriter &writer) {
        if (frame->context == nullptr) return true;
        if (identity.run_epoch != frame->header.run_epoch || identity.task_ref != frame->execution.task_ref ||
            identity.block_idx != frame->header.block_idx || identity.subtask_slot != frame->header.subtask_slot ||
            writer.sealed != 1 || writer.error_code != 0 || writer.count > identity.capacity)
            return false;
        frame->context->evidence.flags = ASYNC_SUBMISSION_FLAGS;
        flush(&frame->context->evidence, sizeof(AsyncSubmissionEvidence));
        return true;
    }
};

inline __aicore__ bool flush_provenance(const SchedulerSubmissionExecutionView &submissions, const Frame &frame) {
    if (frame.header.completion_mode != CoreAsyncMode::DEFERRED) return true;
    const auto &ctx = frame.payload->local_context.async_ctx;
    const uint32_t count = *ctx.completion_count;
    if (*ctx.completion_error_code != 0 || count > frame.execution.condition_capacity) return true;
    for (uint32_t condition = 0; condition < count; ++condition) {
        if (ctx.completion_entries[condition].completion_type != COMPLETION_TYPE_SDMA_EVENT_RECORD) continue;
        if (condition >= frame.execution.provenance_capacity) return false;
        scheduler_writeback_cache_line(
            at<SchedulerSubmissionProvenance>(
                submissions.view, submissions.view.header.provenance_offset,
                frame.execution.provenance_begin + condition
            )
        );
    }
    scheduler_cache_barrier();
    return true;
}

template <typename Backend>
struct SealingPolicy {
    Backend *backend;
    const Frame *frame;
    inline __aicore__ bool bind_condition(
        const SchedulerAsyncGroupIdentity &identity, uint32_t condition, const DeferredCompletionEntry &entry,
        SchedulerAsyncConditionBinding *out
    ) {
        if (entry.completion_type == COMPLETION_TYPE_COUNTER ||
            identity.binding_uses_offset == SCHEDULER_ASYNC_USES_DYNAMIC)
            return backend->bind_condition(identity, condition, entry, out);
        const auto &h = frame->header;
        const SchedulerSubmissionSealIdentity actual{
            h.run_epoch,
            frame->execution.task_ref,
            frame->execution_index,
            h.dispatch_generation,
            h.frame_address,
            h.task_token,
            h.block_idx,
            h.subtask_slot,
            h.frame_sub_block_id
        };
        if (frame->context != nullptr && backend->bind_condition_at_seal(identity, condition, entry, actual, out))
            return true;
        auto &ctx = frame->payload->local_context.async_ctx;
        if (ctx.completion_error_code != nullptr && *ctx.completion_error_code == 0)
            *ctx.completion_error_code = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
        return false;
    }
};
}  // namespace scheduler_submission_execution_detail

template <typename Backend>
inline __aicore__ bool scheduler_submission_execution_mount(
    const SchedulerAsyncExecutionState<Backend> &state, SchedulerSubmissionExecutionView *out
) {
    if (out == nullptr) return false;
    scheduler_reset_local(out);
    if (!state.view.enabled()) return true;
    SchedulerLocalStorage<SchedulerSubmissionProvenanceView> provenance_storage;
    auto &provenance = provenance_storage.get();
    if (!scheduler_submission_provenance_mount(state.view.base, state.view.layout, &provenance)) return false;
    out->async_offset = provenance.async_offset;
    out->view = provenance.submissions;
    return true;
}

// One local transaction encloses the original CPU admission call. No ordinary
// prefix escapes to service, another candidate, or another scheduler iteration.
template <bool RecordProtocol = true, typename Backend>
inline __aicore__ bool scheduler_submission_execution_admit(
    const SchedulerSubmissionExecutionView &submissions, SchedulerAsyncExecutionState<Backend> *state,
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t slot_index, bool *permit, SchedulerSubmissionTransactionStats *stats = nullptr
) {
    using namespace scheduler_submission_execution_detail;
    if (state == nullptr || local == nullptr || permit == nullptr || slot_index >= SCHEDULER_PENDING_SLOT_COUNT)
        return false;
    *permit = false;
    if (!submissions.view.enabled()) {
        if (local->admission != nullptr)
            return scheduler_ordered_can_enter_slot<RecordProtocol>(graph, base, local, run, slot_index, permit);
        *permit = true;
        return true;
    }
    SchedulerLocalStorage<Frame> current_storage;
    auto &current = current_storage.get();
    if (!frame(submissions, *state, graph, base, local, slot_index, &current) || !bind(current))
        return scheduler_async_runtime_detail::fail(graph, local, run, local->slots[slot_index].task_ref);
    SchedulerSubmissionProvisional transaction{};
    if (current.execution_index != SCHEDULER_ASYNC_INDEX_NONE) {
        SchedulerSubmissionCpuGate<Backend> cpu{state, local, slot_index};
        SchedulerLocalStorage<SchedulerSubmissionBundleIdentity> identity_storage;
        auto &identity = identity_storage.get();
        if (!scheduler_submission_detail::bundle(submissions.view, current.execution.bundle_index, &identity))
            return scheduler_async_runtime_detail::fail(graph, local, run, current.execution.task_ref);
        if (identity.kind == SchedulerSubmissionBundleKind::EXECUTION) {
            const auto result = scheduler_submission_try_ordinary(
                submissions.view, current.execution_index, run, cpu, &transaction, stats
            );
            if (result != SchedulerSubmissionProvisionalResult::RESERVED) {
                if (transaction.active &&
                    !scheduler_submission_abort_provisional(submissions.view, &transaction, run, stats))
                    return false;
                return result == SchedulerSubmissionProvisionalResult::PENDING;
            }
        } else {
            SchedulerSubmissionBudget budget{};
            const auto result =
                scheduler_submission_acquire(submissions.view, current.execution_index, run, cpu, &budget);
            if (result != SchedulerSubmissionResult::READY) return result == SchedulerSubmissionResult::PENDING;
        }
    }
    bool allowed = true;
    const bool valid = local->admission == nullptr ||
                       scheduler_ordered_can_enter_slot<RecordProtocol>(graph, base, local, run, slot_index, &allowed);
    if (!valid || !allowed) {
        if (transaction.active && !scheduler_submission_abort_provisional(submissions.view, &transaction, run, stats))
            return false;
        return valid;
    }
    if (transaction.active && !scheduler_submission_commit_provisional(submissions.view, &transaction, run, stats)) {
        if (transaction.active)
            (void)scheduler_submission_abort_provisional(submissions.view, &transaction, run, stats);
        return false;
    }
    *permit = true;
    return true;
}

template <typename Backend>
inline __aicore__ bool scheduler_submission_execution_returned(
    const SchedulerSubmissionExecutionView &submissions, SchedulerAsyncExecutionState<Backend> *state,
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t slot_index, Backend &backend
) {
    using namespace scheduler_submission_execution_detail;
    if (state == nullptr || local == nullptr || slot_index >= SCHEDULER_PENDING_SLOT_COUNT) return false;
    if (!submissions.view.enabled())
        return scheduler_async_execution_returned(state, graph, base, local, run, slot_index, backend);
    SchedulerLocalStorage<Frame> current_storage;
    auto &current = current_storage.get();
    if (!frame(submissions, *state, graph, base, local, slot_index, &current) || !bound(current))
        return scheduler_async_runtime_detail::fail(graph, local, run, local->slots[slot_index].task_ref);
    if (current.context != nullptr) {
        if (state->permitted_slot != slot_index || state->permitted_generation != local->slots[slot_index].generation)
            return scheduler_async_runtime_detail::fail(graph, local, run, current.execution.task_ref);
        // The SDK may have left a dirty failed attempt without calling commit.
        // Flush only the canonical row, after checking the actual frame pointer.
        flush(&current.context->evidence, sizeof(AsyncSubmissionEvidence));
        SchedulerLocalStorage<AsyncSubmissionEvidence> evidence_storage;
        load_into(&current.context->evidence, &evidence_storage.get());
        const auto &evidence = evidence_storage.get();
        if (!evidence_ready(current, evidence))
            return scheduler_async_runtime_detail::fail(
                graph, local, run, current.execution.task_ref, nullptr, SCHEDULER_ASYNC_INDEX_NONE,
                scheduler_async_runtime_detail::writer_error_result(
                    evidence.error_code != 0 ? evidence.error_code : SIMPLER_ERROR_ASYNC_COMPLETION_INVALID
                )
            );
        if (!flush_provenance(submissions, current))
            return scheduler_async_runtime_detail::fail(
                graph, local, run, current.execution.task_ref, nullptr, SCHEDULER_ASYNC_INDEX_NONE,
                SchedulerGraphResult::ASYNC_COMPLETION_INVALID
            );
    }
    SealingPolicy<Backend> policy{&backend, &current};
    if (!scheduler_async_execution_returned(
            state, graph, base, local, run, slot_index, policy, SealPublication{&current}
        ))
        return false;
    if (current.context == nullptr) return true;
    // Synchronous lanes have no group to publish. Their return is the seal
    // boundary; deferred lanes publish this fact before enabling CHECK.
    if (current.header.completion_mode == CoreAsyncMode::SYNCHRONOUS) {
        current.context->evidence.flags = ASYNC_SUBMISSION_FLAGS;
        flush(&current.context->evidence, sizeof(AsyncSubmissionEvidence));
    }
    const SchedulerSubmissionExecutionIdentity identity{current.header.run_epoch, current.execution.task_ref,
                                                        current.execution_index,  current.header.dispatch_generation,
                                                        current.header.block_idx, current.header.subtask_slot};
    return scheduler_submission_release_all(submissions.view, identity, &current.context->evidence, run);
}
