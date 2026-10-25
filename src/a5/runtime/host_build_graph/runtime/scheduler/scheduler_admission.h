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

#include "scheduler_protocol_block.h"
#include "scheduler_protocol_execution.h"

enum class SchedulerAdmissionResult : uint8_t { PROGRESS, RETRY, ERROR };

inline __aicore__ bool scheduler_admission_range_end(uint64_t offset, uint64_t count, uint64_t stride, uint64_t *end) {
    if (stride == 0 || count > (UINT64_MAX - offset) / stride) return false;
    *end = offset + count * stride;
    return true;
}

inline __aicore__ bool
scheduler_initialize_admission(__gm__ void *base, const SchedulerLocalState *local, SchedulerAdmissionLocalState *out) {
    if (base == nullptr || local == nullptr || out == nullptr || local->config.graph == nullptr ||
        local->config.graph_summary_offset == 0 ||
        (reinterpret_cast<uintptr_t>(base) & (SCHEDULER_STATE_ALIGNMENT - 1)) != 0)
        return false;
    auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, local->config.graph_summary_offset);
    for (uint64_t offset = 0; offset < sizeof(*summary); offset += 64)
        scheduler_observe_cache_line(reinterpret_cast<__gm__ uint8_t *>(summary) + offset);
    const uint64_t workers = local->config.runtime_worker_count;
    const uint64_t clusters = local->config.cluster_count;
    if (summary->cooperative_task_count > local->config.graph->task_count ||
        summary->mix_task_count > summary->cooperative_task_count)
        return false;
    SchedulerAdmissionLocalState next{};
    next.epoch = summary->run_epoch;
    next.control_offset = summary->admission_control_offset;
    next.records_offset = summary->admission_records_offset;
    next.record_stride = summary->admission_record_stride;
    next.log_offset = summary->admission_log_offset;
    next.capacity = summary->cooperative_task_count;
    next.member_word_count = summary->admission_member_word_count;
    if (next.capacity == 0) {
        if (next.control_offset != 0 || next.records_offset != 0 || next.record_stride != 0 || next.log_offset != 0 ||
            next.member_word_count != 0)
            return false;
    } else {
        uint64_t control_end = 0;
        uint64_t records_end = 0;
        uint64_t log_end = 0;
        if (workers == 0 || workers > SCHEDULER_WORKER_CAPACITY || clusters == 0 ||
            workers != clusters * PLATFORM_CORES_PER_BLOCKDIM || summary->worker_count != workers ||
            summary->cluster_count != clusters || local->config.graph->task_count > static_cast<uint64_t>(INT32_MAX) ||
            next.member_word_count != (workers + 63) / 64 || next.control_offset == 0 || next.records_offset == 0 ||
            next.log_offset == 0 ||
            ((next.control_offset | next.records_offset | next.log_offset | next.record_stride) & 127) != 0 ||
            next.record_stride < sizeof(SchedulerAdmissionRecord) + next.member_word_count * sizeof(uint64_t) ||
            !scheduler_admission_range_end(next.control_offset, 1, sizeof(SchedulerAdmissionControl), &control_end) ||
            !scheduler_admission_range_end(
                next.records_offset, local->config.graph->task_count, next.record_stride, &records_end
            ) ||
            !scheduler_admission_range_end(next.log_offset, next.capacity, sizeof(int64_t), &log_end) ||
            control_end > next.records_offset || records_end > next.log_offset ||
            log_end > UINT64_MAX - reinterpret_cast<uintptr_t>(base))
            return false;
    }
    if (summary->mix_task_count != 0) {
        uint64_t mailbox_end = 0;
        uint64_t seals_end = 0;
        if (local->config.shared_context == nullptr) return false;
        const uint64_t cluster = local->config.shared_context->cluster_index;
        if (cluster >= clusters || summary->group_mailboxes_offset == 0 ||
            (summary->group_mailboxes_offset & 127) != 0 || (summary->mix_close_seals_offset & 127) != 0 ||
            summary->group_mailboxes_offset < next.log_offset + next.capacity * sizeof(int64_t) ||
            !scheduler_admission_range_end(
                summary->group_mailboxes_offset, clusters, sizeof(SchedulerGroupMailbox), &mailbox_end
            ) ||
            mailbox_end > summary->mix_close_seals_offset ||
            !scheduler_admission_range_end(
                summary->mix_close_seals_offset, summary->mix_task_count * clusters, sizeof(uint64_t), &seals_end
            ) ||
            seals_end > UINT64_MAX - reinterpret_cast<uintptr_t>(base))
            return false;
        next.gm_mix.mailbox_offset = summary->group_mailboxes_offset + cluster * sizeof(SchedulerGroupMailbox);
        next.gm_mix.cluster_index = static_cast<uint32_t>(cluster);
    }
    if (!scheduler_protocol_bind(base, local)) return false;
    next.initialized = true;
    *out = next;
    return true;
}

inline __aicore__ bool scheduler_admission_fail(
    const SchedulerGraphView &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run, int64_t task_ref,
    SchedulerErrorSite site, const SchedulerLocalSlotState *slot = nullptr
) {
    if (run != nullptr)
        scheduler_record_error(run, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local, site, slot);
    return false;
}

inline __aicore__ bool scheduler_admission_context_valid(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    const SchedulerAdmissionLocalState *admission
) {
    if (base == nullptr || local == nullptr || run == nullptr || admission == nullptr || !admission->initialized ||
        local->config.graph == nullptr || graph.task_count != local->config.graph->task_count)
        return scheduler_admission_fail(
            graph, local, run, SCHEDULER_TASK_REF_INVALID, SchedulerErrorSite::ADMISSION_INVALID_CONFIG
        );
    if (scheduler_gm_fetch_add(run->scheduler_error, UINT64_C(0)) != 0) return false;
    // READY freezes the summary for this run. This is a cached configuration
    // read, not a poll for a new epoch; teardown owns cross-run quiescence.
    auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, local->config.graph_summary_offset);
    if (admission->epoch != summary->run_epoch)
        return scheduler_admission_fail(
            graph, local, run, admission->pending_task_ref, SchedulerErrorSite::ADMISSION_EPOCH_MISMATCH
        );
    return true;
}

inline __aicore__ __gm__ SchedulerAdmissionRecord *
scheduler_admission_record_at(__gm__ void *base, const SchedulerAdmissionLocalState *admission, int64_t task_ref) {
    return scheduler_state_at<SchedulerAdmissionRecord>(
        base, admission->records_offset + static_cast<uint64_t>(task_ref) * admission->record_stride
    );
}

inline __aicore__ bool scheduler_admission_static_record_valid(
    __gm__ void *base, SchedulerLocalState *local, const SchedulerAdmissionLocalState *admission, int64_t task_ref,
    __gm__ const SchedulerAdmissionRecord *record
) {
    const SchedulerTaskMetadata meta = scheduler_load_dispatch_metadata(base, local, task_ref, 0);
    const uint32_t active = static_cast<uint32_t>(__builtin_popcount(meta.active_mask));
    uint32_t subtasks = 0;
    if (!scheduler_task_requires_admission(meta) || (meta.active_mask & ~UINT8_C(7)) != 0 ||
        (meta.flags & UINT8_C(128)) != 0 ||
        !scheduler_subtask_count_checked(meta.logical_block_num, active, &subtasks) ||
        scheduler_task_is_mix(meta.flags) != (active > 1) ||
        scheduler_task_is_spmd(meta.flags) != (meta.logical_block_num > 1) ||
        scheduler_task_has_predicate(meta.flags) ||
        (scheduler_task_requires_sync_start(meta.flags) && meta.logical_block_num == 1) || meta.timing_slot < -1 ||
        meta.timing_slot >= SCHEDULER_TASK_TIMING_SLOT_COUNT || record->task_ref != task_ref ||
        record->logical_block_num != meta.logical_block_num || record->active_mask != meta.active_mask ||
        record->flags != meta.flags ||
        record->member_words_offset != admission->records_offset +
                                           static_cast<uint64_t>(task_ref) * admission->record_stride +
                                           sizeof(SchedulerAdmissionRecord))
        return false;
    if (!scheduler_task_is_mix(meta.flags)) return record->mix_close_seals_offset == 0;
    const auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, local->config.graph_summary_offset);
    const uint64_t row_bytes = static_cast<uint64_t>(local->config.cluster_count) * sizeof(uint64_t);
    if (row_bytes == 0 || summary->mix_close_seals_offset == 0 ||
        record->mix_close_seals_offset < summary->mix_close_seals_offset)
        return false;
    const uint64_t delta = record->mix_close_seals_offset - summary->mix_close_seals_offset;
    return delta % row_bytes == 0 && delta / row_bytes < summary->mix_task_count;
}

// Contexts are the READY-published topology. A bitmap bit is a compact worker
// index, never a physical core ID, a type rank, or an assumed interleaved lane.
inline __aicore__ bool scheduler_admission_expected_members(
    __gm__ void *base, SchedulerLocalState *local, __gm__ const SchedulerAdmissionRecord *record, uint64_t *words,
    uint32_t *member_count
) {
    uint32_t cubes = 0;
    uint32_t vectors = 0;
    const bool mix = scheduler_task_is_mix(record->flags);
    for (uint64_t worker = 0; worker < local->config.runtime_worker_count; ++worker) {
        auto *context = scheduler_worker_context_at(base, local, worker);
        scheduler_observe_cache_line(context);
        scheduler_observe_cache_line(&context->worker_index);
        const int32_t type = context->core_type;
        if (context->worker_index != worker ||
            (type != static_cast<int32_t>(CoreType::AIC) && type != static_cast<int32_t>(CoreType::AIV)))
            return false;
        const bool cube = type == static_cast<int32_t>(CoreType::AIC);
        cubes += cube ? 1 : 0;
        vectors += cube ? 0 : 1;
        if (mix || cube == ((record->active_mask & 1U) != 0)) words[worker / 64] |= UINT64_C(1) << (worker % 64);
    }
    if (cubes != local->config.cluster_count || vectors != cubes * PLATFORM_AIV_CORES_PER_BLOCKDIM) return false;
    *member_count = mix ? cubes + vectors : ((record->active_mask & 1U) != 0 ? cubes : vectors);
    const uint32_t positions = mix ? cubes : *member_count;
    return *member_count != 0 &&
           (!scheduler_task_requires_sync_start(record->flags) || record->logical_block_num <= positions);
}

inline __aicore__ bool scheduler_admission_published_record_valid(
    __gm__ void *base, SchedulerLocalState *local, const SchedulerAdmissionLocalState *admission, int64_t task_ref,
    __gm__ const SchedulerAdmissionRecord *record
) {
    if (!scheduler_admission_static_record_valid(base, local, admission, task_ref, record)) return false;
    uint64_t expected[(SCHEDULER_WORKER_CAPACITY + 63) / 64]{};
    uint32_t members = 0;
    if (!scheduler_admission_expected_members(base, local, record, expected, &members) ||
        record->member_count != members)
        return false;
    const auto *words = scheduler_state_at<uint64_t>(base, record->member_words_offset);
    for (uint64_t word = 0; word < admission->member_word_count; ++word)
        if (words[word] != expected[word]) return false;
    return true;
}

// The caller uniquely owns the candidate through ready pop or direct resolve.
// Only PROGRESS transfers it into pending; RETRY leaves that ownership with the caller.
inline __aicore__ SchedulerAdmissionResult scheduler_try_claim_admission(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerAdmissionLocalState *admission, const SchedulerReadyClaim &candidate
) {
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return SchedulerAdmissionResult::ERROR;
    if (admission->pending_task_ref != SCHEDULER_TASK_REF_INVALID) return SchedulerAdmissionResult::RETRY;
    const int64_t task = candidate.task_ref;
    if (admission->capacity == 0 || !scheduler_task_ref_valid(graph, task) ||
        !scheduler_task_requires_admission(scheduler_load_dispatch_metadata(base, local, task, 0))) {
        scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_RECORD);
        return SchedulerAdmissionResult::ERROR;
    }
    auto *control = scheduler_task_control_at(base, local, task);
    if (scheduler_gm_load(control->state) != static_cast<int64_t>(SchedulerTaskState::BLOCKED) ||
        scheduler_gm_compare_exchange(
            control->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED),
            static_cast<uint64_t>(SchedulerAdmissionState::CLAIMED)
        ) != static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED)) {
        scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_STATE);
        return SchedulerAdmissionResult::ERROR;
    }
    admission->pending_task_ref = task;
    return SchedulerAdmissionResult::PROGRESS;
}

inline __aicore__ bool scheduler_prepare_admission_record(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerAdmissionLocalState *admission
) {
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return false;
    const int64_t task = admission->pending_task_ref;
    if (admission->capacity == 0 || !scheduler_task_ref_valid(graph, task))
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_STATE);
    auto *control = scheduler_task_control_at(base, local, task);
    const uint64_t state = scheduler_gm_fetch_add(control->admission_state, UINT64_C(0));
    if (state == static_cast<uint64_t>(SchedulerAdmissionState::PREPARED)) return true;
    if (state != static_cast<uint64_t>(SchedulerAdmissionState::CLAIMED) ||
        scheduler_gm_fetch_add(control->next_block, UINT64_C(0)) != 0 ||
        scheduler_gm_fetch_add(control->completed_blocks, UINT64_C(0)) != 0)
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_STATE);
    auto *record = scheduler_admission_record_at(base, admission, task);
    scheduler_observe_cache_line(record);
    if (!scheduler_admission_static_record_valid(base, local, admission, task, record) ||
        record->order_key != UINT64_MAX || record->member_count != 0 || record->run_epoch != 0)
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_RECORD);
    auto *words = scheduler_state_at<uint64_t>(base, record->member_words_offset);
    scheduler_observe_cache_line(words);
    for (uint64_t word = 0; word < admission->member_word_count; ++word)
        if (words[word] != 0)
            return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_RECORD);
    uint64_t expected[(SCHEDULER_WORKER_CAPACITY + 63) / 64]{};
    uint32_t members = 0;
    if (!scheduler_admission_expected_members(base, local, record, expected, &members))
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_TOPOLOGY);
    record->run_epoch = admission->epoch;
    record->member_count = members;
    for (uint64_t word = 0; word < admission->member_word_count; ++word)
        words[word] = expected[word];
    scheduler_writeback_cache_line(record);
    scheduler_writeback_cache_line(words);
    scheduler_cache_barrier();
    scheduler_gm_store(control->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::PREPARED));
    return true;
}

inline __aicore__ SchedulerAdmissionResult scheduler_try_commit_admission(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerAdmissionLocalState *admission
) {
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return SchedulerAdmissionResult::ERROR;
    const int64_t task = admission->pending_task_ref;
    if (admission->capacity == 0 || !scheduler_task_ref_valid(graph, task)) {
        scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_STATE);
        return SchedulerAdmissionResult::ERROR;
    }
    auto *control = scheduler_task_control_at(base, local, task);
    auto *record = scheduler_admission_record_at(base, admission, task);
    if (scheduler_gm_fetch_add(control->admission_state, UINT64_C(0)) !=
            static_cast<uint64_t>(SchedulerAdmissionState::PREPARED) ||
        scheduler_gm_fetch_add(control->next_block, UINT64_C(0)) != 0 ||
        scheduler_gm_fetch_add(control->completed_blocks, UINT64_C(0)) != 0) {
        scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_STATE);
        return SchedulerAdmissionResult::ERROR;
    }
    if (record->run_epoch != admission->epoch) {
        scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_EPOCH_MISMATCH);
        return SchedulerAdmissionResult::ERROR;
    }
    if (record->order_key != UINT64_MAX ||
        !scheduler_admission_published_record_valid(base, local, admission, task, record)) {
        scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_RECORD);
        return SchedulerAdmissionResult::ERROR;
    }
    auto *shared = scheduler_state_at<SchedulerAdmissionControl>(base, admission->control_offset);
    const uint64_t owner = local->worker_id() + 1;
    if (owner == 0 || owner > local->config.runtime_worker_count) {
        scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_TOPOLOGY);
        return SchedulerAdmissionResult::ERROR;
    }
    if (scheduler_gm_compare_exchange(shared->commit_lock, UINT64_C(0), owner) != 0)
        return SchedulerAdmissionResult::RETRY;
    // Only this owner releases the lock, including a failed commit. No waiting
    // for a worker, kernel, ready queue, or task completion occurs under it.
    const uint64_t tail = scheduler_gm_fetch_add(shared->commit_tail, UINT64_C(0));
    bool valid = scheduler_admission_context_valid(graph, base, local, run, admission);
    if (valid && tail >= admission->capacity)
        valid = scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_LOG_CAPACITY);
    if (valid) {
        scheduler_protocol_admission(base, local, task, tail, record->flags, true);
        record->order_key = tail;
        scheduler_publish_cache_line(record);
        auto *log = scheduler_state_at<int64_t>(base, admission->log_offset);
        scheduler_gm_store(log[tail], task);
        scheduler_gm_store(control->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED));
        scheduler_cache_barrier();
        scheduler_gm_store(shared->commit_tail, tail + 1);
        scheduler_cache_barrier();
        scheduler_protocol_admission(base, local, task, tail, record->flags, false);
    }
    scheduler_gm_store(shared->commit_lock, UINT64_C(0));
    if (!valid) return SchedulerAdmissionResult::ERROR;
    admission->pending_task_ref = SCHEDULER_TASK_REF_INVALID;
    return SchedulerAdmissionResult::PROGRESS;
}

inline __aicore__ bool scheduler_refresh_local_obligation(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerAdmissionLocalState *admission
) {
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return false;
    if (admission->capacity == 0) {
        if (admission->scan_cursor != 0 || admission->current_task_ref != SCHEDULER_TASK_REF_INVALID)
            return scheduler_admission_fail(
                graph, local, run, admission->current_task_ref, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER
            );
        return true;
    }
    if (admission->current_task_ref != SCHEDULER_TASK_REF_INVALID) {
        if (!scheduler_task_ref_valid(graph, admission->current_task_ref) ||
            admission->scan_cursor >= admission->capacity || admission->current_block_num == 0)
            return scheduler_admission_fail(
                graph, local, run, admission->current_task_ref, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER
            );
        return true;
    }
    auto *shared = scheduler_state_at<SchedulerAdmissionControl>(base, admission->control_offset);
    const uint64_t tail = scheduler_gm_fetch_add(shared->commit_tail, UINT64_C(0));
    if (tail > admission->capacity || admission->scan_cursor > tail ||
        local->worker_id() >= local->config.runtime_worker_count)
        return scheduler_admission_fail(
            graph, local, run, admission->current_task_ref, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER
        );
    auto *log = scheduler_state_at<int64_t>(base, admission->log_offset);
    for (uint32_t scanned = 0; scanned < 8 && admission->scan_cursor < tail; ++scanned) {
        const int64_t task = scheduler_gm_load(log[admission->scan_cursor]);
        if (!scheduler_task_ref_valid(graph, task))
            return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_RECORD);
        auto *record = scheduler_admission_record_at(base, admission, task);
        scheduler_observe_cache_line(record);
        if (record->run_epoch != admission->epoch)
            return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_EPOCH_MISMATCH);
        if (!scheduler_admission_static_record_valid(base, local, admission, task, record))
            return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_RECORD);
        auto *words = scheduler_state_at<uint64_t>(base, record->member_words_offset);
        scheduler_observe_cache_line(words);
        auto *control = scheduler_task_control_at(base, local, task);
        if (record->order_key != admission->scan_cursor ||
            scheduler_gm_fetch_add(control->admission_state, UINT64_C(0)) !=
                static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED) ||
            !scheduler_admission_published_record_valid(base, local, admission, task, record))
            return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_RECORD);
        const uint64_t worker = local->worker_id();
        if ((words[worker / 64] & (UINT64_C(1) << (worker % 64))) != 0) {
            if (admission->current_task_ref != SCHEDULER_TASK_REF_INVALID && admission->current_task_ref != task)
                return scheduler_admission_fail(
                    graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER
                );
            admission->current_task_ref = task;
            admission->current_block_num = record->logical_block_num;
            admission->current_active_mask = record->active_mask;
            admission->current_flags = record->flags;
            scheduler_protocol_member_observed(base, local, admission, tail);
            return true;
        }
        if (admission->current_task_ref != SCHEDULER_TASK_REF_INVALID)
            return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER);
        ++admission->scan_cursor;
    }
    return true;
}

inline __aicore__ bool scheduler_can_enter_ordinary(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerAdmissionLocalState *admission, bool *permit, uint64_t *observed_tail = nullptr
) {
    if (permit == nullptr)
        return scheduler_admission_fail(
            graph, local, run, SCHEDULER_TASK_REF_INVALID, SchedulerErrorSite::ADMISSION_INVALID_CONFIG
        );
    *permit = false;
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return false;
    if (admission->pending_task_ref != SCHEDULER_TASK_REF_INVALID ||
        admission->current_task_ref != SCHEDULER_TASK_REF_INVALID)
        return true;
    if (admission->capacity == 0) {
        if (admission->scan_cursor != 0)
            return scheduler_admission_fail(
                graph, local, run, SCHEDULER_TASK_REF_INVALID, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER
            );
        *permit = true;
        if (observed_tail != nullptr) *observed_tail = 0;
        return true;
    }
    auto *shared = scheduler_state_at<SchedulerAdmissionControl>(base, admission->control_offset);
    const uint64_t tail = scheduler_gm_fetch_add(shared->commit_tail, UINT64_C(0));
    if (tail > admission->capacity || admission->scan_cursor > tail)
        return scheduler_admission_fail(
            graph, local, run, SCHEDULER_TASK_REF_INVALID, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER
        );
    // This critical read grants one entry only. A later publication may wait
    // for that finite kernel, but cannot be ignored by its next invocation.
    *permit = tail == admission->scan_cursor;
    if (*permit && observed_tail != nullptr) *observed_tail = tail;
    return true;
}

inline __aicore__ SchedulerReadyClaim *scheduler_ordered_direct_sink(SchedulerAdmissionLocalState *admission) {
    if (admission == nullptr || !admission->initialized) return nullptr;
    for (auto &candidate : admission->candidates) {
        if (scheduler_task_ref_is_null(candidate.ready.task_ref)) {
            candidate = {};
            return &candidate.ready;
        }
    }
    return nullptr;
}

inline __aicore__ bool
scheduler_ordered_retain_candidate(SchedulerAdmissionLocalState *admission, const SchedulerReadyClaim &ready) {
    if (admission == nullptr || scheduler_task_ref_is_null(ready.task_ref) ||
        ready.task_ref == admission->pending_task_ref || ready.task_ref == admission->current_task_ref)
        return false;
    for (const auto &candidate : admission->candidates)
        if (candidate.ready.task_ref == ready.task_ref) return false;
    auto *sink = scheduler_ordered_direct_sink(admission);
    if (sink == nullptr) return false;
    *sink = ready;
    return true;
}

enum class SchedulerBlockClaimResult : uint8_t { CLAIMED, EXHAUSTED, RETRY, ERROR };
enum class SchedulerBlockFinishResult : uint8_t { PARTIAL, TASK_LAST, ERROR };

inline __aicore__ SchedulerBlockClaimResult scheduler_claim_ordered_cursor(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t *block_idx
) {
    auto *admission = local == nullptr ? nullptr : local->admission;
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return SchedulerBlockClaimResult::ERROR;
    if (block_idx == nullptr || !scheduler_task_ref_valid(graph, admission->current_task_ref) ||
        admission->current_block_num == 0 || scheduler_task_requires_sync_start(admission->current_flags)) {
        scheduler_admission_fail(
            graph, local, run, admission->current_task_ref, SchedulerErrorSite::BLOCK_CURSOR_INVALID
        );
        return SchedulerBlockClaimResult::ERROR;
    }
    auto *control = scheduler_task_control_at(base, local, admission->current_task_ref);
    uint64_t observed = scheduler_gm_query(control->next_block);
    bool critical = false;
    for (uint32_t attempt = 0; attempt < 8; ++attempt) {
        if (observed >= admission->current_block_num) {
            if (!critical) observed = scheduler_gm_fetch_add(control->next_block, UINT64_C(0));
            if (observed == admission->current_block_num) return SchedulerBlockClaimResult::EXHAUSTED;
            if (observed > admission->current_block_num) {
                scheduler_admission_fail(
                    graph, local, run, admission->current_task_ref, SchedulerErrorSite::BLOCK_CURSOR_INVALID
                );
                return SchedulerBlockClaimResult::ERROR;
            }
        }
        const uint64_t claim_start = scheduler_protocol_cycles(local);
        const uint64_t previous = scheduler_gm_compare_exchange(control->next_block, observed, observed + 1);
        if (previous == observed) {
            *block_idx = static_cast<uint32_t>(observed);
            scheduler_protocol_block_claim(base, local, *block_idx, claim_start, scheduler_protocol_cycles(local));
            return SchedulerBlockClaimResult::CLAIMED;
        }
        observed = previous;
        critical = true;
    }
    return SchedulerBlockClaimResult::RETRY;
}

inline __aicore__ SchedulerBlockClaimResult scheduler_claim_ordered_block(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t slot_index
) {
    auto *admission = local == nullptr ? nullptr : local->admission;
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return SchedulerBlockClaimResult::ERROR;
    const int64_t task = admission->current_task_ref;
    if (!scheduler_task_ref_valid(graph, task) || slot_index >= SCHEDULER_PENDING_SLOT_COUNT ||
        local->slots[slot_index].state != SchedulerDispatchSlotState::FREE || admission->current_block_num == 0 ||
        scheduler_task_is_mix(admission->current_flags) ||
        scheduler_task_requires_sync_start(admission->current_flags) ||
        scheduler_metadata_single_subtask_slot(admission->current_active_mask) == UINT8_MAX ||
        scheduler_task_ready_queue(admission->current_flags, admission->current_active_mask) !=
            local->config.self_core_type) {
        scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::BLOCK_CURSOR_INVALID);
        return SchedulerBlockClaimResult::ERROR;
    }
    uint32_t block = 0;
    const auto result = scheduler_claim_ordered_cursor(graph, base, local, run, &block);
    if (result != SchedulerBlockClaimResult::CLAIMED) return result;
    auto &slot = local->slots[slot_index];
    slot.task_ref = task;
    slot.block_idx = block;
    slot.cooperative = true;
    slot.gm_mix = false;
    slot.candidate_index = UINT8_MAX;
    slot.predicate_cached = false;
    slot.mix_tracker = UINT8_MAX;
    slot.subtask_slot = scheduler_metadata_single_subtask_slot(admission->current_active_mask);
    slot.state = SchedulerDispatchSlotState::FILLING;
    return SchedulerBlockClaimResult::CLAIMED;
}

inline __aicore__ SchedulerBlockFinishResult scheduler_contribute_ordered_block(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    const SchedulerLocalSlotState *slot = nullptr
) {
    auto *admission = local == nullptr ? nullptr : local->admission;
    if (!scheduler_admission_context_valid(graph, base, local, run, admission))
        return SchedulerBlockFinishResult::ERROR;
    const int64_t task = admission->current_task_ref;
    if (!scheduler_task_ref_valid(graph, task) || admission->current_block_num == 0) {
        scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::BLOCK_COMPLETION_INVALID, slot);
        return SchedulerBlockFinishResult::ERROR;
    }
    auto *control = scheduler_task_control_at(base, local, task);
    const uint32_t block = slot != nullptr                                 ? slot->block_idx :
                           scheduler_task_is_mix(admission->current_flags) ? admission->gm_mix.block_idx :
                                                                             UINT32_MAX;
    scheduler_protocol_block_complete(base, local, block, true, UINT64_MAX);
    scheduler_cache_barrier();
    const uint64_t previous = scheduler_gm_fetch_add(control->completed_blocks, UINT64_C(1));
    scheduler_protocol_block_complete(base, local, block, false, previous);
    if (previous >= admission->current_block_num) {
        scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::BLOCK_COMPLETION_INVALID, slot);
        return SchedulerBlockFinishResult::ERROR;
    }
    return previous + 1 == admission->current_block_num ? SchedulerBlockFinishResult::TASK_LAST :
                                                          SchedulerBlockFinishResult::PARTIAL;
}

// Generation consumption belongs to the completion caller. Its successful
// generation check makes this contribution unique; an error terminates the run.
inline __aicore__ SchedulerBlockFinishResult scheduler_finish_ordered_block(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t slot_index
) {
    auto *admission = local == nullptr ? nullptr : local->admission;
    if (!scheduler_admission_context_valid(graph, base, local, run, admission))
        return SchedulerBlockFinishResult::ERROR;
    if (slot_index >= SCHEDULER_PENDING_SLOT_COUNT) {
        scheduler_admission_fail(
            graph, local, run, admission->current_task_ref, SchedulerErrorSite::BLOCK_COMPLETION_INVALID
        );
        return SchedulerBlockFinishResult::ERROR;
    }
    const auto &slot = local->slots[slot_index];
    if (!slot.cooperative || slot.state != SchedulerDispatchSlotState::READY ||
        slot.task_ref != admission->current_task_ref || !scheduler_task_ref_valid(graph, slot.task_ref) ||
        slot.block_idx >= admission->current_block_num || scheduler_task_is_mix(admission->current_flags)) {
        scheduler_admission_fail(graph, local, run, slot.task_ref, SchedulerErrorSite::BLOCK_COMPLETION_INVALID, &slot);
        return SchedulerBlockFinishResult::ERROR;
    }
    return scheduler_contribute_ordered_block(graph, base, local, run, &slot);
}

inline __aicore__ bool scheduler_close_ordered_obligation(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *closed
) {
    if (closed == nullptr) return false;
    *closed = false;
    auto *admission = local == nullptr ? nullptr : local->admission;
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return false;
    const int64_t task = admission->current_task_ref;
    if (scheduler_task_ref_is_null(task)) return true;
    if (!scheduler_task_ref_valid(graph, task) || scheduler_task_is_mix(admission->current_flags) ||
        scheduler_task_requires_sync_start(admission->current_flags))
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER);
    for (const auto &slot : local->slots)
        if (slot.cooperative && slot.task_ref == task && slot.state != SchedulerDispatchSlotState::FREE) return true;
    const uint64_t cursor =
        scheduler_gm_fetch_add(scheduler_task_control_at(base, local, task)->next_block, UINT64_C(0));
    if (cursor < admission->current_block_num) return true;
    if (cursor != admission->current_block_num || admission->scan_cursor >= admission->capacity)
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::BLOCK_CURSOR_INVALID);
    scheduler_protocol_member_closed(base, local, admission, cursor, 0, 0);
    ++admission->scan_cursor;
    admission->current_task_ref = SCHEDULER_TASK_REF_INVALID;
    admission->current_block_num = 0;
    admission->current_active_mask = 0;
    admission->current_flags = 0;
    *closed = true;
    return true;
}

inline __aicore__ bool scheduler_ordered_park_ordinary(
    const SchedulerGraphView &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    auto *admission = local->admission;
    for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT; ++index) {
        auto &slot = local->slots[index];
        if (slot.state != SchedulerDispatchSlotState::READY || slot.cooperative ||
            (local->local_ready_mask & (1U << index)) == 0)
            continue;
        if (slot.candidate_index >= SCHEDULER_PENDING_SLOT_COUNT)
            return scheduler_admission_fail(
                graph, local, run, slot.task_ref, SchedulerErrorSite::ORDERED_CANDIDATE_INVALID
            );
        auto &candidate = admission->candidates[slot.candidate_index];
        if (candidate.ready.task_ref != slot.task_ref || candidate.slot_index != index)
            return scheduler_admission_fail(
                graph, local, run, slot.task_ref, SchedulerErrorSite::ORDERED_CANDIDATE_INVALID
            );
        candidate.predicate_cached = slot.predicate_cached;
        candidate.predicate_pass = slot.predicate_pass;
        candidate.slot_index = UINT8_MAX;
        local->local_ready_mask &= static_cast<uint8_t>(~(1U << index));
        scheduler_initialize_free_slot(&slot);
        local->set_timing_slot(local->config.self_lane, index, -1);
    }
    return true;
}

inline __aicore__ SchedulerAdmissionResult scheduler_ordered_publish_pending(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    auto *admission = local->admission;
    if (scheduler_task_ref_is_null(admission->pending_task_ref)) return SchedulerAdmissionResult::PROGRESS;
    if (!scheduler_prepare_admission_record(graph, base, local, run, admission)) return SchedulerAdmissionResult::ERROR;
    return scheduler_try_commit_admission(graph, base, local, run, admission);
}

inline __aicore__ bool scheduler_ordered_can_enter_slot(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t slot_index, bool *permit
) {
    if (permit == nullptr) return false;
    *permit = false;
    auto *admission = local == nullptr ? nullptr : local->admission;
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return false;
    if (slot_index >= SCHEDULER_PENDING_SLOT_COUNT)
        return scheduler_admission_fail(
            graph, local, run, SCHEDULER_TASK_REF_INVALID, SchedulerErrorSite::ORDERED_CANDIDATE_INVALID
        );
    auto &slot = local->slots[slot_index];
    if (slot.state != SchedulerDispatchSlotState::READY)
        return scheduler_admission_fail(
            graph, local, run, slot.task_ref, SchedulerErrorSite::ORDERED_CANDIDATE_INVALID
        );
    if (!scheduler_task_ref_is_null(admission->pending_task_ref)) return true;
    if (slot.cooperative) {
        if (slot.task_ref != admission->current_task_ref || slot.block_idx >= admission->current_block_num)
            return scheduler_admission_fail(
                graph, local, run, slot.task_ref, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER
            );
        if (slot.gm_mix &&
            (admission->gm_mix.phase != SchedulerGmMixPhase::RUNNING || admission->gm_mix.slot_index != slot_index ||
             admission->gm_mix.slot_generation != slot.generation || admission->gm_mix.block_idx != slot.block_idx ||
             admission->gm_mix.subtask_slot != slot.subtask_slot))
            return scheduler_admission_fail(graph, local, run, slot.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
        *permit = true;
        scheduler_protocol_execution_permit(base, local, slot_index, UINT64_MAX, true);
        return true;
    }
    uint64_t observed_tail = UINT64_MAX;
    if (!scheduler_can_enter_ordinary(graph, base, local, run, admission, permit, &observed_tail)) return false;
    if (!*permit) return true;
    if (slot.candidate_index >= SCHEDULER_PENDING_SLOT_COUNT)
        return scheduler_admission_fail(
            graph, local, run, slot.task_ref, SchedulerErrorSite::ORDERED_CANDIDATE_INVALID
        );
    auto &candidate = admission->candidates[slot.candidate_index];
    if (candidate.ready.task_ref != slot.task_ref || candidate.slot_index != slot_index)
        return scheduler_admission_fail(
            graph, local, run, slot.task_ref, SchedulerErrorSite::ORDERED_CANDIDATE_INVALID
        );
    scheduler_protocol_execution_permit(base, local, slot_index, observed_tail, admission->capacity != 0);
    admission->next_candidate = static_cast<uint8_t>((slot.candidate_index + 1) % SCHEDULER_PENDING_SLOT_COUNT);
    candidate = {};
    slot.candidate_index = UINT8_MAX;
    return true;
}
