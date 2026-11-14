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

static inline __aicore__ bool
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
        summary->mix_task_count > local->config.graph->task_count ||
        summary->sync_mix_task_count > summary->cooperative_task_count ||
        summary->sync_mix_task_count > summary->mix_task_count)
        return false;
    SchedulerAdmissionLocalState next{};
    next.epoch = summary->run_epoch;
    next.control_offset = summary->admission_control_offset;
    next.records_offset = summary->admission_records_offset;
    next.record_stride = summary->admission_record_stride;
    next.log_offset = summary->admission_log_offset;
    next.capacity = summary->cooperative_task_count;
    if (next.capacity == 0) {
        if (next.control_offset != 0 || next.records_offset != 0 || next.record_stride != 0 || next.log_offset != 0)
            return false;
    } else {
        uint64_t control_end = 0;
        uint64_t records_end = 0;
        uint64_t log_end = 0;
        if (workers == 0 || workers > SCHEDULER_WORKER_CAPACITY || clusters == 0 ||
            workers != clusters * PLATFORM_CORES_PER_BLOCKDIM || summary->worker_count != workers ||
            summary->cluster_count != clusters || local->config.graph->task_count > static_cast<uint64_t>(INT32_MAX) ||
            next.control_offset == 0 || next.records_offset == 0 || next.log_offset == 0 ||
            ((next.control_offset | next.records_offset | next.log_offset | next.record_stride) &
             (SCHEDULER_STATE_ALIGNMENT - 1)) != 0 ||
            next.record_stride != sizeof(SchedulerAdmissionRecord) ||
            !scheduler_admission_range_end(next.control_offset, 1, sizeof(SchedulerAdmissionControl), &control_end) ||
            !scheduler_admission_range_end(
                next.records_offset, local->config.graph->task_count, next.record_stride, &records_end
            ) ||
            !scheduler_admission_range_end(next.log_offset, next.capacity, sizeof(int64_t), &log_end) ||
            control_end > next.records_offset || records_end > next.log_offset ||
            log_end > UINT64_MAX - reinterpret_cast<uintptr_t>(base))
            return false;
    }
    if (summary->sync_mix_task_count != 0) {
        uint64_t mailbox_end = 0;
        uint64_t seals_end = 0;
        if (local->config.shared_context == nullptr) return false;
        const uint64_t cluster = local->config.shared_context->cluster_index;
        if (cluster >= clusters || summary->group_mailboxes_offset == 0 ||
            (summary->group_mailboxes_offset & (alignof(SchedulerGroupMailbox) - 1)) != 0 ||
            (summary->mix_close_seals_offset & (SCHEDULER_STATE_ALIGNMENT - 1)) != 0 ||
            summary->group_mailboxes_offset < next.log_offset + next.capacity * sizeof(int64_t) ||
            !scheduler_admission_range_end(
                summary->group_mailboxes_offset, clusters, sizeof(SchedulerGroupMailbox), &mailbox_end
            ) ||
            mailbox_end > summary->mix_close_seals_offset ||
            !scheduler_admission_range_end(
                summary->mix_close_seals_offset, summary->sync_mix_task_count * clusters, sizeof(SchedulerMixCloseSeal),
                &seals_end
            ) ||
            seals_end > UINT64_MAX - reinterpret_cast<uintptr_t>(base))
            return false;
        next.gm_mix.mailbox_offset = summary->group_mailboxes_offset + cluster * sizeof(SchedulerGroupMailbox);
        next.gm_mix.cluster_index = static_cast<uint32_t>(cluster);
    }
    if (summary->topology_layout != SchedulerTopologyLayout::GENERAL &&
        summary->topology_layout != SchedulerTopologyLayout::COMPACT_LAUNCH)
        return false;
    if (next.capacity != 0 && summary->topology_layout == SchedulerTopologyLayout::COMPACT_LAUNCH) {
        if (local->config.self_lane >= PLATFORM_CORES_PER_BLOCKDIM) return false;
        const uint64_t worker = local->worker_id();
        if (worker >= workers) return false;
        const uint64_t cluster = worker < clusters ? worker : (worker - clusters) / 2;
        const uint32_t lane = worker < clusters ? 0U : 1U + static_cast<uint32_t>((worker - clusters) % 2);
        if (local->config.self_lane != lane || local->config.self_core_type != (lane == 0 ? 0U : 1U) ||
            local->config.worker_ids[0] != cluster || local->config.worker_ids[1] != clusters + 2 * cluster ||
            local->config.worker_ids[2] != clusters + 2 * cluster + 1)
            return false;
        // Discovery validated this compact topology before READY.
        next.topology_worker_count = workers;
    }
    if (!scheduler_protocol_bind(base, local)) return false;
    next.initialized = true;
    *out = next;
    return true;
}

static inline __aicore__ bool scheduler_admission_fail(
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run, int64_t task_ref,
    SchedulerErrorSite site, const SchedulerLocalSlotState *slot = nullptr
) {
    if (run != nullptr)
        scheduler_record_error(run, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local, site, slot);
    return false;
}

namespace scheduler_admission_detail {

// Configuration pointers are validated by the enclosing bounded operation.
// Neither the error signal nor the current run's epoch is a reusable permit.
static inline __aicore__ bool active_run_valid(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    const SchedulerAdmissionLocalState *admission
) {
    const uint64_t error = scheduler_read_worker_error(run, local->worker_id());
    if (error != 0) return false;
    // READY freezes the summary for this run. This is a cached configuration
    // read, not a poll for a new epoch; teardown owns cross-run quiescence.
    auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, local->config.graph_summary_offset);
    if (admission->epoch != summary->run_epoch)
        return scheduler_admission_fail(
            graph, local, run, admission->pending_task_ref, SchedulerErrorSite::ADMISSION_EPOCH_MISMATCH
        );
    return true;
}

}  // namespace scheduler_admission_detail

// Admission operations require the resident's live base/local/run and an
// initialized admission object. run_resident_scheduler installs admission only
// after both initializers succeed; fill/entry dispatch select this path only
// when it is present, and exit drain handles absence before calling here.
// The cached graph binding is immutable until the run fence. Error and epoch
// observations remain per-operation checks, including after lock acquisition.
static inline __aicore__ bool scheduler_admission_context_valid(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    const SchedulerAdmissionLocalState *admission
) {
    return scheduler_admission_detail::active_run_valid(graph, base, local, run, admission);
}

inline __aicore__ __gm__ SchedulerAdmissionRecord *
scheduler_admission_record_at(__gm__ void *base, const SchedulerAdmissionLocalState *admission, int64_t task_ref) {
    return scheduler_state_at<SchedulerAdmissionRecord>(
        base, admission->records_offset + static_cast<uint64_t>(task_ref) * admission->record_stride
    );
}

static inline __aicore__ bool scheduler_admission_static_record_valid(
    __gm__ void *base, SchedulerLocalState *local, const SchedulerAdmissionLocalState *admission, int64_t task_ref,
    __gm__ const SchedulerAdmissionRecord *record
) {
    const SchedulerTaskMetadata meta = scheduler_load_dispatch_metadata(base, local, task_ref, 0);
    const uint32_t active = static_cast<uint32_t>(meta.active_mask.count());
    uint32_t subtasks = 0;
    if (!scheduler_task_requires_admission(meta) || !meta.active_mask.is_valid() || (meta.flags & UINT8_C(128)) != 0 ||
        !scheduler_subtask_count_checked(meta.logical_block_num, active, &subtasks) ||
        scheduler_task_is_mix(meta.flags) != (active > 1) ||
        scheduler_task_is_spmd(meta.flags) != (meta.logical_block_num > 1) ||
        scheduler_task_has_predicate(meta.flags) ||
        (scheduler_task_requires_sync_start(meta.flags) && meta.logical_block_num == 1) || meta.timing_slot < -1 ||
        meta.timing_slot >= SCHEDULER_TASK_TIMING_SLOT_COUNT || record->task_ref != task_ref ||
        record->logical_block_num != meta.logical_block_num ||
        scheduler_lane_mask_load(record->active_mask) != meta.active_mask || record->flags != meta.flags)
        return false;
    if (!scheduler_task_is_mix(meta.flags)) return record->mix_close_seals_offset == 0;
    const auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, local->config.graph_summary_offset);
    const uint64_t row_bytes = static_cast<uint64_t>(local->config.cluster_count) * sizeof(SchedulerMixCloseSeal);
    if (row_bytes == 0 || summary->mix_close_seals_offset == 0 ||
        record->mix_close_seals_offset < summary->mix_close_seals_offset)
        return false;
    const uint64_t delta = record->mix_close_seals_offset - summary->mix_close_seals_offset;
    return delta % row_bytes == 0 && delta / row_bytes < summary->sync_mix_task_count;
}

// GENERAL worker IDs need not encode a role. Validate the complete frozen
// topology once, then use the run-entry-validated local lane and core type.
static inline __aicore__ bool scheduler_admission_validate_topology(
    __gm__ void *base, SchedulerLocalState *local, SchedulerAdmissionLocalState *admission
) {
    const uint64_t workers = local->config.runtime_worker_count;
    const uint64_t cubes = local->config.cluster_count;
    if (workers == 0 || workers > SCHEDULER_WORKER_CAPACITY || cubes == 0 ||
        workers != cubes * PLATFORM_CORES_PER_BLOCKDIM)
        return false;
    if (admission->topology_worker_count != 0) return admission->topology_worker_count == workers;
    uint64_t cube_words[(SCHEDULER_WORKER_CAPACITY + 63) / 64]{};
    uint64_t vector_words[2][(SCHEDULER_WORKER_CAPACITY + 63) / 64]{};
    uint64_t observed_cubes = 0;
    for (uint64_t worker = 0; worker < workers; ++worker) {
        auto *context = scheduler_worker_context_at(base, local, worker);
        scheduler_observe_cache_line(context);
        const int32_t type = context->core_type;
        if (context->worker_index != worker ||
            (type != static_cast<int32_t>(CoreType::AIC) && type != static_cast<int32_t>(CoreType::AIV)))
            return false;
        if (type == static_cast<int32_t>(CoreType::AIC)) {
            ++observed_cubes;
            cube_words[worker / 64] |= UINT64_C(1) << (worker % 64);
            scheduler_invalidate_cache_line(&context->cluster_index);
            scheduler_invalidate_cache_line(&context->cluster_worker_ids[0]);
            scheduler_invalidate_cache_line(&context->cluster_worker_ids[2]);
            scheduler_cache_barrier();
            if (context->cluster_index >= cubes || context->cluster_worker_ids[0] != worker) return false;
            for (uint32_t lane = 1; lane < 3; ++lane) {
                const uint64_t peer = context->cluster_worker_ids[lane];
                if (peer >= workers || peer == worker) return false;
                auto *peer_context = scheduler_worker_context_at(base, local, peer);
                scheduler_invalidate_cache_line(peer_context);
                scheduler_invalidate_cache_line(&peer_context->cluster_index);
                scheduler_invalidate_cache_line(&peer_context->cluster_worker_ids[0]);
                scheduler_invalidate_cache_line(&peer_context->cluster_worker_ids[2]);
                scheduler_cache_barrier();
                if (peer_context->worker_index != peer ||
                    peer_context->core_type != static_cast<int32_t>(CoreType::AIV) ||
                    peer_context->cluster_index != context->cluster_index)
                    return false;
                for (uint32_t member_lane = 0; member_lane < PLATFORM_CORES_PER_BLOCKDIM; ++member_lane)
                    if (peer_context->cluster_worker_ids[member_lane] != context->cluster_worker_ids[member_lane])
                        return false;
                const uint64_t bit = UINT64_C(1) << (peer % 64);
                if (((vector_words[0][peer / 64] | vector_words[1][peer / 64]) & bit) != 0) return false;
                vector_words[lane - 1][peer / 64] |= bit;
            }
        }
    }
    if (observed_cubes != cubes) return false;
    const uint64_t words = (workers + 63) / 64;
    for (uint64_t word = 0; word < words; ++word) {
        const uint64_t remaining = workers - word * 64;
        const uint64_t valid = remaining >= 64 ? UINT64_MAX : (UINT64_C(1) << remaining) - 1;
        const uint64_t vectors = vector_words[0][word] | vector_words[1][word];
        if ((vectors & cube_words[word]) != 0 || (vectors | cube_words[word]) != valid) return false;
    }
    admission->topology_worker_count = workers;
    return true;
}

static inline __aicore__ bool scheduler_admission_member_count(
    const SchedulerLocalState *local, const __gm__ SchedulerAdmissionRecord *record, uint32_t *member_count
) {
    const uint32_t cubes = local->config.cluster_count;
    const LaneMask lanes = scheduler_lane_mask_load(record->active_mask);
    const bool mix = scheduler_task_is_mix(record->flags);
    *member_count = mix ? cubes * scheduler_mix_resource_mask(lanes).count() :
                          cubes * (lanes.has_aic() ? 1U : PLATFORM_AIV_CORES_PER_BLOCKDIM);
    const uint32_t positions = mix ? cubes : *member_count;
    return *member_count != 0 &&
           (!scheduler_task_requires_sync_start(record->flags) || record->logical_block_num <= positions);
}

inline __aicore__ bool
scheduler_admission_is_member(const SchedulerLocalState *local, const __gm__ SchedulerAdmissionRecord *record) {
    const LaneMask lanes = scheduler_lane_mask_load(record->active_mask);
    return scheduler_task_is_mix(record->flags) ?
               scheduler_mix_resource_mask(lanes).has(static_cast<SubtaskSlot>(local->config.self_lane)) :
               local->config.self_core_type == (lanes.has_aic() ? 0U : 1U);
}

// Pending names a local ready observation. Only commit_lock grants publication ownership.
static inline __aicore__ SchedulerAdmissionResult scheduler_stage_admission_candidate(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerAdmissionLocalState *admission, int64_t task_ref
) {
    SchedulerScalarScope scalar_detail(local, SchedPhaseKind::ScalarAdmissionClaim, task_ref);
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return SchedulerAdmissionResult::ERROR;
    if (admission->pending_task_ref != SCHEDULER_TASK_REF_INVALID) return SchedulerAdmissionResult::RETRY;
    if (admission->capacity == 0 || !scheduler_task_ref_valid(graph, task_ref) ||
        !scheduler_task_requires_admission(scheduler_task_metadata_at(base, local, task_ref)->flags)) {
        scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::ADMISSION_INVALID_STATE);
        return SchedulerAdmissionResult::ERROR;
    }
    admission->pending_task_ref = task_ref;
    return SchedulerAdmissionResult::PROGRESS;
}

// Bootstrap is the sole immutable-record writer before the scan gate.
static inline __aicore__ bool scheduler_bootstrap_prepare_admission(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerAdmissionLocalState *admission, int64_t task
) {
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return false;
    if (admission->pending_task_ref != SCHEDULER_TASK_REF_INVALID ||
        scheduler_gm_query(run->bootstrap_scan_complete) != 0)
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_STATE);
    if (admission->capacity == 0 || !scheduler_task_ref_valid(graph, task))
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_STATE);
    auto *control = scheduler_task_control_at(base, local, task);
    const uint64_t state = scheduler_gm_query(control->admission_state);
    if (state != static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED) ||
        scheduler_gm_query(control->claim_seal) != 0 ||
        scheduler_gm_fetch_add(control->claim_tickets, UINT64_C(0)) != 0 ||
        scheduler_gm_fetch_add(control->next_block, UINT64_C(0)) != 0 ||
        scheduler_gm_fetch_add(control->completion_state, UINT64_C(0)) !=
            scheduler_completion_initial_state(
                scheduler_task_metadata_at(base, local, task)->logical_block_num *
                scheduler_task_lanes(*scheduler_task_metadata_at(base, local, task)).count()
            ))
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_STATE);
    auto *record = scheduler_admission_record_at(base, admission, task);
    scheduler_observe_cache_line(record);
    if (!scheduler_admission_static_record_valid(base, local, admission, task, record) ||
        record->order_key != UINT64_MAX || record->member_count != 0 || record->run_epoch != 0)
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_RECORD);
    uint32_t members = 0;
    if (!scheduler_admission_validate_topology(base, local, admission) ||
        !scheduler_admission_member_count(local, record, &members))
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_TOPOLOGY);
    record->run_epoch = admission->epoch;
    record->member_count = members;
    scheduler_writeback_cache_line(record);
    scheduler_cache_barrier();
    scheduler_gm_store(control->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED));
    return true;
}

// The lock serializes order assignment. Multiple local observations may name
// one task; an already committed observation retires without a second log entry.
template <bool RecordProtocol = true>
static inline __aicore__ SchedulerAdmissionResult scheduler_try_commit_admission(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerAdmissionLocalState *admission
) {
    SchedulerScalarScope scalar_detail(local, SchedPhaseKind::ScalarAdmissionCommit, -1);
    const int64_t task = admission->pending_task_ref;
    if (!scheduler_task_ref_valid(graph, task)) {
        scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_STATE);
        return SchedulerAdmissionResult::ERROR;
    }
    auto *control = scheduler_task_control_at(base, local, task);
    auto *record = scheduler_admission_record_at(base, admission, task);
    auto *shared = scheduler_state_at<SchedulerAdmissionControl>(base, admission->control_offset);
    // Run entry validates the immutable worker index against runtime_worker_count.
    const uint64_t owner = local->worker_id() + 1;
    if (scheduler_gm_compare_exchange(shared->commit_lock, UINT64_C(0), owner) != 0) {
        if (!scheduler_admission_context_valid(graph, base, local, run, admission))
            return SchedulerAdmissionResult::ERROR;
        return SchedulerAdmissionResult::RETRY;
    }
    bool valid = scheduler_admission_context_valid(graph, base, local, run, admission);
    const uint64_t state = scheduler_gm_fetch_add(control->admission_state, UINT64_C(0));
    if (valid && state == static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED)) {
        admission->pending_task_ref = SCHEDULER_TASK_REF_INVALID;
        scheduler_gm_store(shared->commit_lock, UINT64_C(0));
        return SchedulerAdmissionResult::PROGRESS;
    }
    if (valid && state != static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED))
        valid = scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_STATE);
    const uint64_t tail = valid ? scheduler_gm_fetch_add(shared->commit_tail, UINT64_C(0)) : 0;
    if (valid) {
        scheduler_observe_cache_line(record);
        if (record->run_epoch != admission->epoch)
            valid = scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_EPOCH_MISMATCH);
        else if (record->order_key != UINT64_MAX)
            valid = scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_RECORD);
    }
    if (valid && tail >= admission->capacity)
        valid = scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_LOG_CAPACITY);
    if (valid) {
        if constexpr (RecordProtocol) scheduler_protocol_admission(base, local, task, tail, record->flags, true);
        record->order_key = tail;
        scheduler_publish_cache_line(record);
        auto *log = scheduler_state_at<int64_t>(base, admission->log_offset);
        scheduler_publish_signal(log[tail], task);
        scheduler_gm_store(control->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED));
        scheduler_cache_barrier();
        scheduler_publish_signal(shared->commit_tail, tail + 1);
        scheduler_cache_barrier();
        if constexpr (RecordProtocol) scheduler_protocol_admission(base, local, task, tail, record->flags, false);
    }
    scheduler_gm_store(shared->commit_lock, UINT64_C(0));
    if (!valid) return SchedulerAdmissionResult::ERROR;
    admission->pending_task_ref = SCHEDULER_TASK_REF_INVALID;
    return SchedulerAdmissionResult::PROGRESS;
}

static inline __aicore__ bool scheduler_read_claim_cursor(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t *cursor, bool *used_seal = nullptr
);

inline __aicore__ void scheduler_reset_current_task_cache(SchedulerAdmissionLocalState &admission) {
    admission.current_task_ref = SCHEDULER_TASK_REF_INVALID;
    admission.current_block_num = 0;
    admission.current_active_mask = LaneMask::none();
    admission.current_flags = 0;
    admission.current_claim_positions = 0;
    admission.current_claim_limit = 0;
    admission.current_uses_ticket_claim = false;
    admission.current_single_subtask = UINT8_MAX;
    admission.current_resource_mask = LaneMask::none();
    admission.current_mix_close_seal_offset = 0;
    admission.sync_claimed = false;
    admission.ticket_exhausted = false;
}

inline __aicore__ void
scheduler_retire_local_obligation(SchedulerLocalState *local, SchedulerAdmissionLocalState &admission) {
    ++admission.scan_cursor;
    scheduler_reset_current_task_cache(admission);
    scheduler_cluster_advance_sync(local->cluster_signal, local->config.self_lane, admission.scan_cursor);
}

template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_refresh_local_obligation(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerAdmissionLocalState *admission
) {
    SchedulerScalarScope scalar_detail(local, SchedPhaseKind::ScalarAdmissionRefresh, -1);
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
    // Retaining an existing local obligation does not change ownership. A
    // scan that may adopt or pass a new obligation still takes the strong gate.
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    auto *shared = scheduler_state_at<SchedulerAdmissionControl>(base, admission->control_offset);
    const uint64_t tail = scheduler_gm_query(shared->commit_tail);
    if (tail > admission->capacity || admission->scan_cursor > tail ||
        local->worker_id() >= local->config.runtime_worker_count)
        return scheduler_admission_fail(
            graph, local, run, admission->current_task_ref, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER
        );
    auto *log = scheduler_state_at<int64_t>(base, admission->log_offset);
    for (uint32_t scanned = 0; scanned < 8 && admission->scan_cursor < tail; ++scanned) {
        const int64_t task = scheduler_gm_query(log[admission->scan_cursor]);
        if (!scheduler_task_ref_valid(graph, task))
            return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_RECORD);
        auto *record = scheduler_admission_record_at(base, admission, task);
        scheduler_invalidate_cache_line(record);
        scheduler_cache_barrier();
        if (record->run_epoch != admission->epoch)
            return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_EPOCH_MISMATCH);
        auto *control = scheduler_task_control_at(base, local, task);
        // Bootstrap validates immutable shape and topology; epoch, order and
        // publication remain dynamic gates for every adoption.
        if (record->order_key != admission->scan_cursor ||
            scheduler_gm_query(control->admission_state) != static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED))
            return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_RECORD);
        if (scheduler_admission_is_member(local, record)) {
            if (admission->current_task_ref != SCHEDULER_TASK_REF_INVALID && admission->current_task_ref != task)
                return scheduler_admission_fail(
                    graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER
                );
            admission->current_task_ref = task;
            admission->current_block_num = record->logical_block_num;
            admission->current_active_mask = scheduler_lane_mask_load(record->active_mask);
            admission->current_flags = record->flags;
            admission->current_claim_positions =
                scheduler_block_claim_positions(admission->current_active_mask, local->config.cluster_count);
            admission->current_claim_limit = admission->current_claim_positions;
            admission->current_uses_ticket_claim = scheduler_task_uses_ticket_claim(
                record->flags, admission->current_active_mask, record->logical_block_num, local->config.cluster_count
            );
            admission->current_single_subtask =
                scheduler_task_is_mix(record->flags) ?
                    (admission->current_active_mask.has(static_cast<SubtaskSlot>(local->config.self_lane)) ?
                         local->config.self_lane :
                         UINT8_MAX) :
                    scheduler_metadata_single_subtask_slot(admission->current_active_mask);
            admission->current_resource_mask = scheduler_task_is_mix(record->flags) ?
                                                   scheduler_mix_resource_mask(admission->current_active_mask) :
                                                   admission->current_active_mask;
            admission->current_mix_close_seal_offset =
                scheduler_task_is_mix(record->flags) ?
                    record->mix_close_seals_offset +
                        static_cast<uint64_t>(admission->gm_mix.cluster_index) * sizeof(SchedulerMixCloseSeal) :
                    0;
            admission->sync_claimed = false;
            admission->ticket_exhausted = false;
            if (scheduler_task_requires_sync_start(record->flags))
                scheduler_cluster_hold_sync(local->cluster_signal, local->config.self_lane, admission->scan_cursor);
            const auto binding_error = scheduler_observe_graph_arguments(graph, task);
            if (binding_error != SchedulerGraphBindingError::NONE)
                return scheduler_admission_fail(
                    graph, local, run, task, scheduler_graph_binding_error_site(binding_error)
                );
            if constexpr (RecordProtocol) scheduler_protocol_member_observed(base, local, admission, tail);
            return true;
        }
        if (admission->current_task_ref != SCHEDULER_TASK_REF_INVALID)
            return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER);
        ++admission->scan_cursor;
        scheduler_cluster_advance_sync(local->cluster_signal, local->config.self_lane, admission->scan_cursor);
    }
    return true;
}

static inline __aicore__ bool scheduler_can_enter_ordinary(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerAdmissionLocalState *admission, bool *permit, uint64_t *observed_tail = nullptr
) {
    SchedulerScalarScope scalar_detail(local, SchedPhaseKind::ScalarOrdinaryPermit, -1);
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
    const uint64_t tail = scheduler_gm_query(shared->commit_tail);
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

enum class SchedulerBlockClaimResult : uint8_t { CLAIMED, EXHAUSTED, RETRY, ERROR };

static inline __aicore__ bool scheduler_read_claim_cursor(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t *cursor, bool *used_seal
) {
    SchedulerScalarScope scalar_detail(local, SchedPhaseKind::ScalarBlockCursorRead, -1);
    SchedulerScalarScope scalar_validate(local, SchedPhaseKind::ScalarCursorValidate);
    if (used_seal != nullptr) *used_seal = false;
    auto *admission = local->admission;
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return false;
    if (cursor == nullptr || !scheduler_task_ref_valid(graph, admission->current_task_ref) ||
        admission->current_block_num == 0)
        return scheduler_admission_fail(
            graph, local, run, admission->current_task_ref, SchedulerErrorSite::BLOCK_CURSOR_INVALID
        );
    auto *control = scheduler_task_control_at(base, local, admission->current_task_ref);
    const uint64_t count = admission->current_block_num;
    scalar_validate.close();
    const uint64_t seal_hint = scheduler_scalar_result(local, SchedPhaseKind::ScalarClaimSealProbe, [&]() __aicore__ {
        return scheduler_gm_query(control->claim_seal);
    });
    const uint64_t seal =
        seal_hint == 0 ? 0 : scheduler_scalar_result(local, SchedPhaseKind::ScalarClaimSealConfirm, [&]() __aicore__ {
            return scheduler_gm_fetch_add(control->claim_seal, UINT64_C(0));
        });
    if (seal != 0 && seal != count)
        return scheduler_admission_fail(
            graph, local, run, admission->current_task_ref, SchedulerErrorSite::BLOCK_CURSOR_INVALID
        );
    if (seal == count && scheduler_scalar_result(local, SchedPhaseKind::ScalarNextBlockProbe, [&]() __aicore__ {
                             return scheduler_gm_query(control->next_block);
                         }) == count) {
        *cursor = count;
        if (used_seal != nullptr) *used_seal = true;
        return true;
    }
    // A missing publication never stalls the original strong-read path. A weak
    // mismatch can be a delayed atomic image, so confirm before rejecting it.
    const uint64_t confirmed = scheduler_scalar_result(local, SchedPhaseKind::ScalarNextBlockConfirm, [&]() __aicore__ {
        return scheduler_gm_fetch_add(control->next_block, UINT64_C(0));
    });
    if (confirmed > count || (seal == count && confirmed != count))
        return scheduler_admission_fail(
            graph, local, run, admission->current_task_ref, SchedulerErrorSite::BLOCK_CURSOR_INVALID
        );
    *cursor = confirmed;
    return true;
}

// Called only by the successful FAA/CAS that confirms valid count N, never by
// an observer of exhaustion or an overshooting reservation ticket.
template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_publish_claim_seal(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t block
) {
    auto *admission = local->admission;
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return false;
    auto *control = scheduler_task_control_at(base, local, admission->current_task_ref);
    if (scheduler_gm_query(control->claim_seal) != 0)
        return scheduler_admission_fail(
            graph, local, run, admission->current_task_ref, SchedulerErrorSite::BLOCK_CURSOR_INVALID
        );
    if constexpr (RecordProtocol) scheduler_protocol_claim_seal(base, local, block, true, admission->current_block_num);
    scheduler_cache_barrier();
    scheduler_gm_store(control->claim_seal, static_cast<uint64_t>(admission->current_block_num));
    if constexpr (RecordProtocol)
        scheduler_protocol_claim_seal(base, local, block, false, admission->current_block_num);
    return true;
}

template <bool RecordProtocol = true>
static inline __aicore__ SchedulerBlockClaimResult scheduler_claim_ordered_cursor(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t *block_idx
) {
    auto *admission = local->admission;
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return SchedulerBlockClaimResult::ERROR;
    if (block_idx == nullptr || !scheduler_task_ref_valid(graph, admission->current_task_ref) ||
        admission->current_block_num == 0) {
        scheduler_admission_fail(
            graph, local, run, admission->current_task_ref, SchedulerErrorSite::BLOCK_CURSOR_INVALID
        );
        return SchedulerBlockClaimResult::ERROR;
    }
    if (admission->sync_claimed) return SchedulerBlockClaimResult::RETRY;
    // A named ordinary rendezvous goes first, before this obligation owns a
    // ticket or physical slot that the rendezvous would need to reclaim.
    if (scheduler_task_requires_sync_start(admission->current_flags) &&
        scheduler_cluster_names_lane(local->cluster_signal, local->config.self_lane))
        return SchedulerBlockClaimResult::RETRY;
    auto *control = scheduler_task_control_at(base, local, admission->current_task_ref);
    if (admission->current_uses_ticket_claim) {
        if (admission->ticket_exhausted) return SchedulerBlockClaimResult::EXHAUSTED;
        const uint64_t count = admission->current_block_num;
        if (scheduler_gm_query(control->next_block) >= count) {
            uint64_t committed = 0;
            if (!scheduler_read_claim_cursor(graph, base, local, run, &committed))
                return SchedulerBlockClaimResult::ERROR;
            if (committed > count) {
                scheduler_admission_fail(
                    graph, local, run, admission->current_task_ref, SchedulerErrorSite::BLOCK_CURSOR_INVALID
                );
                return SchedulerBlockClaimResult::ERROR;
            }
            if (committed == count) {
                admission->ticket_exhausted = true;
                return SchedulerBlockClaimResult::EXHAUSTED;
            }
        }
        const uint64_t positions = admission->current_claim_positions;
        const uint64_t limit = admission->current_claim_limit;
        if (scheduler_gm_query(control->claim_tickets) >= limit &&
            scheduler_gm_fetch_add(control->claim_tickets, UINT64_C(0)) >= limit) {
            scheduler_admission_fail(
                graph, local, run, admission->current_task_ref, SchedulerErrorSite::BLOCK_CURSOR_INVALID
            );
            return SchedulerBlockClaimResult::ERROR;
        }
        const uint64_t claim_start = RecordProtocol ? scheduler_protocol_cycles(local) : 0;
        const uint64_t ticket = scheduler_gm_fetch_add(control->claim_tickets, UINT64_C(1));
        if (ticket >= count) {
            admission->ticket_exhausted = true;
            if (ticket >= limit) {
                scheduler_admission_fail(
                    graph, local, run, admission->current_task_ref, SchedulerErrorSite::BLOCK_CURSOR_INVALID
                );
                return SchedulerBlockClaimResult::ERROR;
            }
            return SchedulerBlockClaimResult::EXHAUSTED;
        }
        // A ticket is irrevocable ownership of one block. The bounded valid
        // count reaches N only after every valid reservation is published.
        admission->sync_claimed = scheduler_task_requires_sync_start(admission->current_flags);
        const uint64_t previous = scheduler_gm_fetch_add(control->next_block, UINT64_C(1));
        if (previous >= count || ticket > previous + positions - 1) {
            scheduler_admission_fail(
                graph, local, run, admission->current_task_ref, SchedulerErrorSite::BLOCK_CURSOR_INVALID
            );
            return SchedulerBlockClaimResult::ERROR;
        }
        *block_idx = static_cast<uint32_t>(ticket);
        if constexpr (RecordProtocol)
            scheduler_protocol_block_claim(
                base, local, *block_idx, claim_start, scheduler_protocol_cycles(local), previous + 1
            );
        if (previous + 1 == count && !scheduler_publish_claim_seal<RecordProtocol>(graph, base, local, run, *block_idx))
            return SchedulerBlockClaimResult::ERROR;
        return SchedulerBlockClaimResult::CLAIMED;
    }
    // A full-residency sync cohort has exactly one eligible claim position per
    // logical block: one AIC lane per cluster (the Mix coordinator when the
    // task is Mix), or the two AIV lanes of a cluster, as
    // `scheduler_block_claim_positions` counts them. The `sync_claimed` gate
    // above allows each position one claim. One fetch-add on next_block hands
    // out the dynamic index and
    // publishes the claim; the actual last claim still publishes the seal, and
    // claim_tickets stays zero for this branch.
    if (scheduler_task_requires_sync_start(admission->current_flags) &&
        admission->current_block_num == admission->current_claim_positions) {
        const uint64_t count = admission->current_block_num;
        const uint64_t claim_start = RecordProtocol ? scheduler_protocol_cycles(local) : 0;
        const uint64_t block = scheduler_gm_fetch_add(control->next_block, UINT64_C(1));
        if (block >= count) {
            // No valid position can arrive here: the cohort has as many
            // positions as blocks and each already claimed its own. Treat the
            // extra reservation as cursor corruption rather than return an
            // index outside the block range.
            scheduler_admission_fail(
                graph, local, run, admission->current_task_ref, SchedulerErrorSite::BLOCK_CURSOR_INVALID
            );
            return SchedulerBlockClaimResult::ERROR;
        }
        *block_idx = static_cast<uint32_t>(block);
        admission->sync_claimed = true;
        if constexpr (RecordProtocol)
            scheduler_protocol_block_claim(
                base, local, *block_idx, claim_start, scheduler_protocol_cycles(local), block + 1
            );
        if (block + 1 == count && !scheduler_publish_claim_seal<RecordProtocol>(graph, base, local, run, *block_idx))
            return SchedulerBlockClaimResult::ERROR;
        return SchedulerBlockClaimResult::CLAIMED;
    }
    scheduler_admission_fail(graph, local, run, admission->current_task_ref, SchedulerErrorSite::BLOCK_CURSOR_INVALID);
    return SchedulerBlockClaimResult::ERROR;
}

template <bool RecordProtocol = true>
static inline __aicore__ SchedulerBlockClaimResult scheduler_claim_ordered_block(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t slot_index
) {
    SchedulerScalarScope scalar_detail(local, SchedPhaseKind::ScalarBlockClaim, -1);
    auto *admission = local->admission;
    // The cursor helper performs the error check before ownership changes.
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return SchedulerBlockClaimResult::ERROR;
    const int64_t task = admission->current_task_ref;
    if (!scheduler_task_ref_valid(graph, task) || slot_index >= SCHEDULER_PENDING_SLOT_COUNT ||
        local->slots[slot_index].state != SchedulerDispatchSlotState::FREE || admission->current_block_num == 0 ||
        scheduler_task_is_mix(admission->current_flags) || admission->current_single_subtask == UINT8_MAX ||
        scheduler_task_execution_engine(admission->current_flags, admission->current_active_mask) !=
            local->config.self_core_type) {
        scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::BLOCK_CURSOR_INVALID);
        return SchedulerBlockClaimResult::ERROR;
    }
    uint32_t block = 0;
    const auto result = scheduler_claim_ordered_cursor<RecordProtocol>(graph, base, local, run, &block);
    if (result != SchedulerBlockClaimResult::CLAIMED) return result;
    auto &slot = local->slots[slot_index];
    slot.task_ref = task;
    slot.block_idx = block;
    slot.args_observed = true;
    slot.gm_mix = false;
    slot.predicate = SchedulerPredicateState::UNEVALUATED;
    slot.subtask_slot = admission->current_single_subtask;
    slot.state = SchedulerDispatchSlotState::FILLING;
    return SchedulerBlockClaimResult::CLAIMED;
}

template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_close_ordered_obligation(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *closed
) {
    SchedulerScalarScope scalar_detail(local, SchedPhaseKind::ScalarAdmissionClose, -1);
    if (closed == nullptr) return false;
    *closed = false;
    auto *admission = local->admission;
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return false;
    const int64_t task = admission->current_task_ref;
    if (scheduler_task_ref_is_null(task)) return true;
    if (!scheduler_task_ref_valid(graph, task) || scheduler_task_is_mix(admission->current_flags))
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER);
    for (const auto &slot : local->slots)
        if (slot.task_ref == task && slot.state != SchedulerDispatchSlotState::FREE) return true;
    uint64_t cursor = 0;
    bool used_claim_seal = false;
    if (!scheduler_read_claim_cursor(graph, base, local, run, &cursor, &used_claim_seal)) return false;
    if (cursor < admission->current_block_num && !admission->sync_claimed) return true;
    if (cursor > admission->current_block_num || (admission->sync_claimed && cursor == 0) ||
        admission->scan_cursor >= admission->capacity)
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::BLOCK_CURSOR_INVALID);
    if constexpr (RecordProtocol)
        scheduler_protocol_member_closed(base, local, admission, cursor, 0, 0, used_claim_seal);
    scheduler_retire_local_obligation(local, *admission);
    *closed = true;
    return true;
}

static inline __aicore__ bool scheduler_ordered_restore_ordinary(
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    SchedulerScalarScope scalar_detail(local, SchedPhaseKind::ScalarOrdinaryRestore, -1);
    auto *admission = local->admission;
    for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT; ++index) {
        auto &slot = local->slots[index];
        if (slot.state != SchedulerDispatchSlotState::READY || slot.cluster_mix ||
            scheduler_slot_uses_admission(local, slot) || (local->local_ready_mask & (1U << index)) == 0)
            continue;
        if (!slot.execution_token || !scheduler_restore_execution_token(local, index))
            return scheduler_admission_fail(
                graph, local, run, slot.task_ref, SchedulerErrorSite::ORDERED_CANDIDATE_INVALID
            );
    }
    return true;
}

template <bool RecordProtocol = true>
static inline __aicore__ SchedulerAdmissionResult scheduler_ordered_publish_pending(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    SchedulerScalarScope scalar_detail(local, SchedPhaseKind::ScalarAdmissionPublish, -1);
    auto *admission = local->admission;
    if (scheduler_task_ref_is_null(admission->pending_task_ref)) return SchedulerAdmissionResult::PROGRESS;
    return scheduler_try_commit_admission<RecordProtocol>(graph, base, local, run, admission);
}

template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_ordered_can_enter_slot(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t slot_index, bool *permit
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarAdmission, -1);
    if (permit == nullptr) return false;
    *permit = false;
    auto *admission = local->admission;
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
    if (scheduler_slot_uses_admission(local, slot)) {
        if (slot.task_ref != admission->current_task_ref || slot.block_idx >= admission->current_block_num)
            return scheduler_admission_fail(
                graph, local, run, slot.task_ref, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER
            );
        if (slot.gm_mix &&
            (admission->gm_mix.phase != SchedulerGmMixPhase::RUNNING || admission->gm_mix.slot_index != slot_index ||
             admission->gm_mix.block_idx != slot.block_idx || admission->gm_mix.subtask_slot != slot.subtask_slot))
            return scheduler_admission_fail(graph, local, run, slot.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
        // Ordinary entry has its own strong gate below; cooperative entry
        // needs this one. Neither path reuses an earlier poll's permission.
        if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
        if (scheduler_task_requires_sync_start(admission->current_flags) &&
            scheduler_cluster_names_lane(local->cluster_signal, local->config.self_lane))
            return true;
        *permit = true;
        if constexpr (RecordProtocol) scheduler_protocol_execution_permit(base, local, slot_index, UINT64_MAX, true);
        return true;
    }
    // The current cooperative obligation is already in the committed prefix.
    // An unpublished candidate cannot precede it; ordinary entry still waits.
    if (!scheduler_task_ref_is_null(admission->pending_task_ref)) return true;
    uint64_t observed_tail = UINT64_MAX;
    if (!scheduler_can_enter_ordinary(graph, base, local, run, admission, permit, &observed_tail)) return false;
    if (!*permit) return true;
    if (!slot.execution_token)
        return scheduler_admission_fail(
            graph, local, run, slot.task_ref, SchedulerErrorSite::ORDERED_CANDIDATE_INVALID
        );
    if constexpr (RecordProtocol)
        scheduler_protocol_execution_permit(base, local, slot_index, observed_tail, admission->capacity != 0);
    return true;
}
