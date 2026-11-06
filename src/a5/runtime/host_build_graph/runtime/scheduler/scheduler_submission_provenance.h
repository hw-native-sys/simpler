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

#include "scheduler_submission_types.h"
#include "scheduler_local_storage.h"
#include "scheduler_memory.h"
#include "scheduler_async_lease.h"
#include "async_submission_context.h"

struct SchedulerSubmissionView {
    __gm__ uint8_t *base{nullptr};
    SchedulerSubmissionHeader header{};
    inline __aicore__ bool enabled() const { return base != nullptr; }
};

namespace scheduler_submission_detail {
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

template <typename T>
inline __aicore__ T load(__gm__ const T *source) {
    static_assert(alignof(T) <= 8, "Over-aligned snapshots require qualified local storage and load_into");
    T result{};
    load_into(source, &result);
    return result;
}
template <typename T>
inline __aicore__ __gm__ T *at(const SchedulerSubmissionView &view, uint64_t offset, uint64_t index = 0) {
    return reinterpret_cast<__gm__ T *>(view.base + offset) + index;
}
inline __aicore__ bool region(uint64_t bytes, uint64_t offset, uint64_t count, uint64_t stride, uint64_t *end) {
    if (offset < *end || offset % 128 != 0 || offset > bytes || count > (bytes - offset) / stride) return false;
    *end = offset + count * stride;
    return true;
}
inline __aicore__ bool task_map(const SchedulerSubmissionView &view, int64_t task, SchedulerSubmissionTaskMap *out) {
    if (!view.enabled() || out == nullptr || task < 0 || static_cast<uint64_t>(task) >= view.header.task_count)
        return false;
    *out = load(at<SchedulerSubmissionTaskMap>(view, view.header.task_maps_offset, task));
    if (out->execution_base == SCHEDULER_ASYNC_INDEX_NONE)
        return out->bundle_base == SCHEDULER_ASYNC_INDEX_NONE && out->bundle_count == 0;
    const uint32_t lanes = __builtin_popcount(out->lane_mask);
    const bool sync = scheduler_task_requires_sync_start(out->flags);
    const uint64_t executions = static_cast<uint64_t>(out->block_count) * lanes;
    return out->flags <= UINT8_MAX && scheduler_task_is_executable(out->flags) &&
           !scheduler_task_is_inline(out->flags) && out->lane_mask != 0 && (out->lane_mask & ~7u) == 0 &&
           out->block_count != 0 && executions <= SCHEDULER_ASYNC_JOIN_LIMIT &&
           scheduler_task_is_mix(out->flags) == (lanes > 1) &&
           scheduler_task_is_spmd(out->flags) == (out->block_count > 1) && (!sync || out->block_count > 1) &&
           out->execution_base < view.header.execution_count &&
           executions <= view.header.execution_count - out->execution_base &&
           out->bundle_count == (sync ? 1u : out->block_count) && out->bundle_base < view.header.bundle_count &&
           out->bundle_count <= view.header.bundle_count - out->bundle_base;
}
inline __aicore__ bool
execution(const SchedulerSubmissionView &view, uint64_t index, SchedulerSubmissionExecution *out) {
    if (index >= view.header.execution_count || out == nullptr) return false;
    *out = load(at<SchedulerSubmissionExecution>(view, view.header.executions_offset, index));
    SchedulerSubmissionTaskMap map{};
    if (!task_map(view, out->task_ref, &map) || map.execution_base == SCHEDULER_ASYNC_INDEX_NONE ||
        out->block_idx >= map.block_count || out->subtask_slot >= 3 ||
        (map.lane_mask & (1u << out->subtask_slot)) == 0 ||
        out->grant_count > SCHEDULER_SUBMISSION_MAX_EXECUTION_GRANTS ||
        out->grant_index_begin > view.header.grant_count ||
        out->grant_count > view.header.grant_count - out->grant_index_begin ||
        (out->subtask_slot == 0 && out->grant_count != 0))
        return false;
    const bool capability_valid = out->completion_mode == CoreAsyncMode::SYNCHRONOUS ?
                                      out->condition_capacity == 0 && out->condition_types == 0 :
                                      out->completion_mode == CoreAsyncMode::DEFERRED && out->condition_capacity >= 1 &&
                                          out->condition_capacity <= CORE_ASYNC_MAX_CONDITIONS &&
                                          out->condition_types != 0 &&
                                          (out->condition_types & ~CORE_ASYNC_CONDITION_TYPES) == 0;
    const uint32_t expected_provenance =
        out->grant_count != 0 && out->completion_mode == CoreAsyncMode::DEFERRED ? out->condition_capacity : 0;
    if (!capability_valid || out->provenance_capacity != expected_provenance ||
        out->provenance_begin > view.header.provenance_count ||
        out->provenance_capacity > view.header.provenance_count - out->provenance_begin ||
        (expected_provenance != 0 && (out->condition_types & CORE_ASYNC_CONDITION_SDMA_POST_DONE) == 0))
        return false;
    const uint64_t expected = map.execution_base + out->block_idx * __builtin_popcount(map.lane_mask) +
                              __builtin_popcount(map.lane_mask & ((1u << out->subtask_slot) - 1));
    return index == expected &&
           out->bundle_index == map.bundle_base + (scheduler_task_requires_sync_start(map.flags) ? 0 : out->block_idx);
}

inline __aicore__ bool
use_map(const SchedulerSubmissionView &view, uint64_t index, SchedulerSubmissionExecutionUseMap *out) {
    if (index >= view.header.execution_count || out == nullptr) return false;
    *out = load(at<SchedulerSubmissionExecutionUseMap>(view, view.header.execution_use_maps_offset, index));
    return out->use_count <= UINT32_MAX && (out->use_count == 0 ? out->use_begin == SCHEDULER_ASYNC_INDEX_NONE :
                                                                  out->use_begin != SCHEDULER_ASYNC_INDEX_NONE &&
                                                                      out->use_count <= UINT64_MAX - out->use_begin);
}
inline __aicore__ bool
bundle(const SchedulerSubmissionView &view, uint64_t index, SchedulerSubmissionBundleIdentity *out) {
    if (index >= view.header.bundle_count || out == nullptr) return false;
    load_into(at<SchedulerSubmissionBundleIdentity>(view, view.header.bundle_identities_offset, index), out);
    SchedulerSubmissionTaskMap map{};
    if (out->run_epoch != view.header.run_epoch || !task_map(view, out->task_ref, &map) ||
        map.bundle_base == SCHEDULER_ASYNC_INDEX_NONE || out->lane_mask != map.lane_mask ||
        out->block_count != map.block_count || out->request_begin > view.header.request_count ||
        out->request_count > view.header.request_count - out->request_begin ||
        out->request_count > view.header.workspace_count)
        return false;
    for (uint64_t reserved : out->reserved)
        if (reserved != 0) return false;
    const bool sync = scheduler_task_requires_sync_start(map.flags);
    const uint64_t lanes = __builtin_popcount(map.lane_mask);
    const auto kind = sync      ? SchedulerSubmissionBundleKind::SYNC_TASK :
                      lanes > 1 ? SchedulerSubmissionBundleKind::MIX_BLOCK :
                                  SchedulerSubmissionBundleKind::EXECUTION;
    return out->kind == kind && (sync ? out->block_scope == UINT64_MAX : out->block_scope < map.block_count) &&
           index == map.bundle_base + (sync ? 0 : out->block_scope) &&
           out->execution_begin == map.execution_base + (sync ? 0 : out->block_scope * lanes) &&
           out->execution_count == (sync ? map.block_count * lanes : lanes);
}
inline __aicore__ bool
workspace(const SchedulerSubmissionView &view, uint64_t index, SchedulerSubmissionWorkspace *out) {
    if (index >= view.header.workspace_count || out == nullptr) return false;
    *out = load(at<SchedulerSubmissionWorkspace>(view, view.header.workspaces_offset, index));
    if (out->address == 0 || out->bytes == 0 || out->bytes > UINT64_MAX - out->address ||
        out->qualification_epoch != view.header.run_epoch ||
        out->qualification != SchedulerSubmissionPoolQualification::EXCLUSIVE_RUN || out->queue_num == 0 ||
        out->physical_channels == 0 || out->physical_channels > SCHEDULER_SUBMISSION_SDMA_CHANNELS ||
        out->queue_num > out->physical_channels || out->reserved0 != 0)
        return false;
    for (uint64_t reserved : out->reserved)
        if (reserved != 0) return false;
    return true;
}
inline __aicore__ bool request(
    const SchedulerSubmissionView &view, const SchedulerSubmissionBundleIdentity &identity, uint64_t ordinal,
    SchedulerSubmissionRequest *out, SchedulerSubmissionWorkspace *workspace_out
) {
    if (ordinal >= identity.request_count || out == nullptr) return false;
    *out = load(at<SchedulerSubmissionRequest>(view, view.header.requests_offset, identity.request_begin + ordinal));
    if (!workspace(view, out->workspace_index, workspace_out) || out->grant_count == 0 ||
        out->grant_count > identity.execution_count ||
        out->grant_count > workspace_out->physical_channels / workspace_out->queue_num ||
        out->grant_begin >= view.header.grant_count || out->grant_count > view.header.grant_count - out->grant_begin)
        return false;
    if (ordinal != 0) {
        const auto previous = load(
            at<SchedulerSubmissionRequest>(view, view.header.requests_offset, identity.request_begin + ordinal - 1)
        );
        if (previous.workspace_index >= out->workspace_index) return false;
    }
    return true;
}
}  // namespace scheduler_submission_detail

inline __aicore__ bool scheduler_submission_mount(
    __gm__ uint8_t *base, uint64_t bytes, uint64_t epoch, uint64_t tasks, SchedulerSubmissionView *out
) {
    using namespace scheduler_submission_detail;
    if (out == nullptr || epoch == 0) return false;
    if (bytes == 0) {
        scheduler_reset_local(out);
        return true;
    }
    if (base == nullptr || reinterpret_cast<uint64_t>(base) % 128 != 0 || bytes < sizeof(SchedulerSubmissionHeader) ||
        bytes > UINT64_MAX - reinterpret_cast<uint64_t>(base))
        return false;
    SchedulerLocalStorage<SchedulerSubmissionView> view_storage;
    auto &view = view_storage.get();
    view.base = base;
    load_into(reinterpret_cast<__gm__ const SchedulerSubmissionHeader *>(base), &view.header);
    const auto &h = view.header;
    if (h.version != SCHEDULER_SUBMISSION_VERSION || h.run_epoch != epoch || h.task_count != tasks || tasks == 0 ||
        tasks > INT32_MAX || h.total_bytes != bytes || h.total_bytes % 128 != 0 || h.workspace_count == 0 ||
        h.bundle_count == 0 || h.execution_count == 0 || h.request_count == 0 || h.grant_count == 0 ||
        h.bundle_count > h.execution_count || h.workspace_count > h.request_count || h.request_count > h.grant_count)
        return false;
    for (uint64_t reserved : h.reserved)
        if (reserved != 0) return false;
    uint64_t end = sizeof(h);
    if (!region(bytes, h.task_maps_offset, h.task_count, sizeof(SchedulerSubmissionTaskMap), &end) ||
        !region(bytes, h.workspaces_offset, h.workspace_count, sizeof(SchedulerSubmissionWorkspace), &end) ||
        !region(bytes, h.pools_offset, h.workspace_count, sizeof(SchedulerSubmissionPool), &end) ||
        !region(bytes, h.bundle_identities_offset, h.bundle_count, sizeof(SchedulerSubmissionBundleIdentity), &end) ||
        !region(bytes, h.bundle_leases_offset, h.bundle_count, sizeof(SchedulerAsyncLease), &end) ||
        !region(bytes, h.bundle_progress_offset, h.bundle_count, sizeof(SchedulerSubmissionProgress), &end) ||
        !region(bytes, h.executions_offset, h.execution_count, sizeof(SchedulerSubmissionExecution), &end) ||
        !region(bytes, h.requests_offset, h.request_count, sizeof(SchedulerSubmissionRequest), &end) ||
        !region(bytes, h.grants_offset, h.grant_count, sizeof(SchedulerSubmissionGrant), &end) ||
        !region(bytes, h.release_states_offset, h.grant_count, sizeof(SchedulerSubmissionReleaseState), &end) ||
        !region(bytes, h.execution_grant_indices_offset, h.grant_count, sizeof(uint64_t), &end) ||
        !region(bytes, h.contexts_offset, h.execution_count, sizeof(AsyncSubmissionContext), &end) ||
        !region(bytes, h.provenance_offset, h.provenance_count, sizeof(SchedulerSubmissionProvenance), &end) ||
        !region(
            bytes, h.execution_use_maps_offset, h.execution_count, sizeof(SchedulerSubmissionExecutionUseMap), &end
        ))
        return false;
    *out = view;
    return true;
}

inline __aicore__ bool scheduler_submission_execution_index(
    const SchedulerSubmissionView &view, int64_t task, uint32_t block, uint32_t lane, uint64_t *index
) {
    using namespace scheduler_submission_detail;
    if (index == nullptr) return false;
    *index = SCHEDULER_ASYNC_INDEX_NONE;
    if (!view.enabled()) return true;
    SchedulerSubmissionTaskMap map{};
    if (!task_map(view, task, &map)) return false;
    if (map.execution_base == SCHEDULER_ASYNC_INDEX_NONE) return true;
    if (block >= map.block_count || lane >= 3 || (map.lane_mask & (1u << lane)) == 0) return false;
    *index = map.execution_base + block * __builtin_popcount(map.lane_mask) +
             __builtin_popcount(map.lane_mask & ((1u << lane) - 1));
    return true;
}

inline __aicore__ bool scheduler_submission_lookup_grant(
    const SchedulerSubmissionView &view, uint64_t execution_index, uint64_t ordinal, SchedulerSubmissionGrant *grant,
    SchedulerSubmissionWorkspace *workspace, uint64_t *grant_index, bool require_active = true
) {
    using namespace scheduler_submission_detail;
    SchedulerSubmissionExecution execution_row{};
    if (grant == nullptr || workspace == nullptr || grant_index == nullptr ||
        !execution(view, execution_index, &execution_row) || ordinal >= execution_row.grant_count)
        return false;
    auto &lease = *at<SchedulerAsyncLease>(view, view.header.bundle_leases_offset, execution_row.bundle_index);
    const uint64_t lease_word = scheduler_async_lease_load(lease);
    if (!scheduler_async_lease_valid(lease_word) ||
        scheduler_async_lease_phase(lease_word) != SchedulerAsyncPhase::TERMINAL)
        return false;
    SchedulerLocalStorage<SchedulerSubmissionProgress> progress_storage;
    load_into(
        at<SchedulerSubmissionProgress>(view, view.header.bundle_progress_offset, execution_row.bundle_index),
        &progress_storage.get()
    );
    const auto &progress = progress_storage.get();
    if (progress.phase != SchedulerSubmissionProgressPhase::GRANTED) return false;
    *grant_index =
        load(at<uint64_t>(view, view.header.execution_grant_indices_offset, execution_row.grant_index_begin + ordinal));
    if (*grant_index >= view.header.grant_count) return false;
    *grant = load(at<SchedulerSubmissionGrant>(view, view.header.grants_offset, *grant_index));
    if (require_active &&
        scheduler_gm_fetch_add(
            at<SchedulerSubmissionReleaseState>(view, view.header.release_states_offset, *grant_index)->phase,
            UINT64_C(0)
        ) != static_cast<uint64_t>(SchedulerSubmissionReleasePhase::RESERVED))
        return false;
    return grant->execution_index == execution_index && grant->reserved == 0 &&
           scheduler_submission_detail::workspace(view, grant->workspace_index, workspace) &&
           grant->group_index < workspace->physical_channels / workspace->queue_num;
}

// This non-owning view is mounted from the retained run image, not from a
// condition address or a token cookie.
struct SchedulerSubmissionProvenanceView {
    SchedulerSubmissionView submissions{};
    __gm__ const uint8_t *async_base{nullptr};
    uint64_t async_bytes{0};
    uint64_t async_offset{0};
};

inline __aicore__ bool scheduler_submission_provenance_mount(
    __gm__ const uint8_t *async_base, const SchedulerAsyncLayout &layout, SchedulerSubmissionProvenanceView *out
) {
    if (out == nullptr) return false;
    scheduler_reset_local(out);
    if (layout.total_bytes == 0) return async_base == nullptr;
    SchedulerLocalStorage<SchedulerAsyncCohortView> cohorts_storage;
    auto &cohorts = cohorts_storage.get();
    if (!SchedulerAsyncCohortView::mount(async_base, layout, &cohorts)) return false;
    if (cohorts.header.auxiliary_bytes == 0) return true;
    SchedulerLocalStorage<SchedulerSubmissionProvenanceView> view_storage;
    auto &view = view_storage.get();
    view.async_base = async_base;
    view.async_bytes = layout.total_bytes;
    view.async_offset = layout.scratch_offset + cohorts.header.auxiliary_offset;
    if (!scheduler_submission_mount(
            cohorts.base + cohorts.header.auxiliary_offset, cohorts.header.auxiliary_bytes, layout.run_epoch,
            layout.task_count, &view.submissions
        ))
        return false;
    *out = view;
    return true;
}

inline __aicore__ __gm__ AsyncSubmissionContext *
scheduler_submission_canonical_context(const SchedulerSubmissionProvenanceView &view, uint64_t execution_index) {
    if (!view.submissions.enabled() || execution_index >= view.submissions.header.execution_count) return nullptr;
    return scheduler_submission_detail::at<AsyncSubmissionContext>(
        view.submissions, view.submissions.header.contexts_offset, execution_index
    );
}

// SDK code uses this immutable-header reader and its own cached evidence. It
// must not read an in-flight dirty attempt through the bypass snapshot below.
inline __aicore__ bool scheduler_submission_context_header(
    const SchedulerSubmissionProvenanceView &view, uint64_t execution_index, AsyncSubmissionContextHeader *out
) {
    using namespace scheduler_submission_detail;
    auto *context = scheduler_submission_canonical_context(view, execution_index);
    SchedulerSubmissionExecution execution_row{};
    SchedulerSubmissionExecutionUseMap uses{};
    if (out == nullptr || context == nullptr || !execution(view.submissions, execution_index, &execution_row) ||
        !use_map(view.submissions, execution_index, &uses))
        return false;
    load_into(&context->header, out);
    const auto &h = *out;
    const uint64_t provenance_offset = execution_row.provenance_capacity == 0 ?
                                           0 :
                                           view.async_offset + view.submissions.header.provenance_offset +
                                               execution_row.provenance_begin * sizeof(SchedulerSubmissionProvenance);
    return h.version == ASYNC_SUBMISSION_VERSION && h.context_bytes == sizeof(AsyncSubmissionContext) &&
           h.run_epoch == view.submissions.header.run_epoch &&
           h.region_address == reinterpret_cast<uint64_t>(view.async_base) && h.region_bytes == view.async_bytes &&
           h.submission_layout_offset == view.async_offset && h.execution_index == execution_index &&
           h.dispatch_generation != 0 && h.frame_address != 0 && h.frame_address % alignof(DispatchPayload) == 0 &&
           h.frame_address <= UINT64_MAX - sizeof(DispatchPayload) && h.frame_sub_block_id <= 1 &&
           h.block_idx == execution_row.block_idx && h.subtask_slot == execution_row.subtask_slot &&
           h.completion_mode == execution_row.completion_mode && h.use_begin == uses.use_begin &&
           h.use_count == uses.use_count && h.provenance_capacity == execution_row.provenance_capacity &&
           h.provenance_offset == provenance_offset &&
           (h.completion_mode == CoreAsyncMode::SYNCHRONOUS ? h.task_token == TaskId::to_uint64(TaskId::invalid()) :
                                                              h.task_token != TaskId::to_uint64(TaskId::invalid()));
}

// Call only after the original writer's explicit writeback, or after acquiring
// a published CHECK lease. No cache operation touches atomic control rows.
inline __aicore__ bool scheduler_submission_context_snapshot(
    const SchedulerSubmissionProvenanceView &view, uint64_t execution_index, AsyncSubmissionContextHeader *header,
    AsyncSubmissionEvidence *evidence
) {
    using namespace scheduler_submission_detail;
    if (evidence == nullptr || !scheduler_submission_context_header(view, execution_index, header)) return false;
    SchedulerSubmissionExecution execution_row{};
    if (!execution(view.submissions, execution_index, &execution_row)) return false;
    load_into(&scheduler_submission_canonical_context(view, execution_index)->evidence, evidence);
    const auto &e = *evidence;
    if (e.version != ASYNC_SUBMISSION_VERSION || e.run_epoch != header->run_epoch ||
        e.task_ref != execution_row.task_ref || e.execution_index != execution_index ||
        e.dispatch_generation != header->dispatch_generation || e.block_idx != header->block_idx ||
        e.subtask_slot != header->subtask_slot || (e.flags & ~ASYNC_SUBMISSION_FLAGS) != 0)
        return false;
    for (uint64_t reserved : e.reserved)
        if (reserved != 0) return false;
    return true;
}

// Constructed from the actual resident slot/frame before calling the sealing
// policy. It is local authority, never recovered from a writable context row.
struct SchedulerSubmissionSealIdentity {
    uint64_t run_epoch{0};
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint64_t execution_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t dispatch_generation{0};
    uint64_t frame_address{0};
    uint64_t task_token{0};
    uint32_t block_idx{0};
    uint32_t subtask_slot{0};
    uint32_t frame_sub_block_id{0};
};

namespace scheduler_submission_provenance_detail {
enum class Phase : uint8_t { SEAL_INPUT, SEALED_CHECK };

inline __aicore__ bool record(
    const SchedulerSubmissionProvenanceView &view, uint64_t execution_index, uint32_t condition,
    const DeferredCompletionEntry &entry, Phase phase, SchedulerSubmissionProvenance *out,
    SchedulerSubmissionWorkspace *workspace
) {
    using namespace scheduler_submission_detail;
    SchedulerLocalStorage<AsyncSubmissionContextHeader> header_storage;
    auto &header = header_storage.get();
    SchedulerLocalStorage<AsyncSubmissionEvidence> evidence_storage;
    auto &evidence = evidence_storage.get();
    SchedulerSubmissionExecution execution_row{};
    if (out == nullptr || workspace == nullptr || !execution(view.submissions, execution_index, &execution_row) ||
        condition >= execution_row.provenance_capacity ||
        !scheduler_submission_context_snapshot(view, execution_index, &header, &evidence) ||
        header.completion_mode != CoreAsyncMode::DEFERRED || evidence.error_code != 0 || evidence.committed == 0 ||
        evidence.attempted != evidence.committed ||
        evidence.flags != (phase == Phase::SEAL_INPUT ? ASYNC_SUBMISSION_METADATA_PUBLISHED : ASYNC_SUBMISSION_FLAGS) ||
        entry.engine != COMPLETION_ENGINE_SDMA || entry.completion_type != COMPLETION_TYPE_SDMA_EVENT_RECORD ||
        entry.expected_value != 0 || entry.backend_cookie == 0 || entry._pad != 0 || entry.addr % 8 != 0)
        return false;
    *out = load(
        at<SchedulerSubmissionProvenance>(
            view.submissions, view.submissions.header.provenance_offset, execution_row.provenance_begin + condition
        )
    );
    if (out->version != 1 || out->completion_type != entry.completion_type || out->address != entry.addr ||
        out->backend_cookie != entry.backend_cookie || out->execution_index != execution_index || out->reserved != 0 ||
        out->use_index < header.use_begin || out->use_index - header.use_begin >= header.use_count ||
        !scheduler_submission_detail::workspace(view.submissions, out->workspace_index, workspace))
        return false;
    // The backing owns the entire line that observation may invalidate. This
    // extent check alone grants no record: the exact public fact above is required.
    const uint64_t line = entry.addr & ~UINT64_C(63);
    if (line < workspace->address || line - workspace->address > workspace->bytes ||
        UINT64_C(64) > workspace->bytes - (line - workspace->address))
        return false;
    // One execution has at most 16 grants. CHECK reads the immutable assignment
    // after release; only the original seal-input path demands active ownership.
    for (uint64_t ordinal = 0; ordinal < execution_row.grant_count; ++ordinal) {
        SchedulerSubmissionGrant grant{};
        SchedulerSubmissionWorkspace selected{};
        uint64_t index = 0;
        if (!scheduler_submission_lookup_grant(
                view.submissions, execution_index, ordinal, &grant, &selected, &index, phase == Phase::SEAL_INPUT
            ))
            return false;
        if (grant.workspace_index == out->workspace_index && grant.group_index == out->channel_group) return true;
    }
    return false;
}
}  // namespace scheduler_submission_provenance_detail

inline __aicore__ bool scheduler_submission_read_seal_provenance(
    const SchedulerSubmissionProvenanceView &view, const SchedulerSubmissionSealIdentity &identity, uint32_t condition,
    const DeferredCompletionEntry &entry, SchedulerSubmissionProvenance *out, SchedulerSubmissionWorkspace *workspace
) {
    SchedulerLocalStorage<AsyncSubmissionContextHeader> header_storage;
    auto &header = header_storage.get();
    SchedulerLocalStorage<AsyncSubmissionEvidence> evidence_storage;
    auto &evidence = evidence_storage.get();
    if (!scheduler_submission_context_snapshot(view, identity.execution_index, &header, &evidence) ||
        identity.run_epoch != header.run_epoch || identity.task_ref != evidence.task_ref ||
        identity.dispatch_generation != header.dispatch_generation || identity.frame_address != header.frame_address ||
        identity.task_token != header.task_token || identity.block_idx != header.block_idx ||
        identity.subtask_slot != header.subtask_slot || identity.frame_sub_block_id != header.frame_sub_block_id)
        return false;
    return scheduler_submission_provenance_detail::record(
        view, identity.execution_index, condition, entry, scheduler_submission_provenance_detail::Phase::SEAL_INPUT,
        out, workspace
    );
}

inline __aicore__ bool scheduler_submission_read_sealed_provenance(
    const SchedulerSubmissionProvenanceView &view, const SchedulerAsyncGroupIdentity &identity, uint32_t condition,
    const DeferredCompletionEntry &entry, uint64_t binding_use_index, SchedulerSubmissionProvenance *out,
    SchedulerSubmissionWorkspace *workspace
) {
    using namespace scheduler_submission_detail;
    uint64_t execution_index = SCHEDULER_ASYNC_INDEX_NONE;
    SchedulerSubmissionExecution execution_row{};
    if (identity.run_epoch != view.submissions.header.run_epoch ||
        !scheduler_submission_execution_index(
            view.submissions, identity.task_ref, identity.block_idx, identity.subtask_slot, &execution_index
        ) ||
        !execution(view.submissions, execution_index, &execution_row) ||
        execution_row.condition_capacity != identity.capacity ||
        execution_row.condition_types != identity.condition_types ||
        !scheduler_submission_provenance_detail::record(
            view, execution_index, condition, entry, scheduler_submission_provenance_detail::Phase::SEALED_CHECK, out,
            workspace
        ))
        return false;
    return out->use_index == binding_use_index;
}
