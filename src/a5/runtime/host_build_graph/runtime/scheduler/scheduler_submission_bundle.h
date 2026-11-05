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
#include "scheduler_async_runtime.h"
#include "scheduler_async_execution.h"
#include "async_submission_context.h"

struct SchedulerSubmissionView {
    __gm__ uint8_t *base{nullptr};
    SchedulerSubmissionHeader header{};
    inline __aicore__ bool enabled() const { return base != nullptr; }
};

namespace scheduler_submission_detail {
using scheduler_async_runtime_detail::load;
using scheduler_async_runtime_detail::publish;
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
    for (uint64_t reserved : out->reserved)
        if (reserved != 0) return false;
    const uint64_t expected = map.execution_base + out->block_idx * __builtin_popcount(map.lane_mask) +
                              __builtin_popcount(map.lane_mask & ((1u << out->subtask_slot) - 1));
    return index == expected &&
           out->bundle_index == map.bundle_base + (scheduler_task_requires_sync_start(map.flags) ? 0 : out->block_idx);
}
inline __aicore__ bool
bundle(const SchedulerSubmissionView &view, uint64_t index, SchedulerSubmissionBundleIdentity *out) {
    if (index >= view.header.bundle_count || out == nullptr) return false;
    *out = load(at<SchedulerSubmissionBundleIdentity>(view, view.header.bundle_identities_offset, index));
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
inline __aicore__ SchedulerSubmissionResult error(__gm__ SchedulerRunControl *run, int64_t task) {
    scheduler_record_error(run, task, SchedulerGraphResult::INVALID_ARGUMENTS);
    return SchedulerSubmissionResult::ERROR;
}
struct Atomics {
    inline __aicore__ uint64_t observe(__gm__ volatile uint64_t &word) {
        return scheduler_gm_fetch_add(word, UINT64_C(0));
    }
    inline __aicore__ uint64_t claim(__gm__ volatile uint64_t &word, uint64_t expected, uint64_t desired) {
        return scheduler_gm_compare_exchange(word, expected, desired);
    }
};
}  // namespace scheduler_submission_detail

inline __aicore__ bool scheduler_submission_mount(
    __gm__ uint8_t *base, uint64_t bytes, uint64_t epoch, uint64_t tasks, SchedulerSubmissionView *out
) {
    using namespace scheduler_submission_detail;
    if (out == nullptr || epoch == 0) return false;
    if (bytes == 0) {
        *out = {};
        return true;
    }
    if (base == nullptr || reinterpret_cast<uint64_t>(base) % 128 != 0 || bytes < sizeof(SchedulerSubmissionHeader) ||
        bytes > UINT64_MAX - reinterpret_cast<uint64_t>(base))
        return false;
    SchedulerSubmissionView view{};
    view.base = base;
    view.header = load(reinterpret_cast<__gm__ const SchedulerSubmissionHeader *>(base));
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
        !region(bytes, h.execution_grant_indices_offset, h.grant_count, sizeof(uint64_t), &end))
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

// This adapter reads the real executor's local preparation/generation and the
// existing shared CPU cohort. Ordinary READY is only a provisional CPU fact;
// its original ordered-admission check follows the fixed try-all transaction.
template <typename Backend>
struct SchedulerSubmissionCpuGate {
    const SchedulerAsyncExecutionState<Backend> *state{nullptr};
    const SchedulerLocalState *local{nullptr};
    uint32_t slot_index{UINT32_MAX};

    inline __aicore__ SchedulerSubmissionResult
    observe_once(const SchedulerSubmissionExecution &execution, const SchedulerSubmissionBundleIdentity &bundle) const {
        using Result = SchedulerSubmissionResult;
        if (state == nullptr || local == nullptr || slot_index >= SCHEDULER_PENDING_SLOT_COUNT ||
            state->view.layout.run_epoch != bundle.run_epoch || state->permitted_slot != UINT32_MAX)
            return Result::ERROR;
        const auto &slot = local->slots[slot_index];
        if (slot.state != SchedulerDispatchSlotState::READY || slot.generation == 0 ||
            slot.task_ref != execution.task_ref || slot.block_idx != execution.block_idx ||
            slot.subtask_slot != execution.subtask_slot)
            return Result::ERROR;
        if (state->prepared_generations[slot_index] == 0) return Result::PENDING;
        if (state->prepared_generations[slot_index] != slot.generation ||
            state->prepared_write_serials[slot_index] != state->core_write_serial)
            return Result::ERROR;
        if (bundle.kind == SchedulerSubmissionBundleKind::EXECUTION) return Result::READY;
        if (state->pinned_slot == UINT32_MAX) return Result::PENDING;
        if (state->pinned_slot != slot_index || state->pinned_generation != slot.generation || !slot.cooperative ||
            local->admission == nullptr || local->admission->current_task_ref != slot.task_ref ||
            local->admission->current_block_num != bundle.block_count || state->cohorts.base == nullptr ||
            state->cohorts.header.run_epoch != bundle.run_epoch || slot.task_ref < 0 ||
            static_cast<uint64_t>(slot.task_ref) >= state->cohorts.header.task_count)
            return Result::ERROR;
        if (slot.gm_mix && (local->admission->gm_mix.phase != SchedulerGmMixPhase::RUNNING ||
                            local->admission->gm_mix.slot_index != slot_index ||
                            local->admission->gm_mix.slot_generation != slot.generation ||
                            local->admission->gm_mix.block_idx != slot.block_idx ||
                            local->admission->gm_mix.subtask_slot != slot.subtask_slot))
            return Result::ERROR;
        const auto map = scheduler_async_cohort_detail::load(
            reinterpret_cast<__gm__ const SchedulerAsyncCohortTaskMap *>(
                state->cohorts.base + state->cohorts.header.maps_offset
            ) +
            slot.task_ref
        );
        const bool sync = bundle.kind == SchedulerSubmissionBundleKind::SYNC_TASK;
        if (map.flags != local->admission->current_flags || map.block_count != bundle.block_count ||
            map.lane_mask != bundle.lane_mask || scheduler_task_requires_sync_start(map.flags) != sync ||
            map.expected != bundle.execution_count || map.cohort_count != (sync ? 1u : map.block_count) ||
            map.cohort_base >= state->cohorts.header.cohort_count ||
            map.cohort_count > state->cohorts.header.cohort_count - map.cohort_base)
            return Result::ERROR;
        const uint64_t index = map.cohort_base + (sync ? 0 : execution.block_idx);
        auto &counter = (reinterpret_cast<__gm__ SchedulerAsyncCohortCounter *>(
                             state->cohorts.base + state->cohorts.header.counters_offset
                         ) +
                         index)
                            ->ready;
        const uint64_t count = scheduler_gm_fetch_add(counter, UINT64_C(0));
        if (count > map.expected) return Result::ERROR;
        return count == map.expected ? Result::READY : Result::PENDING;
    }
};

// CpuGate is mandatory: observe_once(execution,bundle) checks the actual pinned
// CPU reservation and performs at most one authoritative cohort-ready probe.
// A READY slot alone is not sufficient ordinary ordered-admission authority.
template <typename CpuGate, typename Atomics = scheduler_submission_detail::Atomics>
inline __aicore__ SchedulerSubmissionResult scheduler_submission_acquire(
    const SchedulerSubmissionView &view, uint64_t execution_index, __gm__ SchedulerRunControl *run, CpuGate &cpu,
    SchedulerSubmissionBudget *budget, Atomics atomics = {}
) {
    using namespace scheduler_submission_detail;
    using Result = SchedulerSubmissionResult;
    if (budget == nullptr || run == nullptr || !view.enabled()) return Result::ERROR;
    SchedulerSubmissionExecution caller{};
    SchedulerSubmissionBundleIdentity identity{};
    if (!execution(view, execution_index, &caller) || !bundle(view, caller.bundle_index, &identity))
        return error(run, SCHEDULER_TASK_REF_INVALID);
    if (identity.kind == SchedulerSubmissionBundleKind::EXECUTION) return error(run, caller.task_ref);
    if (scheduler_read_error(run) != 0) return Result::ERROR;
    if (budget->work_units == 0) return Result::PENDING;
    --budget->work_units;
    const auto cpu_result = cpu.observe_once(caller, identity);
    if (cpu_result != Result::READY)
        return cpu_result == Result::PENDING ? Result::PENDING : error(run, caller.task_ref);
    if (budget->lease_probes == 0) return Result::PENDING;
    --budget->lease_probes;
    auto &lease = *at<SchedulerAsyncLease>(view, view.header.bundle_leases_offset, caller.bundle_index);
    const uint64_t observed = scheduler_async_lease_load(lease);
    if (!scheduler_async_lease_valid(observed)) return error(run, caller.task_ref);
    const auto phase = scheduler_async_lease_phase(observed);
    if (phase == SchedulerAsyncPhase::TERMINAL) {
        const auto saved =
            load(at<SchedulerSubmissionProgress>(view, view.header.bundle_progress_offset, caller.bundle_index));
        return saved.phase == SchedulerSubmissionProgressPhase::GRANTED &&
                       saved.request_index == identity.request_count && saved.selected_mask == 0 &&
                       saved.selected_count == 0 && saved.snapshot == 0 && scheduler_read_error(run) == 0 ?
                   Result::READY :
                   error(run, caller.task_ref);
    }
    if (phase == SchedulerAsyncPhase::FAILED) return Result::ERROR;
    if (phase != SchedulerAsyncPhase::RUNNABLE || budget->claim_attempts == 0 || budget->work_units == 0)
        return Result::PENDING;
    --budget->claim_attempts;
    const auto claimed = scheduler_async_lease_try_claim(lease, observed);
    if (claimed.result == SchedulerAsyncLeaseResult::CONTENDED) return Result::PENDING;
    if (claimed.result != SchedulerAsyncLeaseResult::SUCCESS) return error(run, caller.task_ref);
    scheduler_cache_barrier();
    auto progress =
        load(at<SchedulerSubmissionProgress>(view, view.header.bundle_progress_offset, caller.bundle_index));
    const uint32_t available = budget->work_units < 8 ? budget->work_units : 8;
    uint32_t left = available;
    bool valid = progress.request_index <= identity.request_count && progress.failure_code == 0 &&
                 (progress.phase == SchedulerSubmissionProgressPhase::OBSERVE ||
                  progress.phase == SchedulerSubmissionProgressPhase::SELECT ||
                  progress.phase == SchedulerSubmissionProgressPhase::CLAIM);
    for (uint64_t reserved : progress.reserved)
        if (reserved != 0) valid = false;
    if (valid && progress.request_index != 0) {
        SchedulerSubmissionRequest previous{};
        SchedulerSubmissionWorkspace previous_workspace{};
        valid = request(view, identity, progress.request_index - 1, &previous, &previous_workspace) &&
                previous.grant_count == static_cast<uint64_t>(__builtin_popcountll(previous.acquired_mask)) &&
                (previous.acquired_mask >> (previous_workspace.physical_channels / previous_workspace.queue_num)) == 0;
    }
    while (valid && left != 0 && progress.phase != SchedulerSubmissionProgressPhase::GRANTED) {
        if (scheduler_read_error(run) != 0) {
            valid = false;
            break;
        }
        if (progress.request_index == identity.request_count) {
            if (progress.phase != SchedulerSubmissionProgressPhase::OBSERVE || progress.selected_count != 0 ||
                progress.selected_mask != 0) {
                valid = false;
                break;
            }
            --left;
            progress.phase = SchedulerSubmissionProgressPhase::GRANTED;
            break;
        }
        SchedulerSubmissionRequest row{};
        SchedulerSubmissionWorkspace workspace_row{};
        if (!request(view, identity, progress.request_index, &row, &workspace_row) || row.acquired_mask != 0) {
            valid = false;
            break;
        }
        const uint32_t groups = workspace_row.physical_channels / workspace_row.queue_num;
        const uint64_t allowed = (UINT64_C(1) << groups) - 1;
        auto &pool = at<SchedulerSubmissionPool>(view, view.header.pools_offset, row.workspace_index)->busy_groups;
        --left;
        if (progress.phase == SchedulerSubmissionProgressPhase::OBSERVE) {
            if (progress.selected_count != 0 || progress.selected_mask != 0) {
                valid = false;
                break;
            }
            progress.snapshot = atomics.observe(pool);
            if ((progress.snapshot & ~allowed) != 0) {
                valid = false;
                break;
            }
            if (static_cast<uint64_t>(__builtin_popcountll(allowed & ~progress.snapshot)) < row.grant_count) break;
            progress.phase = SchedulerSubmissionProgressPhase::SELECT;
        } else if (progress.phase == SchedulerSubmissionProgressPhase::SELECT) {
            if ((progress.snapshot & ~allowed) != 0 || (progress.selected_mask & ~allowed) != 0 ||
                (progress.selected_mask & progress.snapshot) != 0 ||
                progress.selected_count != static_cast<uint64_t>(__builtin_popcountll(progress.selected_mask)) ||
                progress.selected_count >= row.grant_count) {
                valid = false;
                break;
            }
            const uint64_t free = allowed & ~(progress.snapshot | progress.selected_mask);
            if (free == 0) {
                valid = false;
                break;
            }
            const uint32_t group = __builtin_ctzll(free);
            auto grant = load(
                at<SchedulerSubmissionGrant>(view, view.header.grants_offset, row.grant_begin + progress.selected_count)
            );
            if (grant.workspace_index != row.workspace_index || grant.reserved != 0 ||
                grant.execution_index < identity.execution_begin ||
                grant.execution_index - identity.execution_begin >= identity.execution_count) {
                valid = false;
                break;
            }
            grant.group_index = group;
            publish(
                at<SchedulerSubmissionGrant>(
                    view, view.header.grants_offset, row.grant_begin + progress.selected_count
                ),
                grant
            );
            progress.selected_mask |= UINT64_C(1) << group;
            ++progress.selected_count;
            if (progress.selected_count == row.grant_count) progress.phase = SchedulerSubmissionProgressPhase::CLAIM;
        } else if (progress.phase == SchedulerSubmissionProgressPhase::CLAIM) {
            if ((progress.selected_mask & progress.snapshot) != 0 || (progress.selected_mask & ~allowed) != 0 ||
                progress.selected_count != row.grant_count ||
                progress.selected_count != static_cast<uint64_t>(__builtin_popcountll(progress.selected_mask))) {
                valid = false;
                break;
            }
            const uint64_t previous =
                atomics.claim(pool, progress.snapshot, progress.snapshot | progress.selected_mask);
            const bool success = previous == progress.snapshot;
            if (success) {
                // Reservation has no yielding publication gap: the complete
                // grant rows exist and only this fixed checkpoint tail remains.
                row.acquired_mask = progress.selected_mask;
                publish(
                    at<SchedulerSubmissionRequest>(
                        view, view.header.requests_offset, identity.request_begin + progress.request_index
                    ),
                    row
                );
                ++progress.request_index;
            }
            progress.snapshot = progress.selected_mask = progress.selected_count = 0;
            progress.phase = SchedulerSubmissionProgressPhase::OBSERVE;
            if (!success) break;
        } else {
            valid = false;
        }
    }
    const uint32_t used = available - left;
    budget->work_units -= used;
    progress.work_units += used;
    if (scheduler_read_error(run) != 0) valid = false;
    if (!valid) {
        progress.phase = SchedulerSubmissionProgressPhase::FAILED;
        progress.failure_code = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
    }
    publish(at<SchedulerSubmissionProgress>(view, view.header.bundle_progress_offset, caller.bundle_index), progress);
    const auto next = !valid                                                      ? SchedulerAsyncPhase::FAILED :
                      progress.phase == SchedulerSubmissionProgressPhase::GRANTED ? SchedulerAsyncPhase::TERMINAL :
                                                                                    SchedulerAsyncPhase::RUNNABLE;
    if (scheduler_async_lease_handoff(lease, claimed.word, next).result != SchedulerAsyncLeaseResult::SUCCESS)
        return error(run, caller.task_ref);
    if (!valid) return error(run, caller.task_ref);
    return next == SchedulerAsyncPhase::TERMINAL ? Result::READY : Result::PENDING;
}

// Ordinary work never retains a channel across a scheduler iteration. This
// tail also runs after an ordered-admission denial, before any other service.
inline __aicore__ bool scheduler_submission_abort_provisional(
    const SchedulerSubmissionView &view, SchedulerSubmissionProvisional *transaction, __gm__ SchedulerRunControl *run,
    SchedulerSubmissionTransactionStats *stats = nullptr
) {
    using namespace scheduler_submission_detail;
    if (transaction == nullptr || !transaction->active ||
        transaction->count > SCHEDULER_SUBMISSION_MAX_EXECUTION_GRANTS ||
        transaction->bundle_index >= view.header.bundle_count)
        return false;
    SchedulerSubmissionExecution execution_row{};
    if (!execution(view, transaction->execution_index, &execution_row) ||
        execution_row.bundle_index != transaction->bundle_index)
        return error(run, SCHEDULER_TASK_REF_INVALID), false;
    auto &lease = *at<SchedulerAsyncLease>(view, view.header.bundle_leases_offset, transaction->bundle_index);
    if (stats != nullptr) ++stats->lease_probes;
    if (scheduler_async_lease_load(lease) != transaction->owned_word) return error(run, execution_row.task_ref), false;
    for (uint64_t i = 0; i < transaction->count; ++i) {
        if (transaction->request_indices[i] >= view.header.request_count ||
            transaction->workspace_indices[i] >= view.header.workspace_count ||
            __builtin_popcountll(transaction->masks[i]) != 1)
            return error(run, execution_row.task_ref), false;
        const auto row =
            load(at<SchedulerSubmissionRequest>(view, view.header.requests_offset, transaction->request_indices[i]));
        if (row.workspace_index != transaction->workspace_indices[i] || row.acquired_mask != transaction->masks[i] ||
            row.grant_count != 1)
            return error(run, execution_row.task_ref), false;
    }
    for (uint64_t i = 0; i < transaction->count; ++i) {
        auto &pool =
            at<SchedulerSubmissionPool>(view, view.header.pools_offset, transaction->workspace_indices[i])->busy_groups;
        const uint64_t previous = scheduler_gm_fetch_add(pool, UINT64_C(0) - transaction->masks[i]);
        if (stats != nullptr) ++stats->pool_returns;
        if ((previous & transaction->masks[i]) != transaction->masks[i])
            return error(run, execution_row.task_ref), false;
        auto row =
            load(at<SchedulerSubmissionRequest>(view, view.header.requests_offset, transaction->request_indices[i]));
        row.acquired_mask = 0;
        publish(
            at<SchedulerSubmissionRequest>(view, view.header.requests_offset, transaction->request_indices[i]), row
        );
        auto grant = load(at<SchedulerSubmissionGrant>(view, view.header.grants_offset, row.grant_begin));
        grant.group_index = SCHEDULER_ASYNC_INDEX_NONE;
        publish(at<SchedulerSubmissionGrant>(view, view.header.grants_offset, row.grant_begin), grant);
    }
    publish(
        at<SchedulerSubmissionProgress>(view, view.header.bundle_progress_offset, transaction->bundle_index),
        SchedulerSubmissionProgress{}
    );
    if (stats != nullptr) ++stats->lease_cas;
    if (scheduler_async_lease_handoff(lease, transaction->owned_word, SchedulerAsyncPhase::RUNNABLE).result !=
        SchedulerAsyncLeaseResult::SUCCESS)
        return error(run, execution_row.task_ref), false;
    transaction->active = false;
    return true;
}

// A RESERVED result is provisional, never kernel permission. The caller must
// immediately execute the original ordinary admission check and then commit or
// abort this transaction. At most 16 workspaces each cost one probe and one CAS;
// contention returns every acquired prefix in the fixed <=16-return tail.
template <typename CpuGate, typename Atomics = scheduler_submission_detail::Atomics>
inline __aicore__ SchedulerSubmissionProvisionalResult scheduler_submission_try_ordinary(
    const SchedulerSubmissionView &view, uint64_t execution_index, __gm__ SchedulerRunControl *run, CpuGate &cpu,
    SchedulerSubmissionProvisional *transaction, SchedulerSubmissionTransactionStats *stats = nullptr,
    Atomics atomics = {}
) {
    using namespace scheduler_submission_detail;
    using Result = SchedulerSubmissionProvisionalResult;
    if (transaction == nullptr || transaction->active || run == nullptr || !view.enabled()) return Result::ERROR;
    SchedulerSubmissionExecution caller{};
    SchedulerSubmissionBundleIdentity identity{};
    if (!execution(view, execution_index, &caller) || !bundle(view, caller.bundle_index, &identity) ||
        identity.kind != SchedulerSubmissionBundleKind::EXECUTION || identity.execution_count != 1 ||
        identity.execution_begin != execution_index || identity.request_count != caller.grant_count ||
        identity.request_count > SCHEDULER_SUBMISSION_MAX_EXECUTION_GRANTS)
        return error(run, SCHEDULER_TASK_REF_INVALID), Result::ERROR;
    if (scheduler_read_error(run) != 0) return Result::ERROR;
    if (stats != nullptr) ++stats->cpu_checks;
    const auto cpu_result = cpu.observe_once(caller, identity);
    if (cpu_result != SchedulerSubmissionResult::READY)
        return cpu_result == SchedulerSubmissionResult::PENDING ? Result::PENDING :
                                                                  (error(run, caller.task_ref), Result::ERROR);
    auto &lease = *at<SchedulerAsyncLease>(view, view.header.bundle_leases_offset, caller.bundle_index);
    if (stats != nullptr) ++stats->lease_probes;
    const uint64_t observed = scheduler_async_lease_load(lease);
    if (!scheduler_async_lease_valid(observed)) return error(run, caller.task_ref), Result::ERROR;
    if (scheduler_async_lease_phase(observed) != SchedulerAsyncPhase::RUNNABLE) return Result::PENDING;
    if (stats != nullptr) ++stats->lease_cas;
    const auto claimed = scheduler_async_lease_try_claim(lease, observed);
    if (claimed.result == SchedulerAsyncLeaseResult::CONTENDED) return Result::PENDING;
    if (claimed.result != SchedulerAsyncLeaseResult::SUCCESS) return error(run, caller.task_ref), Result::ERROR;
    scheduler_cache_barrier();
    *transaction = {};
    transaction->active = true;
    transaction->bundle_index = caller.bundle_index;
    transaction->execution_index = execution_index;
    transaction->owned_word = claimed.word;
    const auto progress =
        load(at<SchedulerSubmissionProgress>(view, view.header.bundle_progress_offset, caller.bundle_index));
    if (progress.phase != SchedulerSubmissionProgressPhase::OBSERVE || progress.request_index != 0 ||
        progress.selected_mask != 0 || progress.selected_count != 0 || progress.failure_code != 0)
        return error(run, caller.task_ref), Result::ERROR;
    for (uint64_t ordinal = 0; ordinal < identity.request_count; ++ordinal) {
        SchedulerSubmissionRequest row{};
        SchedulerSubmissionWorkspace workspace_row{};
        if (!request(view, identity, ordinal, &row, &workspace_row) || row.grant_count != 1 || row.acquired_mask != 0)
            return error(run, caller.task_ref), Result::ERROR;
        auto grant = load(at<SchedulerSubmissionGrant>(view, view.header.grants_offset, row.grant_begin));
        if (grant.execution_index != execution_index || grant.workspace_index != row.workspace_index ||
            grant.reserved != 0)
            return error(run, caller.task_ref), Result::ERROR;
        auto &pool = at<SchedulerSubmissionPool>(view, view.header.pools_offset, row.workspace_index)->busy_groups;
        if (stats != nullptr) ++stats->pool_probes;
        const uint64_t occupied = atomics.observe(pool);
        const uint32_t groups = workspace_row.physical_channels / workspace_row.queue_num;
        const uint64_t allowed = (UINT64_C(1) << groups) - 1;
        if ((occupied & ~allowed) != 0) return error(run, caller.task_ref), Result::ERROR;
        const uint64_t available = allowed & ~occupied;
        if (available == 0) {
            if (!scheduler_submission_abort_provisional(view, transaction, run, stats)) return Result::ERROR;
            return Result::PENDING;
        }
        const uint64_t mask = available & (UINT64_C(0) - available);
        if (stats != nullptr) ++stats->pool_cas;
        if (atomics.claim(pool, occupied, occupied | mask) != occupied) {
            if (!scheduler_submission_abort_provisional(view, transaction, run, stats)) return Result::ERROR;
            return Result::PENDING;
        }
        // No scheduling return can separate the reservation from its local and
        // GM ownership checkpoint; the next operation is bounded publication.
        grant.group_index = __builtin_ctzll(mask);
        publish(at<SchedulerSubmissionGrant>(view, view.header.grants_offset, row.grant_begin), grant);
        row.acquired_mask = mask;
        publish(
            at<SchedulerSubmissionRequest>(view, view.header.requests_offset, identity.request_begin + ordinal), row
        );
        const uint64_t i = transaction->count++;
        transaction->request_indices[i] = identity.request_begin + ordinal;
        transaction->workspace_indices[i] = row.workspace_index;
        transaction->masks[i] = mask;
    }
    if (scheduler_read_error(run) != 0) {
        (void)scheduler_submission_abort_provisional(view, transaction, run, stats);
        return Result::ERROR;
    }
    return Result::RESERVED;
}

inline __aicore__ bool scheduler_submission_commit_provisional(
    const SchedulerSubmissionView &view, SchedulerSubmissionProvisional *transaction, __gm__ SchedulerRunControl *run,
    SchedulerSubmissionTransactionStats *stats = nullptr
) {
    using namespace scheduler_submission_detail;
    if (transaction == nullptr || !transaction->active || run == nullptr) return false;
    if (scheduler_read_error(run) != 0) {
        (void)scheduler_submission_abort_provisional(view, transaction, run, stats);
        return false;
    }
    SchedulerSubmissionExecution caller{};
    SchedulerSubmissionBundleIdentity identity{};
    if (!execution(view, transaction->execution_index, &caller) ||
        !bundle(view, transaction->bundle_index, &identity) || caller.bundle_index != transaction->bundle_index ||
        identity.kind != SchedulerSubmissionBundleKind::EXECUTION || transaction->count != identity.request_count)
        return error(run, SCHEDULER_TASK_REF_INVALID), false;
    auto &lease = *at<SchedulerAsyncLease>(view, view.header.bundle_leases_offset, transaction->bundle_index);
    if (stats != nullptr) ++stats->lease_probes;
    if (scheduler_async_lease_load(lease) != transaction->owned_word) return error(run, caller.task_ref), false;
    SchedulerSubmissionProgress progress{};
    progress.phase = SchedulerSubmissionProgressPhase::GRANTED;
    progress.request_index = identity.request_count;
    publish(
        at<SchedulerSubmissionProgress>(view, view.header.bundle_progress_offset, transaction->bundle_index), progress
    );
    if (stats != nullptr) ++stats->lease_cas;
    if (scheduler_async_lease_handoff(lease, transaction->owned_word, SchedulerAsyncPhase::TERMINAL).result !=
        SchedulerAsyncLeaseResult::SUCCESS)
        return error(run, caller.task_ref), false;
    transaction->active = false;
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
    const auto progress =
        load(at<SchedulerSubmissionProgress>(view, view.header.bundle_progress_offset, execution_row.bundle_index));
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

// Evidence is the original writer's GM record, not caller-supplied booleans.
// SDK publication and runtime seal have completed before this fixed <=16-grant
// tail. Each RESERVED->RELEASING winner performs its matching pool subtraction
// and RELEASED publication without yielding. Failed runs retain remaining bits.
inline __aicore__ bool scheduler_submission_release_all(
    const SchedulerSubmissionView &view, const SchedulerSubmissionExecutionIdentity &identity,
    __gm__ const AsyncSubmissionEvidence *evidence_source, __gm__ SchedulerRunControl *run
) {
    using namespace scheduler_submission_detail;
    if (evidence_source == nullptr || run == nullptr || scheduler_read_error(run) != 0) return false;
    const auto evidence = load(evidence_source);
    SchedulerSubmissionExecution execution_row{};
    if (!execution(view, identity.execution_index, &execution_row) || identity.run_epoch != view.header.run_epoch ||
        identity.dispatch_generation == 0 || execution_row.task_ref != identity.task_ref ||
        execution_row.block_idx != identity.block_idx || execution_row.subtask_slot != identity.subtask_slot ||
        evidence.version != ASYNC_SUBMISSION_VERSION || evidence.run_epoch != identity.run_epoch ||
        evidence.task_ref != identity.task_ref || evidence.execution_index != identity.execution_index ||
        evidence.dispatch_generation != identity.dispatch_generation || evidence.block_idx != identity.block_idx ||
        evidence.subtask_slot != identity.subtask_slot || evidence.error_code != 0 ||
        evidence.attempted != evidence.committed || evidence.flags != ASYNC_SUBMISSION_FLAGS)
        return error(run, identity.task_ref), false;
    for (uint64_t reserved : evidence.reserved)
        if (reserved != 0) return error(run, identity.task_ref), false;
    SchedulerSubmissionGrant grants[SCHEDULER_SUBMISSION_MAX_EXECUTION_GRANTS]{};
    uint64_t indices[SCHEDULER_SUBMISSION_MAX_EXECUTION_GRANTS]{};
    for (uint64_t ordinal = 0; ordinal < execution_row.grant_count; ++ordinal) {
        SchedulerSubmissionWorkspace workspace_row{};
        if (!scheduler_submission_lookup_grant(
                view, identity.execution_index, ordinal, &grants[ordinal], &workspace_row, &indices[ordinal], false
            ) ||
            (ordinal != 0 && grants[ordinal - 1].workspace_index >= grants[ordinal].workspace_index))
            return error(run, identity.task_ref), false;
    }
    for (uint64_t ordinal = 0; ordinal < execution_row.grant_count; ++ordinal) {
        if (scheduler_read_error(run) != 0) return false;
        auto &state =
            at<SchedulerSubmissionReleaseState>(view, view.header.release_states_offset, indices[ordinal])->phase;
        const uint64_t previous = scheduler_gm_compare_exchange(state, UINT64_C(0), UINT64_C(1));
        if (previous == static_cast<uint64_t>(SchedulerSubmissionReleasePhase::RELEASED)) continue;
        if (previous != static_cast<uint64_t>(SchedulerSubmissionReleasePhase::RESERVED))
            return error(run, identity.task_ref), false;
        scheduler_cache_barrier();
        const uint64_t mask = UINT64_C(1) << grants[ordinal].group_index;
        auto &pool =
            at<SchedulerSubmissionPool>(view, view.header.pools_offset, grants[ordinal].workspace_index)->busy_groups;
        const uint64_t before = scheduler_gm_fetch_add(pool, UINT64_C(0) - mask);
        if ((before & mask) == 0) return error(run, identity.task_ref), false;
        scheduler_cache_barrier();
        scheduler_gm_store(state, static_cast<uint64_t>(SchedulerSubmissionReleasePhase::RELEASED));
    }
    return true;
}
