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

#include "scheduler_submission_provenance.h"
#include "scheduler_async_runtime.h"
#include "scheduler_async_execution.h"
#include "async_submission_context.h"

namespace scheduler_submission_detail {
using scheduler_async_runtime_detail::publish;
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
    SchedulerLocalStorage<SchedulerSubmissionBundleIdentity> identity_storage;
    auto &identity = identity_storage.get();
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
        SchedulerLocalStorage<SchedulerSubmissionProgress> saved_storage;
        load_into(
            at<SchedulerSubmissionProgress>(view, view.header.bundle_progress_offset, caller.bundle_index),
            &saved_storage.get()
        );
        const auto &saved = saved_storage.get();
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
    SchedulerLocalStorage<SchedulerSubmissionProgress> progress_storage;
    load_into(
        at<SchedulerSubmissionProgress>(view, view.header.bundle_progress_offset, caller.bundle_index),
        &progress_storage.get()
    );
    auto &progress = progress_storage.get();
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
    SchedulerLocalStorage<SchedulerSubmissionBundleIdentity> identity_storage;
    auto &identity = identity_storage.get();
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
    SchedulerLocalStorage<SchedulerSubmissionProgress> progress_storage;
    load_into(
        at<SchedulerSubmissionProgress>(view, view.header.bundle_progress_offset, caller.bundle_index),
        &progress_storage.get()
    );
    const auto &progress = progress_storage.get();
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
    SchedulerLocalStorage<SchedulerSubmissionBundleIdentity> identity_storage;
    auto &identity = identity_storage.get();
    if (!execution(view, transaction->execution_index, &caller) ||
        !bundle(view, transaction->bundle_index, &identity) || caller.bundle_index != transaction->bundle_index ||
        identity.kind != SchedulerSubmissionBundleKind::EXECUTION || transaction->count != identity.request_count)
        return error(run, SCHEDULER_TASK_REF_INVALID), false;
    auto &lease = *at<SchedulerAsyncLease>(view, view.header.bundle_leases_offset, transaction->bundle_index);
    if (stats != nullptr) ++stats->lease_probes;
    if (scheduler_async_lease_load(lease) != transaction->owned_word) return error(run, caller.task_ref), false;
    SchedulerLocalStorage<SchedulerSubmissionProgress> progress_storage;
    auto &progress = progress_storage.get();
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
    SchedulerLocalStorage<AsyncSubmissionEvidence> evidence_storage;
    load_into(evidence_source, &evidence_storage.get());
    const auto &evidence = evidence_storage.get();
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
