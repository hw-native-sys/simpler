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

#include "submission_bundle_plan.h"
#include "async_submission_context.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <stdexcept>
#include <tuple>

namespace {
bool fail(std::string *error, const char *message) {
    if (error != nullptr) *error = message;
    return false;
}

template <typename T>
void write(std::vector<std::byte> &image, uint64_t offset, const T &value) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::memcpy(image.data() + offset, &value, sizeof(T));
}
template <typename T>
void write_rows(std::vector<std::byte> &image, uint64_t offset, const std::vector<T> &rows) {
    if (!rows.empty()) std::memcpy(image.data() + offset, rows.data(), rows.size() * sizeof(T));
}
bool reserve(uint64_t count, uint64_t stride, uint64_t &cursor, uint64_t &offset) {
    uint64_t bytes = 0;
    return scheduler_layout_checked_mul(count, stride, &bytes) &&
           scheduler_layout_reserve(&cursor, bytes, 128, &offset);
}
uint32_t lane_mask(LaneMask mask) {
    uint32_t bits = 0;
    for (uint32_t lane = 0; lane < 3; ++lane)
        if (mask.has(static_cast<SubtaskSlot>(lane))) bits |= 1u << lane;
    return bits;
}

bool build(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    const std::vector<SchedulerSubmissionWorkspaceInput> &workspace_inputs,
    const std::vector<SchedulerSubmissionExecutionUse> &uses,
    const std::vector<SchedulerSubmissionExecutionUseSet> &use_sets, uint64_t epoch, uint64_t budget,
    SchedulerSubmissionHostPlan *out, std::string *error
) {
    if (out == nullptr || epoch == 0 || metadata.size() > INT32_MAX || capabilities.size() != metadata.size())
        return fail(error, "submission plan requires a live epoch and bounded task references");
    SchedulerSubmissionHostPlan next;
    auto &header = next.header;
    header.run_epoch = epoch;
    header.task_count = metadata.size();
    if (uses.empty()) {
        *out = std::move(next);
        return true;
    }
    if (budget < sizeof(header) || metadata.size() > (budget - sizeof(header)) / sizeof(SchedulerSubmissionTaskMap))
        return fail(error, "submission task map exceeds its storage budget");

    using ExecutionKey = std::tuple<int64_t, uint32_t, uint32_t>;
    using ContextKey = std::tuple<int64_t, uint32_t, uint32_t, uint32_t>;
    std::map<ExecutionKey, SchedulerSubmissionExecutionUseMap> allowed_sets;
    for (const auto &set : use_sets) {
        if (set.task_ref < 0 || static_cast<uint64_t>(set.task_ref) >= metadata.size() || set.subtask_slot >= 3 ||
            !metadata[set.task_ref].active_mask.has(static_cast<SubtaskSlot>(set.subtask_slot)) ||
            set.block_idx >= metadata[set.task_ref].logical_block_num ||
            !scheduler_task_is_executable(metadata[set.task_ref].flags) ||
            scheduler_task_is_inline(metadata[set.task_ref].flags) || set.use_count > UINT32_MAX ||
            (set.use_count == 0 ?
                 set.use_begin != SCHEDULER_ASYNC_INDEX_NONE :
                 (set.use_begin == SCHEDULER_ASYNC_INDEX_NONE || set.use_count > UINT64_MAX - set.use_begin)) ||
            !allowed_sets
                 .emplace(
                     ExecutionKey{set.task_ref, set.block_idx, set.subtask_slot},
                     SchedulerSubmissionExecutionUseMap{set.use_begin, set.use_count}
                 )
                 .second)
            return fail(error, "submission allowed-use map must uniquely name real executions and bounded ranges");
    }
    std::map<uint64_t, SchedulerSubmissionWorkspace> canonical;
    std::map<ContextKey, uint64_t> contexts;
    std::map<ExecutionKey, std::set<uint64_t>> execution_workspaces;
    std::set<uint64_t> referenced_inputs;
    std::set<int64_t> submitting_tasks;
    for (const auto &use : uses) {
        if (use.task_ref < 0 || static_cast<uint64_t>(use.task_ref) >= metadata.size() ||
            use.workspace_input >= workspace_inputs.size() || use.context_scalar >= CORE_MAX_SCALAR_ARGS ||
            use.subtask_slot == 0 || use.subtask_slot >= 3)
            return fail(error, "submission use must name a real AIV execution and scalar context");
        const auto &meta = metadata[use.task_ref];
        uint32_t physical = 0;
        const uint32_t lanes = meta.active_mask.count();
        if (!scheduler_task_is_executable(meta.flags) || scheduler_task_is_inline(meta.flags) ||
            !meta.active_mask.is_valid() || (meta.flags & UINT8_C(128)) != 0 ||
            !scheduler_subtask_count_checked(meta.logical_block_num, lanes, &physical) ||
            !meta.active_mask.has(static_cast<SubtaskSlot>(use.subtask_slot)) ||
            use.block_idx >= meta.logical_block_num || scheduler_task_is_mix(meta.flags) != (lanes > 1) ||
            scheduler_task_is_spmd(meta.flags) != (meta.logical_block_num > 1) ||
            (scheduler_task_requires_sync_start(meta.flags) && meta.logical_block_num == 1) ||
            (scheduler_task_has_predicate(meta.flags) && (meta.logical_block_num != 1 || lanes != 1)))
            return fail(error, "submission use disagrees with its final task shape");
        const auto &source = workspace_inputs[use.workspace_input];
        if (source.address == 0 || source.bytes == 0 || source.bytes > UINT64_MAX - source.address ||
            source.qualification != SchedulerSubmissionPoolQualification::EXCLUSIVE_RUN ||
            source.qualification_epoch != epoch || source.physical_channels == 0 ||
            source.physical_channels > SCHEDULER_SUBMISSION_SDMA_CHANNELS || source.queue_num == 0 ||
            source.queue_num > source.physical_channels)
            return fail(error, "submission workspace lacks supported exclusive-run or fixed-queue qualification");
        SchedulerSubmissionWorkspace workspace{};
        workspace.address = source.address;
        workspace.bytes = source.bytes;
        workspace.qualification_epoch = source.qualification_epoch;
        workspace.queue_num = source.queue_num;
        workspace.physical_channels = source.physical_channels;
        workspace.qualification = source.qualification;
        const auto [where, inserted] = canonical.emplace(source.address, workspace);
        if (!inserted && std::memcmp(&where->second, &workspace, sizeof(workspace)) != 0)
            return fail(error, "workspace aliases disagree on extent, queue configuration or authority");
        const ContextKey context{use.task_ref, use.block_idx, use.subtask_slot, use.context_scalar};
        const auto [previous, new_context] = contexts.emplace(context, source.address);
        if (!new_context && previous->second != source.address)
            return fail(error, "one scalar CommContext cannot name two SDMA workspaces");
        auto &addresses = execution_workspaces[{use.task_ref, use.block_idx, use.subtask_slot}];
        addresses.insert(source.address);
        if (addresses.size() > SCHEDULER_SUBMISSION_MAX_EXECUTION_GRANTS)
            return fail(error, "submission execution exceeds its scalar-context workspace bound");
        referenced_inputs.insert(use.workspace_input);
        submitting_tasks.insert(use.task_ref);
    }

    std::vector<SchedulerSubmissionWorkspace> workspaces;
    std::map<uint64_t, uint64_t> workspace_indices;
    uint64_t previous_end = 0;
    for (const auto &[address, workspace] : canonical) {
        if (address < previous_end)
            return fail(error, "overlapping workspace views cannot own independent channel pools");
        previous_end = address + workspace.bytes;
        workspace_indices.emplace(address, workspaces.size());
        workspaces.push_back(workspace);
    }
    next.workspace_remap.assign(workspace_inputs.size(), SCHEDULER_ASYNC_INDEX_NONE);
    for (uint64_t input : referenced_inputs)
        next.workspace_remap[input] = workspace_indices.at(workspace_inputs[input].address);
    std::vector<SchedulerSubmissionTaskMap> task_maps(metadata.size());
    std::vector<SchedulerSubmissionExecution> executions;
    std::vector<SchedulerSubmissionExecutionUseMap> execution_use_maps;
    std::vector<SchedulerSubmissionBundleIdentity> bundles;
    for (size_t task = 0; task < metadata.size(); ++task) {
        const auto &meta = metadata[task];
        auto &map = task_maps[task];
        map.block_count = meta.logical_block_num;
        map.lane_mask = lane_mask(meta.active_mask);
        map.flags = meta.flags;
        if (submitting_tasks.count(static_cast<int64_t>(task)) == 0) continue;
        const bool sync = scheduler_task_requires_sync_start(meta.flags);
        const uint64_t active = meta.active_mask.count();
        const uint64_t count = active * meta.logical_block_num;
        if (count > budget / sizeof(SchedulerSubmissionExecution) ||
            executions.size() > budget / sizeof(SchedulerSubmissionExecution) - count)
            return fail(error, "submission physical execution rows exceed their storage budget");
        map.execution_base = executions.size();
        map.bundle_base = bundles.size();
        map.bundle_count = sync ? 1 : map.block_count;
        for (uint32_t index = 0; index < map.bundle_count; ++index) {
            SchedulerSubmissionBundleIdentity bundle{};
            bundle.run_epoch = epoch;
            bundle.task_ref = static_cast<int64_t>(task);
            bundle.execution_begin = map.execution_base + (sync ? 0 : index * active);
            bundle.execution_count = sync ? count : active;
            bundle.block_scope = sync ? UINT64_MAX : index;
            bundle.kind = sync       ? SchedulerSubmissionBundleKind::SYNC_TASK :
                          active > 1 ? SchedulerSubmissionBundleKind::MIX_BLOCK :
                                       SchedulerSubmissionBundleKind::EXECUTION;
            bundle.lane_mask = map.lane_mask;
            bundle.block_count = map.block_count;
            bundles.push_back(bundle);
        }
        for (uint32_t block = 0; block < map.block_count; ++block)
            for (uint32_t lane = 0; lane < 3; ++lane) {
                if ((map.lane_mask & (1u << lane)) == 0) continue;
                SchedulerSubmissionExecution execution{};
                execution.task_ref = task;
                execution.bundle_index = map.bundle_base + (sync ? 0 : block);
                execution.block_idx = block;
                execution.subtask_slot = lane;
                const auto capability = core_async_effective_capability(capabilities[task][lane]);
                if (!core_async_capability_valid(capability))
                    return fail(error, "submission execution requires a valid registered callable capability");
                execution.completion_mode = capability.mode;
                execution.condition_capacity = capability.max_conditions;
                execution.condition_types = capability.condition_types;
                const auto allowed = allowed_sets.find({static_cast<int64_t>(task), block, lane});
                if (allowed == allowed_sets.end())
                    return fail(error, "submission execution is missing its immutable allowed-use map");
                execution_use_maps.push_back(allowed->second);
                executions.push_back(execution);
            }
    }

    std::vector<SchedulerSubmissionRequest> requests;
    std::vector<SchedulerSubmissionGrant> grants;
    std::vector<std::vector<uint64_t>> execution_grants(executions.size());
    for (auto &bundle : bundles) {
        std::map<uint64_t, std::vector<uint64_t>> demands;
        for (uint64_t execution = bundle.execution_begin; execution < bundle.execution_begin + bundle.execution_count;
             ++execution) {
            const auto &row = executions[execution];
            const auto found = execution_workspaces.find({row.task_ref, row.block_idx, row.subtask_slot});
            if (found == execution_workspaces.end()) continue;
            for (uint64_t address : found->second)
                demands[workspace_indices.at(address)].push_back(execution);
        }
        bundle.request_begin = requests.size();
        bundle.request_count = demands.size();
        for (const auto &[workspace_index, owners] : demands) {
            const auto &workspace = workspaces[workspace_index];
            if (owners.size() > workspace.physical_channels / workspace.queue_num)
                return fail(error, "whole SDMA cohort exceeds the workspace's physical channel capacity");
            SchedulerSubmissionRequest request{};
            request.workspace_index = workspace_index;
            request.grant_begin = grants.size();
            request.grant_count = owners.size();
            for (uint64_t execution : owners) {
                execution_grants[execution].push_back(grants.size());
                grants.push_back({execution, workspace_index, SCHEDULER_ASYNC_INDEX_NONE, 0});
            }
            requests.push_back(request);
        }
    }
    std::vector<uint64_t> grant_indices;
    for (size_t i = 0; i < executions.size(); ++i) {
        executions[i].grant_index_begin = grant_indices.size();
        executions[i].grant_count = execution_grants[i].size();
        auto &execution = executions[i];
        execution.provenance_begin = header.provenance_count;
        if (execution.grant_count != 0 && execution.completion_mode == CoreAsyncMode::DEFERRED) {
            if ((execution.condition_types & CORE_ASYNC_CONDITION_SDMA_POST_DONE) == 0)
                return fail(error, "deferred SDMA submit lane has no registered SDMA completion capability");
            execution.provenance_capacity = execution.condition_capacity;
            if (!scheduler_layout_checked_add(
                    header.provenance_count, execution.provenance_capacity, &header.provenance_count
                ))
                return fail(error, "submission provenance count overflows");
        }
        grant_indices.insert(grant_indices.end(), execution_grants[i].begin(), execution_grants[i].end());
    }
    header.workspace_count = workspaces.size();
    header.bundle_count = bundles.size();
    header.execution_count = executions.size();
    header.request_count = requests.size();
    header.grant_count = grants.size();
    uint64_t cursor = sizeof(header);
    if (!reserve(header.task_count, sizeof(SchedulerSubmissionTaskMap), cursor, header.task_maps_offset) ||
        !reserve(header.workspace_count, sizeof(SchedulerSubmissionWorkspace), cursor, header.workspaces_offset) ||
        !reserve(header.workspace_count, sizeof(SchedulerSubmissionPool), cursor, header.pools_offset) ||
        !reserve(
            header.bundle_count, sizeof(SchedulerSubmissionBundleIdentity), cursor, header.bundle_identities_offset
        ) ||
        !reserve(header.bundle_count, sizeof(SchedulerAsyncLease), cursor, header.bundle_leases_offset) ||
        !reserve(header.bundle_count, sizeof(SchedulerSubmissionProgress), cursor, header.bundle_progress_offset) ||
        !reserve(header.execution_count, sizeof(SchedulerSubmissionExecution), cursor, header.executions_offset) ||
        !reserve(header.request_count, sizeof(SchedulerSubmissionRequest), cursor, header.requests_offset) ||
        !reserve(header.grant_count, sizeof(SchedulerSubmissionGrant), cursor, header.grants_offset) ||
        !reserve(header.grant_count, sizeof(SchedulerSubmissionReleaseState), cursor, header.release_states_offset) ||
        !reserve(header.grant_count, sizeof(uint64_t), cursor, header.execution_grant_indices_offset) ||
        !reserve(header.execution_count, sizeof(AsyncSubmissionContext), cursor, header.contexts_offset) ||
        !reserve(header.provenance_count, sizeof(SchedulerSubmissionProvenance), cursor, header.provenance_offset) ||
        !reserve(
            header.execution_count, sizeof(SchedulerSubmissionExecutionUseMap), cursor, header.execution_use_maps_offset
        ) ||
        !scheduler_layout_checked_align(cursor, 128, &header.total_bytes) || header.total_bytes > budget ||
        header.total_bytes > std::numeric_limits<size_t>::max())
        return fail(error, "submission image exceeds its byte budget or addressable storage");
    next.image.resize(static_cast<size_t>(header.total_bytes));
    write(next.image, 0, header);
    write_rows(next.image, header.task_maps_offset, task_maps);
    write_rows(next.image, header.workspaces_offset, workspaces);
    write_rows(next.image, header.bundle_identities_offset, bundles);
    write_rows(next.image, header.executions_offset, executions);
    write_rows(next.image, header.requests_offset, requests);
    write_rows(next.image, header.grants_offset, grants);
    write_rows(next.image, header.execution_grant_indices_offset, grant_indices);
    write_rows(next.image, header.execution_use_maps_offset, execution_use_maps);
    for (uint64_t index = 0; index < header.execution_count; ++index)
        write(next.image, header.contexts_offset + index * sizeof(AsyncSubmissionContext), AsyncSubmissionContext{});
    for (uint64_t index = 0; index < header.provenance_count; ++index)
        write(
            next.image, header.provenance_offset + index * sizeof(SchedulerSubmissionProvenance),
            SchedulerSubmissionProvenance{}
        );
    for (uint64_t index = 0; index < header.bundle_count; ++index) {
        const uint64_t lease = scheduler_async_lease_word(1, SchedulerAsyncPhase::RUNNABLE);
        write(next.image, header.bundle_leases_offset + index * sizeof(SchedulerAsyncLease), lease);
        write(
            next.image, header.bundle_progress_offset + index * sizeof(SchedulerSubmissionProgress),
            SchedulerSubmissionProgress{}
        );
    }
    *out = std::move(next);
    return true;
}
}  // namespace

bool scheduler_plan_submission_bundles(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    const std::vector<SchedulerSubmissionWorkspaceInput> &workspaces,
    const std::vector<SchedulerSubmissionExecutionUse> &uses,
    const std::vector<SchedulerSubmissionExecutionUseSet> &use_sets, uint64_t run_epoch, uint64_t byte_budget,
    SchedulerSubmissionHostPlan *out, std::string *error
) {
    try {
        return build(metadata, capabilities, workspaces, uses, use_sets, run_epoch, byte_budget, out, error);
    } catch (const std::bad_alloc &) {
        return fail(error, "cannot allocate the submission bundle plan");
    } catch (const std::length_error &) {
        return fail(error, "submission bundle plan exceeds container capacity");
    }
}
