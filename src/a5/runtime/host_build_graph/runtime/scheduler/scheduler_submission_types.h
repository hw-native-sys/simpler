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

#include "scheduler_async_cohort.h"
#include "async_capability.h"

inline constexpr uint64_t SCHEDULER_SUBMISSION_VERSION = 2;
// Capacity of the pinned SDMA backend's per-workspace physical queue array.
inline constexpr uint32_t SCHEDULER_SUBMISSION_SDMA_CHANNELS = 48;
inline constexpr uint32_t SCHEDULER_SUBMISSION_MAX_EXECUTION_GRANTS = CORE_MAX_SCALAR_ARGS;
static_assert(SCHEDULER_SUBMISSION_MAX_EXECUTION_GRANTS == 16);

enum class SchedulerSubmissionPoolQualification : uint32_t { UNKNOWN, EXCLUSIVE_RUN, OWNER_SHARED };
enum class SchedulerSubmissionBundleKind : uint64_t { EXECUTION, MIX_BLOCK, SYNC_TASK };
enum class SchedulerSubmissionProgressPhase : uint64_t { OBSERVE, SELECT, CLAIM, GRANTED, FAILED };
enum class SchedulerSubmissionReleasePhase : uint64_t { RESERVED, RELEASING, RELEASED };

struct alignas(128) SchedulerSubmissionHeader {
    uint64_t version{SCHEDULER_SUBMISSION_VERSION};
    uint64_t total_bytes{0};
    uint64_t run_epoch{0};
    uint64_t task_count{0};
    uint64_t workspace_count{0};
    uint64_t bundle_count{0};
    uint64_t execution_count{0};
    uint64_t request_count{0};
    uint64_t grant_count{0};
    uint64_t task_maps_offset{0};
    uint64_t workspaces_offset{0};
    uint64_t pools_offset{0};
    uint64_t bundle_identities_offset{0};
    uint64_t bundle_leases_offset{0};
    uint64_t bundle_progress_offset{0};
    uint64_t executions_offset{0};
    uint64_t requests_offset{0};
    uint64_t grants_offset{0};
    uint64_t release_states_offset{0};
    uint64_t execution_grant_indices_offset{0};
    uint64_t contexts_offset{0};
    uint64_t provenance_offset{0};
    uint64_t provenance_count{0};
    uint64_t execution_use_maps_offset{0};
    uint64_t reserved[8]{};
};

struct SchedulerSubmissionTaskMap {
    uint64_t execution_base{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t bundle_base{SCHEDULER_ASYNC_INDEX_NONE};
    uint32_t block_count{0};
    uint32_t lane_mask{0};
    uint32_t flags{0};
    uint32_t bundle_count{0};
};

struct SchedulerSubmissionWorkspace {
    uint64_t address{0};
    uint64_t bytes{0};
    uint64_t qualification_epoch{0};
    uint32_t queue_num{0};
    uint32_t physical_channels{0};
    SchedulerSubmissionPoolQualification qualification{SchedulerSubmissionPoolQualification::UNKNOWN};
    uint32_t reserved0{0};
    uint64_t reserved[3]{};
};

// Each bit owns one qNum-wide group. All groups in one workspace use the same
// qNum. These atomic-only lines belong to the run's proven exclusive authority.
struct alignas(128) SchedulerSubmissionPool {
    volatile uint64_t busy_groups{0};
    uint8_t reserved[120]{};
};

struct alignas(128) SchedulerSubmissionBundleIdentity {
    uint64_t run_epoch{0};
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint64_t execution_begin{0};
    uint64_t execution_count{0};
    uint64_t request_begin{0};
    uint64_t request_count{0};
    uint64_t block_scope{0};
    SchedulerSubmissionBundleKind kind{SchedulerSubmissionBundleKind::EXECUTION};
    uint64_t lane_mask{0};
    uint64_t block_count{0};
    uint64_t reserved[6]{};
};

struct SchedulerSubmissionExecution {
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint64_t bundle_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t grant_index_begin{0};
    uint64_t grant_count{0};
    uint32_t block_idx{0};
    uint32_t subtask_slot{0};
    CoreAsyncMode completion_mode{CoreAsyncMode::UNKNOWN};
    uint32_t condition_capacity{0};
    uint32_t condition_types{0};
    uint32_t provenance_capacity{0};
    uint64_t provenance_begin{0};
};

struct SchedulerSubmissionExecutionUseMap {
    uint64_t use_begin{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t use_count{0};
};
static_assert(sizeof(SchedulerSubmissionExecutionUseMap) == 16);

// Captured from the public event only after the qualified SDK submission.
// The enclosing run/execution owns this row; zero version is never evidence.
struct SchedulerSubmissionProvenance {
    uint32_t version{0};
    int32_t completion_type{0};
    uint64_t address{0};
    uint64_t backend_cookie{0};
    uint64_t use_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t workspace_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t execution_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t channel_group{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t reserved{0};
};
static_assert(sizeof(SchedulerSubmissionProvenance) == 64);

// Grant rows for one workspace are contiguous and contain distinct executions.
// acquired_mask is written only by the bundle lease owner before GRANTED.
struct SchedulerSubmissionRequest {
    uint64_t workspace_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t grant_begin{0};
    uint64_t grant_count{0};
    uint64_t acquired_mask{0};
};

struct SchedulerSubmissionGrant {
    uint64_t execution_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t workspace_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t group_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t reserved{0};
};

struct alignas(128) SchedulerSubmissionProgress {
    SchedulerSubmissionProgressPhase phase{SchedulerSubmissionProgressPhase::OBSERVE};
    uint64_t request_index{0};
    uint64_t snapshot{0};
    uint64_t selected_mask{0};
    uint64_t selected_count{0};
    uint64_t work_units{0};
    int64_t failure_code{0};
    uint64_t reserved[9]{};
};

struct alignas(128) SchedulerSubmissionReleaseState {
    volatile uint64_t phase{static_cast<uint64_t>(SchedulerSubmissionReleasePhase::RESERVED)};
    uint8_t reserved[120]{};
};

// Expected identity is supplied from the original executing slot generation.
// This local value is not by itself evidence of metadata publication or seal.
struct SchedulerSubmissionExecutionIdentity {
    uint64_t run_epoch{0};
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint64_t execution_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t dispatch_generation{0};
    uint32_t block_idx{0};
    uint32_t subtask_slot{0};
};

struct SchedulerSubmissionBudget {
    uint32_t lease_probes{1};
    uint32_t claim_attempts{1};
    uint32_t work_units{8};
};
enum class SchedulerSubmissionResult : uint8_t { PENDING, READY, ERROR };
enum class SchedulerSubmissionProvisionalResult : uint8_t { PENDING, RESERVED, ERROR };

struct SchedulerSubmissionTransactionStats {
    uint64_t cpu_checks{0};
    uint64_t lease_probes{0};
    uint64_t lease_cas{0};
    uint64_t pool_probes{0};
    uint64_t pool_cas{0};
    uint64_t pool_returns{0};
};

// Local, single-owner transaction. It must commit or abort before the caller
// returns to scheduling; copying it does not create another ownership grant.
struct SchedulerSubmissionProvisional {
    uint64_t bundle_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t execution_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t owned_word{0};
    uint64_t count{0};
    uint64_t request_indices[SCHEDULER_SUBMISSION_MAX_EXECUTION_GRANTS]{};
    uint64_t workspace_indices[SCHEDULER_SUBMISSION_MAX_EXECUTION_GRANTS]{};
    uint64_t masks[SCHEDULER_SUBMISSION_MAX_EXECUTION_GRANTS]{};
    bool active{false};
};

static_assert(sizeof(SchedulerSubmissionHeader) == 256);
static_assert(sizeof(SchedulerSubmissionTaskMap) == 32 && sizeof(SchedulerSubmissionWorkspace) == 64);
static_assert(sizeof(SchedulerSubmissionPool) == 128 && sizeof(SchedulerSubmissionBundleIdentity) == 128);
static_assert(sizeof(SchedulerSubmissionExecution) == 64 && sizeof(SchedulerSubmissionRequest) == 32);
static_assert(sizeof(SchedulerSubmissionGrant) == 32 && sizeof(SchedulerSubmissionProgress) == 128);
static_assert(sizeof(SchedulerSubmissionReleaseState) == 128);
#define SCHEDULER_SUBMISSION_POD(type) \
    static_assert(std::is_standard_layout_v<type> && std::is_trivially_copyable_v<type>)
SCHEDULER_SUBMISSION_POD(SchedulerSubmissionHeader);
SCHEDULER_SUBMISSION_POD(SchedulerSubmissionTaskMap);
SCHEDULER_SUBMISSION_POD(SchedulerSubmissionWorkspace);
SCHEDULER_SUBMISSION_POD(SchedulerSubmissionPool);
SCHEDULER_SUBMISSION_POD(SchedulerSubmissionBundleIdentity);
SCHEDULER_SUBMISSION_POD(SchedulerSubmissionExecution);
SCHEDULER_SUBMISSION_POD(SchedulerSubmissionExecutionUseMap);
SCHEDULER_SUBMISSION_POD(SchedulerSubmissionProvenance);
SCHEDULER_SUBMISSION_POD(SchedulerSubmissionRequest);
SCHEDULER_SUBMISSION_POD(SchedulerSubmissionGrant);
SCHEDULER_SUBMISSION_POD(SchedulerSubmissionProgress);
SCHEDULER_SUBMISSION_POD(SchedulerSubmissionReleaseState);
#undef SCHEDULER_SUBMISSION_POD
