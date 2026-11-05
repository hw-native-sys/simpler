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

#include <cstddef>
#include <string>
#include <vector>

#include "scheduler/scheduler_submission_types.h"

// Qualification comes from the retained workspace owner and the real run
// admission boundary. UNKNOWN and unimplemented OWNER_SHARED are rejected.
struct SchedulerSubmissionWorkspaceInput {
    uint64_t address{0};
    uint64_t bytes{0};
    uint64_t qualification_epoch{0};
    uint32_t queue_num{0};
    uint32_t physical_channels{0};
    SchedulerSubmissionPoolQualification qualification{SchedulerSubmissionPoolQualification::UNKNOWN};
};

// Each input names an actual SDMA-capable physical execution and its resolved
// scalar context. Possible branch uses are included before any kernel entry.
struct SchedulerSubmissionExecutionUse {
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint32_t block_idx{0};
    uint32_t subtask_slot{0};
    uint32_t context_scalar{UINT32_MAX};
    uint64_t workspace_input{SCHEDULER_ASYNC_INDEX_NONE};
};

struct SchedulerSubmissionHostPlan {
    SchedulerSubmissionHeader header{};
    std::vector<std::byte> image;
    std::vector<uint64_t> workspace_remap;
};

// Normalize exact workspace aliases, reject overlapping/conflicting views,
// reserve whole CPU-cohort demands in stable workspace-address order, and
// serialize the private image. Failure leaves out unchanged.
bool scheduler_plan_submission_bundles(
    const std::vector<SchedulerTaskMetadata> &metadata,
    const std::vector<SchedulerSubmissionWorkspaceInput> &workspaces,
    const std::vector<SchedulerSubmissionExecutionUse> &uses, uint64_t run_epoch, uint64_t byte_budget,
    SchedulerSubmissionHostPlan *out, std::string *error
);
