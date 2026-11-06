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

#include "async_plan.h"
#include "native_graph_plan.h"
#include "submission_bundle_plan.h"
#include "backend/async_resource_binding.h"
#include "common/host_api.h"
#include "host_build_graph/host/async_declarations.h"

// The already compacted Host mirror of the device task image. All embedded
// tensor addresses have been rebased to actual GM; self-relative argument
// regions still resolve inside this Host mirror. No orchestration virtual heap.
struct AsyncBindingTaskImage {
    const void *host_base{nullptr};
    uint64_t bytes{0};
    uint64_t device_base{0};
    uint64_t storage_offset{0};
    uint32_t task_count{0};
};

struct AsyncBindingGroupUses {
    uint64_t begin{0};
    uint64_t count{0};
};

struct AsyncBindingHostPlan {
    uint64_t run_epoch{0};
    uint64_t execution_contract{ASYNC_EXECUTION_UNQUALIFIED};
    SchedulerAsyncAuxiliaryBytes auxiliary{};
    std::vector<AsyncRetainedSpan> spans;
    std::vector<AsyncResourceRange> ranges;
    std::vector<AsyncConditionContract> conditions;
    std::vector<AsyncBindingGroupUses> groups;
    std::vector<SchedulerSubmissionWorkspaceInput> submission_workspaces;
    std::vector<SchedulerSubmissionExecutionUse> submission_uses;
    std::vector<SchedulerSubmissionExecutionUseSet> execution_use_sets;
};

// Inspect only declarations of this run's actual expanded executions. A cached
// Graph definition that is not instantiated does not require a submission image.
bool scheduler_has_sdma_uses(
    const std::vector<SchedulerTaskMetadata> &metadata, const NativeGraphHostPlan &native,
    const AsyncHostDeclarations &declarations, bool *out, std::string *error
);

// Queries only owners already selected by this native preparation. Explicit
// PUBLISHED_CLEAN_LINES declarations provide the publication/clean-alias
// contract; owner lookup alone supplies no such permission. Executions without
// condition declarations use the existing kernel-owned live-address contract.
// Invalid explicit declarations never fall back to dynamic registration.
bool scheduler_plan_async_bindings(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    const AsyncBindingTaskImage &image, const NativeGraphHostPlan &native, const AsyncHostDeclarations &declarations,
    const HostApi &api, uint64_t byte_budget, AsyncBindingHostPlan *out, std::string *error
);

// Writes the private side sections into the already initialized async image and
// binds each real group's allowed-use set. No allocation or pointer relocation.
bool scheduler_initialize_async_bindings(
    const AsyncBindingHostPlan &plan, const SchedulerAsyncLayout &layout, void *region, uint64_t region_bytes,
    std::string *error
);
