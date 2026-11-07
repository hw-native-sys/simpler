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

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "async_capability.h"
#include "scheduler/scheduler_async_layout.h"

using SchedulerAsyncCapabilities = std::array<CoreAsyncCapability, 3>;

// Optional backend declarations share the run's retained scheduler allocation.
struct SchedulerAsyncAuxiliaryBytes {
    uint64_t backend{0};
    uint64_t sdma_guard_count{0};
    uint64_t execution_contract{HBG_ASYNC_CONTRACT_UNQUALIFIED};
};

struct SchedulerAsyncHostPlan {
    SchedulerAsyncLayout layout{};
    std::vector<std::byte> image;
};

// Count and validate without allocating an image. Offsets in the result are
// relative to the async region, independent of its final scheduler placement.
bool scheduler_plan_async_layout(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    uint64_t run_epoch, uint64_t byte_budget, const SchedulerAsyncAuxiliaryBytes &auxiliary, SchedulerAsyncLayout *out,
    std::string *error
);

// Append to the same retained scheduler backing. Zero bytes produce locator 0.
// A failure leaves both output values unchanged; byte_budget covers the whole
// backing, not just the async region.
bool scheduler_append_async_layout(
    uint64_t prefix_bytes, const SchedulerAsyncLayout &layout, uint64_t byte_budget, uint64_t *async_offset,
    uint64_t *total_bytes, std::string *error
);

// Initialize directly in the retained Host mirror. The complete layout is
// revalidated before any write; only its own extent is initialized. No device
// address or pointer fix-up is needed, and no second full image is allocated.
bool scheduler_initialize_async_state(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    const SchedulerAsyncLayout &layout, void *region, uint64_t region_bytes, std::string *error
);

// Metadata is the final expanded native-Graph/TOP node sequence. Capabilities
// belong to actual registered lane callables, never profiling/trace identities.
// Failure leaves out unchanged. A fully synchronous run has an empty image.
bool scheduler_plan_async_state(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    uint64_t run_epoch, uint64_t byte_budget, const SchedulerAsyncAuxiliaryBytes &auxiliary,
    SchedulerAsyncHostPlan *out, std::string *error
);
