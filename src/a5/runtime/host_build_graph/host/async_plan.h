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

// The resource owner supplies these sections after validating concrete bindings.
// Reserving their byte ranges does not prove that the bindings are admissible.
struct SchedulerAsyncAuxiliaryBytes {
    uint64_t resources{0};
    uint64_t uses{0};
    uint64_t consumers{0};
    uint64_t scratch{0};
};

struct SchedulerAsyncHostPlan {
    SchedulerAsyncLayout layout{};
    std::vector<std::byte> image;
};

// Metadata is the final expanded native-Graph/TOP node sequence. Capabilities
// belong to actual registered lane callables, never profiling/trace identities.
// Failure leaves out unchanged. A fully synchronous run has an empty image.
bool scheduler_plan_async_state(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    uint64_t run_epoch, uint64_t byte_budget, const SchedulerAsyncAuxiliaryBytes &auxiliary,
    SchedulerAsyncHostPlan *out, std::string *error
);
