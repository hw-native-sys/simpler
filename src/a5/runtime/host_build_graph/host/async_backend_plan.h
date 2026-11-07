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
#include "backend/async_backend.h"
#include "common/host_api.h"
#include "host_build_graph/host/async_declarations.h"

struct AsyncBackendTaskImage {
    const void *host_base{nullptr};
    uint64_t bytes{0};
    uint64_t device_base{0};
    uint64_t storage_offset{0};
    uint32_t task_count{0};
};

struct AsyncBackendHostPlan {
    uint64_t run_epoch{0};
    uint64_t execution_contract{0};
    bool needs_sdma_guards{false};
    std::vector<std::byte> image;
};

bool async_backend_has_sdma_uses(
    const std::vector<SchedulerTaskMetadata> &metadata, const NativeGraphHostPlan &native,
    const AsyncHostDeclarations &declarations, bool *out, std::string *error
);

bool scheduler_plan_async_backend(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    const AsyncBackendTaskImage &image, const NativeGraphHostPlan &native, const AsyncHostDeclarations &declarations,
    const HostApi &api, uint64_t byte_budget, AsyncBackendHostPlan *out, std::string *error
);
