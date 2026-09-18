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

#include <memory>

#include "host_build_graph/kernel_graph_template.h"
#include "host_build_graph/kernel_resource_requirements.h"
#include "task_interface/task_args.h"

namespace hbg {

struct KernelCallableLaunchState {
    GraphLaunchTemplate graph_template;
    std::unique_ptr<ChipStorageTaskArgs> argument_snapshot;
    uint64_t argument_hash{0};
};

struct KernelContextLaunchState {
    KernelResourcePlan resource_plan;
    GraphSlotRegistration slot_registration{};
    bool resources_prepared{false};
    bool slot_registered{false};
};

}  // namespace hbg
