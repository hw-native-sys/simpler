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
#include <mutex>
#include <unordered_map>
#include <vector>
#include "../common/wse_submission.h"
#include "host_build_graph/host/orchestrator.h"
#include "wse_launch.h"

struct WseBlockArguments {
    WseBlockPlan plan{};
    std::vector<simpler::hbg::Tensor> tensors;
    std::vector<TensorArgType> directions;
    std::vector<uint64_t> scalars;
    std::vector<int32_t> scalar_parameters;
};
struct WseKernelLaunch {
    std::vector<WseBlockArguments> blocks;
};
struct WseLaunchRecords {
    std::unordered_map<uint32_t, WseKernelLaunch> ordinary;
    std::unordered_map<uint64_t, std::unordered_map<uint32_t, WseKernelLaunch>> definitions;
};
struct WseLaunchState {
    std::mutex launches_mutex;
    WseLaunchRecords launches;
};
void wse_bind_ops(OrchContext *, WseOrchOps &, WseLaunchState &);
