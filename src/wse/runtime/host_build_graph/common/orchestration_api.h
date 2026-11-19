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
#include "host_build_graph/host/orchestration/orchestration_api.h"
#include "wse_submission.h"

static inline TaskOutputTensors
rt_submit_wse_task(int32_t kernel_id, const CoreTaskArgs &args, const WseLaunchSpec &launch = {}) {
    auto *ctx = current_orch_context();
    if (ctx->ops->is_fatal(ctx)) return {};
    return static_cast<const WseOrchOps *>(ctx->ops)->submit_wse(ctx, kernel_id, args, launch);
}
static inline int32_t rt_available_core_count() { return rt_available_cluster_count(); }
