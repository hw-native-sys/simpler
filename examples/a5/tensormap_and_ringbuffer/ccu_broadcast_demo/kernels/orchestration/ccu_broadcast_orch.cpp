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
#include "orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig ccu_broadcast_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 5};
}

__attribute__((visibility("default"))) void ccu_broadcast_orchestration(const ChipTaskArgs &args) {
    if (args.tensor_count() != 0 || args.scalar_count() != 5) {
        LOG_ERROR("CCU Broadcast expects five scalar arguments");
        return;
    }
    CoreTaskArgs task;
    for (int i = 0; i < 5; ++i)
        task.add_scalar(args.scalar(i));
    rt_submit_aiv_task(0, task);
}

}  // extern "C"
