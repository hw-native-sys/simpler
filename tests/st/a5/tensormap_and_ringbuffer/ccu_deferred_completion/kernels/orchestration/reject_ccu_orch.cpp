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

extern "C" __attribute__((visibility("default"))) OrchestrationConfig ccu_rejection_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 7};
}

extern "C" __attribute__((visibility("default"))) void ccu_rejection_orchestration(const ChipTaskArgs &args) {
    for (uint64_t round = 0; round < args.scalar<uint64_t>(1); ++round) {
        SIMPLER_SCOPE() {
            for (int32_t function = 0; function < 3; ++function) {
                CoreTaskArgs task;
                task.add_inout(args.tensor(0).ref());
                task.add_inout(args.tensor(1).ref());
                task.add_inout(args.tensor(2).ref());
                task.add_inout(args.tensor(3).ref());
                if (function == 1) task.add_input(args.tensor(4).ref());
                task.add_scalar(args.scalar<uint64_t>(0));
                task.add_scalar(round);
                rt_submit_aiv_task(function, task);
            }
        }
    }
}
