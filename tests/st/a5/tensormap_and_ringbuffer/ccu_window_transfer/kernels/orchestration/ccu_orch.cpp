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

extern "C" __attribute__((visibility("default"))) OrchestrationConfig ccu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 5};
}

extern "C" __attribute__((visibility("default"))) void ccu_orchestration(const ChipTaskArgs &args) {
    auto b = args.tensor(0).ref();
    auto a = args.tensor(1).ref();
    auto evidence = args.tensor(2).ref();
    const auto context = args.scalar<uint64_t>(0);
    const auto value = args.scalar<uint64_t>(1);
    const auto mode = value >> 32;
    const auto mask = mode == 0 ? 3 : mode - 1;
    {
        CoreTaskArgs task;
        task.add_output(b);
        task.add_scalar(context);
        task.add_scalar(value + 2);
        task.add_scalar((mask & 2) != 0);
        rt_submit_aiv_task(0, task);
    }
    {
        CoreTaskArgs task;
        task.add_input(b);
        task.add_output(a);
        task.add_scalar(context);
        task.add_scalar(value + 1);
        task.add_scalar((mask & 1) != 0);
        rt_submit_aiv_task(1, task);
    }
    CoreTaskArgs verify;
    verify.add_input(b);
    verify.add_input(a);
    verify.add_output(evidence);
    verify.add_scalar(context);
    verify.add_scalar(value);
    rt_submit_aiv_task(2, verify);
}
