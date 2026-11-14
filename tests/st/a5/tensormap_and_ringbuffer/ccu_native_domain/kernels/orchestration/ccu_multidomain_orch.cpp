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
    return OrchestrationConfig{.expected_arg_count = 6};
}

extern "C" __attribute__((visibility("default"))) void ccu_orchestration(const ChipTaskArgs &args) {
    auto b = args.tensor(0).ref();
    auto a = args.tensor(1).ref();
    const auto context_a = args.scalar<uint64_t>(0);
    const auto context_b = args.scalar<uint64_t>(1);
    const auto value = args.scalar<uint64_t>(2);
    CoreTaskArgs trigger_b;
    trigger_b.add_output(b);
    trigger_b.add_scalar(context_b);
    trigger_b.add_scalar(value + 2);
    trigger_b.add_scalar(1);
    rt_submit_aiv_task(0, trigger_b);
    CoreTaskArgs trigger_a;
    trigger_a.add_input(b);
    trigger_a.add_output(a);
    trigger_a.add_scalar(context_a);
    trigger_a.add_scalar(value + 1);
    trigger_a.add_scalar(1);
    rt_submit_aiv_task(1, trigger_a);
    CoreTaskArgs trigger_a_second;
    trigger_a_second.add_input(a);
    trigger_a_second.add_output(args.tensor(2).ref());
    trigger_a_second.add_scalar(context_b);
    trigger_a_second.add_scalar(value + 3);
    trigger_a_second.add_scalar(1);
    rt_submit_aiv_task(1, trigger_a_second);
}
