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

#include "arg_with_deps.h"
#include "orchestration_api.h"

extern "C" __attribute__((visibility("default"))) OrchestrationConfig ccu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 6};
}

extern "C" __attribute__((visibility("default"))) void ccu_orchestration(const ChipTaskArgs &args) {
    auto b = args.tensor(0).ref();
    auto a = args.tensor(1).ref();
    auto second_a = args.tensor(2).ref();
    const auto context_a = args.scalar<uint64_t>(0);
    const auto context_b = args.scalar<uint64_t>(1);
    const auto value = args.scalar<uint64_t>(2);
    CoreTaskArgs trigger_b;
    trigger_b.add_output(b);
    trigger_b.add_scalar(context_b);
    trigger_b.add_scalar(value + 2);
    trigger_b.add_scalar(1);
    const auto first = rt_submit_aiv_task(0, trigger_b);
    CoreTaskArgsWithDeps<1> trigger_a;
    trigger_a.add_input(b);
    trigger_a.add_output(a);
    trigger_a.add_scalar(context_a);
    trigger_a.add_scalar(value + 1);
    trigger_a.add_scalar(1);
    trigger_a.add_dep(first.task_id());
    const auto second = rt_submit_aiv_task(1, trigger_a);
    CoreTaskArgsWithDeps<1> trigger_second_a;
    trigger_second_a.add_input(a);
    trigger_second_a.add_output(second_a);
    trigger_second_a.add_scalar(context_b);
    trigger_second_a.add_scalar(value + 3);
    trigger_second_a.add_scalar(1);
    trigger_second_a.add_dep(second.task_id());
    const auto third = rt_submit_aiv_task(1, trigger_second_a);

    uint32_t shape[] = {1};
    uint32_t offset[] = {1};
    auto exit_b_output = b.view(shape, offset);
    auto exit_a_output = a.view(shape, offset);
    auto exit_second_a_output = second_a.view(shape, offset);
    CoreTaskArgsWithDeps<1> exit_b;
    exit_b.add_output(exit_b_output);
    exit_b.add_scalar(context_b);
    exit_b.add_scalar(value + 12);
    exit_b.add_scalar(0);
    exit_b.add_dep(third.task_id());
    const auto fourth = rt_submit_aiv_task(0, exit_b);
    CoreTaskArgsWithDeps<1> exit_a;
    exit_a.add_input(b);
    exit_a.add_output(exit_a_output);
    exit_a.add_scalar(context_a);
    exit_a.add_scalar(value + 11);
    exit_a.add_scalar(0);
    exit_a.add_dep(fourth.task_id());
    const auto fifth = rt_submit_aiv_task(1, exit_a);
    CoreTaskArgsWithDeps<1> exit_second_a;
    exit_second_a.add_input(a);
    exit_second_a.add_output(exit_second_a_output);
    exit_second_a.add_scalar(context_b);
    exit_second_a.add_scalar(value + 13);
    exit_second_a.add_scalar(0);
    exit_second_a.add_dep(fifth.task_id());
    rt_submit_aiv_task(1, exit_second_a);
}
