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

#include <cstdint>

#include "orchestration_api.h"

namespace {

constexpr int kWriteKernel = 0;
constexpr int kEmptyKernel = 1;
constexpr int kFanout = 32;

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 1};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &input = args.tensor(0).ref();

    CoreTaskArgs root_args;
    root_args.launch_spec.set_block_num(1);
    const TaskId root = rt_submit_aiv_task(kEmptyKernel, root_args).task_id();

    TaskId children[kFanout];
    for (int child = 0; child < kFanout; ++child) {
        CoreTaskArgs child_args;
        child_args.set_dependencies(&root, 1);
        child_args.launch_spec.set_block_num(1);
        children[child] = rt_submit_aiv_task(kEmptyKernel, child_args).task_id();
    }

    CoreTaskArgs final_args;
    final_args.add_inout(input);
    final_args.set_dependencies(children, kFanout);
    final_args.launch_spec.set_block_num(1);
    const TaskId final_task = rt_submit_aiv_task(kWriteKernel, final_args).task_id();

    CoreTaskArgs dummy_args;
    dummy_args.set_dependencies(&final_task, 1);
    rt_submit_dummy_task(dummy_args);
}

}  // extern "C"
