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
#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {
void body(const GraphTaskArgs &args) {
    CoreTaskArgs task;
    task.add_inout(args.tensor(0).ref());
    const int64_t base = 0;
    task.add_scalar(base);
    task.dump(args.tensor(0).ref(), base);
    task.launch_spec.set_block_num(73);
    MixedKernels kernels;
    kernels.aic_kernel_id = 0;
    kernels.aiv0_kernel_id = 1;
    kernels.aiv1_kernel_id = 2;
    rt_submit_task(kernels, task);
}
}  // namespace

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 1};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    GraphTaskArgs graph;
    graph.add_inout(args.tensor(0).ref());
    rt_submit_graph(&body, graph);
}
