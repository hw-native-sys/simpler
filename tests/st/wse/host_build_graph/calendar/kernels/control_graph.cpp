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

namespace {
void wait_for(TaskId producer) {
    CoreTaskArgs barrier;
    barrier.set_dependencies(&producer, 1);
    rt_submit_dummy_task(barrier);
}

void kernel_and_barrier(const GraphTaskArgs &args) {
    CoreTaskArgs task;
    task.add_input(args.tensor(0).ref(), args.tensor(1).ref());
    task.add_output(args.tensor(2).ref());
    task.add_scalar(1);
    WseLaunchSpec launch;
    launch.placement = {{7}};
    wait_for(rt_submit_wse_task(0, task, launch).task_id());
}
}  // namespace

extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) { return {.expected_arg_count = 3}; }

extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    GraphTaskArgs graph;
    graph.add_input(args.tensor(0).ref(), args.tensor(1).ref());
    graph.add_output(args.tensor(2).ref());
    const auto result = rt_submit_graph(kernel_and_barrier, graph);
    if (!result.has_error) wait_for(result.task_id);
}
