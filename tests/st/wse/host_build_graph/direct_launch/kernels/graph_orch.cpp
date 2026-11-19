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
void layer(const GraphTaskArgs &args) {
    CoreTaskArgs task;
    task.add_inout(args.tensor(0).ref());
    task.launch_spec.set_block_num(rt_available_core_count());
    rt_submit_wse_task(0, task);
}
}  // namespace
extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 3};
}
extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    for (int i = 0; i < 3; ++i) {
        GraphTaskArgs graph;
        graph.add_inout(args.tensor(i).ref());
        rt_submit_graph(&layer, graph);
    }
}
