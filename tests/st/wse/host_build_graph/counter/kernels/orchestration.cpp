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
template <bool split>
void protocol(const GraphTaskArgs &args) {
    auto counters = args.tensor(0).ref();
    auto payload = args.tensor(1).ref();
    auto result = args.tensor(2).ref();
    // These disjoint participant writes are synchronized by the explicit counter protocol.
    counters.manual_dep = true;
    payload.manual_dep = true;
    result.manual_dep = true;
    for (int child = 0; child < (split ? 2 : 1); ++child) {
        CoreTaskArgs task;
        task.add_inout(counters, payload, result);
        for (int scalar = 3; scalar < 8; ++scalar)
            task.add_scalar(args.scalar(scalar - 3));
        WseLaunchSpec launch;
        launch.thread_num = 2;
        task.launch_spec.set_block_num(split ? 1 : 2);
        if (split) launch.placement = {{static_cast<uint32_t>(child), {0, 1}}};
        rt_submit_wse_task(split ? child + 1 : 0, task, launch);
    }
}
}  // namespace
extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) { return {.expected_arg_count = 9}; }
extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    GraphTaskArgs graph;
    for (int t = 0; t < 3; ++t)
        graph.add_inout(args.tensor(t).ref());
    for (int scalar = 3; scalar < 9; ++scalar)
        graph.add_scalar(args.scalar<int64_t>(scalar - 3));
    if (args.scalar<int64_t>(5)) rt_submit_graph(&protocol<true>, graph);
    else rt_submit_graph(&protocol<false>, graph);
}
