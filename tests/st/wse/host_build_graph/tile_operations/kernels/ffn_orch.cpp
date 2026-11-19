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
#include <stdexcept>
namespace {
void layer(const GraphTaskArgs &a) {
    WseLaunchSpec launch;
    launch.thread_num = 4;
    CoreTaskArgs first;
    first.add_input(a.tensor(0).ref());
    first.add_input(a.tensor(1).ref());
    first.add_output(a.tensor(3).ref());
    first.add_output(a.tensor(6).ref());
    first.launch_spec.set_block_num(40);
    rt_submit_wse_task(0, first, launch);
    CoreTaskArgs relu;
    relu.add_input(a.tensor(3).ref());
    relu.add_output(a.tensor(4).ref());
    relu.launch_spec.set_block_num(40);
    rt_submit_wse_task(1, relu, launch);
    CoreTaskArgs second;
    second.add_input(a.tensor(4).ref());
    second.add_input(a.tensor(2).ref());
    second.add_output(a.tensor(5).ref());
    second.launch_spec.set_block_num(40);
    rt_submit_wse_task(2, second, launch);
}
}  // namespace
extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) { return {.expected_arg_count = 7}; }
extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &a) {
    if (rt_available_core_count() != 40) throw std::invalid_argument("FFN qualification requires forty cores");
    GraphTaskArgs graph;
    for (int i = 0; i < 3; ++i)
        graph.add_input(a.tensor(i).ref());
    for (int i = 3; i < 7; ++i)
        graph.add_output(a.tensor(i).ref());
    rt_submit_graph(&layer, graph);
}
