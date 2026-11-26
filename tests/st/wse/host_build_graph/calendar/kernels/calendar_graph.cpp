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
void submit_ss(const GraphTaskArgs &args, int input, int output, int block_count = 1, int tiles_per_block = 1) {
    CoreTaskArgs task;
    task.add_input(args.tensor(input).ref());
    task.add_input(args.tensor(1).ref());
    task.add_output(args.tensor(output).ref());
    task.add_scalar(tiles_per_block);
    task.launch_spec.set_block_num(block_count);
    rt_submit_wse_task(0, task);
}

void fan_out_and_join(const GraphTaskArgs &args) {
    constexpr int branches = 4;
    submit_ss(args, 0, 2);
    submit_ss(args, 2, 3, branches);
    submit_ss(args, 3, 4, 1, branches);
}

void independent_leaf_and_chain(const GraphTaskArgs &args) {
    submit_ss(args, 0, 2);  // Independent leaf, submitted first.
    submit_ss(args, 0, 3);  // The other ready node feeds the final SS.
    submit_ss(args, 3, 4);
}
}  // namespace

extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    return {.expected_arg_count = args.scalar<int64_t>(0) ? 6 : 11};
}

extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const bool independent = args.scalar<int64_t>(0);
    for (int instance = 0; instance < (independent ? 1 : 2); ++instance) {
        GraphTaskArgs graph;
        const int first = 5 * instance;
        graph.add_input(args.tensor(first).ref());
        graph.add_input(args.tensor(first + 1).ref());
        for (int i = 2; i < 5; ++i)
            graph.add_output(args.tensor(first + i).ref());
        const auto result = rt_submit_graph(independent ? independent_leaf_and_chain : fan_out_and_join, graph);
        if (result.has_error || result.recording != (instance == 0)) {
            rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "%s", "expected one Graph definition recording per bind");
            return;
        }
    }
}
