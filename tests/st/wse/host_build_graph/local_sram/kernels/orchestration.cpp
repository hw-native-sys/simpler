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
struct Config : simpler::hbg::ConfigBase {
    bool skew;
    explicit Config(bool value) :
        skew(value) {}
    uint64_t hash() const { return simpler::hbg::config_hash(skew); }
    bool operator==(const Config &other) const { return skew == other.skew; }
};
void graph(const GraphTaskArgs &a, const Config &config) {
    for (int stage = 0; stage < 2; ++stage) {
        CoreTaskArgs task;
        task.add_inout(a.tensor(0).ref());
        task.add_inout(a.tensor(1).ref());
        task.add_inout(a.tensor(2).ref());
        task.add_scalar(stage);
        task.launch_spec.set_block_num(a.tensor(0).ref().shapes[0]);
        WseLaunchSpec launch;
        if (config.skew) launch.placement.assign(task.launch_spec.block_num(), WseBlockPlacement{1, {0}});
        rt_submit_wse_task(0, task, launch);
    }
}
}  // namespace
extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) { return {.expected_arg_count = 4}; }
extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &a) {
    GraphTaskArgs args;
    for (int i = 0; i < 3; ++i)
        args.add_inout(a.tensor(i).ref());
    const auto mode = a.scalar<int64_t>(0);
    const Config config((mode & 2) != 0);
    if (mode & 1) rt_submit_graph(&graph, args, config);
    else graph(args, config);
}
