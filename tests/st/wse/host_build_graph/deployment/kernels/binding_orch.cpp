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
struct PredicateConfig : simpler::hbg::ConfigBase {
    bool enabled;
    explicit PredicateConfig(bool value) :
        enabled(value) {}
    uint64_t hash() const { return simpler::hbg::config_hash(enabled); }
    bool operator==(const PredicateConfig &other) const { return enabled == other.enabled; }
};
void body(const GraphTaskArgs &args, const PredicateConfig &config) {
    for (uint32_t stage = 0; stage < 2; ++stage) {
        CoreTaskArgs task;
        task.add_input(args.tensor(0).ref());
        task.add_inout(args.tensor(1).ref(), args.tensor(2).ref());
        task.add_scalar(stage);
        task.add_scalar(args.scalar(0));
        WseLaunchSpec launch;
        launch.placement = {{39, {3 - stage}}};
        CoreTaskPredicate predicate;
        predicate.operand.tensor = &args.tensor(2).ref();
        predicate.operand.ndims = 3;
        predicate.operand.indices[0] = stage;
        predicate.op = PredicateOp::EQ;
        predicate.target = config.enabled ? 999 : 0;
        task.set_predicate(predicate);
        rt_submit_wse_task(0, task, launch);
    }
}
}  // namespace
extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) { return {.expected_arg_count = 7}; }
extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    GraphTaskArgs graph;
    graph.add_input(args.tensor(0).ref());
    graph.add_inout(args.tensor(1).ref(), args.tensor(2).ref());
    graph.add_scalar(args.scalar(3));
    const PredicateConfig config(args.scalar<int64_t>(1) != 0);
    if (args.scalar<int64_t>(2)) rt_submit_graph(&body, graph, config);
    else body(graph, config);
}
