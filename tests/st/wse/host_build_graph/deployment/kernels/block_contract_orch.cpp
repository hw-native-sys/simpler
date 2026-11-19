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
#include <memory>
namespace {
struct Config : simpler::hbg::ConfigBase {
    int threads;
    explicit Config(int t) :
        threads(t) {}
    uint64_t hash() const { return simpler::hbg::config_hash(threads); }
    bool operator==(const Config &other) const { return threads == other.threads; }
};
void body(const GraphTaskArgs &a, const Config &config) {
    for (int stage = 0; stage < 2; ++stage) {
        CoreTaskArgs invocation;
        invocation.launch_spec.set_block_num(5);
        WseLaunchSpec launch;
        launch.thread_num = config.threads;
        auto args = std::make_unique<CoreTaskArgs[]>(5);
        std::vector<simpler::hbg::Tensor> views;
        views.reserve(5);
        for (uint32_t b = 0; b < 5; ++b) {
            views.push_back(a.tensor(0).ref().slice(0, b * 32, (b + 1) * 32));
            args[b].add_inout(views.back());
            args[b].add_scalar(stage);
            args[b].add_scalar(a.scalar(0));
            launch.block_args.push_back(&args[b]);
            WseBlockPlacement placement{b % 2, {}};
            for (int t = 0; t < config.threads; ++t)
                placement.thread_pes.push_back((t + 3) % 4);
            launch.placement.push_back(std::move(placement));
        }
        rt_submit_wse_task(0, invocation, launch);
    }
}
}  // namespace
extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) { return {.expected_arg_count = 4}; }
extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &a) {
    GraphTaskArgs graph;
    graph.add_inout(a.tensor(0).ref());
    graph.add_scalar(a.scalar(2));
    Config config(a.scalar<int64_t>(0));
    if (a.scalar<int64_t>(1)) rt_submit_graph(&body, graph, config);
    else body(graph, config);
}
