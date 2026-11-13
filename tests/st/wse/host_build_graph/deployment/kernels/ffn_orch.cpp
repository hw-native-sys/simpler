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
#include <array>
#include <memory>
#include <stdexcept>
namespace {
struct Deployment : simpler::hbg::ConfigBase {
    int64_t mode;
    explicit Deployment(int64_t value) :
        mode(value) {
        if (mode < -1 || mode > 4) throw std::invalid_argument("unknown FFN deployment");
    }
    uint64_t hash() const { return simpler::hbg::config_hash(mode); }
    bool operator==(const Deployment &other) const { return mode == other.mode; }
};
simpler::hbg::Tensor shard(const GraphTaskArgs &a, uint32_t core, uint32_t begin, uint32_t count) {
    return a.tensor(core * 2).ref().slice(0, begin, begin + count);
}
void layer(const GraphTaskArgs &a, const Deployment &deployment) {
    for (int stage = 0; stage < 3; ++stage) {
        CoreTaskArgs task;
        task.launch_spec.set_block_num(40);
        WseLaunchSpec launch;
        launch.thread_num = 4;
        auto arguments = std::make_unique<CoreTaskArgs[]>(40);
        std::vector<std::vector<simpler::hbg::Tensor>> views(40);
        for (uint32_t b = 0; b < 40; ++b) {
            const uint32_t group = b / 4;
            const uint32_t core = deployment.mode == 1 || deployment.mode == 2 ? (b + 1) % 40 : b;
            const uint32_t rotation = deployment.mode == 2 ? 1 : 0;
            if (deployment.mode != -1) {
                WseBlockPlacement placement{
                    core, {rotation, (rotation + 1) % 4, (rotation + 2) % 4, (rotation + 3) % 4}
                };
                if (deployment.mode == 3) placement.core_id = 40;
                if (deployment.mode == 4) placement.thread_pes[3] = placement.thread_pes[0];
                launch.placement.push_back(std::move(placement));
            }
            auto &v = views[b];
            if (stage == 1) {
                v.push_back(shard(a, b, 6144, 4096));
                v.push_back(shard(a, b, 10240, 4096));
            } else {
                for (uint32_t k = 0; k < 4; ++k)
                    v.push_back(shard(a, group * 4 + k, stage == 0 ? 0 : 10240, stage == 0 ? 1024 : 4096));
                v.push_back(shard(a, b, stage == 0 ? 1024 : 2048, stage == 0 ? 1024 : 4096));
                v.push_back(shard(a, b, stage == 0 ? 6144 : 14336, 4096));
            }
            v.push_back(a.tensor(b * 2 + 1).ref().slice(0, stage * 80, (stage + 1) * 80));
            auto &args = arguments[b];
            const size_t inputs = stage == 1 ? 1 : 5;
            for (size_t i = 0; i < inputs; ++i)
                args.add_input(v[i]);
            args.add_output(v[inputs]);
            args.add_inout(v.back());
            args.add_scalar(rotation);
            args.add_scalar(a.scalar(0));
            launch.block_args.push_back(&args);
        }
        rt_submit_wse_task(stage, task, launch);
    }
}
}  // namespace
extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &a) {
    return {.expected_arg_count = a.tensor_count() + a.scalar_count()};
}
extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &a) {
    const int64_t instances = a.scalar<int64_t>(0);
    if (rt_available_core_count() != 40 || (instances != 1 && instances != 2) || a.tensor_count() != instances * 80 ||
        a.scalar_count() != 3)
        throw std::invalid_argument("FFN requires forty cores and eighty SRAM Buffers per instance");
    const Deployment deployment(a.scalar<int64_t>(1));
    for (int64_t instance = 0; instance < instances; ++instance) {
        GraphTaskArgs graph;
        for (int i = 0; i < 80; ++i)
            graph.add_inout(a.tensor(instance * 80 + i).ref());
        graph.add_scalar(instance);
        if (a.scalar<int64_t>(2)) rt_submit_graph(&layer, graph, deployment);
        else layer(graph, deployment);
    }
}
