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
namespace {
struct Deployment : simpler::hbg::ConfigBase {
    int64_t variant;
    explicit Deployment(int64_t value) :
        variant(value) {}
    uint64_t hash() const { return simpler::hbg::config_hash(variant); }
    bool operator==(const Deployment &other) const { return variant == other.variant; }
};
void protocol(const GraphTaskArgs &args, const Deployment &deployment) {
    CoreTaskArgs task;
    std::array<simpler::hbg::Tensor, 3> tensors;
    for (int i = 0; i < 3; ++i) {
        tensors[i] = args.tensor(i).ref();
        tensors[i].manual_dep = true;
        task.add_inout(tensors[i]);
    }
    for (int i = 0; i < 5; ++i)
        task.add_scalar(args.scalar(i));
    task.launch_spec.set_block_num(2);
    WseLaunchSpec launch;
    launch.thread_num = 2;
    launch.placement = {{0, {0, 1}}, {1, {2, 3}}};
    if (deployment.variant)
        for (auto &block : launch.placement) {
            block.core_id = 1 - block.core_id;
            for (auto &pe : block.thread_pes)
                pe = (pe + 1) % 4;
        }
    rt_submit_wse_task(0, task, launch);
}
}  // namespace
extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) { return {.expected_arg_count = 9}; }
extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &a) {
    if (rt_available_core_count() != 40) throw std::invalid_argument("counter deployment requires forty active cores");
    GraphTaskArgs graph;
    for (int i = 0; i < 3; ++i)
        graph.add_inout(a.tensor(i).ref());
    for (int i = 0; i < 5; ++i)
        graph.add_scalar(a.scalar(i));
    rt_submit_graph(&protocol, graph, Deployment(a.scalar<int64_t>(5)));
}
