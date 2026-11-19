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

#include <vector>

#include "orchestration_api.h"

namespace {

struct Topology : simpler::hbg::ConfigBase {
    int32_t nodes;
    int32_t kind;
    Topology(int32_t count, int32_t shape) :
        nodes(count),
        kind(shape) {}
    uint64_t hash() const { return simpler::hbg::config_hash(nodes, kind); }
    bool operator==(const Topology &other) const { return nodes == other.nodes && kind == other.kind; }
};

void body(const GraphTaskArgs &args, const Topology &topology) {
    std::vector<TaskId> ids;
    ids.reserve(topology.nodes);
    for (int32_t i = 0; i < topology.nodes; ++i) {
        CoreTaskArgs child;
        const bool dummy = topology.kind >= 2 && (topology.kind == 3 || i + 1 < topology.nodes);
        if (topology.kind >= 2 && i > 0) child.set_dependencies(&ids.back(), 1);
        if (topology.kind == 1 && i + 1 == topology.nodes)
            child.set_dependencies(ids.data(), static_cast<uint32_t>(ids.size()));
        if (dummy) {
            child.set_allow_early_resolve(true);
            ids.push_back(rt_submit_dummy_task(child).task_id());
            continue;
        }
        child.add_no_dep(args.tensor(0).ref());
        child.add_scalar(args.scalar(0), args.scalar(1));
        child.add_scalar(static_cast<int64_t>(i + 1));
        const bool join = topology.kind == 1 && i + 1 == topology.nodes;
        child.add_scalar(static_cast<int64_t>(join ? 1 : 0));
        child.add_scalar(static_cast<int64_t>(join ? topology.nodes - 1 : 1));
        ids.push_back(((i & 1) == 0 ? rt_submit_aic_task(0, child) : rt_submit_aiv_task(1, child)).task_id());
    }
}

TaskId submit_check(
    const simpler::hbg::Tensor &state, int64_t base, int64_t token, int64_t local, int64_t begin, int64_t count,
    const TaskId *dependency
) {
    CoreTaskArgs task;
    if (dependency == nullptr) task.add_inout(state);
    else task.add_no_dep(state);
    task.add_scalar(base, token, local, begin, count);
    if (dependency != nullptr) task.set_dependencies(dependency, 1);
    return rt_submit_aiv_task(1, task).task_id();
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const auto &state = args.tensor(0).ref();
    const int32_t nodes = static_cast<int32_t>(args.scalar<int64_t>(0));
    const int32_t kind = static_cast<int32_t>(args.scalar<int64_t>(1));
    const int32_t instances = static_cast<int32_t>(args.scalar<int64_t>(2));
    for (int32_t instance = 0; instance < instances; ++instance) {
        const int64_t base = static_cast<int64_t>(instance) * (nodes + 2);
        const int64_t token = 10000 + static_cast<int64_t>(instance) * 1000;
        (void)submit_check(state, base, token, 0, 0, 0, nullptr);
        GraphTaskArgs graph;
        graph.add_inout(state);
        graph.add_scalar(base, token);
        const auto result = rt_submit_graph(&body, graph, Topology{nodes, kind});
        const TaskId shell = result.task_id;
        const int64_t begin = kind >= 2 ? (kind == 3 ? 0 : nodes) : 1;
        const int64_t count = kind >= 2 ? 1 : nodes;
        (void)submit_check(state, base, token, nodes + 1, begin, count, &shell);
    }
}

}  // extern "C"
