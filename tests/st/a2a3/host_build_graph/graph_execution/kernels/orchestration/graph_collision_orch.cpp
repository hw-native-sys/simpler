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

#include <atomic>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {
struct Config : simpler::hbg::ConfigBase {
    explicit Config(int v) :
        variant(v) {}
    int variant;
    uint64_t hash() const { return 0; }
    bool operator==(const Config &other) const { return variant == other.variant; }
};
std::atomic<int> recordings{0};

void body(const GraphTaskArgs &args, const Config &config) {
    recordings.fetch_add(1, std::memory_order_relaxed);
    CoreTaskArgs task;
    task.add_input(args.tensor(0).ref());
    task.add_output(args.tensor(1).ref());
    task.add_scalar(args.scalar(config.variant));
    rt_submit_aiv_task(0, task);
}
}  // namespace

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 5};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    recordings.store(0, std::memory_order_relaxed);
    for (int i = 0; i < 4; ++i) {
        GraphTaskArgs graph;
        graph.add_input(args.tensor(0).ref());
        graph.add_output(args.tensor(i + 1).ref());
        graph.add_scalar(static_cast<float>(i + 1), static_cast<float>(10 * (i + 1)));
        rt_submit_graph(&body, graph, Config{i % 2});
        // Complete publication so the repeated A and B submissions exercise cache hits.
        rt_graph_commit();
    }
    if (recordings.load(std::memory_order_relaxed) != 2) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "Expected exactly two colliding Graph recordings");
    }
}
}  // extern "C"
