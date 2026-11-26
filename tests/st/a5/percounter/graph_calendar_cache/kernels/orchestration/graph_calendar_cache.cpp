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

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

struct BlockCount : simpler::hbg::ConfigBase {
    explicit BlockCount(int32_t value) :
        value(value) {}

    // Exercise value equality even when different Configs hash to the same bucket.
    uint64_t hash() const { return 0; }
    bool operator==(const BlockCount &other) const { return value == other.value; }

    int32_t value;
};

void graph_body(const GraphTaskArgs &args, const BlockCount &blocks) {
    const auto &out = args.tensor(0).ref();
    CoreTaskArgs stamp;
    stamp.add_inout(out);
    stamp.launch_spec.set_block_num(blocks.value == 0 ? out.shapes[0] / 64 : blocks.value);
    rt_submit_aic_task(0, stamp);

    CoreTaskArgs mark;
    mark.add_inout(out);
    mark.add_scalar(args.scalar(0));
    rt_submit_aiv_task(1, mark);
}

void submit_graph(uint64_t graph_id, const simpler::hbg::Tensor &out, BlockCount blocks, int64_t mark, int64_t bucket) {
    GraphTaskArgs graph;
    graph.add_inout(out);
    graph.add_scalar(mark);
    if (bucket < 0) {
        rt_submit_graph(graph_id, &graph_body, graph, blocks);
    } else {
        rt_submit_graph(
            graph_id, &graph_body, graph, blocks,
            simpler::hbg::GraphSubmitOptions{.calendar_hash = static_cast<uint64_t>(bucket)}
        );
    }
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 9};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const BlockCount blocks(static_cast<int32_t>(args.scalar<int64_t>(0)));
    const auto graph_id = args.scalar<uint64_t>(1);
    const auto prefix_graph_id = args.scalar<uint64_t>(4);
    if (prefix_graph_id != 0) {
        submit_graph(prefix_graph_id, args.tensor(1).ref(), blocks, args.scalar<int64_t>(5), args.scalar<int64_t>(6));
    }
    submit_graph(graph_id, args.tensor(0).ref(), blocks, args.scalar<int64_t>(2), args.scalar<int64_t>(3));
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry_alternate(const ChipTaskArgs &args) {
    aicpu_orchestration_entry(args);
}

}  // extern "C"
