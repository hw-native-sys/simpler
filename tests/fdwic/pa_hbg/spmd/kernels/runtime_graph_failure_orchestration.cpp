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
struct Shape : simpler::hbg::ConfigBase {
    int16_t blocks;
    uint32_t mask;
    Shape(int16_t count, uint32_t lanes) :
        blocks(count),
        mask(lanes) {}
    uint64_t hash() const { return simpler::hbg::config_hash(blocks, mask); }
    bool operator==(const Shape &other) const { return blocks == other.blocks && mask == other.mask; }
};

void body(const GraphTaskArgs &args, const Shape &shape) {
    CoreTaskArgs root;
    root.add_inout(args.tensor(0).ref());
    root.add_scalar(static_cast<int64_t>(shape.blocks), static_cast<int64_t>(shape.mask));
    root.add_scalar(args.scalar(0), args.scalar(1));
    root.launch_spec.set_block_num(shape.blocks);
    rt_submit_task(
        MixedKernels{
            (shape.mask & 1U) ? 0 : INVALID_KERNEL_ID, (shape.mask & 2U) ? 1 : INVALID_KERNEL_ID,
            (shape.mask & 4U) ? 2 : INVALID_KERNEL_ID
        },
        root
    );
    CoreTaskArgs verify;
    verify.add_input(args.tensor(0).ref());
    verify.add_inout(args.tensor(1).ref());
    verify.add_scalar(static_cast<int64_t>(shape.blocks), static_cast<int64_t>(shape.mask), INT64_C(1));
    rt_submit_aiv_task(3, verify);
}
}  // namespace

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 6};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    GraphTaskArgs graph;
    graph.add_inout(args.tensor(0).ref(), args.tensor(1).ref());
    graph.add_scalar(args.scalar<int64_t>(2), args.scalar<int64_t>(3));
    rt_submit_graph(
        &body, graph,
        Shape{static_cast<int16_t>(args.scalar<int64_t>(0)), static_cast<uint32_t>(args.scalar<int64_t>(1))}
    );
}
