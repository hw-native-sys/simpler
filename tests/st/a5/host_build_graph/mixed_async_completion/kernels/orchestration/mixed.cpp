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
#include <array>
#include <cstdint>
#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {
struct ShapeConfig : simpler::hbg::ConfigBase {
    int32_t mask;
    int32_t blocks;
    int32_t elements;
    ShapeConfig(int32_t lanes, int32_t count, int32_t size) :
        mask(lanes),
        blocks(count),
        elements(size) {}
    uint64_t hash() const { return simpler::hbg::config_hash(mask, blocks, elements); }
    bool operator==(const ShapeConfig &other) const {
        return mask == other.mask && blocks == other.blocks && elements == other.elements;
    }
};
MixedKernels kernels(int32_t mask) {
    return {mask & 1 ? 0 : INVALID_KERNEL_ID, mask & 2 ? 1 : INVALID_KERNEL_ID, mask & 4 ? 1 : INVALID_KERNEL_ID};
}
void launch(CoreTaskArgs &args, const ShapeConfig &config) {
    args.launch_spec.set_block_num(config.blocks);
    args.launch_spec.set_require_sync_start(config.blocks > 1);
}
void graph_body(const GraphTaskArgs &args, const ShapeConfig &config) {
    CoreTaskArgs producer;
    for (uint32_t i = 0; i < 9; ++i)
        producer.add_no_dep(args.tensor(i).ref());
    producer.add_scalar(args.scalar(0), config.elements, config.mask, args.scalar(1), 0);
    launch(producer, config);
    (void)rt_submit_task(kernels(config.mask), producer);
}
void core_args(
    CoreTaskArgs &args, const std::array<simpler::hbg::Tensor, 9> &tensors, const ShapeConfig &config, uint64_t context,
    int64_t tag, int64_t role = 0
) {
    for (const auto &tensor : tensors)
        args.add_no_dep(tensor);
    args.add_scalar(context, config.elements, config.mask, tag, role);
}
}  // namespace
extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 16};
}
extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const uint64_t context = args.scalar<uint64_t>(0);
    const ShapeConfig config(args.scalar<int64_t>(2), args.scalar<int64_t>(3), args.scalar<int64_t>(1));
    const bool graph = args.scalar<int64_t>(4) != 0, late = args.scalar<int64_t>(5) != 0;
    const int64_t tag = args.scalar<int64_t>(6);
    constexpr uint32_t instances = 2;
    std::array<simpler::hbg::Tensor, 9> all;
    for (uint32_t i = 0; i < 9; ++i)
        all[i] = args.tensor(i).ref();
    for (uint32_t instance = 0; instance < instances; ++instance) {
        std::array<simpler::hbg::Tensor, 9> views;
        for (uint32_t i = 0; i < 9; ++i) {
            if (i == 3) {
                views[i] = all[i];
                continue;
            }
            const uint32_t shape[]{all[i].shapes[0] / instances};
            const uint32_t offset[]{instance * shape[0]};
            views[i] = all[i].view(shape, offset);
        }
        TaskId dependency;
        if (graph) {
            GraphTaskArgs invocation;
            for (const auto &view : views)
                invocation.add_no_dep(view);
            invocation.add_scalar(context, tag);
            dependency = rt_submit_graph(&graph_body, invocation, config).task_id;
        } else {
            CoreTaskArgs producer;
            core_args(producer, views, config, context, tag);
            launch(producer, config);
            dependency = rt_submit_task(kernels(config.mask), producer).task_id();
        }
        CoreTaskArgs consumer;
        core_args(consumer, views, config, context, tag);
        consumer.launch_spec.set_block_num(config.blocks * 3);
        consumer.set_dependencies(&dependency, 1);
        (void)rt_submit_aiv_task(3, consumer);
    }
    if (!late) return;
    // Readiness is based on registration, never the waiting tasks' logical completion.
    CoreTaskArgs gate;
    core_args(gate, all, config, context, tag, 0);
    const auto registered = rt_submit_aiv_task(2, gate).task_id();
    CoreTaskArgs broadcast;
    core_args(broadcast, all, config, context, tag, 1);
    broadcast.set_dependencies(&registered, 1);
    const auto sent = rt_submit_aiv_task(2, broadcast).task_id();
    CoreTaskArgs ready;
    core_args(ready, all, config, context, tag, 2);
    ready.set_dependencies(&sent, 1);
    const auto everyone_ready = rt_submit_aiv_task(2, ready).task_id();
    CoreTaskArgs publisher;
    core_args(publisher, all, config, context, tag, 3);
    publisher.set_dependencies(&everyone_ready, 1);
    (void)rt_submit_aiv_task(2, publisher);
}
