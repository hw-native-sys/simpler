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
constexpr uint32_t instances = 2;
using Tensors = std::array<simpler::hbg::Tensor, 9>;

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

template <typename Args>
Tensors tensors(const Args &args) {
    Tensors result;
    for (uint32_t i = 0; i < result.size(); ++i)
        result[i] = args.tensor(i).ref();
    return result;
}

Tensors slice(const Tensors &all, uint32_t instance) {
    Tensors views;
    for (uint32_t i = 0; i < all.size(); ++i) {
        if (i == 3) {
            views[i] = all[i];
            continue;
        }
        const uint32_t shape[]{all[i].shapes[0] / instances};
        const uint32_t offset[]{instance * shape[0]};
        views[i] = all[i].view(shape, offset);
    }
    return views;
}

template <typename Context, typename Tag>
void core_args(
    CoreTaskArgs &args, const Tensors &tensors, const ShapeConfig &config, Context context, Tag tag, int role
) {
    for (const auto &tensor : tensors)
        args.add_no_dep(tensor);
    args.add_scalar(context, config.elements, config.mask, tag, role);
}

template <typename Context, typename Tag>
TaskId submit_producer(const Tensors &views, const ShapeConfig &config, Context context, Tag tag) {
    CoreTaskArgs producer;
    core_args(producer, views, config, context, tag, 0);
    launch(producer, config);
    return rt_submit_task(kernels(config.mask), producer).task_id();
}

template <typename Context, typename Tag>
void submit_late_notifications(const Tensors &all, const ShapeConfig &config, Context context, Tag tag) {
    // The gate follows actual registrations, which can precede logical producer completion.
    TaskId preceding;
    for (int role = 0; role < 4; ++role) {
        CoreTaskArgs task;
        core_args(task, all, config, context, tag, role);
        if (preceding.is_valid()) task.set_dependencies(&preceding, 1);
        preceding = rt_submit_aiv_task(2, task).task_id();
    }
}

void graph_body(const GraphTaskArgs &args, const ShapeConfig &config) {
    submit_producer(tensors(args), config, args.scalar(0), args.scalar(1));
}

void late_graph_body(const GraphTaskArgs &args, const ShapeConfig &config) {
    const auto all = tensors(args);
    for (uint32_t instance = 0; instance < instances; ++instance)
        submit_producer(slice(all, instance), config, args.scalar(0), args.scalar(1));
    submit_late_notifications(all, config, args.scalar(0), args.scalar(1));
}

template <auto Body>
TaskId submit_graph(const Tensors &tensors, const ShapeConfig &config, uint64_t context, int64_t tag) {
    GraphTaskArgs invocation;
    for (const auto &tensor : tensors)
        invocation.add_no_dep(tensor);
    invocation.add_scalar(context, tag);
    return rt_submit_graph(Body, invocation, config).task_id;
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
    const auto all = tensors(args);
    TaskId late_graph;
    if (graph && late) late_graph = submit_graph<&late_graph_body>(all, config, context, tag);
    for (uint32_t instance = 0; instance < instances; ++instance) {
        const auto views = slice(all, instance);
        TaskId dependency = late_graph;
        if (!graph) dependency = submit_producer(views, config, context, tag);
        else if (!late) dependency = submit_graph<&graph_body>(views, config, context, tag);
        CoreTaskArgs consumer;
        core_args(consumer, views, config, context, tag, 0);
        consumer.launch_spec.set_block_num(config.blocks * 3);
        consumer.set_dependencies(&dependency, 1);
        (void)rt_submit_aiv_task(3, consumer);
    }
    if (!graph && late) submit_late_notifications(all, config, context, tag);
}
