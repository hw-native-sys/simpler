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
struct ShapeConfig : simpler::hbg::ConfigBase {
    int32_t mask;
    int32_t blocks;
    int32_t mode;

    ShapeConfig(int32_t lanes, int32_t count, int32_t kind) :
        mask(lanes),
        blocks(count),
        mode(kind) {}
    uint64_t hash() const { return simpler::hbg::config_hash(mask, blocks, mode); }
    bool operator==(const ShapeConfig &other) const {
        return mask == other.mask && blocks == other.blocks && mode == other.mode;
    }
};

void submit_shape(const simpler::hbg::Tensor &out, const ShapeConfig &config) {
    const uint32_t shape[]{static_cast<uint32_t>(1 + 3 * config.blocks * 65) * 32};
    TensorCreateInfo state_info(shape, 1, DataType::INT32);
    CoreTaskArgs initialize;
    initialize.add_output(state_info);
    initialize.add_no_dep(out);
    initialize.add_scalar(0, config.mask, config.blocks, config.mode == 3 ? 1 : 0);
    const auto initialized = rt_submit_aiv_task(1, initialize);
    const auto state = initialized.get_ref(0);
    const TaskId init = initialized.task_id();
    const MixedKernels kernels{
        config.mask & 1 ? 0 : INVALID_KERNEL_ID,
        config.mask & 2 ? 1 : INVALID_KERNEL_ID,
        config.mask & 4 ? 1 : INVALID_KERNEL_ID,
    };
    CoreTaskArgs waiter;
    waiter.add_no_dep(state, out);
    waiter.add_scalar(1, config.mask, config.blocks, config.mode == 3 ? 1 : 0);
    waiter.set_dependencies(&init, 1);
    waiter.launch_spec.set_block_num(config.blocks);
    waiter.launch_spec.set_require_sync_start(config.mode == 2);
    if (config.mode >= 3) {
        CoreTaskPredicate predicate;
        predicate.operand.tensor = &state;
        predicate.operand.ndims = 1;
        predicate.operand.indices[0] = 0;
        predicate.op = PredicateOp::GT;
        predicate.target = 0;
        waiter.set_predicate(predicate);
    }
    const TaskId waiting = rt_submit_task(kernels, waiter).task_id();
    CoreTaskArgs publisher;
    publisher.add_no_dep(state, out);
    publisher.add_scalar(2, config.mask, config.blocks, config.mode == 3 ? 1 : 0);
    publisher.set_dependencies(&init, 1);
    publisher.launch_spec.set_block_num(config.blocks);
    publisher.launch_spec.set_require_sync_start(config.mode == 2);
    (void)rt_submit_task(kernels, publisher);
    // No publisher dependency: only the deferred waiter may release this reader.
    CoreTaskArgs consumer;
    consumer.add_no_dep(state);
    consumer.add_output(out);
    consumer.add_scalar(3, config.mask, config.blocks, config.mode == 3 ? 1 : 0);
    consumer.set_dependencies(&waiting, 1);
    (void)rt_submit_aiv_task(1, consumer);
}

void graph_shape(const GraphTaskArgs &args, const ShapeConfig &config) { submit_shape(args.tensor(0).ref(), config); }
}  // namespace

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 4};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int32_t mask = static_cast<int32_t>(args.scalar<int64_t>(0));
    const int32_t mode = static_cast<int32_t>(args.scalar<int64_t>(1));
    const bool graph = args.scalar<int64_t>(2) != 0;
    const int32_t capacity = mask == 2 ? rt_available_aiv_count() : rt_available_cluster_count();
    const int32_t blocks = mode == 1 ? capacity + 1 : mode == 2 ? capacity : 1;
    const ShapeConfig config{mask, blocks, mode};
    const auto &out = args.tensor(0).ref();
    const uint32_t shape[]{64};
    for (int32_t instance = 0; instance < (graph ? 2 : 1); ++instance) {
        const uint32_t offset[]{static_cast<uint32_t>(instance) * 64};
        const auto view = out.view(shape, offset);
        if (graph) {
            GraphTaskArgs task;
            task.add_output(view);
            (void)rt_submit_graph(&graph_shape, task, config);
        } else {
            submit_shape(view, config);
        }
    }
}
