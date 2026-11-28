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
constexpr uint32_t spmd = 0;
constexpr uint32_t mixed_graph = 1;
constexpr uint32_t pending = 2;
constexpr uint32_t pending_aiv = 3;
constexpr uint32_t release_gate = 4;
constexpr uint32_t pending_blocks = 128;

TaskId consume(
    const simpler::hbg::Tensor &input, const simpler::hbg::Tensor &out, uint32_t count, const TaskId *deps = nullptr,
    uint32_t ndeps = 0
) {
    CoreTaskArgs task;
    task.add_input(input);
    task.add_output(out);
    task.add_scalar(count);
    task.set_dependencies(deps, ndeps);
    return rt_submit_aiv_task(1, task).task_id();
}

void submit_top_level(const ChipTaskArgs &args, uint32_t mode, uint32_t blocks) {
    const uint32_t size = args.scalar<uint32_t>(1);
    const auto &out = args.tensor(2).ref();
    CoreTaskArgs gets;
    gets.add_no_dep(args.tensor(0).ref(), args.tensor(1).ref(), out);
    gets.add_scalar(args.scalar<uint64_t>(0), size, mode, blocks);
    gets.launch_spec.set_block_num(blocks);
    gets.launch_spec.set_require_sync_start(mode == spmd);
    auto transfer = rt_submit_aiv_task(0, gets).task_id();

    if (mode == pending || mode == pending_aiv) {
        // Every AIV consumer waits on all deferred blocks.
        const uint32_t shape[]{size / blocks};
        TaskId consumers[pending_blocks];
        for (uint32_t i = 0; i < blocks; ++i) {
            const uint32_t offset[]{i * shape[0]};
            consumers[i] =
                consume(out.view(shape, offset), args.tensor(3).ref().view(shape, offset), shape[0], &transfer, 1);
        }
        CoreTaskArgs notifier;
        notifier.add_no_dep(args.tensor(0).ref(), args.tensor(1).ref(), out);
        notifier.add_scalar(args.scalar<uint64_t>(0), size, mode, blocks);
        auto observed = rt_submit_aic_task(2, notifier).task_id();
        if (mode == pending_aiv) {
            // The AIC observes pending work. Only this AIV can publish the gate.
            CoreTaskArgs release;
            release.add_no_dep(args.tensor(0).ref(), args.tensor(1).ref(), out);
            release.add_scalar(args.scalar<uint64_t>(0), size, release_gate, blocks);
            release.set_dependencies(&observed, 1);
            rt_submit_aiv_task(4, release);

            CoreTaskArgs cohort;
            cohort.add_no_dep(args.tensor(0).ref(), args.tensor(1).ref(), out);
            cohort.add_scalar(args.scalar<uint64_t>(0), size, mixed_graph, uint32_t{2});
            cohort.set_dependencies(consumers, blocks);
            cohort.launch_spec.set_block_num(2);
            cohort.launch_spec.set_require_sync_start(true);
            rt_submit_aiv_task(4, cohort);
        }
    } else {
        consume(out, args.tensor(3).ref(), size, &transfer, 1);
    }
}

void predicate_body(const GraphTaskArgs &args) {
    const uint32_t shape[]{32};
    CoreTaskArgs gate_args;
    const TensorCreateInfo gate_output(shape, 1, DataType::INT32);
    gate_args.add_output(gate_output);
    gate_args.add_scalar(args.scalar(2));
    auto gate = rt_submit_aic_task(3, gate_args);
    auto gate_tensor = gate.get_ref(0);
    auto gate_id = gate.task_id();
    CoreTaskPredicate pred;
    pred.operand.tensor = &gate_tensor;
    pred.operand.ndims = 1;
    pred.operand.indices[0] = 0;
    pred.op = PredicateOp::EQ;
    pred.target = 1;
    CoreTaskArgs gets;
    gets.add_input(args.tensor(0).ref(), gate_tensor);
    gets.add_inout(args.tensor(1).ref());
    gets.add_scalar(args.scalar(0), args.scalar(1), spmd, uint32_t{1});
    gets.set_dependencies(&gate_id, 1);
    gets.set_predicate(pred);
    rt_submit_aiv_task(0, gets);
}

void mix_body(const GraphTaskArgs &args) {
    MixedKernels kernels;
    kernels.aic_kernel_id = 2;
    kernels.aiv0_kernel_id = 0;
    kernels.aiv1_kernel_id = 4;
    CoreTaskArgs gets;
    gets.add_input(args.tensor(0).ref(), args.tensor(1).ref());
    gets.add_inout(args.tensor(2).ref());
    gets.add_scalar(args.scalar(0), args.scalar(1), mixed_graph, uint32_t{2});
    gets.launch_spec.set_block_num(2);
    gets.launch_spec.set_require_sync_start(true);
    rt_submit_task(kernels, gets);
}
}  // namespace

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 8};
}

extern "C" __attribute__((visibility("default"))) void spmd_get(const ChipTaskArgs &args) {
    submit_top_level(args, spmd, 4);
}

extern "C" __attribute__((visibility("default"))) void pending_gets(const ChipTaskArgs &args) {
    submit_top_level(args, pending, pending_blocks);
}

extern "C" __attribute__((visibility("default"))) void pending_gets_aiv(const ChipTaskArgs &args) {
    submit_top_level(args, pending_aiv, pending_blocks);
}

extern "C" __attribute__((visibility("default"))) void predicated_graph(const ChipTaskArgs &args) {
    const uint32_t size = args.scalar<uint32_t>(1);
    const uint32_t shape[]{size / 2};
    // Leave a known local value in every destination before the false predicate skips the GET.
    const auto seeded = consume(args.tensor(0).ref(), args.tensor(2).ref(), size);
    for (uint32_t active : {0u, 1u}) {
        const uint32_t offset[]{active * shape[0]};
        const auto source = args.tensor(0).ref().view(shape, offset);
        const auto out = args.tensor(2).ref().view(shape, offset);
        const auto result = args.tensor(3).ref().view(shape, offset);
        GraphTaskArgs invocation;
        invocation.add_input(source);
        invocation.add_inout(out);
        invocation.add_scalar(args.scalar<uint64_t>(0), shape[0], active);
        invocation.set_dependencies(&seeded, 1);
        const auto graph = rt_submit_graph(predicate_body, invocation);
        consume(out, result, shape[0], &graph.task_id, 1);
    }
}

extern "C" __attribute__((visibility("default"))) void graph_mix(const ChipTaskArgs &args) {
    const uint32_t size = args.scalar<uint32_t>(1);
    GraphTaskArgs invocation;
    invocation.add_input(args.tensor(0).ref(), args.tensor(1).ref());
    invocation.add_inout(args.tensor(2).ref());
    invocation.add_scalar(args.scalar<uint64_t>(0), size);
    const auto graph = rt_submit_graph(mix_body, invocation);
    consume(args.tensor(2).ref(), args.tensor(3).ref(), size, &graph.task_id, 1);
}
