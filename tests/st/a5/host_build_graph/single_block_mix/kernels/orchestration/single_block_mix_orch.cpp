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

constexpr int32_t kMatmulFuncId = 0;
constexpr int32_t kAddFuncId = 1;
constexpr int32_t kMulFuncId = 2;
constexpr int32_t kTaskCount = 16;
constexpr uint32_t kTileElems = 128 * 128;

TaskId submit_mix(
    const ChipTaskArgs &orch_args, int32_t task_index, uint8_t mask, const TaskId *dependency, uint32_t dependency_count
) {
    const uint32_t shape[1] = {kTileElems};
    const uint32_t offset[1] = {static_cast<uint32_t>(task_index) * kTileElems};
    simpler::hbg::Tensor c = orch_args.tensor(2).ref().view(shape, offset);
    simpler::hbg::Tensor f = orch_args.tensor(5).ref().view(shape, offset);
    simpler::hbg::Tensor i = orch_args.tensor(8).ref().view(shape, offset);

    const uint32_t previous_offset[1] = {static_cast<uint32_t>(task_index == 0 ? 0 : task_index - 1) * kTileElems};
    const bool chained = dependency_count != 0;
    auto a = chained && (mask & 1U) != 0 ? orch_args.tensor(2).ref().view(shape, previous_offset) :
                                           orch_args.tensor(0).ref();
    auto d = chained && (mask & 2U) != 0 ? orch_args.tensor(5).ref().view(shape, previous_offset) :
                                           orch_args.tensor(3).ref();
    auto g = chained && (mask & 4U) != 0 ? orch_args.tensor(8).ref().view(shape, previous_offset) :
                                           orch_args.tensor(6).ref();
    CoreTaskArgs args;
    args.add_input(a);
    args.add_input(orch_args.tensor(1).ref());
    args.add_output(c);
    args.add_input(d);
    args.add_input(orch_args.tensor(4).ref());
    args.add_output(f);
    args.add_input(g);
    args.add_input(orch_args.tensor(7).ref());
    args.add_output(i);
    args.set_dependencies(dependency, dependency_count);

    MixedKernels kernels;
    kernels.aic_kernel_id = (mask & 1U) != 0 ? kMatmulFuncId : INVALID_KERNEL_ID;
    kernels.aiv0_kernel_id = (mask & 2U) != 0 ? kAddFuncId : INVALID_KERNEL_ID;
    kernels.aiv1_kernel_id = (mask & 4U) != 0 ? kMulFuncId : INVALID_KERNEL_ID;
    return rt_submit_task(kernels, args).task_id();
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 11};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const int64_t graph_case = orch_args.scalar<int64_t>(0);
    if (graph_case < 0 || graph_case > 2) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "unsupported graph case %ld", graph_case);
        return;
    }

    const uint8_t mask = static_cast<uint8_t>(orch_args.scalar<int64_t>(1));
    TaskId previous = TaskId::invalid();
    for (int32_t task_index = 0; task_index < kTaskCount; ++task_index) {
        const bool chained = graph_case == 0 && task_index != 0;
        previous = submit_mix(
            orch_args, task_index, graph_case == 2 && task_index % 2 ? (task_index % 4 == 1 ? 1U : 2U) : mask,
            chained ? &previous : nullptr, chained ? 1U : 0U
        );
    }
}

}  // extern "C"
