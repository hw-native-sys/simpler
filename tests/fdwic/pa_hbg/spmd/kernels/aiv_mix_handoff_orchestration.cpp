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

#include "orchestration_api.h"

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 7};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const auto blocks = static_cast<int16_t>(orch_args.scalar<int64_t>(0));
    const auto mask = static_cast<uint32_t>(orch_args.scalar<int64_t>(1));
    const auto tasks = static_cast<uint32_t>(orch_args.scalar<int64_t>(2));
    const auto delay = orch_args.scalar<int64_t>(3);
    const uint32_t shape[1] = {(static_cast<uint32_t>(blocks) * 3 + 1) * 16};
    MixedKernels kernels;
    kernels.aic_kernel_id = (mask & 1U) ? 0 : INVALID_KERNEL_ID;
    kernels.aiv0_kernel_id = (mask & 2U) ? 1 : INVALID_KERNEL_ID;
    kernels.aiv1_kernel_id = (mask & 4U) ? 2 : INVALID_KERNEL_ID;
    TaskId previous_mix = TaskId::invalid();
    for (uint32_t task = 0; task < tasks; ++task) {
        const uint32_t offset[1] = {task * shape[0]};
        auto root_state = orch_args.tensor(2).ref().view(shape, offset);
        CoreTaskArgsWithDeps<1> root;
        root.add_inout(root_state);
        root.add_scalar(static_cast<int64_t>(blocks));
        root.add_scalar(2);  // The ordinary AIV task uses record_slot_1.
        root.add_scalar(static_cast<int64_t>(task));
        root.add_scalar(delay);
        root.launch_spec.set_block_num(blocks);
        if (previous_mix.is_valid()) {
            root.add_dep(previous_mix);
        }
        const TaskId predecessor = rt_submit_aiv_task(1, root).task_id();

        auto mix_state = orch_args.tensor(0).ref().view(shape, offset);
        CoreTaskArgsWithDeps<1> mix;
        mix.add_inout(mix_state);
        mix.add_scalar(static_cast<int64_t>(blocks));
        mix.add_scalar(static_cast<int64_t>(mask));
        mix.add_scalar(static_cast<int64_t>(task));
        mix.add_scalar(delay);
        mix.launch_spec.set_block_num(blocks);
        // The record kernel ABI is one tensor followed by four scalars.
        // Each AIV has exactly one successor, exercising the direct Mix-ready path.
        mix.add_dep(predecessor);
        previous_mix = rt_submit_task(kernels, mix).task_id();
    }

    CoreTaskArgs verify;
    verify.add_input(orch_args.tensor(0).ref());
    verify.add_inout(orch_args.tensor(1).ref());
    verify.add_scalar(static_cast<int64_t>(blocks));
    verify.add_scalar(static_cast<int64_t>(mask));
    verify.add_scalar(static_cast<int64_t>(tasks));
    rt_submit_aiv_task(3, verify);
}
