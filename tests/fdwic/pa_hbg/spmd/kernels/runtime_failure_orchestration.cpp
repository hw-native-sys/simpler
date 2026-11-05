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
    return OrchestrationConfig{.expected_arg_count = 6};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const auto mask = static_cast<uint32_t>(orch_args.scalar<int64_t>(1));
    // A negative block count asks for two full waves of this device's own
    // topology, which the host cannot read before the run. One wave is one
    // block per execution position, and a cluster offers two of them when a
    // single AIV lane carries the task and one otherwise.
    int32_t requested = static_cast<int32_t>(orch_args.scalar<int64_t>(0));
    if (requested < 0) {
        const int32_t positions = rt_available_cluster_count() * ((mask == 2U || mask == 4U) ? 2 : 1);
        requested = positions * 2;
    }
    const auto blocks = static_cast<int16_t>(requested);
    CoreTaskArgs root;
    root.add_inout(orch_args.tensor(0).ref());
    root.add_scalar(static_cast<int64_t>(blocks));
    for (int32_t scalar = 1; scalar < 4; ++scalar)
        root.add_scalar(orch_args.scalar<int64_t>(scalar));
    root.launch_spec.set_block_num(blocks);
    const MixedKernels kernels{
        (mask & 1U) != 0 ? 0 : INVALID_KERNEL_ID,
        (mask & 2U) != 0 ? 1 : INVALID_KERNEL_ID,
        (mask & 4U) != 0 ? 2 : INVALID_KERNEL_ID,
    };
    rt_submit_task(kernels, root);

    CoreTaskArgs verify;
    verify.add_input(orch_args.tensor(0).ref());
    verify.add_inout(orch_args.tensor(1).ref());
    verify.add_scalar(static_cast<int64_t>(blocks));
    verify.add_scalar(static_cast<int64_t>(mask));
    verify.add_scalar(INT64_C(1));
    rt_submit_aiv_task(3, verify);
}
