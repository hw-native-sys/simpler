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
/**
 * Fan-Out Orchestration (tensormap_and_ringbuffer Runtime)
 *
 * Builds a fan-out DAG: 1 source task -> N independent consumer tasks.
 *
 *   seed -> [Source] -> intermediate -> [Consumer_0] -> result[0]
 *                                    -> [Consumer_1] -> result[1]
 *                                    -> ...
 *                                    -> [Consumer_{N-1}] -> result[N-1]
 *
 * Source produces a runtime tensor via OUTPUT. All N consumers read
 * that tensor (INPUT) and write to separate cache-line-aligned slots
 * in the result tensor (INOUT).
 *
 * Tests: parallel dispatch capability, core utilization with N independent
 * ready-to-run tasks.
 *
 * Arg layout: [seed, result, fanout_width]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_INC 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(
    const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(
    const ChipStorageTaskArgs& orch_args, int orch_thread_num, int orch_thread_index) {
    if (orch_thread_index != 0) {
        return;
    }

    Tensor seed = from_tensor_arg(orch_args.tensor(0));
    Tensor result = from_tensor_arg(orch_args.tensor(1));
    int fanout_width = static_cast<int>(orch_args.scalar(0));

    LOG_ALWAYS("[fanout_N] fanout_width=%d", fanout_width);

    uint32_t scalar_shape[1] = {1};
    TensorCreateInfo ci(scalar_shape, 1, DataType::FLOAT32);

    // Source task: seed -> intermediate
    Arg src_params;
    src_params.add_input(seed);
    src_params.add_output(ci);
    TaskOutputTensors source_outs = pto2_rt_submit_aiv_task(FUNC_INC, src_params);

    // Consumer tasks: each reads source output, writes to separate result slot
    constexpr uint32_t CACHE_LINE_ELEMS = 16;
    uint32_t slot_shape[1] = {1};

    for (int i = 0; i < fanout_width; i++) {
        uint32_t view_offsets[1] = {static_cast<uint32_t>(i * CACHE_LINE_ELEMS)};
        Tensor result_slot = result.view(slot_shape, view_offsets);

        Arg params;
        params.add_input(source_outs.get_ref(0));
        params.add_inout(result_slot);
        pto2_rt_submit_aiv_task(FUNC_INC, params);
    }

    LOG_ALWAYS("[fanout_N] Submitted 1 source + %d consumer tasks", fanout_width);
}

}  // extern "C"
