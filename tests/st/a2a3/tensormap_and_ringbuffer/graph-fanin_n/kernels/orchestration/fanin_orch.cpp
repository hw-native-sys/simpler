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
 * Fan-In Orchestration (tensormap_and_ringbuffer Runtime)
 *
 * Builds a fan-in DAG: N independent producer tasks -> 1 barrier task.
 *
 *   seed -> [Producer_0] -> prod_out_0 -.
 *   seed -> [Producer_1] -> prod_out_1 -+-> [Barrier] -> result
 *   ...                                 |
 *   seed -> [Producer_{N-1}] -> prod_out_{N-1} -'
 *
 * Each producer reads seed (INPUT) and writes to an independent runtime
 * tensor (OUTPUT). The barrier task reads all N producer outputs (INPUT
 * for dependency tracking) and writes to result (INOUT).
 *
 * The barrier kernel (FUNC_NOOP) only uses args[0] (result INOUT).
 * Producer output refs at args[1..N] are unused by the kernel but create
 * runtime dependencies that force the barrier to wait for all producers.
 *
 * Tests: dependency convergence overhead, tracking N predecessors efficiently.
 *
 * Arg layout: [seed, result, fanin_width]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_INC 0
#define FUNC_NOOP 1

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
    int fanin_width = static_cast<int>(orch_args.scalar(0));

    LOG_ALWAYS("[fanin_N] fanin_width=%d", fanin_width);

    uint32_t scalar_shape[1] = {1};
    TensorCreateInfo ci(scalar_shape, 1, DataType::FLOAT32);

    // Build barrier args incrementally: result (INOUT) + all producer outputs (INPUT)
    Arg barrier_params;
    barrier_params.add_inout(result);

    // Submit N independent producers, collecting their output refs
    for (int i = 0; i < fanin_width; i++) {
        Arg p;
        p.add_input(seed);
        p.add_output(ci);
        TaskOutputTensors outs = pto2_rt_submit_aiv_task(FUNC_INC, p);
        barrier_params.add_input(outs.get_ref(0));
    }

    // Barrier task: waits for all producers, then increments result
    pto2_rt_submit_aiv_task(FUNC_NOOP, barrier_params);

    LOG_ALWAYS("[fanin_N] Submitted %d producers + 1 barrier task", fanin_width);
}

}  // extern "C"
