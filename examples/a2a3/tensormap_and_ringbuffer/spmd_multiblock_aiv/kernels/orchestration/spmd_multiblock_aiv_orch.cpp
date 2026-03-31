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
 * SPMD Multi-Block AIV Orchestration
 *
 * Submits three AIV tasks with increasing block_num to exercise:
 *   T0: block_num=4   — fits within a single sched thread
 *   T1: block_num=16  — saturates one sched thread (8 clusters × 2 AIV)
 *   T2: block_num=24  — forces cross-thread re-push via ready_queue
 *
 * Each task writes to a disjoint region of the output tensor using the
 * base_cl scalar to offset the block writes.
 *
 * Args layout: [output]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_SPMD_WRITE_AIV 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(
    const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

static void submit_spmd_aiv(int32_t kernel_id, Tensor& out, int16_t block_num, int64_t base_cl) {
    Arg args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.launch_spec.set_block_num(block_num);
    pto2_rt_submit_aiv_task(kernel_id, args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(
    const ChipStorageTaskArgs& orch_args, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;  // NOLINT(readability/casting)
    if (orch_thread_index != 0) return;

    Tensor ext_output = from_tensor_arg(orch_args.tensor(0));

    // T0: 4 blocks — basic multi-block
    submit_spmd_aiv(FUNC_SPMD_WRITE_AIV, ext_output, 4, 0);

    // T1: 16 blocks — saturate one sched thread's AIV cores (8 clusters × 2 AIV)
    submit_spmd_aiv(FUNC_SPMD_WRITE_AIV, ext_output, 16, 4);

    // T2: 24 blocks — cross-thread dispatch via ready_queue re-push
    submit_spmd_aiv(FUNC_SPMD_WRITE_AIV, ext_output, 24, 20);

    // T3: 48 blocks — occupy all AIV cores across all 3 sched threads
    submit_spmd_aiv(FUNC_SPMD_WRITE_AIV, ext_output, 48, 44);

    // T4: 96 blocks — two full rounds of all AIV cores
    submit_spmd_aiv(FUNC_SPMD_WRITE_AIV, ext_output, 96, 92);

    LOG_ALWAYS("[spmd_multiblock_aiv] Submitted 5 AIV tasks: block_num=4,16,24,48,96");
}

}  // extern "C"
