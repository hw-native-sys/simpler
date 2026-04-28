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
 * SPMD Starvation Prevention Orchestration
 *
 * Submits a large wave of normal MIX tasks followed by sync_start tasks,
 * then another wave of normal tasks.  The drain mechanism must ensure the
 * sync_start tasks are not indefinitely delayed by the surrounding load.
 *
 * Layout: 3 waves × 6 normal tasks (block_num=4) + 2 sync_start tasks (block_num=6)
 *
 * Normal task: block_num=4, require_sync_start=false  → 4 blocks × 3 slots = 12 CL each
 * Sync task:   block_num=6, require_sync_start=true   → 6 blocks × 3 slots = 18 CL each
 *
 * Total CL: 3×6×12 + 2×18 = 216 + 36 = 252
 *
 * Args layout: [output]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_SPMD_MIX_AIC 0
#define FUNC_SPMD_MIX_AIV0 1
#define FUNC_SPMD_MIX_AIV1 2

static constexpr int32_t SLOTS_PER_BLOCK = 3;  // AIC, AIV0, AIV1
static constexpr int32_t NORMAL_BLOCK_NUM = 4;
static constexpr int32_t SYNC_BLOCK_NUM = 6;
static constexpr int32_t NORMAL_CL = NORMAL_BLOCK_NUM * SLOTS_PER_BLOCK;  // 12
static constexpr int32_t SYNC_CL = SYNC_BLOCK_NUM * SLOTS_PER_BLOCK;      // 18

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

static void submit_mix(Tensor &out, int16_t block_num, int64_t base_cl, bool sync_start) {
    MixedKernels mk;
    mk.aic_kernel_id = FUNC_SPMD_MIX_AIC;
    mk.aiv0_kernel_id = FUNC_SPMD_MIX_AIV0;
    mk.aiv1_kernel_id = FUNC_SPMD_MIX_AIV1;

    Arg args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.launch_spec.set_block_num(block_num);
    args.launch_spec.set_require_sync_start(sync_start);
    rt_submit_task(mk, args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_output = from_tensor_arg(orch_args.tensor(0));

    int64_t cl = 0;

    // Wave 1: 6 normal MIX tasks
    for (int i = 0; i < 6; i++, cl += NORMAL_CL)
        submit_mix(ext_output, NORMAL_BLOCK_NUM, cl, false);

    // Sync-start task 0: must not be starved by wave 1 or wave 2
    submit_mix(ext_output, SYNC_BLOCK_NUM, cl, true);
    cl += SYNC_CL;

    // Wave 2: 6 normal MIX tasks
    for (int i = 0; i < 6; i++, cl += NORMAL_CL)
        submit_mix(ext_output, NORMAL_BLOCK_NUM, cl, false);

    // Sync-start task 1: must not be starved by wave 2 or wave 3
    submit_mix(ext_output, SYNC_BLOCK_NUM, cl, true);
    cl += SYNC_CL;

    // Wave 3: 6 normal MIX tasks
    for (int i = 0; i < 6; i++, cl += NORMAL_CL)
        submit_mix(ext_output, NORMAL_BLOCK_NUM, cl, false);

    LOG_ALWAYS("[spmd_starvation] Submitted 20 tasks (18 normal + 2 sync_start)");
}

}  // extern "C"
