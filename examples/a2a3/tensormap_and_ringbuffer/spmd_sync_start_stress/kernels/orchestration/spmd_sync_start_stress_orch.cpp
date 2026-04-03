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
 * SPMD Sync-Start Stress Orchestration (mixed shapes)
 *
 * Submits 6 rounds of mixed MIX + AIV tasks to stress-test:
 *   - Drain CAS contention (multiple sync_start tasks per round)
 *   - Ack barrier correctness (normal tasks occupy clusters during drain entry)
 *   - State cleanup between consecutive drain cycles
 *
 * Each round (9 tasks):
 *   4 × normal MIX  (block_num=4,  sync=false) -> 4 × 4 × 3 = 48 CL
 *   2 × sync   MIX  (block_num=12, sync=true)  -> 2 × 12 × 3 = 72 CL
 *   2 × sync   AIV  (block_num=8,  sync=true)  -> 2 × 8 × 1 = 16 CL
 *   1 × normal AIV  (block_num=4,  sync=false) -> 1 × 4 × 1 = 4 CL
 *   Round total: 140 CL
 *
 * 6 rounds → 54 tasks total, 840 CL grand total.
 *
 * Args layout: [output]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_SPMD_MIX_AIC 0
#define FUNC_SPMD_MIX_AIV0 1
#define FUNC_SPMD_MIX_AIV1 2
#define FUNC_SPMD_WRITE_AIV 3

static constexpr int32_t MIX_SLOTS = 3;
static constexpr int32_t NORMAL_MIX_BN = 4;
static constexpr int32_t SYNC_MIX_BN = 12;
static constexpr int32_t SYNC_AIV_BN = 8;
static constexpr int32_t NORMAL_AIV_BN = 4;
static constexpr int32_t ROUNDS = 6;

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 1};
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
    pto2_rt_submit_task(mk, args);
}

static void submit_aiv(Tensor &out, int16_t block_num, int64_t base_cl, bool sync_start) {
    Arg args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.launch_spec.set_block_num(block_num);
    args.launch_spec.set_require_sync_start(sync_start);
    pto2_rt_submit_aiv_task(FUNC_SPMD_WRITE_AIV, args);
}

__attribute__((visibility("default"))) void
aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;
    if (orch_thread_index != 0) return;

    Tensor ext_output = from_tensor_arg(orch_args.tensor(0));

    int64_t cl = 0;

    for (int32_t r = 0; r < ROUNDS; r++) {
        // 4 × normal MIX
        for (int i = 0; i < 4; i++, cl += NORMAL_MIX_BN * MIX_SLOTS)
            submit_mix(ext_output, NORMAL_MIX_BN, cl, false);

        // 2 × sync MIX — CAS contention: second sync task may arrive while first is draining
        for (int i = 0; i < 2; i++, cl += SYNC_MIX_BN * MIX_SLOTS)
            submit_mix(ext_output, SYNC_MIX_BN, cl, true);

        // 2 × sync AIV — cross-shape drain contention with the MIX drain above
        for (int i = 0; i < 2; i++, cl += SYNC_AIV_BN)
            submit_aiv(ext_output, SYNC_AIV_BN, cl, true);

        // 1 × normal AIV
        submit_aiv(ext_output, NORMAL_AIV_BN, cl, false);
        cl += NORMAL_AIV_BN;
    }

    LOG_ALWAYS("[spmd_sync_start_stress] Submitted %d tasks over %d rounds", 9 * ROUNDS, ROUNDS);
}

}  // extern "C"
