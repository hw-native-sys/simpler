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
 *   4 x normal MIX  (block_num=4,  sync=false) -> 4 x 4 x 3 = 48 CL
 *   2 x sync   MIX  (block_num=12, sync=true)  -> 2 x 12 x 3 = 72 CL
 *   2 x sync   AIV  (block_num=8,  sync=true)  -> 2 x 8 x 1 = 16 CL
 *   1 x normal AIV  (block_num=4,  sync=false) -> 1 x 4 x 1 = 4 CL
 *   Round total: 140 CL
 *
 * 6 rounds -> 54 tasks total, 840 CL grand total.
 *
 * Args layout: [output]
 */

#include <stddef.h>
#include <stdint.h>

#include "orchestration_api.h"

#define FUNC_SPMD_MIX_AIC 0
#define FUNC_SPMD_MIX_AIV0 1
#define FUNC_SPMD_MIX_AIV1 2
#define FUNC_SPMD_WRITE_AIV 3

static constexpr int32_t MIX_SLOTS = 3;
// Cohort widths scale with this run's cluster count, keeping the shape the
// case exercises (a mix of narrow and near-capacity cohorts contending on the
// drain path) on any device. At 24 clusters these are the original 4/12/8/4.
static int16_t cohort(int32_t total, int32_t divisor) {
    int32_t n = total / divisor;
    return static_cast<int16_t>(n < 1 ? 1 : n);
}
static constexpr int32_t ROUNDS = 6;

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 1};
}

static void submit_mix(const simpler::tmr::Tensor &out, int16_t block_num, int64_t base_cl, bool sync_start) {
    MixedKernels mk;
    mk.aic_kernel_id = FUNC_SPMD_MIX_AIC;
    mk.aiv0_kernel_id = FUNC_SPMD_MIX_AIV0;
    mk.aiv1_kernel_id = FUNC_SPMD_MIX_AIV1;
    CoreTaskArgs args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.launch_spec.set_block_num(block_num);
    args.launch_spec.set_require_sync_start(sync_start);
    rt_submit_task(mk, args);
}

static void submit_aiv(const simpler::tmr::Tensor &out, int16_t block_num, int64_t base_cl, bool sync_start) {
    CoreTaskArgs args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.launch_spec.set_block_num(block_num);
    args.launch_spec.set_require_sync_start(sync_start);
    rt_submit_aiv_task(FUNC_SPMD_WRITE_AIV, args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &ext_output = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &layout = orch_args.tensor(1).ref();

    const int32_t total = rt_available_cluster_count();
    const int16_t normal_mix_bn = cohort(total, 6);
    const int16_t sync_mix_bn = cohort(total, 2);
    const int16_t sync_aiv_bn = cohort(total, 3);
    const int16_t normal_aiv_bn = cohort(total, 6);

    // The host replays this same round structure from the four widths; it cannot
    // predict them, so the run reports what it used.
    uint32_t idx[1] = {0};
    set_tensor_data<int32_t>(layout, 1, idx, normal_mix_bn);
    idx[0] = 1;
    set_tensor_data<int32_t>(layout, 1, idx, sync_mix_bn);
    idx[0] = 2;
    set_tensor_data<int32_t>(layout, 1, idx, sync_aiv_bn);
    idx[0] = 3;
    set_tensor_data<int32_t>(layout, 1, idx, normal_aiv_bn);

    int64_t cl = 0;

    for (int32_t r = 0; r < ROUNDS; r++) {
        // 4 × normal MIX
        for (int i = 0; i < 4; i++, cl += normal_mix_bn * MIX_SLOTS)
            submit_mix(ext_output, normal_mix_bn, cl, false);

        // 2 × sync MIX — CAS contention: second sync task may arrive while first is draining
        for (int i = 0; i < 2; i++, cl += sync_mix_bn * MIX_SLOTS)
            submit_mix(ext_output, sync_mix_bn, cl, true);

        // 2 × sync AIV — cross-shape drain contention with the MIX drain above
        for (int i = 0; i < 2; i++, cl += sync_aiv_bn)
            submit_aiv(ext_output, sync_aiv_bn, cl, true);

        // 1 × normal AIV
        submit_aiv(ext_output, normal_aiv_bn, cl, false);
        cl += normal_aiv_bn;
    }

    LOG_INFO("[spmd_sync_start_stress] Submitted %d tasks over %d rounds", 9 * ROUNDS, ROUNDS);
}

}  // extern "C"
