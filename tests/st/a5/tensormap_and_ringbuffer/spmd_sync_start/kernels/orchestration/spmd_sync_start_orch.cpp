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
 * SPMD Sync-Start Orchestration
 *
 * Submits MIX tasks with require_sync_start=true to verify that the scheduler
 * atomically launches all blocks before any can run.
 *
 * Tasks:
 *   T0: block_num=2,  require_sync_start=true   (basic sync launch)
 *   T1: block_num=8,  require_sync_start=true   (larger batch)
 *   T2: block_num=2,  require_sync_start=false  (normal, as baseline)
 *   T3: block_num=12, require_sync_start=true   (cross-thread batch)
 *
 * Each block writes float(block_idx) to its allocated cache-line slot,
 * identical to spmd_multiblock_mix so the same kernel binaries can be reused.
 *
 * Args layout: [output]
 */

#include <stddef.h>
#include <stdint.h>

#include "orchestration_api.h"

#define FUNC_SPMD_MIX_AIC 0
#define FUNC_SPMD_MIX_AIV0 1
#define FUNC_SPMD_MIX_AIV1 2

static constexpr int32_t SLOTS_PER_BLOCK = 3;

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return OrchestrationConfig{
        .expected_arg_count = 2,
    };
}

// The cohort widths are fractions of this run's cluster count rather than
// literals: the run always takes the whole device, and that width differs
// between sim and silicon. The shape the case exercises — two narrow cohorts,
// one mid, one wide enough to contend for most of the device — is what matters,
// so each is derived and clamped to at least one block.
static int16_t cohort(int32_t clusters, int32_t divisor) {
    int32_t n = clusters / divisor;
    return static_cast<int16_t>(n < 1 ? 1 : n);
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

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &ext_output = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &layout = orch_args.tensor(1).ref();

    const int32_t clusters = rt_available_cluster_count();
    // T0/T2 narrow, T1 mid, T3 wide; T2 is the non-sync_start baseline.
    const int16_t block_nums[4] = {
        cohort(clusters, 12), cohort(clusters, 3), cohort(clusters, 12), cohort(clusters, 2)
    };
    const bool sync_start[4] = {true, true, false, true};

    // layout[2i] = block_num, layout[2i+1] = base cache line. The host cannot
    // predict either, so the run reports the geometry it used and the golden is
    // rebuilt from it.
    int32_t base_cl = 0;
    for (int32_t i = 0; i < 4; i++) {
        submit_mix(ext_output, block_nums[i], base_cl, sync_start[i]);
        uint32_t idx[1] = {static_cast<uint32_t>(2 * i)};
        set_tensor_data<int32_t>(layout, 1, idx, block_nums[i]);
        idx[0] = static_cast<uint32_t>(2 * i + 1);
        set_tensor_data<int32_t>(layout, 1, idx, base_cl);
        base_cl += block_nums[i] * SLOTS_PER_BLOCK;
    }

    LOG_INFO("[spmd_sync_start] Submitted 4 tasks over %d clusters", clusters);
}

}  // extern "C"
