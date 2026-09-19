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
 * available_aicore_counts: spend the counts rt_available_*_count() reports.
 *
 * The counts are an AICPU-side runtime query with no host-side equivalent, so
 * the run is made self-describing rather than compared against a pinned number:
 * `shape` carries the reported cluster / AIV counts, and a MIX cohort of
 * exactly `cluster_count` blocks writes its block index into `blocks`. The host
 * reads the width out of `shape` and checks that many block slots.
 *
 * require_sync_start on the cohort is what makes the count falsifiable: the
 * cohort needs every block co-resident, so an over-reported cluster count trips
 * the sync-start deadlock guard instead of silently passing.
 */

#include <stdint.h>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_SPMD_MIX_AIC 0
#define FUNC_SPMD_MIX_AIV0 1
#define FUNC_SPMD_MIX_AIV1 2

// LaunchSpec spells the SPMD block-count setter differently per arch
// (a5: set_core_num, a2a3: set_block_num) for the same field. Bridge it so this
// one fixture compiles on both; keyed off the arch's types.h include guard,
// which orchestration_api.h pulls in transitively.
static inline void set_block_count(CoreTaskArgs &args, int16_t n) {
#if defined(SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_TYPES_H_)
    args.launch_spec.set_core_num(n);
#else
    args.launch_spec.set_block_num(n);
#endif
}

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{
        .expected_arg_count = 2,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &blocks = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &shape = orch_args.tensor(1).ref();

    const int32_t cluster_count = rt_available_cluster_count();
    const int32_t aiv_count = rt_available_aiv_count();
    LOG_INFO("[available_aicore_counts] clusters=%d aiv=%d", cluster_count, aiv_count);

    MixedKernels mk;
    mk.aic_kernel_id = FUNC_SPMD_MIX_AIC;
    mk.aiv0_kernel_id = FUNC_SPMD_MIX_AIV0;
    mk.aiv1_kernel_id = FUNC_SPMD_MIX_AIV1;

    CoreTaskArgs args;
    args.add_inout(blocks);
    args.add_scalar(static_cast<int64_t>(0));  // base cache line
    set_block_count(args, static_cast<int16_t>(cluster_count));
    args.launch_spec.set_require_sync_start(true);
    rt_submit_task(mk, args);

    // shape carries no producer or consumer task, so set_tensor_data writes it
    // straight through. Giving it one would hang host_build_graph: its
    // orchestrator runs to completion on the host before the device executes
    // anything, so a producer's task_state can never reach COMPLETED and
    // wait_for_tensor_ready would spin until the tensor-data deadline.
    uint32_t idx[1] = {0};
    set_tensor_data<int32_t>(shape, 1, idx, cluster_count);
    idx[0] = 1;
    set_tensor_data<int32_t>(shape, 1, idx, aiv_count);
}

}  // extern "C"
