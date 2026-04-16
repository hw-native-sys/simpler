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
 * SPMD Sync-Start AIV Orchestration
 *
 * Submits AIV-only tasks with require_sync_start=true to exercise:
 *   - AIV fast path: count_idle_aiv_cores() >= block_num (small block_num)
 *   - AIV drain path: block_num exceeds local AIV cores (cross-thread drain)
 *
 * Tasks:
 *   T0: block_num=4,  require_sync_start=true   (fast path)
 *   T1: block_num=16, require_sync_start=true   (saturate one thread: 8 clusters x 2 AIV)
 *   T2: block_num=4,  require_sync_start=false  (baseline)
 *   T3: block_num=24, require_sync_start=true   (cross-thread drain)
 *
 * Each block writes float(block_idx) at (base_cl + block_idx) x FLOATS_PER_CACHE_LINE,
 * reusing the kernel from spmd_multiblock_aiv.
 *
 * Args layout: [output]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_SPMD_WRITE_AIV 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

static void submit_aiv(Tensor &out, int16_t block_num, int64_t base_cl, bool sync_start) {
    Arg args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.launch_spec.set_core_num(block_num);
    args.launch_spec.set_require_sync_start(sync_start);
    pto2_rt_submit_aiv_task(FUNC_SPMD_WRITE_AIV, args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_output = from_tensor_arg(orch_args.tensor(0));

    // T0: 4 blocks, sync_start=true (fast path: 4 <= idle AIV cores on one thread)
    submit_aiv(ext_output, 4, 0, true);
    // T1: 16 blocks, sync_start=true (saturate: 8 clusters x 2 AIV = 16 cores)
    submit_aiv(ext_output, 16, 4, true);
    // T2: 4 blocks, sync_start=false (baseline)
    submit_aiv(ext_output, 4, 20, false);
    // T3: 24 blocks, sync_start=true (cross-thread drain)
    submit_aiv(ext_output, 24, 24, true);

    LOG_ALWAYS("[spmd_sync_start_aiv] Submitted 4 AIV tasks (3 sync_start + 1 baseline)");
}

}  // extern "C"
