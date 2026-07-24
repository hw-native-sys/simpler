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

#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_SPMD_MIX_AIC 0
#define FUNC_SPMD_MIX_AIV0 1
#define FUNC_SPMD_MIX_AIV1 2

static constexpr int16_t HOLDER_BLOCKS = 1;
static constexpr int16_t SYNC_BLOCKS = 24;
static constexpr int64_t HOLDER_SPIN_ITERS = 50000000;

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 3};
}

static MixedKernels mix_kernels() {
    MixedKernels mk;
    mk.aic_kernel_id = FUNC_SPMD_MIX_AIC;
    mk.aiv0_kernel_id = FUNC_SPMD_MIX_AIV0;
    mk.aiv1_kernel_id = FUNC_SPMD_MIX_AIV1;
    return mk;
}

static void submit_mix(
    const Tensor &out, const Tensor &scratch, int16_t block_num, int64_t base_cl, int64_t spin_iters, bool sync_start
) {
    L0TaskArgs args;
    args.add_inout(out);
    args.add_input(scratch);
    args.add_scalar(base_cl);
    args.add_scalar(spin_iters);
    args.launch_spec.set_block_num(block_num);
    args.launch_spec.set_require_sync_start(sync_start);
    rt_submit_task(mix_kernels(), args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &output = orch_args.tensor(0).ref();
    const Tensor &holder = orch_args.tensor(1).ref();
    const Tensor &scratch = orch_args.tensor(2).ref();

    submit_mix(holder, scratch, HOLDER_BLOCKS, 0, HOLDER_SPIN_ITERS, false);
    submit_mix(output, scratch, SYNC_BLOCKS, 0, 0, true);

    LOG_INFO_V9("[spmd_sync_start_drain_aba_hook] Submitted holder + sync_start drain reproducer");
}

}  // extern "C"
