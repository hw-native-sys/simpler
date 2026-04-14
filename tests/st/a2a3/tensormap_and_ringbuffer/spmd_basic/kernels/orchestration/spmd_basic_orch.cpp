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
 * SPMD Basic Orchestration
 *
 * Submits a single MIX task (AIC + AIV0 + AIV1) with a shared output
 * tensor. Each subtask writes its SPMD context at a sub_block_id-based
 * offset, so the host can verify all three slots independently.
 *
 * Args layout: [output]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_SPMD_READ_AIC 0
#define FUNC_SPMD_READ_AIV0 1
#define FUNC_SPMD_READ_AIV1 2

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_output = from_tensor_arg(orch_args.tensor(0));

    MixedKernels mk;
    mk.aic_kernel_id = FUNC_SPMD_READ_AIC;
    mk.aiv0_kernel_id = FUNC_SPMD_READ_AIV0;
    mk.aiv1_kernel_id = FUNC_SPMD_READ_AIV1;

    Arg args;
    args.add_inout(ext_output);

    pto2_rt_submit_task(mk, args);

    LOG_ALWAYS("[spmd_basic_orch] Submitted 1 MIX task (AIC+AIV0+AIV1)");
}

}  // extern "C"
