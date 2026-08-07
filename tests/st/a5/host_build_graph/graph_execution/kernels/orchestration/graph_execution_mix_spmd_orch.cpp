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

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_SPMD_MIX_AIC 0
#define FUNC_SPMD_MIX_AIV0 1
#define FUNC_SPMD_MIX_AIV1 2

namespace {

void mix_spmd_layer(const CoreTaskArgs &args) {
    MixedKernels kernels;
    kernels.aic_kernel_id = FUNC_SPMD_MIX_AIC;
    kernels.aiv0_kernel_id = FUNC_SPMD_MIX_AIV0;
    kernels.aiv1_kernel_id = FUNC_SPMD_MIX_AIV1;

    CoreTaskArgs task_args;
    task_args.add_inout(args.tensor(0).ref());
    task_args.add_scalar(int64_t{0});
    task_args.launch_spec.set_block_num(static_cast<int16_t>(rt_available_cluster_count()));
    task_args.launch_spec.set_require_sync_start(true);
    rt_submit_task(kernels, task_args);
}

void submit_layer(const CoreTaskArgs &args) { rt_submit_graph(&mix_spmd_layer, args); }

}  // namespace

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    for (int32_t output_index = 0; output_index < 3; ++output_index) {
        CoreTaskArgs layer_args;
        layer_args.add_inout(args.tensor(output_index).ref());
        submit_layer(layer_args);
    }
}

}  // extern "C"
