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

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int32_t mask = static_cast<int32_t>(args.scalar<int64_t>(0));
    const int16_t blocks = static_cast<int16_t>(rt_available_cluster_count() * 8);
    MixedKernels kernels;
    kernels.aic_kernel_id = (mask & 1) != 0 ? 0 : INVALID_KERNEL_ID;
    kernels.aiv0_kernel_id = (mask & 2) != 0 ? 1 : INVALID_KERNEL_ID;
    kernels.aiv1_kernel_id = (mask & 4) != 0 ? 2 : INVALID_KERNEL_ID;
    CoreTaskArgs mixed;
    mixed.add_inout(args.tensor(0).ref());
    mixed.add_scalar(0);
    mixed.add_scalar(20000);
    mixed.launch_spec.set_block_num(blocks);
    rt_submit_task(kernels, mixed);

    CoreTaskArgs pure;
    pure.add_inout(args.tensor(1).ref());
    pure.add_scalar(0);
    pure.add_scalar(20000);
    pure.launch_spec.set_block_num(blocks);
    if (mask == 6) rt_submit_aic_task(4, pure);
    else rt_submit_aiv_task(3, pure);

    uint32_t index[1] = {0};
    set_tensor_data<int32_t>(args.tensor(2).ref(), 1, index, blocks);
}
}
