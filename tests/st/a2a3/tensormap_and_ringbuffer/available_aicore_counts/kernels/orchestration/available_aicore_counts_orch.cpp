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
 * available_aicore_counts: surface rt_available_cluster_count() to the host.
 *
 * The this-run MIX cluster (= AIC) count is an AICPU-side runtime query, so no
 * AICore kernel computes it: orchestration reads it through the ops table and
 * writes it into a single int32 output tensor. A dummy_task stands in as that
 * tensor's producer so the run has a completable task graph and set_tensor_data
 * has a producer to wait on; the host then compares the value directly.
 */

#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_out = orch_args.tensor(0).ref();

    // dummy_task stands in as ext_out's producer so the run has a completable
    // task graph and set_tensor_data has a producer to wait on.
    {
        L0TaskArgs args;
        args.add_inout(ext_out);
        rt_submit_dummy_task(args);
    }

    int32_t cluster_count = rt_available_cluster_count();
    LOG_INFO_V0("[available_aicore_counts] rt_available_cluster_count=%d", cluster_count);
    uint32_t idx[1] = {0};
    set_tensor_data<int32_t>(ext_out, 1, idx, cluster_count);
}

}  // extern "C"
