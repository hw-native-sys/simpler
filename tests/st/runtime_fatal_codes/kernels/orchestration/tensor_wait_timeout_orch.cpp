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
 * Negative ST orchestration: PTO2_ERROR_TENSOR_WAIT_TIMEOUT (code 8).
 *
 * Submits an AIC kernel that spins forever, then reads its output with
 * get_tensor_data. Because the output has a producer in the TensorMap,
 * get_tensor_data spin-waits for that producer to complete — which never
 * happens — so after PTO2_TENSOR_DATA_TIMEOUT_CYCLES (15e9 cycles == 15 s at the
 * 1 GHz AICPU counter) the orchestrator latches TENSOR_WAIT_TIMEOUT.
 *
 * Onboard only (the hang kernel would spin the simulator forever). The test
 * raises the AICPU scheduler / STARS op / host stream-sync timeouts all above
 * 15 s so the tensor-data wait wins the race; otherwise the no-progress watchdog
 * (code 100) or STARS reaps the hang first.
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_AIC_HANG 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 0,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    (void)orch_args;

    uint32_t shape[1] = {1};
    TensorCreateInfo ci(shape, 1, DataType::INT32);

    L0TaskArgs args;
    args.add_output(ci);
    TaskOutputTensors outs = rt_submit_aic_task(FUNC_AIC_HANG, args);

    // Reading the hung producer's output blocks until it completes (it never
    // does) -> TENSOR_WAIT_TIMEOUT after the fixed data-wait timeout.
    const Tensor &out = outs.get_ref(0);
    uint32_t idx[1] = {0};
    (void)get_tensor_data(out, 1, idx);
}

}  // extern "C"
