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
 * FFN local linear orchestration — AIC matmul shim.
 *
 *   tensor(0) x_shard       INPUT
 *   tensor(1) w_shard       INPUT
 *   tensor(2) partial_local OUTPUT_EXISTING
 */

#include <stdint.h>

#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
ffn_local_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 3};
}

__attribute__((visibility("default"))) void ffn_local_orchestration(const ChipStorageTaskArgs &orch_args) {
    Tensor x_shard = from_tensor_arg(orch_args.tensor(0));
    Tensor w_shard = from_tensor_arg(orch_args.tensor(1));
    Tensor partial_local = from_tensor_arg(orch_args.tensor(2));

    Arg params;
    params.add_input(x_shard);
    params.add_input(w_shard);
    params.add_output(partial_local);
    pto2_rt_submit_aic_task(0, params);
}

}  // extern "C"
