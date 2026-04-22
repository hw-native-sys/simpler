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
 * AllReduce-sum orchestration — AIV publish/notify/wait/accumulate shim.
 *
 *   tensor(0) partial_local INPUT           (per-rank device mem; producer = ffn_local kernel)
 *   tensor(1) y             OUTPUT_EXISTING (per-rank host-backed)
 *   tensor(2) scratch       INOUT           (HCCL-window slot — mailbox + signal tail)
 *   scalar(0) nranks
 *   scalar(1) CommContext device pointer
 */

#include <stdint.h>

#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
allreduce_sum_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 5};  // 3 tensors + 2 scalars
}

__attribute__((visibility("default"))) void allreduce_sum_orchestration(const ChipStorageTaskArgs &orch_args) {
    Tensor partial_local = from_tensor_arg(orch_args.tensor(0));
    Tensor y = from_tensor_arg(orch_args.tensor(1));
    Tensor scratch = from_tensor_arg(orch_args.tensor(2));

    Arg params;
    params.add_input(partial_local);
    params.add_output(y);
    params.add_inout(scratch);
    params.add_scalar(orch_args.scalar(0));  // nranks
    params.add_scalar(orch_args.scalar(1));  // CommContext
    pto2_rt_submit_aiv_task(1, params);
}

}  // extern "C"
