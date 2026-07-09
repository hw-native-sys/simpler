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
 * Broadcast orchestration shim.
 *
 *   tensor(0) input   INPUT           (COUNT_PER_RANK floats)
 *   tensor(1) output  OUTPUT_EXISTING (COUNT_PER_RANK floats)
 *   tensor(2) scratch INOUT           (HCCL window slot)
 *   scalar(0) nranks
 *   scalar(1) root
 *   scalar(2) CommContext device pointer
 */

#include <stdint.h>

#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
broadcast_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 6,  // 3 tensors + 3 scalars
    };
}

__attribute__((visibility("default"))) void broadcast_orchestration(const L2TaskArgs &orch_args) {
    const Tensor &input = orch_args.tensor(0).ref();
    const Tensor &output = orch_args.tensor(1).ref();
    const Tensor &scratch = orch_args.tensor(2).ref();

    L0TaskArgs params;
    params.add_input(input);
    params.add_output(output);
    params.add_inout(scratch);
    params.add_scalar(orch_args.scalar(0));  // nranks
    params.add_scalar(orch_args.scalar(1));  // root
    params.add_scalar(orch_args.scalar(2));  // CommContext
    rt_submit_aiv_task(0, params);
}

}  // extern "C"
