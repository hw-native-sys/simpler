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
 * Two-Phase Mesh AllReduce orchestration — RS+AG kernel shim.
 *
 * Three Tensor args (the kernel reads ``Tensor->buffer.addr`` + start_offset
 * to get the real device pointer) plus two scalars:
 *
 *   tensor(0) input   INPUT           (plain device mem, staged in by bootstrap)
 *   tensor(1) output  OUTPUT_EXISTING (plain device mem, flushed by bootstrap)
 *   tensor(2) scratch INOUT           (HCCL window slot; RS+AG in AIV kernel)
 *   scalar(0) nranks
 *   scalar(1) CommContext device pointer
 *
 * INOUT on scratch expresses that the kernel both writes (stage-in, reduced
 * chunk) and reads (allgather remote TLOAD) it.
 */

#include <stdint.h>

#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
allreduce_twophase_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 5,  // 3 tensors + 2 scalars
    };
}

__attribute__((visibility("default"))) void allreduce_twophase_orchestration(const ChipStorageTaskArgs &orch_args) {
    Tensor input = from_tensor_arg(orch_args.tensor(0));
    Tensor output = from_tensor_arg(orch_args.tensor(1));
    Tensor scratch = from_tensor_arg(orch_args.tensor(2));

    Arg params;
    params.add_input(input);
    params.add_output(output);
    params.add_inout(scratch);
    params.add_scalar(orch_args.scalar(0));  // nranks
    params.add_scalar(orch_args.scalar(1));  // CommContext
    rt_submit_aiv_task(0, params);
}

}  // extern "C"
