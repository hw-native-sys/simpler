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
 * AllReduce orchestration using tensor args for buffers plus one scalar
 * device-context pointer. This matches the publish/notify/wait kernel ABI.
 */

#include <stdint.h>

#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
allreduce_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default"))) void allreduce_orchestration(const ChipStorageTaskArgs &orch_args) {
    Tensor input = from_tensor_arg(orch_args.tensor(0));
    Tensor recvWindow = from_tensor_arg(orch_args.tensor(1));
    Tensor output = from_tensor_arg(orch_args.tensor(2));
    Tensor notifyCounter = from_tensor_arg(orch_args.tensor(3));
    uint64_t deviceCtx = orch_args.scalar(0);

    Arg params;
    params.add_input(input);
    params.add_inout(recvWindow);
    params.add_output(output);
    params.add_inout(notifyCounter);
    params.add_scalar(deviceCtx);
    pto2_rt_submit_aiv_task(0, params);
}

}  // extern "C"
