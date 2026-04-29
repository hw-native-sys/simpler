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

#include "platform_comm/comm_context.h"
#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
sdma_async_completion_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    return sdma_async_completion_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void sdma_async_completion_orchestration(const ChipStorageTaskArgs &orch_args) {
    if (orch_args.tensor_count() + orch_args.scalar_count() != 4) {
        LOG_ERROR("sdma_async_completion_demo: expected 4 args");
        return;
    }

    Tensor input = from_tensor_arg(orch_args.tensor(0));
    Tensor out = from_tensor_arg(orch_args.tensor(1));
    Tensor result = from_tensor_arg(orch_args.tensor(2));
    auto *comm_ctx = reinterpret_cast<CommContext *>(static_cast<uintptr_t>(orch_args.scalar(0)));

    Arg producer_args;
    producer_args.add_input(input);
    producer_args.add_output(out);
    producer_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    pto2_rt_submit_aiv_task_deferred(0, producer_args);

    Arg consumer_args;
    consumer_args.add_input(out);
    consumer_args.add_output(result);
    pto2_rt_submit_aiv_task(1, consumer_args);
}

}  // extern "C"
