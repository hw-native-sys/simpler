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
urma_deferred_completion_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    return urma_deferred_completion_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void urma_deferred_completion_orchestration(const L2TaskArgs &orch_args) {
    if (orch_args.tensor_count() != 3 || orch_args.scalar_count() != 1) {
        LOG_ERROR("urma_deferred_completion_demo: expected 3 tensors and 1 scalar");
        return;
    }

    const Tensor &input = orch_args.tensor(0).ref();
    const Tensor &out = orch_args.tensor(1).ref();
    const Tensor &result = orch_args.tensor(2).ref();
    auto *comm_ctx = reinterpret_cast<CommContext *>(static_cast<uintptr_t>(orch_args.scalar(0)));

    L0TaskArgs producer_args;
    producer_args.add_input(input);
    producer_args.add_output(out);
    producer_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    rt_submit_aiv_task(0, producer_args);

    L0TaskArgs consumer_args;
    consumer_args.add_input(out);
    consumer_args.add_output(result);
    rt_submit_aiv_task(1, consumer_args);
}

}  // extern "C"
