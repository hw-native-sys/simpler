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
async_notify_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 5};
}

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    return async_notify_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void async_notify_orchestration(const ChipTaskArgs &orch_args) {
    if (orch_args.tensor_count() + orch_args.scalar_count() != 5) {
        LOG_ERROR("async_notify_demo: expected 5 args");
        return;
    }

    const ChipTensor &input = orch_args.tensor(0).ref();
    const ChipTensor &output = orch_args.tensor(1).ref();
    const ChipTensor &result = orch_args.tensor(2).ref();
    const ChipTensor &notify_counter = orch_args.tensor(3).ref();
    auto *comm_ctx = reinterpret_cast<CommContext *>(static_cast<uintptr_t>(orch_args.scalar(0)));

    CoreTaskArgs params_producer;
    params_producer.add_input(input);
    params_producer.add_output(output);
    params_producer.add_scalar(notify_counter.buffer.addr);
    params_producer.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    rt_submit_aiv_task(0, params_producer);

    uint32_t notify_token_shape[1] = {1};
    TensorCreateInfo notify_token_info(notify_token_shape, 1, DataType::INT32);
    CoreTaskArgs params_notify;
    params_notify.add_output(notify_token_info);
    params_notify.add_scalar(notify_counter.buffer.addr);
    params_notify.add_scalar(static_cast<uint64_t>(1));
    TaskOutputTensors notify_outputs = rt_submit_aiv_task(2, params_notify);
    ChipTensor notify_token = notify_outputs.get_ref(0);

    CoreTaskArgs params_consumer;
    params_consumer.add_input(notify_token);
    params_consumer.add_input(output);
    params_consumer.add_output(result);
    params_consumer.add_scalar(notify_counter.buffer.addr);
    rt_submit_aiv_task(1, params_consumer);
}

}  // extern "C"
