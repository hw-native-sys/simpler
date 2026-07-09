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

constexpr uint32_t kElemCount = 1024;

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
urma_async_completion_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 6};
}

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    return urma_async_completion_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void urma_async_completion_orchestration(const L2TaskArgs &orch_args) {
    if (orch_args.tensor_count() + orch_args.scalar_count() != 6) {
        LOG_ERROR("urma_async_completion_demo: expected 6 args");
        return;
    }

    const Tensor &src = orch_args.tensor(0).ref();
    const Tensor &tget_result = orch_args.tensor(2).ref();
    const Tensor &tput_result = orch_args.tensor(3).ref();
    const Tensor &workspace = orch_args.tensor(4).ref();
    auto *comm_ctx = reinterpret_cast<CommContext *>(static_cast<uintptr_t>(orch_args.scalar(0)));
    const uint32_t remote_rank = (comm_ctx->rankId + 1u) % comm_ctx->rankNum;
    uint32_t output_shape[1] = {kElemCount};
    uint32_t token_shape[1] = {1};

    L0TaskArgs reset_args;
    TensorCreateInfo reset_token_info(token_shape, 1, DataType::INT32);
    reset_args.add_input(workspace);
    reset_args.add_output(reset_token_info);
    TaskOutputTensors reset_outputs = rt_submit_aiv_task(4, reset_args);
    Tensor reset_token = reset_outputs.get_ref(0);

    L0TaskArgs tget_args;
    TensorCreateInfo tget_output_info(output_shape, 1, DataType::FLOAT32);
    tget_args.add_input(src);
    tget_args.add_output(tget_output_info);
    tget_args.add_input(workspace);
    tget_args.add_input(reset_token);
    tget_args.add_scalar(remote_rank);
    TaskOutputTensors tget_outputs = rt_submit_aiv_task(0, tget_args);
    Tensor tget_tmp = tget_outputs.get_ref(0);

    L0TaskArgs tput_args;
    TensorCreateInfo tput_output_info(output_shape, 1, DataType::FLOAT32);
    tput_args.add_input(src);
    tput_args.add_output(tput_output_info);
    tput_args.add_input(workspace);
    tput_args.add_input(reset_token);
    tput_args.add_scalar(remote_rank);
    TaskOutputTensors tput_outputs = rt_submit_aiv_task(1, tput_args);
    Tensor tput_tmp = tput_outputs.get_ref(0);

    L0TaskArgs complete_head_1_args;
    complete_head_1_args.add_input(workspace);
    complete_head_1_args.add_input(reset_token);
    complete_head_1_args.add_scalar(remote_rank);
    complete_head_1_args.add_scalar(static_cast<uint64_t>(1));
    rt_submit_aiv_task(2, complete_head_1_args);

    L0TaskArgs complete_head_2_args;
    complete_head_2_args.add_input(workspace);
    complete_head_2_args.add_input(reset_token);
    complete_head_2_args.add_scalar(remote_rank);
    complete_head_2_args.add_scalar(static_cast<uint64_t>(2));
    rt_submit_aiv_task(2, complete_head_2_args);

    L0TaskArgs consumer_args;
    consumer_args.add_input(tget_tmp);
    consumer_args.add_input(tput_tmp);
    consumer_args.add_output(tget_result);
    consumer_args.add_output(tput_result);
    rt_submit_aiv_task(3, consumer_args);
}

}  // extern "C"
