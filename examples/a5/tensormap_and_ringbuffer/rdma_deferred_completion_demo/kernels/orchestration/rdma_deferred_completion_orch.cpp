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
#include "orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig
rdma_deferred_completion_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 6};
}

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    return rdma_deferred_completion_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void rdma_deferred_completion_orchestration(const ChipTaskArgs &orch_args) {
    if (orch_args.tensor_count() + orch_args.scalar_count() != 6) {
        LOG_ERROR("rdma_deferred_completion_demo: expected 6 args");
        return;
    }

    const simpler::tmr::Tensor &send = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &tget_recv = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &tput_recv = orch_args.tensor(2).ref();
    const simpler::tmr::Tensor &status = orch_args.tensor(3).ref();
    auto *comm_ctx = reinterpret_cast<CommContext *>(static_cast<uintptr_t>(orch_args.scalar(0)));
    const uint32_t elem_count = static_cast<uint32_t>(orch_args.scalar(1));

    uint32_t marker_shape[1] = {1};
    uint32_t tget_view_shape[1] = {elem_count};
    uint32_t tget0_view_offset[1] = {0};
    uint32_t tget1_view_offset[1] = {elem_count};
    simpler::tmr::Tensor tget0_recv = tget_recv.view(tget_view_shape, tget0_view_offset);
    simpler::tmr::Tensor tget1_recv = tget_recv.view(tget_view_shape, tget1_view_offset);

    // The HNS1825 RDMA session uses one SQ per peer; serialize producer posts through marker dependencies.

    CoreTaskArgs tget0_args;
    TensorCreateInfo tget_marker_info(marker_shape, 1, DataType::INT32);
    tget0_args.add_input(send);
    tget0_args.add_output(tget0_recv);
    tget0_args.add_output(tget_marker_info);
    tget0_args.add_input(send);
    tget0_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    tget0_args.add_scalar(elem_count);
    TaskOutputTensors tget0_outputs = rt_submit_aiv_task(0, tget0_args);
    simpler::tmr::Tensor tget0_tmp = tget0_recv;
    simpler::tmr::Tensor tget0_marker = tget0_outputs.get_ref(0);

    CoreTaskArgs tget1_args;
    tget1_args.add_input(send);
    tget1_args.add_output(tget1_recv);
    tget1_args.add_output(tget_marker_info);
    tget1_args.add_input(tget0_marker);
    tget1_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    tget1_args.add_scalar(elem_count);
    TaskOutputTensors tget1_outputs = rt_submit_aiv_task(0, tget1_args);
    simpler::tmr::Tensor tget1_tmp = tget1_recv;
    simpler::tmr::Tensor tget1_marker = tget1_outputs.get_ref(0);

    CoreTaskArgs tput0_args;
    TensorCreateInfo marker_info(marker_shape, 1, DataType::INT32);
    tput0_args.add_input(send);
    tput0_args.add_input(tput_recv);
    tput0_args.add_output(marker_info);
    tput0_args.add_input(tget1_marker);
    tput0_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    tput0_args.add_scalar(elem_count);
    TaskOutputTensors tput0_outputs = rt_submit_aiv_task(1, tput0_args);
    simpler::tmr::Tensor tput0_marker = tput0_outputs.get_ref(0);

    CoreTaskArgs tput1_args;
    tput1_args.add_input(send);
    tput1_args.add_input(tput_recv);
    tput1_args.add_output(marker_info);
    tput1_args.add_input(tput0_marker);
    tput1_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    tput1_args.add_scalar(elem_count);
    TaskOutputTensors tput1_outputs = rt_submit_aiv_task(1, tput1_args);
    simpler::tmr::Tensor tput1_marker = tput1_outputs.get_ref(0);

    CoreTaskArgs consumer_args;
    consumer_args.add_input(tget0_tmp);
    consumer_args.add_input(tget1_tmp);
    consumer_args.add_input(tput_recv);
    consumer_args.add_input(tget0_marker);
    consumer_args.add_input(tget1_marker);
    consumer_args.add_input(tput0_marker);
    consumer_args.add_input(tput1_marker);
    consumer_args.add_output(status);
    consumer_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    consumer_args.add_scalar(elem_count);
    rt_submit_aiv_task(2, consumer_args);
}

}  // extern "C"
