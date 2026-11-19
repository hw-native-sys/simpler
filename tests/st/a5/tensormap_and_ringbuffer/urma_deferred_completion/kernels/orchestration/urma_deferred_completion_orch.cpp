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
urma_deferred_completion_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 8};
}

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    return urma_deferred_completion_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void urma_deferred_completion_orchestration(const ChipTaskArgs &orch_args) {
    if (orch_args.tensor_count() != 4 || orch_args.scalar_count() != 4) {
        LOG_ERROR("urma_deferred_completion_demo: expected 4 tensors and 4 scalars");
        return;
    }
    const simpler::tmr::Tensor &input = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &receive = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &out = orch_args.tensor(2).ref();
    const simpler::tmr::Tensor &result = orch_args.tensor(3).ref();
    const uint64_t comm_ctx = orch_args.scalar<uint64_t>(0);
    const uint32_t elems = orch_args.scalar<uint32_t>(1);
    const uint32_t mode = orch_args.scalar<uint32_t>(2);
    const uint32_t repeats = orch_args.scalar<uint32_t>(3);

    simpler::tmr::Tensor gate = input;
    if (mode == 1) {
        uint32_t marker_shape[1] = {1};
        TensorCreateInfo marker(marker_shape, 1, DataType::INT32);
        CoreTaskArgs put;
        put.add_input(input);
        put.add_input(receive);
        put.add_output(marker);
        put.add_input(input);
        put.add_scalar(comm_ctx);
        put.add_scalar(elems);
        put.add_scalar(1);
        put.add_scalar(repeats);
        TaskOutputTensors put_outputs = rt_submit_aiv_task(0, put);
        gate = put_outputs.get_ref(0);
    }

    // A PUT successor reads the region written on the peer only after PUT completion.
    CoreTaskArgs get;
    get.add_input(mode == 1 ? receive : input);
    get.add_input(mode == 1 ? receive : input);
    get.add_output(out);
    get.add_input(gate);
    get.add_scalar(comm_ctx);
    get.add_scalar(elems);
    get.add_scalar(0);
    get.add_scalar(mode == 1 ? 1 : repeats);
    rt_submit_aiv_task(0, get);

    CoreTaskArgs consumer;
    consumer.add_input(out);
    consumer.add_output(result);
    consumer.add_scalar(elems);
    rt_submit_aiv_task(1, consumer);
}

}  // extern "C"
