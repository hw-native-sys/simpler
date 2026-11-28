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
    const auto &input = orch_args.tensor(0).ref();
    const auto &receive = orch_args.tensor(1).ref();
    const auto &out = orch_args.tensor(2).ref();
    const auto &result = orch_args.tensor(3).ref();
    const uint64_t comm_ctx = orch_args.scalar<uint64_t>(0);
    const uint32_t elems = orch_args.scalar<uint32_t>(1);
    const uint32_t mode = orch_args.scalar<uint32_t>(2);
    const uint32_t repeats = orch_args.scalar<uint32_t>(3);

    auto submit_transfer = [mode](CoreTaskArgs &args) {
        if (mode >= 2) {
            // Both vector lanes in one cluster share the native CCE block index.
            MixedKernels lanes;
            lanes.aiv0_kernel_id = 0;
            lanes.aiv1_kernel_id = 0;
            return rt_submit_task(lanes, args);
        }
        return rt_submit_aiv_task(0, args);
    };
    auto gate = input;
    if (mode == 1 || mode == 2 || mode == 4) {
        uint32_t marker_shape[1] = {1};
        TensorCreateInfo marker(marker_shape, 1, DataType::INT32);
        CoreTaskArgs put;
        put.add_input(input);
        put.add_input(receive);
        put.add_output(marker);
        put.add_input(input);
        put.add_scalar(comm_ctx);
        put.add_scalar(elems);
        put.add_scalar(mode);
        put.add_scalar(repeats);
        TaskOutputTensors put_outputs = submit_transfer(put);
        gate = put_outputs.get_ref(0);
    }

    // A PUT successor reads the region written on the peer only after PUT completion.
    CoreTaskArgs get;
    get.add_input(mode == 0 || mode == 6 ? input : receive);
    get.add_input(mode != 0 ? receive : input);
    get.add_output(out);
    get.add_input(gate);
    get.add_scalar(comm_ctx);
    get.add_scalar(elems);
    get.add_scalar(mode == 6 ? mode : (mode >= 2 ? mode + 1 : 0));
    get.add_scalar(mode == 1 ? 1 : repeats);
    submit_transfer(get);

    CoreTaskArgs consumer;
    consumer.add_input(out);
    consumer.add_output(result);
    consumer.add_scalar(elems);
    rt_submit_aiv_task(1, consumer);
}

}  // extern "C"
