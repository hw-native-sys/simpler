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

#include "common/comm_context.h"
#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 5};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor partial_local = from_tensor_arg(orch_args.tensor(0));
    Tensor partial_window = from_tensor_arg(orch_args.tensor(1));
    Tensor y = from_tensor_arg(orch_args.tensor(2));
    Tensor notify_counter = from_tensor_arg(orch_args.tensor(3));
    auto *comm_ctx = reinterpret_cast<CommDeviceContext *>(static_cast<uintptr_t>(orch_args.scalar(0)));

    Arg params;
    params.add_input(partial_local);
    params.add_inout(partial_window);
    params.add_output(y);
    params.add_inout(notify_counter);
    params.add_scalar((uint64_t)(uintptr_t)comm_ctx);
    // Keep publish, notify, wait, and accumulation in one device kernel so
    // the peer notification cannot race ahead of the corresponding TPUT.
    pto2_rt_submit_aiv_task(1, params);
}

}  // extern "C"
