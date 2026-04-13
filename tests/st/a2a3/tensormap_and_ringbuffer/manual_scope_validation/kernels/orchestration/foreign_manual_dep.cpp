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

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 0,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;

    uint32_t shape[1] = {1};
    TensorCreateInfo ci(shape, 1, DataType::FLOAT32);
    PTO2TaskId foreign_dep = PTO2TaskId::invalid();

    PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
        TaskOutputTensors outs = alloc_tensors(ci);
        foreign_dep = outs.task_id();
    }

    PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
        Arg args;
        args.add_dep(foreign_dep);
        (void)pto2_rt_submit_aiv_task(0, args);
    }
}

}  // extern "C"
