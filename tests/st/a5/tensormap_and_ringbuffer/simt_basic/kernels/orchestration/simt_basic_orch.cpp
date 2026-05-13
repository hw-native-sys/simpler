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
 * SIMT basic orchestration: submit a single AIV SIMT scatter task.
 *
 * Args layout: [src, indices, out]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_SIMT_SCATTER 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor src = from_tensor_arg(orch_args.tensor(0));
    Tensor indices = from_tensor_arg(orch_args.tensor(1));
    Tensor out = from_tensor_arg(orch_args.tensor(2));

    // PTO2_SCOPE ensures rt_submit_aiv_task flushes through the task
    // ringbuffer before the entry returns. No set_core_num — let the
    // runtime use the config's block_dim, matching the ptoas-validated
    // mscatter reference.
    PTO2_SCOPE() {
        Arg args;
        args.add_input(src);
        args.add_input(indices);
        args.add_output(out);
        rt_submit_aiv_task(FUNC_SIMT_SCATTER, args);
    }
}

}  // extern "C"
