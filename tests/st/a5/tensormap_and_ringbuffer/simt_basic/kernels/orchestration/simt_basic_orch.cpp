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

#include "orchestration_api.h"

#define FUNC_SIMT_SCATTER 0

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &src = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &indices = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &out = orch_args.tensor(2).ref();

    // SIMPLER_SCOPE ensures rt_submit_aiv_task flushes through the task
    // ringbuffer before the entry returns. With no set_block_num call,
    // LaunchSpec uses its default of one block.
    SIMPLER_SCOPE() {
        CoreTaskArgs args;
        args.add_input(src);
        args.add_input(indices);
        args.add_output(out);
        rt_submit_aiv_task(FUNC_SIMT_SCATTER, args);
    }
}

}  // extern "C"
