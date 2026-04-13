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
 * Task Scaling Orchestration (tensormap_and_ringbuffer Runtime)
 *
 * Measures dispatch overhead growth as task count scales. Submits N
 * independent noop tasks, each writing 1.0 to a separate slot in the
 * output tensor. Tasks are independent (no inter-task data dependency)
 * to isolate pure scheduling overhead from serialization effects.
 *
 * Three modes:
 *   mode=0: AIC-only  — N independent AIC noop tasks
 *   mode=1: AIV-only  — N independent AIV noop tasks
 *   mode=2: AIC+AIV   — alternating AIC/AIV independent noop tasks
 *
 * Arg layout: [output, num_tasks, mode]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_NOOP_AIC 0
#define FUNC_NOOP_AIV 1

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(
    const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(
    const ChipStorageTaskArgs& orch_args, int orch_thread_num, int orch_thread_index) {
    if (orch_thread_index != 0) {
        return;
    }

    Tensor output = from_tensor_arg(orch_args.tensor(0));
    int num_tasks = static_cast<int>(orch_args.scalar(0));
    int mode = static_cast<int>(orch_args.scalar(1));

    LOG_ALWAYS("[task_scaling] num_tasks=%d, mode=%d (0=AIC, 1=AIV, 2=AIC+AIV)", num_tasks, mode);

    // Each task writes to a separate cache line (64B = 16 float32 elements)
    // to avoid false sharing across non-coherent AICore L1 caches.
    constexpr uint32_t CACHE_LINE_ELEMS = 16;
    uint32_t slot_shapes[1] = {1};

    for (int i = 0; i < num_tasks; i++) {
        uint32_t view_offsets[1] = {static_cast<uint32_t>(i * CACHE_LINE_ELEMS)};
        Tensor slot = output.view(slot_shapes, view_offsets);

        if (mode == 0) {
            Arg params;
            params.add_inout(slot);
            pto2_rt_submit_aic_task(FUNC_NOOP_AIC, params);
        } else if (mode == 1) {
            Arg params;
            params.add_inout(slot);
            pto2_rt_submit_aiv_task(FUNC_NOOP_AIV, params);
        } else {
            if (i % 2 == 0) {
                Arg params;
                params.add_inout(slot);
                pto2_rt_submit_aic_task(FUNC_NOOP_AIC, params);
            } else {
                Arg params;
                params.add_inout(slot);
                pto2_rt_submit_aiv_task(FUNC_NOOP_AIV, params);
            }
        }
    }

    LOG_ALWAYS("[task_scaling] Submitted %d independent tasks", num_tasks);
}

}  // extern "C"
