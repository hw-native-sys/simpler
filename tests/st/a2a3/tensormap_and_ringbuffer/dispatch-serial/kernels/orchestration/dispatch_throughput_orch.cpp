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
 * Dispatch Throughput Orchestration (tensormap_and_ringbuffer Runtime)
 *
 * Measures scheduler throughput by submitting N noop tasks serially.
 * Each task increments a counter by 1.0, so the final output equals N.
 *
 * Three modes:
 *   mode=0: AIC-only  — N AIC noop tasks
 *   mode=1: AIV-only  — N AIV noop tasks
 *   mode=2: AIC+AIV   — alternating AIC/AIV noop tasks (N total)
 *
 * All tasks are chained through the same output tensor (INOUT) to enforce
 * serial execution order — each task must wait for the previous one to
 * complete before it can read the accumulated value.
 *
 * Arg layout: [out_aic, out_aiv, num_tasks, mode]
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
        .expected_arg_count = 4,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(
    const ChipStorageTaskArgs& orch_args, int orch_thread_num, int orch_thread_index) {
    if (orch_thread_index != 0) {
        return;
    }

    Tensor out_aic = from_tensor_arg(orch_args.tensor(0));
    Tensor out_aiv = from_tensor_arg(orch_args.tensor(1));
    int num_tasks = static_cast<int>(orch_args.scalar(0));
    int mode = static_cast<int>(orch_args.scalar(1));

    LOG_ALWAYS("[dispatch_throughput] num_tasks=%d, mode=%d (0=AIC, 1=AIV, 2=AIC+AIV)", num_tasks, mode);

    for (int i = 0; i < num_tasks; i++) {
        if (mode == 0) {
            Arg params;
            params.add_inout(out_aic);
            pto2_rt_submit_aic_task(FUNC_NOOP_AIC, params);
        } else if (mode == 1) {
            Arg params;
            params.add_inout(out_aiv);
            pto2_rt_submit_aiv_task(FUNC_NOOP_AIV, params);
        } else {
            // Alternating AIC/AIV
            if (i % 2 == 0) {
                Arg params;
                params.add_inout(out_aic);
                pto2_rt_submit_aic_task(FUNC_NOOP_AIC, params);
            } else {
                Arg params;
                params.add_inout(out_aiv);
                pto2_rt_submit_aiv_task(FUNC_NOOP_AIV, params);
            }
        }
    }

    LOG_ALWAYS("[dispatch_throughput] Submitted %d tasks", num_tasks);
}

}  // extern "C"
