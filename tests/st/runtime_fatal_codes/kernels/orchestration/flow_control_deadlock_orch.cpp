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
 * Negative ST orchestration: PTO2_ERROR_FLOW_CONTROL_DEADLOCK (code 3).
 *
 * Scope depth maps to ring index via min(scope_depth, PTO2_MAX_RING_DEPTH - 1),
 * so every scope nested at depth >= 3 lands on the SAME ring (ring 3). Nest well
 * past depth 3 and submit a few tasks per level: each level's own scope-task
 * count stays below the window, so the scope-admission check (SCOPE_DEADLOCK, 1)
 * does not fire -- but the deep levels collectively overfill the shared ring 3's
 * physical slots, and the task allocator latches FLOW_CONTROL_DEADLOCK ("Task
 * Ring Full"). The test pins ring_task_window tiny so the shared ring fills fast.
 *
 * (A single scope exhausting one ring latches SCOPE_DEADLOCK(1) instead -- see
 * scope_deadlock_orch.cpp; this same-ring-nesting path is what reaches code 3.)
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

static constexpr int32_t DEPTH = 8;      // > PTO2_MAX_RING_DEPTH so depths 3..7 share ring 3
static constexpr int32_t PER_LEVEL = 3;  // < window so no single scope trips SCOPE_DEADLOCK

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 0,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    (void)orch_args;

    uint32_t shape[1] = {1};
    TensorCreateInfo ci(shape, 1, DataType::INT32);

    for (int32_t d = 0; d < DEPTH; d++) {
        rt_scope_begin();
        for (int32_t t = 0; t < PER_LEVEL; t++) {
            L0TaskArgs args;
            args.add_output(ci);
            rt_submit_dummy_task(args);
        }
    }
    for (int32_t d = 0; d < DEPTH; d++) {
        rt_scope_end();
    }
}

}  // extern "C"
