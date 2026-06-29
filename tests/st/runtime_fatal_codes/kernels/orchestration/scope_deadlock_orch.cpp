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
 * Negative ST orchestration: PTO2_ERROR_SCOPE_DEADLOCK (code 1).
 *
 * Submits far more dummy tasks than the (deliberately tiny) per-ring task window
 * inside a single open scope. The orchestrator's scope-admission check
 * (check_scope_can_accept_task) fires first: once tasks-in-scope reaches
 * window_size-1, no slot can be reclaimed while the scope is open, so it latches
 * SCOPE_DEADLOCK. Here check_scope_can_accept_task (per-scope count vs window)
 * fires before the ring's own flow-control detector. The other way to overfill a
 * ring -- nesting many scopes onto the SAME ring so the per-scope check passes but
 * the shared ring fills physically -- latches FLOW_CONTROL_DEADLOCK (code 3)
 * instead; see flow_control_deadlock_orch.cpp.
 *
 * The test drives ring_task_window down via CallConfig.runtime_env.
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

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

    // Just exceed the small ring_task_window the test pins (4): once
    // tasks-in-scope reaches window-1 the scope-admission check latches
    // SCOPE_DEADLOCK. A handful of tasks is enough — a large batch only piles up
    // post-emergency_shutdown teardown state (which destabilises a5 cleanup).
    PTO2_SCOPE() {
        for (int32_t i = 0; i < 8; i++) {
            L0TaskArgs args;
            args.add_output(ci);
            rt_submit_dummy_task(args);
        }
    }
}

}  // extern "C"
