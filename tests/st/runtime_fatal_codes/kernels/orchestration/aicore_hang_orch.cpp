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
 * Negative ST orchestration: PTO2_ERROR_SCHEDULER_TIMEOUT (code 100), sub_class=S1.
 *
 * Dispatches a single AIC kernel that spins forever. The task lands on a core and
 * never completes, so when the AICPU no-progress watchdog fires it sees a RUNNING
 * task (cnt_running > 0) and classifies the stall as S1 (running-stalled) — the
 * AICore-silent-hang case that is issue #1180's primary motivation. Onboard only:
 * the test pins the timeout chain so the AICPU scheduler watchdog (2 s) fires
 * before the STARS op watchdog (3 s), latching code 100 with sub_class=S1 before
 * the device is force-reset.
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_AIC_HANG 0

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

    L0TaskArgs args;
    args.add_output(ci);
    rt_submit_aic_task(FUNC_AIC_HANG, args);
}

}  // extern "C"
