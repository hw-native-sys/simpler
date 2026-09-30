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
 * Orchestration that latches its own fatal code and returns.
 *
 * Takes the same argument layout as the chain beside it so one test can submit
 * either callable with one argument builder. Nothing is submitted to the rings
 * and no resource is exhausted, so the failure is deterministic, needs no
 * watchdog and leaves no hung core behind — which is what makes it usable as
 * the successor of a run that must still produce its own correct result.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{
        .expected_arg_count = 4,  // 3 tensors + the spin scalar, as the chain takes
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "early-enqueue error-attribution case");
}

}  // extern "C"
