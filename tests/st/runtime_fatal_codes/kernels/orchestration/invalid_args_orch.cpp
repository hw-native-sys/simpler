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
 * Negative ST orchestration: PTO2_ERROR_INVALID_ARGS (code 5).
 *
 * Builds an Arg that fails its own validation (set_dependencies with a null
 * pointer but a non-zero count records an error on the Arg) and submits it.
 * submit_*_task checks Arg::has_error up front and latches INVALID_ARGS without
 * ever touching the rings — the simplest deterministic orchestration-bug code.
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

    L0TaskArgs args;
    args.add_output(ci);
    args.set_dependencies(nullptr, 4);  // null deps with count > 0 -> Arg records has_error
    rt_submit_dummy_task(args);
}

}  // extern "C"
