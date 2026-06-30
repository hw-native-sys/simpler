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
 * Negative ST orchestration: PTO2_ERROR_HEAP_RING_DEADLOCK (code 2).
 *
 * Requests an output buffer larger than the (deliberately tiny) per-ring heap.
 * The allocator can never satisfy it, so it latches HEAP_RING_DEADLOCK. The task
 * window is left ample so the slot ring is not the bottleneck (that would be
 * FLOW_CONTROL_DEADLOCK instead).
 *
 * The test drives ring_heap down via CallConfig.runtime_env.
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

    // 8192 INT32 = 32 KiB output; the test pins ring_heap far below this so the
    // single allocation can never fit.
    uint32_t shape[1] = {8192};
    TensorCreateInfo ci(shape, 1, DataType::INT32);

    L0TaskArgs args;
    args.add_output(ci);
    rt_submit_dummy_task(args);
}

}  // extern "C"
