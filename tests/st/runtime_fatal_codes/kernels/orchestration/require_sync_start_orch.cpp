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
 * Negative ST orchestration: PTO2_ERROR_REQUIRE_SYNC_START_INVALID (code 7).
 *
 * Submits one SPMD AIV task that asks for require_sync_start with a block_num
 * far larger than the available cores. A sync-start launch needs every block
 * resident at once, so block_num > core limit is a guaranteed deadlock; the
 * orchestrator catches it proactively at submit time (before any dispatch) and
 * latches REQUIRE_SYNC_START_INVALID. core_num=1000 exceeds the AIV core count
 * on every supported platform. The noop kernel never runs.
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_NOOP_KERNEL 0

// PTO2LaunchSpec spells the SPMD block-count setter differently per arch
// (a5: set_core_num, a2a3: set_block_num) for the same field. Bridge it so this
// one fixture compiles on both; keyed off the arch's pto_types.h include guard,
// which pto_orchestration_api.h pulls in transitively.
static inline void set_block_count(L0TaskArgs &args, int16_t n) {
#if defined(SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_TYPES_H_)
    args.launch_spec.set_core_num(n);
#else
    args.launch_spec.set_block_num(n);
#endif
}

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
    set_block_count(args, 1000);                    // >> available AIV cores
    args.launch_spec.set_require_sync_start(true);  // arm the sync-start deadlock guard
    rt_submit_aiv_task(FUNC_NOOP_KERNEL, args);
}

}  // extern "C"
