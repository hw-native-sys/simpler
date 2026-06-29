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
 * Negative ST orchestration: PTO2_ERROR_SCHEDULER_TIMEOUT (code 100), sub_class=S3.
 *
 * Submits exactly PTO2_FANIN_INLINE_CAP (64) producer dummy tasks inside one
 * open scope (so their ids stay live), then one consumer depending on every
 * producer, with the dep pool pinned tiny via CallConfig.runtime_env. The count
 * is chosen at the inline-cap boundary on purpose: all 64 fanin edges fit the
 * consumer's inline fanin slots, so the fanin spill pool is NEVER allocated and
 * this does not (and must not) latch DEP_POOL_OVERFLOW(4) — that needs > 64
 * edges, see dep_pool_overflow_orch.cpp. Instead the tiny ring_dep_pool starves
 * the scheduler: the consumer's producers all retire (completed=64/65) and the
 * consumer goes READY, but the scheduler cannot make forward progress
 * dispatching it from the undersized pool, so it sits ready-but-idle and the
 * AICPU no-progress watchdog latches SCHEDULER_TIMEOUT, classified S3
 * (ready-but-all-idle) — exactly the stall issue #1180's sub-classification
 * diagnoses. Verified deterministic: it stays code 100 even with the watchdog
 * raised to 30 s, and a generously sized dep pool completes cleanly. Window and
 * heap are left ample so only the dep pool is the bottleneck. The test lowers
 * PTO2_SCHEDULER_TIMEOUT_MS so the watchdog fires quickly.
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

static constexpr int32_t PRODUCER_COUNT = 64;  // == PTO2_FANIN_INLINE_CAP (boundary: no spill)

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

    PTO2_SCOPE() {
        PTO2TaskId producers[PRODUCER_COUNT];
        for (int32_t i = 0; i < PRODUCER_COUNT; i++) {
            L0TaskArgs args;
            args.add_output(ci);
            producers[i] = rt_submit_dummy_task(args).task_id();
        }

        // One consumer depending on every producer. PRODUCER_COUNT fanin edges
        // fit inline (== cap), so they never spill; the tiny dep pool instead
        // starves the scheduler's dispatch of the ready consumer.
        L0TaskArgs consumer;
        consumer.set_dependencies(producers, PRODUCER_COUNT);
        rt_submit_dummy_task(consumer);
    }
}

}  // extern "C"
