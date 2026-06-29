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
 * Negative ST orchestration: PTO2_ERROR_DEP_POOL_OVERFLOW (code 4).
 *
 * One consumer with more fanin edges than the dep pool can hold. The first
 * PTO2_FANIN_INLINE_CAP (64) edges live in the consumer's inline fanin slots;
 * only edge 65+ spills into the per-ring fanin spill pool. With the pool pinned
 * to 4 entries (CallConfig.runtime_env.ring_dep_pool) and 64 spill edges that
 * all belong to the one in-construction consumer (so none can be reclaimed), the
 * spill allocator exhausts and the orchestrator latches DEP_POOL_OVERFLOW.
 *
 * PRODUCER_COUNT must exceed PTO2_FANIN_INLINE_CAP, otherwise every edge fits
 * inline, the spill pool is never touched, and the run stalls into
 * SCHEDULER_TIMEOUT (code 100) instead — see scheduler_timeout_orch.cpp, which
 * pins exactly that boundary. The scope keeps all producer ids live so the
 * consumer really takes PRODUCER_COUNT distinct fanin edges.
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

static constexpr int32_t PRODUCER_COUNT = 128;  // > PTO2_FANIN_INLINE_CAP (64)

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

        L0TaskArgs consumer;
        consumer.set_dependencies(producers, PRODUCER_COUNT);
        rt_submit_dummy_task(consumer);
    }
}

}  // extern "C"
