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
 * One plain Mix task whose 1C2V must be running together.
 *
 * A Mix task is one task spanning an AIC and both AIVs of a cluster, and real
 * Mix kernels make them wait on each other: the AIC publishes a flag the AIVs
 * block on, and the AIVs publish flags the AIC blocks on. So a Mix task is a
 * co-residency requirement at width one -- the same thing sync-start asks for
 * across blocks, asked for across subtasks instead.
 *
 * The runtime places the three subtasks atomically, but placement is not
 * execution: a subtask sits in its slot until its core reaches it. What this
 * case asks is whether the three therefore reach their kernels together.
 *
 * `occupy_count` AIV tasks can be submitted first to put the AIVs to work, so
 * the Mix task's AIC subtask starts while its AIV subtasks are still queued
 * behind something else. That is the shape where "placed atomically" and "runs
 * together" come apart.
 *
 * Args layout: [out] + {occupy_count, occupy_spin, barrier_spin}.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {
constexpr int32_t kBarrierAicFuncId = 0;
constexpr int32_t kBarrierAivLane1FuncId = 1;
constexpr int32_t kBarrierAivLane2FuncId = 2;
// Lane 0 of the same barrier. At width one with one lane it waits only for
// itself, so it returns immediately and serves purely as core occupancy.
constexpr int32_t kBarrierAivLane0FuncId = 3;

// Slot map: the occupying tasks' arrival flags, then their results, then the
// Mix task's arrival flags and results. One Mix block, three lanes.
constexpr int64_t kMaxOccupy = 160;
constexpr int64_t kOccupyArriveBase = 0;
constexpr int64_t kOccupyBase = kOccupyArriveBase + kMaxOccupy;
constexpr int64_t kArriveBase = kOccupyBase + kMaxOccupy;
constexpr int64_t kResultBase = kArriveBase + 8;
constexpr int64_t kLanesPerBlock = 3;
}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &out = args.tensor(0).ref();
    const int64_t occupy_count = args.scalar<int64_t>(0);
    const int64_t occupy_spin = args.scalar<int64_t>(1);
    const int64_t barrier_spin = args.scalar<int64_t>(2);

    if (occupy_count > kMaxOccupy) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "occupy_count %lld exceeds the reserved slots", occupy_count);
        return;
    }

    for (int64_t i = 0; i < occupy_count; ++i) {
        // The barrier kernel at width one, not a kernel of this case's own: a
        // single block waits for a single arrival, so it returns at once and
        // the only thing it contributes is having occupied a core. Reusing a
        // kernel the cohort cases already exercise keeps this case from
        // depending on a signature only it uses.
        CoreTaskArgs busy;
        // No inferred dependency: the occupying tasks and the Mix task write
        // disjoint slots of one tensor, and what this case needs is for them
        // to be ready at the same time. Declaring the tensor as a dependency
        // would serialize them and test nothing.
        busy.add_no_dep(out);
        busy.add_scalar(kOccupyArriveBase + i);
        busy.add_scalar(kOccupyBase + i);
        busy.add_scalar(occupy_spin);
        busy.add_scalar(static_cast<int64_t>(1));
        rt_submit_aiv_task(kBarrierAivLane0FuncId, busy);
    }

    // A negative barrier spin submits the occupying tasks and nothing else, so
    // a case can tell "the occupying tasks alone are fine" from "the pair is
    // not" without changing anything else about the graph.
    if (barrier_spin < 0) return;

    CoreTaskArgs mix;
    mix.add_no_dep(out);
    mix.add_scalar(kArriveBase);
    mix.add_scalar(kResultBase);
    mix.add_scalar(barrier_spin);
    mix.add_scalar(kLanesPerBlock);
    // Plain Mix: one block, no sync-start. Whether its subtasks run together
    // is exactly the question.
    MixedKernels mixed;
    mixed.aic_kernel_id = kBarrierAicFuncId;
    mixed.aiv0_kernel_id = kBarrierAivLane1FuncId;
    mixed.aiv1_kernel_id = kBarrierAivLane2FuncId;
    rt_submit_task(mixed, mix);
}

}  // extern "C"
