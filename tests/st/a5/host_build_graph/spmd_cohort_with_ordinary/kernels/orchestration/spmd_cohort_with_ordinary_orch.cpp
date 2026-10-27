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
 * A few full-width cohorts submitted alongside a batch of ordinary tasks.
 *
 * spmd_cohort_saturation runs cohorts back to back with nothing else in the
 * graph, so every core it needs is either idle or holding one of its own
 * blocks. This asks the other question: a cohort needs every core of its kind
 * at once, and ordinary tasks are competing for exactly those cores. The
 * cohort has to assemble anyway.
 *
 * Everything is independent, so it is all ready at the same moment and the
 * ordinary tasks are already running when the cohorts are claimed. The
 * ordinary tasks occupy their cores rather than returning at once: one that
 * finished immediately would free its slot before the claim pass looked, and
 * there would have been no competition to speak of.
 *
 * Kept small on purpose. Making spmd_mixed_waves' cohorts real would have
 * serialized a thousand-task stress case -- its cohorts are a quarter of the
 * palette and each wants the whole machine -- and destroyed what that case is
 * for, which is many shapes in flight at once.
 *
 * Args layout: [out] + {cohort_count, ordinary_count, ordinary_hold_ms,
 * barrier_wait_ms}.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {
constexpr int32_t kBarrierAicFuncId = 0;
constexpr int32_t kBarrierAivLane0FuncId = 1;
constexpr int32_t kBarrierAivLane1FuncId = 2;
constexpr int32_t kBarrierAivLane2FuncId = 3;
constexpr int32_t kBusyAicFuncId = 4;
constexpr int32_t kBusyAivFuncId = 5;

// Slot map: the ordinary tasks' marks, then one arrival/result pair per
// cohort, each sized for the widest cohort this run can submit.
constexpr int64_t kMaxOrdinary = 1536;
constexpr int64_t kMaxLanesPerCohort = 320;
constexpr int64_t kCohortBase = kMaxOrdinary;
constexpr int64_t kSlotsPerCohort = 2 * kMaxLanesPerCohort;
}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 5};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &out = args.tensor(0).ref();
    const int64_t cohort_count = args.scalar<int64_t>(0);
    const int64_t ordinary_count = args.scalar<int64_t>(1);
    const int64_t ordinary_hold_ms = args.scalar<int64_t>(2);
    const int64_t barrier_spin = args.scalar<int64_t>(3);

    const int32_t clusters = rt_available_cluster_count();
    const int32_t aivs = rt_available_aiv_count();
    if (clusters < 1 || aivs < 1) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "topology reports %d clusters, %d AIV", clusters, aivs);
        return;
    }
    if (ordinary_count > kMaxOrdinary || aivs > kMaxLanesPerCohort || 3 * clusters > kMaxLanesPerCohort) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "run is wider than the slot map this case reserves");
        return;
    }

    // All on the AIVs, and submitted first so they are already holding those
    // cores when the cohorts are claimed. AIV-only on purpose: a Mix cohort's
    // AIV subtasks are the ones that wait in Mix slots, and it is the backlog
    // in front of them that this case measures. Putting half of this on the
    // AICs would instead delay the cohort's placement, which is a different
    // thing and would blunt the measurement.
    for (int64_t i = 0; i < ordinary_count; ++i) {
        CoreTaskArgs busy;
        busy.add_no_dep(out);
        busy.add_scalar(i);
        busy.add_scalar(ordinary_hold_ms);
        rt_submit_aiv_task(kBusyAivFuncId, busy);
    }

    for (int64_t cohort = 0; cohort < cohort_count; ++cohort) {
        const int64_t kind = cohort % 3;
        const int32_t width = kind == 2 ? aivs : clusters;
        const int64_t lanes_per_block = kind == 1 ? 3 : 1;
        const int64_t arrive_base = kCohortBase + cohort * kSlotsPerCohort;
        const int64_t result_base = arrive_base + kMaxLanesPerCohort;

        CoreTaskArgs task;
        task.add_no_dep(out);
        task.add_scalar(arrive_base);
        task.add_scalar(result_base);
        task.add_scalar(barrier_spin);
        task.add_scalar(lanes_per_block);
        task.launch_spec.set_block_num(width);
        task.launch_spec.set_require_sync_start(true);
        if (kind == 0) {
            rt_submit_aic_task(kBarrierAicFuncId, task);
        } else if (kind == 2) {
            rt_submit_aiv_task(kBarrierAivLane0FuncId, task);
        } else {
            MixedKernels mixed;
            mixed.aic_kernel_id = kBarrierAicFuncId;
            mixed.aiv0_kernel_id = kBarrierAivLane1FuncId;
            mixed.aiv1_kernel_id = kBarrierAivLane2FuncId;
            rt_submit_task(mixed, task);
        }
    }
}

}  // extern "C"
