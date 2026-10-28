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
 * Back-to-back cohorts, each as wide as the resource it runs on.
 *
 * Every cohort here needs every core of its kind at once, so there is no slack
 * anywhere: a core still holding a block of the previous cohort is a core the
 * next one cannot have, and the next one cannot assemble without it. The
 * cohorts are independent -- no dependencies between them -- so they are all
 * ready at once and queue up behind a single head, which makes every task
 * boundary a head handover under maximum contention.
 *
 * The kinds alternate so that consecutive cohorts land on different channels:
 * an AIC-only cohort fills each core's own dispatch slots, an AIV-only cohort
 * fills the AIVs' own slots, and a Mix cohort reaches the AIVs through their
 * cluster coordinator's Mix slots instead. A core that carries an obligation
 * from one kind into the next is what this case is looking for.
 *
 * Args layout: [out] + {cohort_count, spin_limit}.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {
constexpr int32_t kBarrierAicFuncId = 0;
constexpr int32_t kBarrierAivFuncId = 1;
constexpr int32_t kBarrierAivLane1FuncId = 2;
constexpr int32_t kBarrierAivLane2FuncId = 3;
// Slots reserved per cohort: arrival flags then results, each sized for the
// widest cohort this run can submit times the most lanes a block can have. A
// fixed stride keeps the host's golden arithmetic independent of which kind
// landed where.
constexpr int64_t kMaxLanesPerCohort = 320;
constexpr int64_t kSlotsPerCohort = 2 * kMaxLanesPerCohort;
}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &out = args.tensor(0).ref();
    const int64_t cohort_count = args.scalar<int64_t>(0);
    const int64_t spin_limit = args.scalar<int64_t>(1);
    const int64_t oversubscribed_blocks = args.scalar<int64_t>(2);

    const int32_t clusters = rt_available_cluster_count();
    const int32_t aivs = rt_available_aiv_count();
    if (clusters < 1 || aivs < 1) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "topology reports %d clusters, %d AIV", clusters, aivs);
        return;
    }
    if (clusters > kMaxLanesPerCohort || aivs > kMaxLanesPerCohort || 2 * clusters > kMaxLanesPerCohort) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "topology wider than the slot stride this case reserves");
        return;
    }

    if (oversubscribed_blocks > 0) {
        // The positive control: more blocks than there are cores, and no
        // sync-start. Such a task runs in waves by construction, so its blocks
        // are provably not co-resident and the barrier they share must time
        // out. If this ever assembles, the barrier has stopped being live --
        // optimized away, or the give-up path no longer reaches the output --
        // and every co-residency assertion built on it is worthless.
        CoreTaskArgs control;
        control.add_inout(out);
        control.add_scalar(static_cast<int64_t>(0));
        control.add_scalar(kMaxLanesPerCohort);
        control.add_scalar(spin_limit);
        control.add_scalar(static_cast<int64_t>(1));
        control.launch_spec.set_block_num(static_cast<int32_t>(oversubscribed_blocks));
        rt_submit_aic_task(kBarrierAicFuncId, control);
        return;
    }

    for (int64_t cohort = 0; cohort < cohort_count; ++cohort) {
        // AIC-only, then Mix, then AIV-only. The order is what matters, not the
        // set: a Mix cohort reaches an AIV through its coordinator's Mix slots,
        // and an AIV-only cohort reaches the same AIV through its own dispatch
        // slots. Putting them adjacent is what gives a core the chance to carry
        // an obligation from one channel into the other.
        const int64_t kind = cohort % 3;
        const int32_t width = kind == 2 ? aivs : clusters;
        // A Mix block is 1C2V, and all three have to be running before the body
        // starts -- the same co-residency a sync-start cohort needs, at width
        // one. So a Mix cohort's barrier spans three lanes per block.
        const int64_t lanes_per_block = kind == 1 ? 3 : 1;
        const int64_t arrive_base = cohort * kSlotsPerCohort;
        const int64_t result_base = arrive_base + kMaxLanesPerCohort;

        CoreTaskArgs task;
        task.add_inout(out);
        task.add_scalar(arrive_base);
        task.add_scalar(result_base);
        task.add_scalar(spin_limit);
        task.add_scalar(lanes_per_block);
        task.launch_spec.set_block_num(width);
        task.launch_spec.set_require_sync_start(true);
        if (kind == 0) {
            rt_submit_aic_task(kBarrierAicFuncId, task);
        } else if (kind == 2) {
            rt_submit_aiv_task(kBarrierAivFuncId, task);
        } else {
            // One AIC and both AIVs per block, so a block spans the whole
            // cluster and its AIV subtasks land on Mix slots rather than the
            // AIVs' own. All three have to be running before the body starts.
            MixedKernels mixed;
            mixed.aic_kernel_id = kBarrierAicFuncId;
            mixed.aiv0_kernel_id = kBarrierAivLane1FuncId;
            mixed.aiv1_kernel_id = kBarrierAivLane2FuncId;
            rt_submit_task(mixed, task);
        }
    }
}

}  // extern "C"
