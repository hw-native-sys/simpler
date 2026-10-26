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
 * One AIC cohort exactly as wide as the machine.
 *
 * The width comes from the run rather than from a constant, because this case
 * is only the case it means to be at the boundary: one block fewer and a core
 * is spare, so an assembly that accidentally put two blocks on one core would
 * still succeed. At full width every core must take exactly one, and the rule
 * that each claiming core takes at most one block of a cohort is the only
 * reason that is reachable.
 *
 * Args layout: [out] + {} -- nothing to configure, the topology decides.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 1};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int32_t clusters = rt_available_cluster_count();
    if (clusters < 1) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "topology reports %d clusters", clusters);
        return;
    }
    CoreTaskArgs cohort;
    cohort.add_inout(args.tensor(0).ref());
    // One AIC per cluster, so the cluster count is the AIC count and this is
    // the widest cohort the submit-time guard will admit.
    cohort.launch_spec.set_block_num(clusters);
    cohort.launch_spec.set_require_sync_start(true);
    rt_submit_aic_task(0, cohort);
}

}  // extern "C"
