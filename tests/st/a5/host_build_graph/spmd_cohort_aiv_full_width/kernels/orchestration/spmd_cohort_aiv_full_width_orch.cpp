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
 * One AIV cohort exactly as wide as the machine's AIV population.
 *
 * The AIC cohort of spmd_cohort_full_width is one block per cluster; this one
 * is one block per AIV, which is several per cluster. So the widest cohort of
 * this kind is a different number, read from a different query, and placing it
 * means filling every AIV of every cluster rather than one core of each.
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
    const int32_t aivs = rt_available_aiv_count();
    if (aivs < 1) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "topology reports %d AIV cores", aivs);
        return;
    }
    CoreTaskArgs cohort;
    cohort.add_inout(args.tensor(0).ref());
    cohort.launch_spec.set_block_num(aivs);
    cohort.launch_spec.set_require_sync_start(true);
    rt_submit_aiv_task(0, cohort);
}

}  // extern "C"
