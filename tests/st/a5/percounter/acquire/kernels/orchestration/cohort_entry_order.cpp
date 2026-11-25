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

#include <cstdint>

#include "orchestration_api.h"

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 2};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    CoreTaskArgs slow;
    slow.add_inout(args.tensor(0).ref());
    slow.add_scalar(int64_t{0});
    slow.add_scalar(int64_t{50});
    rt_submit_aic_task(0, slow);

    CoreTaskArgs cohort;
    cohort.add_inout(args.tensor(1).ref());
    cohort.launch_spec.set_block_num(rt_available_cluster_count());
    cohort.launch_spec.set_require_sync_start(true);
    rt_submit_aic_task(1, cohort);
}
}
