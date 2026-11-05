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
 * One task, as wide as the caller asks and sync-start or not as the caller
 * asks. The pair is the point: the same width is refused for a cohort and run
 * for an ordinary task, because only a cohort needs every block resident at
 * once.
 *
 * Args layout: [out] + {block_num, sync_start} scalars.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 3};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int64_t block_num = args.scalar<int64_t>(0);
    const int64_t sync_start = args.scalar<int64_t>(1);

    CoreTaskArgs task;
    task.add_inout(args.tensor(0).ref());
    task.launch_spec.set_block_num(static_cast<int32_t>(block_num));
    if (sync_start != 0) task.launch_spec.set_require_sync_start(true);
    rt_submit_aic_task(0, task);
}

}  // extern "C"
