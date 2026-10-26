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

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 2};
}

// A Mix task with no AIC subtask: two AIV kernels and an empty AIC slot. The
// flags call this MIX because more than one subtask is live, so its blocks are
// placed by the cluster's AIC even though that AIC runs none of them -- which
// is what makes this shape the one where the coordinator's own dispatch slots
// stay empty while its cluster is fully committed.
//
// sync_start on top: every block has to be resident at once, and no core may
// hold two blocks of the cohort. For this shape "no core" has to be read
// through the Mix slots, because the coordinator's own slots never see it.
__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int64_t block_num = args.scalar<int64_t>(0);
    MixedKernels mixed;
    mixed.aiv0_kernel_id = 0;
    mixed.aiv1_kernel_id = 1;
    CoreTaskArgs cohort;
    cohort.add_inout(args.tensor(0).ref());
    cohort.launch_spec.set_block_num(static_cast<int32_t>(block_num));
    cohort.launch_spec.set_require_sync_start(true);
    rt_submit_task(mixed, cohort);
}

}  // extern "C"
