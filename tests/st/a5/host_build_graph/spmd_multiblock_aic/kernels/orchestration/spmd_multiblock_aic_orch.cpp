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
    return OrchestrationConfig{.expected_arg_count = 3};
}

// One SPMD task, its shape and width chosen by the caller: scalar 0 is the
// block count and scalar 1 picks the core type (0 = AIC, 1 = AIV). Both come
// from the test rather than being fixed here, because what a case needs depends
// on the topology it runs on -- how many cores of a type a box has decides
// whether a given width can demonstrate anything.
__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int64_t block_num = args.scalar<int64_t>(0);
    const int64_t aiv = args.scalar<int64_t>(1);
    CoreTaskArgs root;
    root.add_inout(args.tensor(0).ref());
    root.launch_spec.set_block_num(static_cast<int32_t>(block_num));
    if (aiv != 0) {
        rt_submit_aiv_task(1, root);
    } else {
        rt_submit_aic_task(0, root);
    }
}

}  // extern "C"
