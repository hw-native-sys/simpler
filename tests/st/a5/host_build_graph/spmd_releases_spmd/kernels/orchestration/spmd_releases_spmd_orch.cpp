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
 * A chain of SPMD tasks, each released by the last block of the one before it.
 *
 * Every stage is SPMD, so the successor a completion resolves is a task no
 * single core may run: it belongs in the queue the whole machine claims blocks
 * from, never in the slot that just freed. The chain puts one stage at a time
 * in that queue and empties it in between, so each stage arrives at a queue
 * every core has already seen empty.
 *
 * Args layout: [out] + {block_num, stages, repeats} scalars.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {
constexpr int32_t kStageFuncId = 0;
}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &out = args.tensor(0).ref();
    const int64_t block_num = args.scalar<int64_t>(0);
    const int64_t stages = args.scalar<int64_t>(1);
    const int64_t repeats = args.scalar<int64_t>(2);

    TaskId previous{};
    for (int64_t stage = 0; stage < stages; ++stage) {
        CoreTaskArgs task;
        task.add_no_dep(out);
        task.add_scalar(block_num);
        task.add_scalar(stage);
        task.add_scalar(repeats);
        task.launch_spec.set_block_num(static_cast<int32_t>(block_num));
        // Declared rather than inferred from the tensor, so a stage waits on
        // the whole stage before it and not on whichever blocks overlap it.
        if (stage > 0) task.set_dependencies(&previous, 1);
        previous = rt_submit_aic_task(kStageFuncId, task).task_id();
    }
}

}  // extern "C"
