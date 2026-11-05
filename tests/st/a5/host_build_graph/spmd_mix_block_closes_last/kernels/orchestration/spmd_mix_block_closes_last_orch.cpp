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
 * A Mix SPMD producer whose halves finish far apart, and one consumer.
 *
 * Aggregation here is two levels. A block closes when its own subtasks have
 * all reported, which the claiming AIC tracks locally; the task closes when
 * every block has, which a counter the task owns tracks across the machine.
 * Collapsing the two -- counting subtasks where blocks are meant -- reaches
 * the task's total early and releases the consumer onto half-written data.
 *
 * Args layout: [out] + {block_num, repeats} scalars.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {
constexpr int32_t kSlowFuncId = 0;
constexpr int32_t kFastFuncId = 1;
constexpr int32_t kSumFuncId = 2;
}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 3};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &out = args.tensor(0).ref();
    const int64_t block_num = args.scalar<int64_t>(0);
    const int64_t repeats = args.scalar<int64_t>(1);

    // The AIC subtask is the slow one and the two AIV subtasks return at once,
    // so a block's first report and its last are far apart.
    MixedKernels mixed;
    mixed.aic_kernel_id = kSlowFuncId;
    mixed.aiv0_kernel_id = kFastFuncId;
    mixed.aiv1_kernel_id = kFastFuncId;

    CoreTaskArgs producer;
    producer.add_no_dep(out);
    producer.add_scalar(repeats);
    producer.launch_spec.set_block_num(static_cast<int32_t>(block_num));
    const TaskId produced = rt_submit_task(mixed, producer).task_id();

    CoreTaskArgs consumer;
    consumer.add_no_dep(out);
    consumer.add_scalar(block_num);
    consumer.set_dependencies(&produced, 1);
    (void)rt_submit_aic_task(kSumFuncId, consumer);
}

}  // extern "C"
