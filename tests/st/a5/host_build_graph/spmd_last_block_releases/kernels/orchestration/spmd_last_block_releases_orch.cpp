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
 * An SPMD producer and a single consumer that depends on all of it.
 *
 * Only the producer's last block may release the consumer. A block reports
 * completion like any other dispatch, and what keeps the task from being
 * declared done on the first of them is a counter the task owns; every block
 * but the last leaves the task open and resolves nothing.
 *
 * Args layout: [out] + {block_num, repeats} scalars.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {
constexpr int32_t kBlockFuncId = 0;
constexpr int32_t kSumFuncId = 1;
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

    CoreTaskArgs producer;
    producer.add_no_dep(out);
    producer.add_scalar(repeats);
    producer.launch_spec.set_block_num(static_cast<int32_t>(block_num));
    const TaskId produced = rt_submit_aic_task(kBlockFuncId, producer).task_id();

    // The dependency is declared rather than inferred from the tensor, so what
    // the consumer waits on is the whole producer task and not some overlap
    // the runtime worked out per block.
    CoreTaskArgs consumer;
    consumer.add_no_dep(out);
    consumer.add_scalar(block_num);
    consumer.set_dependencies(&produced, 1);
    (void)rt_submit_aic_task(kSumFuncId, consumer);
}

}  // extern "C"
