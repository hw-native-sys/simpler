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

constexpr int32_t kTaskCount = 512;
constexpr int32_t kPattern[] = {1, 0, 64, 0, 0, 1, 0, 64};

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 2};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::tmr::Tensor &counters = args.tensor(0).ref();
    const simpler::tmr::Tensor &records = args.tensor(1).ref();
    TaskId previous = TaskId::invalid();
    for (int32_t i = 0; i < kTaskCount; ++i) {
        CoreTaskArgs task;
        task.add_input(counters);
        task.add_inout(records);
        task.add_scalar(static_cast<int64_t>(i));
        task.add_scalar(static_cast<int64_t>(kPattern[i % 8]));
        task.launch_spec.set_block_num(1);
        if (i != 0) task.set_dependencies(&previous, 1);
        previous = rt_submit_aiv_task(0, task).task_id();
    }
}

}  // extern "C"
