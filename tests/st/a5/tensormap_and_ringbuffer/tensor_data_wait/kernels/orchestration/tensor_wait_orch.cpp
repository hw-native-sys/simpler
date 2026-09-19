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
extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 2};
}
extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    uint32_t shape[1] = {1}, index[1] = {0};
    CoreTaskArgs task;
    TensorCreateInfo info(shape, 1, DataType::INT32);
    task.add_output(info);
    task.add_scalar(args.scalar<int64_t>(0));
    auto outputs = rt_submit_aiv_task(0, task);

    // Independent completions keep the CI 5 s scheduler watchdog progressing.
    // They must not renew the scalar producer's tensor-data deadline.
    TaskId previous = TaskId::invalid();
    for (int64_t elapsed_ms = 0; elapsed_ms < args.scalar<int64_t>(0); elapsed_ms += 2000) {
        CoreTaskArgs progress;
        progress.add_output(info);
        progress.add_scalar(int64_t{2000});
        if (previous.is_valid()) progress.set_dependencies(&previous, 1);
        previous = rt_submit_aiv_task(0, progress).task_id();
    }
    int32_t value = get_tensor_data<int32_t>(outputs.get_ref(0), 1, index);
    if (rt_is_fatal()) return;
    set_tensor_data(args.tensor(0).ref(), 1, index, value);
}
