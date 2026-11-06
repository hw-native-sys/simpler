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

#include "orchestration_api.h"

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 7};
}
static void run_sdma(const ChipTaskArgs &args, bool declared) {
    CoreTaskArgs task;
    for (int32_t i = 0; i < 4; ++i)
        task.add_no_dep(args.tensor(i).ref());
    task.add_scalar(args.scalar<int64_t>(0), args.scalar<int64_t>(1), args.scalar<int64_t>(2));
    const TaskId transfer = rt_submit_aiv_task(0, task).task_id();
    constexpr auto clean = AsyncUseCacheContract::PUBLISHED_CLEAN_LINES;
    if (declared && !rt_declare_async_sdma_get(transfer, 0, 1, 0, 2, clean))
        throw std::runtime_error("SDMA transfer resource declaration failed");
    // The independent probe observes only the public event receipt; it never
    // reads a payload that may still be written by SDMA.
    CoreTaskArgs probe;
    probe.add_no_dep(args.tensor(3).ref());
    rt_submit_aiv_task(2, probe);
    task.set_dependencies(&transfer, 1);
    rt_submit_aiv_task(1, task);
}
__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    run_sdma(args, true);
}
__attribute__((visibility("default"))) void unannotated_orchestration_entry(const ChipTaskArgs &args) {
    run_sdma(args, false);
}
}
