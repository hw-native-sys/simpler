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
#include "orchestration_api.h"  // NOLINT(build/include_subdir)
namespace {
void depends(CoreTaskArgs &task, const TaskId &dependency) { task.set_dependencies(&dependency, 1); }
TaskId wait(const ChipTaskArgs &args, uint32_t signal, TaskId dependency) {
    CoreTaskArgs task;
    task.add_no_dep(args.tensor(signal).ref());
    task.add_scalar(args.scalar<int64_t>(1), args.scalar<int64_t>(2), args.scalar<int64_t>(3));
    depends(task, dependency);
    return rt_submit_aiv_task(4, task).task_id();
}
}  // namespace
extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 17};
}
__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const bool aic = args.scalar<int64_t>(5) == 0;
    const int32_t blocks = aic ? rt_available_cluster_count() : rt_available_aiv_count();
    always_assert(blocks > 0 && blocks <= 128);
    CoreTaskArgs warm;
    for (uint32_t field = 2; field < 6; ++field)
        warm.add_no_dep(args.tensor(field).ref());
    warm.add_no_dep(args.tensor(10).ref());
    warm.add_scalar(args.scalar<int64_t>(1), args.scalar<int64_t>(4));
    warm.launch_spec.set_block_num(blocks);
    warm.launch_spec.set_require_sync_start(true);
    const auto warmed = (aic ? rt_submit_aic_task(0, warm) : rt_submit_aiv_task(1, warm)).task_id();

    CoreTaskArgs ready;
    ready.add_no_dep(args.tensor(1).ref());
    ready.add_scalar(args.scalar<uint64_t>(0));
    depends(ready, warmed);
    const auto sent_ready = rt_submit_aiv_task(5, ready).task_id();
    const auto peers_ready = wait(args, 1, sent_ready);
    const auto arrived = wait(args, 0, warmed);

    CoreTaskArgs publish;
    publish.add_no_dep(args.tensor(0).ref());
    for (uint32_t field = 2; field < 6; ++field)
        publish.add_no_dep(args.tensor(field).ref());
    publish.add_scalar(args.scalar<uint64_t>(0), args.scalar<int64_t>(3));
    depends(publish, peers_ready);
    (void)rt_submit_aiv_task(6, publish);

    CoreTaskArgs gather;
    for (uint32_t field = 2; field < 11; ++field)
        gather.add_no_dep(args.tensor(field).ref());
    gather.add_scalar(args.scalar<int64_t>(1));
    gather.launch_spec.set_block_num(blocks);
    gather.launch_spec.set_require_sync_start(true);
    depends(gather, arrived);
    if (aic) (void)rt_submit_aic_task(2, gather);
    else (void)rt_submit_aiv_task(args.scalar<int64_t>(5) == 2 ? 7 : 3, gather);
}
}  // extern "C"
