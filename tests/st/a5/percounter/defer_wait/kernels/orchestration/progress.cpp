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

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 3};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const auto &out = args.tensor(0).ref();
    const int32_t waiters = static_cast<int32_t>(args.scalar<int64_t>(0));
    const int32_t consumers = static_cast<int32_t>(args.scalar<int64_t>(1));
    const uint32_t shape[]{32};
    TensorCreateInfo info(shape, 1, DataType::INT32);
    CoreTaskArgs initialize;
    initialize.add_output(info);
    initialize.add_no_dep(out);
    initialize.add_scalar(0, 0);
    const auto initialized = rt_submit_aiv_task(0, initialize);
    const auto state = initialized.get_ref(0);
    const TaskId init = initialized.task_id();
    TaskId first = TaskId::invalid();
    for (int32_t i = 0; i < waiters; ++i) {
        CoreTaskArgs waiter;
        waiter.add_no_dep(state, out);
        waiter.add_scalar(1, i);
        waiter.set_dependencies(&init, 1);
        const auto task = rt_submit_aiv_task(0, waiter).task_id();
        if (i == 0) first = task;
    }
    for (int32_t i = 0; i < consumers; ++i) {
        CoreTaskArgs consumer;
        consumer.add_no_dep(state, out);
        consumer.add_scalar(3, waiters + i);
        consumer.set_dependencies(&first, 1);
        (void)rt_submit_aiv_task(0, consumer);
    }
    CoreTaskArgs notifier;
    notifier.add_no_dep(state, out);
    notifier.add_scalar(2, 0);
    notifier.set_dependencies(&init, 1);
    (void)rt_submit_aiv_task(0, notifier);
}
