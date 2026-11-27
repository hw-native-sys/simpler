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
    return OrchestrationConfig{.expected_arg_count = 2};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const auto &state = args.tensor(0).ref();
    const auto deferred = args.scalar<int64_t>(0);

    CoreTaskArgs producer;
    producer.add_no_dep(state);
    producer.add_scalar(deferred);
    const auto produced = rt_submit_aiv_task(0, producer).task_id();

    CoreTaskArgs consumer;
    consumer.add_no_dep(state);
    consumer.set_dependencies(&produced, 1);
    (void)rt_submit_aic_task(1, consumer);

    if (deferred != 0) {
        // Publish independently while the dependent consumer waits for DONE.
        CoreTaskArgs publisher;
        publisher.add_no_dep(state);
        (void)rt_submit_aiv_task(2, publisher);
    }
}
