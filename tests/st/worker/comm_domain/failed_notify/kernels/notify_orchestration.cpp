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
#include "arg_with_deps.h"

extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 5};
}
extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const auto &output = args.tensor(0).ref();
    const auto &signals = args.tensor(1).ref();
    const auto context = args.scalar(0);
    const auto mode = args.scalar(1);
    const auto rank = args.scalar(2);
    CoreTaskArgs producer;
    uint32_t shape[1] = {1};
    TensorCreateInfo token_info(shape, 1, DataType::UINT32);
    if (rank == 0) {
        producer.add_output(token_info);
    } else {
        producer.add_output(output);
    }
    producer.add_scalar(signals.buffer.addr);
    producer.add_scalar(context);
    producer.add_scalar(mode);
    auto token = rt_submit_aiv_task(0, producer);
    if (rank == 0) {
        CoreTaskArgsWithDeps<1> consumer;
        consumer.add_output(output);
        consumer.add_scalar(signals.buffer.addr);
        consumer.add_scalar(context);
        consumer.add_scalar(mode);
        consumer.add_dep(token.task_id());
        rt_submit_aiv_task(1, consumer);
    }
}
