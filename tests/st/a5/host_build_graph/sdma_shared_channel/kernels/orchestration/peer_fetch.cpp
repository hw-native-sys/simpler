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
#include <stdexcept>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 8};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const auto &input = args.tensor(0).ref();
    const auto &out = args.tensor(1).ref();
    const auto &result = args.tensor(2).ref();
    CoreTaskArgs producer;
    producer.add_input(input);
    producer.add_output(out);
    producer.add_inout(args.tensor(3).ref());
    producer.add_scalar(
        args.scalar<uint64_t>(0), args.scalar<uint64_t>(1), args.scalar<uint64_t>(2), args.scalar<uint64_t>(3)
    );
    rt_submit_task({INVALID_KERNEL_ID, 0, 0}, producer);
    CoreTaskArgs consumer;
    consumer.add_input(out);
    consumer.add_output(result);
    rt_submit_aiv_task(1, consumer);
}
