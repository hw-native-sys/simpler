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
    return OrchestrationConfig{.expected_arg_count = 8};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const auto &input = args.tensor(0).ref();
    const auto &receive = args.tensor(1).ref();
    const auto &out = args.tensor(2).ref();
    const auto &result = args.tensor(3).ref();
    const auto ctx = args.scalar<uint64_t>(0);
    const auto elems = args.scalar<uint32_t>(1);
    const auto mode = args.scalar<uint32_t>(2);
    if (mode == 1) {
        CoreTaskArgs write;
        write.add_no_dep(input, receive, out, input);
        write.add_scalar(ctx, elems, uint32_t{1}, uint32_t{1});
        rt_submit_aiv_task(0, write);
        return;
    }
    CoreTaskArgs read;
    const auto &source = mode == 2 ? receive : input;
    read.add_no_dep(source, source);
    read.add_output(out);
    read.add_no_dep(input);
    read.add_scalar(ctx, elems, mode, uint32_t{1});
    const auto fetched = rt_submit_aiv_task(0, read).task_id();
    CoreTaskArgs consumer;
    consumer.add_input(out);
    consumer.add_output(result);
    consumer.add_scalar(elems);
    consumer.set_dependencies(&fetched, 1);
    rt_submit_aiv_task(1, consumer);
}
