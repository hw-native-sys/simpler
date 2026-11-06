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

#include <stdexcept>
#include "orchestration_api.h"
#include "../common.h"
using namespace sdma_series;
extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 7};
}
__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int64_t posts = args.scalar<int64_t>(2);
    if (posts < 1 || posts > kMaxPosts) throw std::runtime_error("SDMA series post count exceeds its fixture bound");
    constexpr auto clean = AsyncUseCacheContract::PUBLISHED_CLEAN_LINES;
    for (int64_t index = 0; index < posts; ++index) {
        const uint32_t shape[1]{static_cast<uint32_t>(elements(index))};
        const uint32_t start[1]{static_cast<uint32_t>(offset(index))};
        const auto source = args.tensor(0).ref().view(shape, start);
        const auto destination = args.tensor(1).ref().view(shape, start);
        const auto output = args.tensor(2).ref().view(shape, start);
        CoreTaskArgs task;
        task.add_no_dep(source);
        task.add_no_dep(destination);
        task.add_no_dep(output);
        task.add_no_dep(args.tensor(3).ref());
        task.add_scalar(args.scalar<int64_t>(0), args.scalar<int64_t>(1), index, elements(index), posts);
        const TaskId transfer = rt_submit_aiv_task(0, task).task_id();
        if (!rt_declare_async_sdma_get(transfer, 0, 1, 0, 1, clean))
            throw std::runtime_error("SDMA series transfer declaration failed");
        task.set_dependencies(&transfer, 1);
        rt_submit_aiv_task(1, task);
    }
}
}
