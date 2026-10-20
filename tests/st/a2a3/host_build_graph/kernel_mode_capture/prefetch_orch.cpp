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

extern "C" __attribute__((visibility("default"))) OrchestrationConfig
prefetch_async_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 2};
}

extern "C" __attribute__((visibility("default"))) void prefetch_async_orchestration(const ChipTaskArgs &args) {
    const auto &input = args.tensor(0).ref();
    const auto &output = args.tensor(1).ref();
    TensorCreateInfo intermediate(input.shapes, input.ndims, DataType::FLOAT32);
    CoreTaskArgs first;
    first.add_input(input);
    first.add_output(intermediate);
    auto result = rt_submit_aiv_task(0, first);
    CoreTaskArgs second;
    second.add_input(result.get_ref(0));
    second.add_output(output);
    rt_submit_aiv_task(0, second);
}
