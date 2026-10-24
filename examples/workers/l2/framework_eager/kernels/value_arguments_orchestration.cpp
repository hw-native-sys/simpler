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
void submit_values(const ChipTaskArgs &args, bool host_control) {
    const float scale = args.scalar<float>(0);
    const int8_t signed_bias = args.scalar<int8_t>(1);
    const uint64_t selector = args.scalar<uint64_t>(2);
    const double precise = args.scalar<double>(3);
    float bias = static_cast<float>(signed_bias) + static_cast<float>(selector & 0xFFu);
    if ((selector >> 63u) != 0) bias += 16.0F;
    if (precise > 1.0) bias += 32.0F;
    if (precise == 0.0 && (args.scalar<uint64_t>(3) >> 63u) != 0) bias += 64.0F;
    if (host_control) {
        const uint32_t index[] = {0};
        bias += static_cast<float>(get_tensor_data<int32_t>(args.tensor(2).ref(), 1, index));
    }
    CoreTaskArgs task;
    task.add_input(args.tensor(0).ref());
    task.add_output(args.tensor(1).ref());
    task.add_scalar(to_u64(scale));
    task.add_scalar(to_u64(bias));
    rt_submit_aiv_task(0, task);
}
}  // namespace

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig framework_value_arguments_config(const ChipTaskArgs &) {
    return {.expected_arg_count = 6};
}
__attribute__((visibility("default"))) void framework_value_arguments(const ChipTaskArgs &args) {
    submit_values(args, false);
}
__attribute__((visibility("default"))) OrchestrationConfig framework_host_value_arguments_config(const ChipTaskArgs &) {
    return {.expected_arg_count = 7};
}
__attribute__((visibility("default"))) void framework_host_value_arguments(const ChipTaskArgs &args) {
    submit_values(args, true);
}
}
