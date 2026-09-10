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
#include <cstring>
#include "orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 1};
}
__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const auto &state = orch_args.tensor(0).ref();
    uint32_t index[] = {0};
    float value = get_tensor_data<float>(state, 1, index);
    set_tensor_data<float>(state, 1, index, value + 2.0f);
    CoreTaskArgs args;
    args.add_inout(state);
    uint64_t bits = 0;
    float increment = 1.0f;
    std::memcpy(&bits, &increment, sizeof(increment));
    args.add_scalar(bits);
    rt_submit_aiv_task(0, args);
}
}
