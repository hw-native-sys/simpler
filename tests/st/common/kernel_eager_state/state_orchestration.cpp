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

#include "orchestration_api.h"

static void submit_state_write(const ChipTaskArgs &args, int32_t active, int32_t value) {
    for (int32_t i = 0; i < active; ++i) {
        CoreTaskArgs task;
        task.add_inout(args.tensor(0).ref());
        task.add_scalar(static_cast<int64_t>(value));
        rt_submit_aic_task(0, task);
    }
}

extern "C" void state_from_scalars(const ChipTaskArgs &args) {
    submit_state_write(args, args.scalar<int32_t>(0), args.scalar<int32_t>(1));
}

extern "C" void state_from_host(const ChipTaskArgs &args) {
    const uint32_t active_index[] = {0};
    const uint32_t value_index[] = {1};
    const auto &control = args.tensor(1).ref();
    submit_state_write(
        args, get_tensor_data<int32_t>(control, 1, active_index), get_tensor_data<int32_t>(control, 1, value_index)
    );
}
