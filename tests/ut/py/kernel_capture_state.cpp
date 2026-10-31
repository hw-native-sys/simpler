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

extern "C" void captured_state(const ChipTaskArgs &args) {
    const uint32_t command_index[] = {0};
    const uint32_t entered_index[] = {1};
    const auto &control = args.tensor(1).ref();
    const int32_t command = get_tensor_data<int32_t>(control, 1, command_index);
    const int32_t entered = get_tensor_data<int32_t>(control, 1, entered_index);
    set_tensor_data(control, 1, entered_index, entered + 1);
    if (command == 2) {
        rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "injected captured orchestration failure");
        return;
    }
    if (command == 0) return;
    CoreTaskArgs task;
    task.add_inout(args.tensor(0).ref());
    task.add_scalar(static_cast<int64_t>(args.scalar<int32_t>(0)));
    rt_submit_aic_task(0, task);
}
