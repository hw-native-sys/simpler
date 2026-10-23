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

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" void deferred_counter(const ChipTaskArgs &args) {
    const auto &output = args.tensor(1).ref();
    const uint32_t shape[1] = {16};
    for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t offset[1] = {i * 16};
        auto view = output.view(shape, offset);
        CoreTaskArgs task;
        task.add_input(args.tensor(0).ref());
        task.add_output(view);
        task.add_scalar(args.scalar(0));
        rt_submit_aiv_task(0, task);
    }
}
