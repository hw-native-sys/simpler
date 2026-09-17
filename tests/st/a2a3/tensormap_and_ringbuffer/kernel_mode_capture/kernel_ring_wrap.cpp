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

extern "C" void kernel_ring_wrap(const ChipTaskArgs &args) {
    // Each scope releases its task references; 257 tasks exceed the 64-slot ring four times.
    for (int i = 0; i < 257; ++i) {
        SIMPLER_SCOPE_GUARD();
        CoreTaskArgs task;
        task.add_input(args.tensor(i == 0 ? 0 : 1).ref());
        task.add_output(args.tensor(1).ref());
        task.add_scalar(args.scalar(0));
        rt_submit_aiv_task(0, task);
    }
}
