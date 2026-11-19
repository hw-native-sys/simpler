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

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig framework_persistent_state_config(const ChipTaskArgs &args) {
    (void)args;
    return {.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void framework_persistent_state(const ChipTaskArgs &args) {
    CoreTaskArgs task;
    task.add_input(args.tensor(0).ref());
    task.add_inout(args.tensor(1).ref());
    task.add_input(args.tensor(2).ref());
    task.add_input(args.tensor(3).ref());
    rt_submit_aiv_task(0, task);
}
}
