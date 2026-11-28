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

extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) { return {.expected_arg_count = 4}; }

extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    auto submit = [&](int output, WseBlockPlacement placement, const TaskId *wait = nullptr) {
        CoreTaskArgs task;
        task.add_output(args.tensor(output).ref());
        if (wait) task.set_dependencies(wait, 1);
        return rt_submit_wse_task(0, task, {.placement = {placement}}).task_id();
    };
    const auto first = submit(0, {0}), second = submit(1, {1});
    submit(2, {2, {3, 2}}, &first);
    submit(3, {2, {3}}, &second);
}
