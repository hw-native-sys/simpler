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

extern "C" void queue_pressure(const ChipTaskArgs &args) {
    const uint32_t count = args.scalar<uint32_t>(1);
    const uint32_t blocks = args.scalar<uint32_t>(2);
    const bool dependencies = args.scalar<uint32_t>(3) != 0;
    rt_scope_begin(ScopeMode::MANUAL);
    TaskId tasks[128];
    for (uint32_t i = 0; i < count; ++i) {
        CoreTaskArgsWithDeps<128> task;
        task.add_output(args.tensor(0).ref());
        task.add_scalar(args.scalar(0));
        task.add_scalar(i);
        task.add_scalar(blocks);
        task.launch_spec.set_block_num(blocks);
        task.launch_spec.set_require_sync_start(dependencies);
        if (dependencies && i > 0) {
            if (i == count - 1) {
                for (uint32_t j = 1; j < i; ++j)
                    task.add_dep(tasks[j]);
            } else {
                task.add_dep(tasks[0]);
            }
        }
        tasks[i] = rt_submit_aiv_task(0, task).task_id();
    }
    rt_scope_end();
}
