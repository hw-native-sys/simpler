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
#include <algorithm>
#include "orchestration_api.h"

extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) { return {.expected_arg_count = 8}; }

extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const bool sync = args.scalar<int64_t>(0);
    const int blocks = args.scalar<int64_t>(1);
    const uint32_t second_core = args.scalar<uint32_t>(2);
    const int64_t probe = args.scalar<int64_t>(3);

    // Zero selects the independent SS that occupies one PE on Core 2.
    const int producer_count = args.scalar<int64_t>(4);
    const uint32_t producer_core = producer_count ? second_core : 2U;
    TaskId producer = TaskId::invalid();
    for (int i = 0; i < std::max(1, producer_count); ++i) {
        CoreTaskArgs busy;
        busy.add_output(args.tensor(0).ref());
        busy.add_scalar(0);
        busy.add_scalar(sync);
        busy.add_scalar(probe);
        busy.add_scalar(0);
        if (i) busy.set_dependencies(&producer, 1);
        producer = rt_submit_wse_task(0, busy, {.placement = {{producer_core, {3}}}}).task_id();
    }

    CoreTaskArgs members;
    members.add_output(args.tensor(1).ref());
    members.add_scalar(1);
    members.add_scalar(sync);
    members.add_scalar(probe);
    const Tensor &seed = args.tensor(0).ref();
    members.add_scalar(producer_count ? seed.buffer.addr + sizeof(float) * seed.start_offset : 0);
    if (producer_count) members.set_dependencies(&producer, 1);
    members.launch_spec.set_block_num(blocks);
    members.launch_spec.set_require_sync_start(sync);
    WseLaunchSpec launch{.thread_num = 2, .placement = {{2, {2, 3}}}};
    if (blocks == 2) launch.placement.push_back({second_core, {2, 3}});
    rt_submit_wse_task(0, members, launch);

    CoreTaskArgs join;
    join.add_input(args.tensor(0).ref());
    join.add_input(args.tensor(1).ref());
    join.add_output(args.tensor(2).ref());
    rt_submit_wse_task(1, join, {.placement = {{3}}});
}
