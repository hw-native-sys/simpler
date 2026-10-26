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

#pragma once

#include "execution_record.h"
#include "inner_kernel.h"

static inline __aicore__ void record_bounded_stall(__gm__ int64_t *args, int32_t lane) {
    auto *records = reinterpret_cast<__gm__ ExecutionRecord *>(tensor_data(args[0]));
    const auto expected_blocks = static_cast<int32_t>(args[1]);
    const auto mask = static_cast<uint32_t>(args[2]);
    const auto stalled_lane = static_cast<int32_t>(args[3]);
    const auto stall_ms = static_cast<uint64_t>(args[4]);
    const auto block = get_block_idx(args);
    const auto blocks = get_block_num(args);
    if (block < 0 || block >= expected_blocks || blocks != expected_blocks || (mask & (1U << lane)) == 0) {
        scheduler_gm_fetch_add(records[expected_blocks * 3].visits, 1);
        return;
    }
    auto &record = records[block * 3 + lane];
    const uint64_t start = get_sys_cnt_aicore();
    record.block_plus_one = block + 1;
    record.block_num = blocks;
    record.slot_plus_one = lane + 1;
    record.task_plus_one = 1;
    record.metadata_padding[0] = static_cast<int64_t>(start);
    scheduler_publish_cache_line(&record.block_plus_one);
    scheduler_gm_fetch_add(record.visits, 1);
    if (block == 0 && lane == stalled_lane) {
        const uint64_t duration = stall_ms * (PLATFORM_PROF_SYS_CNT_FREQ / 1000);
        while (get_sys_cnt_aicore() - start < duration) {}
    }
    record.metadata_padding[1] = static_cast<int64_t>(get_sys_cnt_aicore());
    scheduler_publish_cache_line(&record.block_plus_one);
    scheduler_gm_fetch_add(record.atomic_padding[0], 1);
}
