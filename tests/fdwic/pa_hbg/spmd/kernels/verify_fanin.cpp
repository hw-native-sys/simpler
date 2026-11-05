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

#include "record.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *records = reinterpret_cast<__gm__ ExecutionRecord *>(tensor_data(args[0]));
    auto *summary = tensor_data(args[1]);
    const auto blocks = static_cast<int32_t>(args[2]);
    const auto mask = static_cast<uint32_t>(args[3]);
    const auto tasks = static_cast<int32_t>(args[4]);
    int64_t good = 0;
    int64_t bad = 0;
    int64_t guards = 0;
    int64_t bad_context = 0;
    for (int32_t task = 0; task < tasks; ++task) {
        for (int32_t block = 0; block < blocks; ++block) {
            for (int32_t slot = 0; slot < 3; ++slot) {
                auto *record = &records[task * (blocks * 3 + 1) + block * 3 + slot];
                const auto visits = scheduler_gm_fetch_add(record->visits, 0);
                scheduler_observe_cache_line(&record->block_plus_one);
                if (mask & (1U << slot)) {
                    const bool valid = visits == 1 && record->block_plus_one == block + 1 &&
                                       record->block_num == blocks && record->slot_plus_one == slot + 1 &&
                                       record->task_plus_one == task + 1;
                    good += valid ? 1 : 0;
                    bad += valid ? 0 : 1;
                } else {
                    bad += visits == 0 ? 0 : 1;
                }
            }
        }
        guards += static_cast<int64_t>(scheduler_gm_fetch_add(records[task * (blocks * 3 + 1) + blocks * 3].visits, 0));
        bad_context += static_cast<int64_t>(
            scheduler_gm_fetch_add(records[task * (blocks * 3 + 1) + blocks * 3].atomic_padding[0], 0)
        );
    }
    summary[0] = good;
    summary[1] = bad;
    summary[2] = guards;
    summary[3] = bad_context;
    scheduler_publish_cache_line(summary);
}
