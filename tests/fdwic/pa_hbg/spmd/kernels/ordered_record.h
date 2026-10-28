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

#include "cla_record.h"
#include "record.h"

static inline __aicore__ void
ordered_record_execution(__gm__ int64_t *args, uint32_t slot, bool sync_start = false, int64_t wait_peer = -1) {
    auto *output = reinterpret_cast<__gm__ ExecutionRecord *>(tensor_data(args[0]));
    auto *all = reinterpret_cast<__gm__ ExecutionRecord *>(tensor_data(args[1]));
    auto *plan = tensor_data(args[2]);
    auto *row = plan + args[3] * cla_contract::kPlanWidth;
    const int64_t blocks = row[static_cast<uint32_t>(cla_contract::Field::Blocks)];
    const auto mask = static_cast<uint32_t>(row[static_cast<uint32_t>(cla_contract::Field::Mask)]);
    if (args[SPMD_LOCAL_CONTEXT_INDEX] == 0 || args[SPMD_GLOBAL_CONTEXT_INDEX] == 0) {
        scheduler_gm_fetch_add(output[blocks * 3].visits, 1);
        return;
    }
    const int32_t block = get_block_idx(args);
    if (block < 0 || block >= blocks || get_block_num(args) != blocks || !(mask & (1U << slot))) {
        scheduler_gm_fetch_add(output[blocks * 3].atomic_padding[0], 1);
        return;
    }
    // The first task's guard owns this atomic-only event counter. Diagnostic
    // order does not depend on synchronized clocks.
    const int64_t first_blocks = plan[static_cast<uint32_t>(cla_contract::Field::Blocks)];
    const int64_t first_base = plan[static_cast<uint32_t>(cla_contract::Field::Base)];
    auto &events = all[first_base + first_blocks * 3].atomic_padding[6];
    const uint64_t entered = scheduler_gm_fetch_add(events, 1) + 1;
    if (sync_start || wait_peer >= 0) scheduler_gm_fetch_add(output[blocks * 3].atomic_padding[3], 1);
    const int64_t errors = cla_check_predecessors(all, plan, row);
    if (sync_start) {
        record_sync_arrival(output[blocks * 3], static_cast<uint64_t>(blocks) * __builtin_popcount(mask), true);
    }
    if (block == row[static_cast<uint32_t>(cla_contract::Field::DelayedBlock)]) {
        if (wait_peer >= 0) {
            scheduler_gm_fetch_add(output[blocks * 3].atomic_padding[5], 1);
            auto *peer = plan + wait_peer * cla_contract::kPlanWidth;
            const auto peer_guard = peer[static_cast<uint32_t>(cla_contract::Field::Base)] +
                                    3 * peer[static_cast<uint32_t>(cla_contract::Field::Blocks)];
            const uint64_t started = get_sys_cnt_aicore();
            // Only the tail block waits. Other execution positions can close
            // their local obligation and enter the independent peer task.
            while (scheduler_gm_fetch_add(all[peer_guard].atomic_padding[3], 0) == 0) {
                if (get_sys_cnt_aicore() - started > PLATFORM_PROF_SYS_CNT_FREQ * 5) {
                    scheduler_gm_fetch_add(output[blocks * 3].atomic_padding[4], 1);
                    break;
                }
            }
        } else {
            for (volatile uint32_t delay = 0;
                 delay < static_cast<uint32_t>(row[static_cast<uint32_t>(cla_contract::Field::Delay)]); ++delay) {}
        }
    }
    auto *record = &output[block * 3 + slot];
    scheduler_gm_fetch_add(record->visits, 1);
    record->block_plus_one = block + 1;
    record->block_num = blocks;
    record->slot_plus_one = slot + 1;
    record->task_plus_one = row[static_cast<uint32_t>(cla_contract::Field::Tag)];
    record->metadata_padding[0] = errors;
    record->metadata_padding[1] = static_cast<int64_t>(entered);
    record->metadata_padding[2] = static_cast<int64_t>(scheduler_gm_fetch_add(events, 1) + 1);
    scheduler_publish_cache_line(&record->block_plus_one);
}
