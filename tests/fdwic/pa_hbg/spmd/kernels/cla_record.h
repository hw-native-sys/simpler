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

#include "cla_plan.h"
#include "execution_record.h"

static inline __aicore__ int64_t
cla_check_predecessors(__gm__ ExecutionRecord *all, __gm__ int64_t *plan, __gm__ int64_t *row) {
    int64_t errors = 0;
    const auto count = static_cast<uint32_t>(row[static_cast<uint32_t>(cla_contract::Field::CheckCount)]);
    for (uint32_t i = 0; i < count; ++i) {
        auto *pred = plan + row[cla_contract::kCheckOffset + i] * cla_contract::kPlanWidth;
        const int64_t n = pred[static_cast<uint32_t>(cla_contract::Field::Blocks)];
        const auto mask = static_cast<uint32_t>(pred[static_cast<uint32_t>(cla_contract::Field::Mask)]);
        const int64_t base = pred[static_cast<uint32_t>(cla_contract::Field::Base)];
        const int64_t tag = pred[static_cast<uint32_t>(cla_contract::Field::Tag)];
        for (int64_t block = 0; block < n; ++block) {
            for (uint32_t slot = 0; slot < 3; ++slot) {
                if ((mask & (1U << slot)) == 0) continue;
                auto *record = &all[base + block * 3 + slot];
                const auto visits = scheduler_gm_fetch_add(record->visits, 0);
                scheduler_observe_cache_line(&record->block_plus_one);
                errors += (visits != 1 || record->block_plus_one != block + 1 || record->block_num != n ||
                           record->slot_plus_one != slot + 1 || record->task_plus_one != tag) ?
                              1 :
                              0;
            }
        }
    }
    return errors;
}

static inline __aicore__ void cla_record_execution(__gm__ int64_t *args, uint32_t slot) {
    auto *output = reinterpret_cast<__gm__ ExecutionRecord *>(tensor_data(args[0]));
    auto *all = reinterpret_cast<__gm__ ExecutionRecord *>(tensor_data(args[1]));
    auto *plan = tensor_data(args[2]);
    auto *row = plan + args[3] * cla_contract::kPlanWidth;
    const int64_t n = row[static_cast<uint32_t>(cla_contract::Field::Blocks)];
    const auto mask = static_cast<uint32_t>(row[static_cast<uint32_t>(cla_contract::Field::Mask)]);
    if (args[SPMD_LOCAL_CONTEXT_INDEX] == 0 || args[SPMD_GLOBAL_CONTEXT_INDEX] == 0) {
        scheduler_gm_fetch_add(output[n * 3].visits, 1);
        return;
    }
    const int32_t block = get_block_idx(args);
    if (block < 0 || block >= n || get_block_num(args) != n || !(mask & (1U << slot))) {
        scheduler_gm_fetch_add(output[n * 3].atomic_padding[0], 1);
        return;
    }
    // Errors are sticky per execution: a later producer return cannot repair an early consumer.
    const int64_t errors = cla_check_predecessors(all, plan, row);
    if (block == row[static_cast<uint32_t>(cla_contract::Field::DelayedBlock)]) {
        for (volatile uint32_t delay = 0;
             delay < static_cast<uint32_t>(row[static_cast<uint32_t>(cla_contract::Field::Delay)]); ++delay) {}
    }
    auto *record = &output[block * 3 + slot];
    scheduler_gm_fetch_add(record->visits, 1);
    record->block_plus_one = block + 1;
    record->block_num = n;
    record->slot_plus_one = slot + 1;
    record->task_plus_one = row[static_cast<uint32_t>(cla_contract::Field::Tag)];
    record->metadata_padding[0] = errors;
    scheduler_publish_cache_line(&record->block_plus_one);
}
