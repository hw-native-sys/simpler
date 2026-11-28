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

#include <cstdint>

#include "common/lane_mask.h"
#include "common/platform_config.h"
#include "task_image_predicate.h"
#include "scheduler_types.h"

enum class SchedulerGraphResult : uint64_t {
    OK = 0,
    EMPTY = 1,
    INVALID_TASK_COUNT = 2,
    INVALID_TASK_ID = 3,
    UNSUPPORTED_SHAPE = 5,
    INVALID_ARGUMENTS = 6,
    INVALID_CALLABLE = 7,
    INVALID_FANIN_COUNT = 8,
    INVALID_FANIN_ID = 9,
    TIMEOUT = 10,
    ASYNC_COMPLETION_INVALID = 11,
    ASYNC_WAIT_OVERFLOW = 12,
    ASYNC_REGISTRATION_FAILED = 13,
};

inline constexpr bool
scheduler_task_shape_supported(uint32_t active_subtasks, uint32_t logical_block_num, bool sync_start) {
    return active_subtasks >= 1 && active_subtasks <= 3 && logical_block_num >= 1 &&
           logical_block_num <= static_cast<uint32_t>(INT16_MAX) / active_subtasks &&
           (!sync_start || logical_block_num > 1);
}

struct SchedulerMinimumWorkerDemand {
    uint32_t aic;
    uint32_t aiv;
};

inline constexpr SchedulerMinimumWorkerDemand scheduler_minimum_worker_demand(LaneMask active_mask) {
    const uint32_t vectors = (active_mask.has_aiv0() ? 1U : 0U) + (active_mask.has_aiv1() ? 1U : 0U);
    return {active_mask.has_aic() ? 1U : 0U, vectors};
}

inline constexpr bool scheduler_worker_demand_checked(
    LaneMask active_mask, uint32_t blocks, bool sync_start, uint64_t workers, SchedulerMinimumWorkerDemand *demand
) {
    if (demand == nullptr || active_mask.is_empty() || !active_mask.is_valid() || workers == 0 ||
        workers > PLATFORM_MAX_CORES || workers % PLATFORM_CORES_PER_BLOCKDIM != 0 ||
        !scheduler_task_shape_supported(static_cast<uint32_t>(active_mask.count()), blocks, sync_start))
        return false;
    auto required = scheduler_minimum_worker_demand(active_mask);
    const uint64_t clusters = workers / PLATFORM_CORES_PER_BLOCKDIM;
    if (sync_start) {
        if (active_mask.is_mix() && blocks > clusters) return false;
        required.aic *= blocks;
        required.aiv *= blocks;
    }
    if (required.aic > clusters || required.aiv > clusters * PLATFORM_AIV_CORES_PER_BLOCKDIM) return false;
    *demand = required;
    return true;
}

inline __aicore__ bool
scheduler_dispatch_predicate_metadata_valid(uint64_t address, uint8_t element_size, uint8_t operation) {
    if (operation == 0) return true;
    const bool valid_width = element_size == 1 || element_size == 2 || element_size == 4 || element_size == 8;
    return operation <= 6 && valid_width && address != 0 && (address & (static_cast<uint64_t>(element_size) - 1)) == 0;
}
