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

#include "scheduler/scheduler_ready.h"

inline uint32_t hbg_test_eligible_mask(const SchedulerLocalState *local) {
    uint32_t mask = 0;
    for (uint32_t id = 0; id < SCHEDULER_LOCAL_EXECUTION_COUNT; ++id)
        if (local->execution_at(id).state == SchedulerDispatchSlotState::READY) mask |= 1U << id;
    return mask;
}

inline bool hbg_test_find_eligible(const SchedulerLocalState *local, uint32_t *index) {
    *index = scheduler_cooperative_execution(local);
    if (*index != UINT32_MAX) return true;
    for (uint32_t id = 0; id < SCHEDULER_EXECUTION_TOKEN_COUNT; ++id) {
        if (local->execution_at(id).state != SchedulerDispatchSlotState::READY) continue;
        *index = id;
        return true;
    }
    return false;
}

inline bool hbg_test_begin_execution(SchedulerLocalState *local, uint32_t index) {
    if (index >= SCHEDULER_LOCAL_EXECUTION_COUNT) return false;
    auto &owner = local->execution_at(index);
    if (owner.state != SchedulerDispatchSlotState::READY) return false;
    owner.state = SchedulerDispatchSlotState::EXECUTING;
    return true;
}
