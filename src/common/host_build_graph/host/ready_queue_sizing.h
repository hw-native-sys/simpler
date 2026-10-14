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

#include "scheduler/scheduler.h"

struct ReadyQueuePopulations {
    uint64_t ready[NUM_RESOURCE_SHAPES]{};
    uint64_t ready_sync[NUM_RESOURCE_SHAPES]{};
    uint64_t dummy{0};
    uint64_t graph_prepare{0};

    void add_task(ActiveMask active_mask, TaskAttrs task_attrs, TaskKind task_kind, uint64_t count = 1);
    void add(const ReadyQueuePopulations &other);
    bool derive_capacities(ReadyQueueCapacities *capacities) const;
};

int32_t derive_ready_queue_capacities(const ReadyQueuePopulations &populations, ReadyQueueCapacities *capacities);
