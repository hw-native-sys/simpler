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

#include <cstddef>
#include <cstdint>

#include "common/lane_mask.h"

struct ReadyQueueCapacities {
    uint64_t ready[NUM_RESOURCE_SHAPES]{};
    uint64_t ready_sync[NUM_RESOURCE_SHAPES]{};
    uint64_t dummy{0};
    uint64_t graph_prepare{0};
};

struct SchedulerLayout {
    size_t off_ready_queue_slots[NUM_RESOURCE_SHAPES];
    size_t off_ready_sync_queue_slots[NUM_RESOURCE_SHAPES];
    size_t off_dummy_ready_queue_slots;
    size_t off_graph_prepare_queue_slots;
    size_t off_early_dispatch_queue_slots[NUM_RESOURCE_SHAPES];
    size_t off_early_sync_start_queue_slots;
    size_t off_ed_publish_drain_queue_slots;
    ReadyQueueCapacities capacities;
};
