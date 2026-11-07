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

#include "host_build_graph/host/ready_queue_sizing.h"

#include "host_build_graph/runtime_status.h"

namespace {

uint64_t capacity_for_population(uint64_t population) {
    uint64_t capacity = 2;
    while (capacity < population)
        capacity <<= 1;
    return capacity;
}

}  // namespace

void ReadyQueuePopulations::add(const ReadyQueuePopulations &other) {
    for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i) {
        add_population(&ready[i], other.ready[i]);
        add_population(&ready_sync[i], other.ready_sync[i]);
    }
    add_population(&dummy, other.dummy);
    add_population(&graph_prepare, other.graph_prepare);
}

bool ReadyQueuePopulations::derive_capacities(ReadyQueueCapacities *capacities, uint64_t reservation) const {
    if (capacities == nullptr || reservation < 2 || reservation > static_cast<uint64_t>(TaskId::GLOBAL_TASK_MAX_NUM) ||
        (reservation & (reservation - 1)) != 0) {
        return false;
    }
    for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i) {
        if (ready[i] > reservation || ready_sync[i] > reservation) return false;
    }
    if (dummy > reservation || graph_prepare > reservation) {
        return false;
    }

    *capacities = ReadyQueueCapacities{};
    for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i) {
        capacities->ready[i] = capacity_for_population(ready[i]);
        capacities->ready_sync[i] = capacity_for_population(ready_sync[i]);
    }
    capacities->dummy = capacity_for_population(dummy);
    capacities->graph_prepare = 2 * capacity_for_population(graph_prepare);
    return true;
}

int32_t derive_ready_queue_capacities(
    const ReadyQueuePopulations &populations, ReadyQueueCapacities *capacities, uint64_t reservation
) {
    if (populations.derive_capacities(capacities, reservation)) return 0;

    return runtime_status_from_error_code(SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
}
