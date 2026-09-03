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

constexpr uint32_t SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING = 1;
constexpr uint32_t SCHEDULER_RUNTIME_MODE_RESIDENT_READY = 2;
// A zeroed Handshake is not a scheduler choice. Every supported fallback is
// published explicitly so an ordinary run cannot silently drift back to the
// legacy dependency scheduler while continuing to pass result-only tests.
constexpr uint32_t SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH = 3;
constexpr uint32_t SCHEDULER_RUNTIME_MODE_LEGACY_UNSUPPORTED_SHAPE = 4;

struct AicoreSchedulerLayout {
    uint64_t total_size;
    uint64_t task_count;
    uint64_t aic_task_count;
    uint64_t aiv_task_count;
    uint64_t run_control_offset;
    uint64_t aicpu_lifecycle_traces_offset;
    uint64_t worker_contexts_offset;
    uint64_t dispatch_payloads_offset;
    uint64_t dispatch_slots_offset;
    uint64_t callable_addresses_offset;
    uint64_t task_metadata_offset;
    uint64_t task_controls_offset;
    uint64_t completion_inboxes_offset;
    uint64_t ready_inboxes_offset;
    uint64_t ready_owner_states_offset;
    uint64_t ready_directory_offset;
    uint64_t trace_cells_offset;
    uint64_t activity_buffers_offset;
    uint64_t gang_coordinator_offset;
    uint64_t gang_cohorts_offset;
    uint64_t gang_participants_offset;
    uint64_t gang_commands_offset;
    uint64_t executable_task_count;
    uint64_t executable_subtask_count;
    uint64_t gang_task_count;
    uint64_t aic_worker_demand;
    uint64_t aiv_worker_demand;
};
