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
#include <cstddef>
#include "common/hardware_spec.h"
#include "scheduler/scheduler_graph.h"

inline constexpr uint32_t WSE_PE_PER_CORE = wse::PE_PER_CORE;

// Immutable host-published image. Mutable scheduling state belongs to the PE runtime.
struct WseBlockPlan {
    uint32_t core_id, thread_num;
    uint32_t thread_pes[WSE_PE_PER_CORE];
    uint32_t tensor_count, scalar_count;
    uint64_t tensors_offset, scalars_offset;
};

struct WseNodePlan {
    uint64_t entry;
    uint32_t blocks;
    uint32_t host_done;
    uint32_t has_predicate;
    uint32_t block_plan_count;
    uint64_t block_plans_offset;
    // Descriptor-relative array of producer node IDs. A producer completes after all its Blocks.
    uint32_t wait_count;
    uint32_t sync_start;
    uint64_t waits_offset;
};

namespace wse {
// One row dispatches every Thread of a logical Block on its assigned Core.
struct CalendarRow {
    uint32_t task, block;
    // Unit-cost earliest start. This is an ordering witness, independent of wall time.
    uint32_t tick;
};
}  // namespace wse

struct WseGraphDescriptor {
    SchedulerGraphView graph;
    uint64_t nodes_address;
    uint64_t expected_tasks;
    uint64_t descriptor_bytes;
    uint32_t calendar_core_count, calendar_row_count;
    // Descriptor-relative offsets to Core row boundaries and packed rows.
    uint64_t calendar_indices_offset, calendar_rows_offset;
};

enum class WseEventKind : uint32_t {
    CONFIG_READY,
    CORE_INIT,
    PE_READY,
    GO,
    BLOCK_BEGIN,
    BLOCK_END,
    TASK_DONE,
    QUIESCENT,
    CORE_RETIRED,
    RUN_DONE,
    STOP,
    BLOCK_SKIPPED,
    OP_PUBLISH,
    OP_WAIT,
    OP_READ,
    OP_CONSUMED,
    THREAD_BEGIN,
    THREAD_END,
    BLOCK_READY,
    SYNC_RELEASE
};
struct WseEvent {
    uint64_t sequence;
    uint64_t task;
    uint64_t value;
    uint32_t core;
    uint32_t pe;
    uint32_t block;
    WseEventKind kind;
    uint32_t slot;
    uint64_t generation;
};

enum class WseSectionKind : uint32_t {
    LOCAL_TASK = 2,
    REGION_MAP = 4,
    TENSORS = 5,
    SCALARS = 6,
    DEPENDENCIES = 7,
    CORE_CONTROL = 8,
    BLOCK_QUEUE = 9,
    ALLOCATIONS = 10,
    TASK_SIGNAL = 11,
};
inline const char *wse_section_name(WseSectionKind kind) {
    switch (kind) {
    case WseSectionKind::LOCAL_TASK:
        return "local_task";
    case WseSectionKind::REGION_MAP:
        return "region_map";
    case WseSectionKind::TENSORS:
        return "tensors";
    case WseSectionKind::SCALARS:
        return "scalars";
    case WseSectionKind::DEPENDENCIES:
        return "dependencies";
    case WseSectionKind::CORE_CONTROL:
        return "core_control";
    case WseSectionKind::BLOCK_QUEUE:
        return "block_queue";
    case WseSectionKind::ALLOCATIONS:
        return "allocations";
    case WseSectionKind::TASK_SIGNAL:
        return "task_signal";
    }
    return "unknown";
}
struct WseMemorySection {
    uint64_t address;
    uint64_t bytes;
    uint32_t owner;
    WseSectionKind kind;
};

struct WseRun;
extern "C" {
// Version 3 publishes per-Core SRAM sections and a separate Host event stream.
// A mismatched Host/PE pair must fail at load time.
WseRun *wse_create_run_v3(
    const WseGraphDescriptor *, uint32_t cores, uint64_t epoch, const wse::SramRegion *, uint32_t region_count,
    const wse::SramAllocation *, uint32_t allocation_count, char *error, size_t error_size
);
uint64_t wse_runtime_used(const WseRun *, uint32_t core);
const WseMemorySection *wse_memory_sections(const WseRun *, uint64_t *count);
uint64_t wse_control_address(const WseRun *, uint32_t core);
void wse_destroy_run(WseRun *);
int wse_execute_pe(WseRun *, uint32_t worker);
// UINT32_MAX selects logical completions emitted during host preparation.
const WseEvent *wse_run_events(const WseRun *, uint32_t worker, uint64_t *count);
uint64_t wse_completed_tasks(const WseRun *);
}
