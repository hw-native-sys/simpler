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

#ifndef PTO_ORCHESTRATOR_H
#define PTO_ORCHESTRATOR_H

#include "common/l2_swimlane_profiling.h"
#include "utils/device_arena.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_submit_types.h"
#include "scheduler/pto_scheduler.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"

#include <stdarg.h>
#include <stdio.h>
#include <limits>
#include "aicpu/dep_gen_collector_aicpu.h"
#include "common/dep_gen.h"
#include "pto_dep_compute.h"
#include "tensor.h"

struct PTO2OrchestratorState;

// Helper aggregate types that pto_orchestrator.cpp constructs by value while
// preparing and submitting tasks.
struct PTO2PreparedTask {
    PTO2TaskId task_id = PTO2TaskId::invalid();
    PTO2TaskAllocResult alloc_result = {-1, 0, nullptr, nullptr};
    PTO2TaskDescriptor *task = nullptr;
    PTO2TaskPayload *payload = nullptr;
    PTO2TaskSlotState *slot_state = nullptr;
};

struct PTO2FaninBuilder {
    int32_t count{0};
    PTO2TaskSlotState *slots[PTO2_MAX_FANIN];
    int32_t local_ids[PTO2_MAX_FANIN];
    uint8_t ring_ids[PTO2_MAX_FANIN];

    bool contains(PTO2TaskSlotState *prod_state) const {
        for (int32_t i = 0; i < count; i++)
            if (slots[i] == prod_state) return true;
        return false;
    }
};

struct PTO2OrchestratorLayout {
    size_t off_scope_tasks;
    size_t off_scope_begins;
    PTO2TensorMapLayout tensor_map;
    int32_t dep_pool_capacity;
    int32_t scope_tasks_cap;
    uint64_t scope_stack_capacity;
};

struct PTO2OrchestratorState {
    // L2 swimlane profiling level — read by upstream aicpu_executor when
    // bridging orchestrator init into the scheduler context. The polling
    // design doesn't gate behavior on this directly, but the field must
    // exist for the upstream code path to compile.
    L2SwimlaneLevel l2_swimlane_level{L2SwimlaneLevel::DISABLED};

    // === SHARED MEMORY ACCESS ===
    PTO2SharedMemoryHeader *sm_header;

    // === PER-RING RESOURCES ===
    PTO2RingSet rings[PTO2_MAX_RING_DEPTH];
    uint32_t *fanin_seen_epoch[PTO2_MAX_RING_DEPTH];
    uint32_t fanin_seen_current_epoch{1};

    // === TENSOR MAP (Private) ===
    PTO2TensorMap tensor_map;  // Producer lookup

    PTO2TaskSlotState **scope_tasks;  // Flat buffer of taskSlotState (all scopes concatenated)
    int32_t scope_tasks_size;         // Number of task IDs currently in the buffer
    int32_t scope_tasks_capacity;     // Allocated capacity of scope_tasks
    int32_t *scope_begins;            // scope_begins[i] = start index of scope i in scope_tasks
    int32_t scope_stack_top;          // Current top of stack (-1 = no scope open)
    uint64_t scope_stack_capacity;    // Max nesting depth (PTO2_MAX_SCOPE_DEPTH)
    int32_t manual_begin_depth{PTO2_MAX_SCOPE_DEPTH};

    PTO2SchedulerState *scheduler;  // For simulated mode only

    // Total core counts set once at executor init; used for submit-time deadlock detection.
    int32_t total_cluster_count{0};  // AIC cores = MIX clusters
    int32_t total_aiv_count{0};      // AIV cores (= 2 × clusters on standard hardware)

    // === GM HEAP (for output buffers) ===
    void *gm_heap_base;     // Base address of GM heap
    uint64_t gm_heap_size;  // Total size of GM heap (all rings)

    bool fatal;

    int64_t inline_completed_tasks{0};

    // === STATISTICS ===

    uint8_t current_ring_id() const {
        int32_t depth = scope_stack_top;
        if (depth < 0) depth = 0;
        return depth < PTO2_MAX_RING_DEPTH ? static_cast<uint8_t>(depth) : PTO2_MAX_RING_DEPTH - 1;
    }

    bool in_manual_scope() const { return scope_stack_top >= manual_begin_depth; }

    // === Cold-path API ===

    static PTO2OrchestratorLayout
    reserve_layout(DeviceArena &arena, const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH], int32_t dep_pool_capacity);
    bool init_data_from_layout(
        const PTO2OrchestratorLayout &layout, DeviceArena &arena, void *sm_dev_base, void *gm_heap, uint64_t heap_size,
        uint64_t task_window_size
    );
    void wire_arena_pointers(const PTO2OrchestratorLayout &layout, DeviceArena &arena, PTO2SchedulerState *scheduler);
    void reset_for_reuse();
    void destroy();
    void set_scheduler(PTO2SchedulerState *scheduler);
    void report_fatal(int32_t error_code, const char *func, const char *fmt, ...);
    void begin_scope(PTO2ScopeMode mode);
    void end_scope();
    TaskOutputTensors submit_task(const MixedKernels &mixed_kernels, const L0TaskArgs &args);
    TaskOutputTensors submit_dummy_task(const L0TaskArgs &args);
    TaskOutputTensors alloc_tensors(const L0TaskArgs &args);
    void mark_done();
};

#endif  // PTO_ORCHESTRATOR_H
