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

/**
 * PTO Runtime2 - Orchestrator Implementation
 *
 * Implements orchestrator state management, scope handling, and task submission.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_orchestrator.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/unified_log.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"
#include "tensor.h"

// =============================================================================
// Orchestrator Profiling (compile-time toggle)
// =============================================================================
#if PTO2_ORCH_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/performance_collector_aicpu.h"
// Weak fallback for builds that don't link device_time.cpp (e.g. host).
// The strong symbol from platform/.../device_time.cpp wins in the AICPU build.
//
// IMPORTANT: visibility("hidden") is required to prevent the HOST .so from
// exporting this weak fallback into the global dynamic symbol table via
// RTLD_GLOBAL. Without it, when the AICPU .so is loaded and its PLT entry
// for get_sys_cnt_aicpu is resolved, the dynamic linker finds the HOST .so's
// weak definition first (already in global table) and uses it — returning 0.
// With hidden visibility, the HOST .so does not export this symbol globally,
// so the AICPU .so's PLT resolves to its own strong definition from
// device_time.cpp.
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }
// Weak fallback for builds that don't link performance_collector_aicpu.cpp.
// The strong symbol from the AICPU build wins when profiling is available.
// Also hidden to prevent HOST .so from polluting the global symbol table.
__attribute__((weak, visibility("hidden"))) void
perf_aicpu_record_orch_phase(AicpuPhaseId, uint64_t, uint64_t, uint32_t, uint64_t) {}
// Accumulated cycles per sub-step (only needed for ORCH_PROFILING export)
static uint64_t g_orch_sync_cycle = 0;       // tensormap sync
static uint64_t g_orch_alloc_cycle = 0;      // unified task+heap alloc
static uint64_t g_orch_args_cycle = 0;       // param copy
static uint64_t g_orch_lookup_cycle = 0;     // tensormap lookup + dep building
static uint64_t g_orch_insert_cycle = 0;     // tensormap insert
static uint64_t g_orch_fanin_cycle = 0;      // fanin list + early-return check
static uint64_t g_orch_scope_end_cycle = 0;  // scope_end overhead
static int64_t g_orch_submit_count = 0;
static uint32_t g_orch_submit_idx = 0;
uint64_t g_orch_alloc_wait_cycle = 0;
uint64_t g_orch_fanin_wait_cycle = 0;
uint64_t g_orch_alloc_atomic_count = 0;
uint64_t g_orch_args_atomic_count = 0;
uint64_t g_orch_fanin_atomic_count = 0;
uint64_t g_orch_finalize_atomic_count = 0;
uint64_t g_orch_scope_end_atomic_count = 0;
#define CYCLE_COUNT_START() uint64_t _t0 = get_sys_cnt_aicpu(), _t1
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)                                    \
    do {                                                                              \
        _t1 = get_sys_cnt_aicpu();                                                    \
        acc += (_t1 - _t0);                                                           \
        perf_aicpu_record_orch_phase((phase_id), _t0, _t1, g_orch_submit_idx, (tid)); \
        _t0 = _t1;                                                                    \
    } while (0)
#elif PTO2_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/performance_collector_aicpu.h"
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }
__attribute__((weak, visibility("hidden"))) void
perf_aicpu_record_orch_phase(AicpuPhaseId, uint64_t, uint64_t, uint32_t, uint64_t) {}
// submit_idx needed for swimlane task_id tagging (no cycle accumulation at this level)
static uint32_t g_orch_submit_idx = 0;
#define CYCLE_COUNT_START()                     \
    bool _prof_active = orch->enable_profiling; \
    uint64_t _t0 = _prof_active ? get_sys_cnt_aicpu() : 0, _t1 = 0
#define CYCLE_COUNT_LAP(acc) \
    do {                     \
    } while (0)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)                                        \
    do {                                                                                  \
        if (_prof_active) {                                                               \
            _t1 = get_sys_cnt_aicpu();                                                    \
            perf_aicpu_record_orch_phase((phase_id), _t0, _t1, g_orch_submit_idx, (tid)); \
            _t0 = _t1;                                                                    \
        }                                                                                 \
    } while (0)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)
#endif

static bool pto2_append_fanin_or_fail(
    PTO2OrchestratorState *orch, PTO2TaskId task_id, int32_t tensor_arg_index, TensorArgType ptype,
    PTO2TaskSlotState *prod_state, PTO2TaskSlotState *fanin_states[], int32_t *fanin_count, const char *reason
) {
    for (int32_t j = 0; j < *fanin_count; j++) {
        if (fanin_states[j] == prod_state) {
            return true;
        }
    }

    if (*fanin_count >= PTO2_MAX_INPUTS) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Dependency Overflow Detected!");
        LOG_ERROR("========================================");
        LOG_ERROR("Task requires more than PTO2_MAX_INPUTS unique fanin dependencies.");
        LOG_ERROR("  task_id.raw:        %" PRIu64, task_id.raw);
        LOG_ERROR("  tensor_arg_index:   %d", tensor_arg_index);
        LOG_ERROR("  tensor_arg_type:    %d", static_cast<int>(ptype));
        LOG_ERROR("  fanin_count:        %d / %d", *fanin_count, PTO2_MAX_INPUTS);
        LOG_ERROR("  reason:             %s", reason);
        LOG_ERROR("This is a runtime dependency-tracking limit.");
        LOG_ERROR("========================================");
        orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_DEPENDENCY_OVERFLOW, std::memory_order_release);
        orch->fatal = true;
        return false;
    }

    fanin_states[(*fanin_count)++] = prod_state;
    return true;
}

// =============================================================================
// Orchestrator Initialization
// =============================================================================

bool pto2_orchestrator_init(
    PTO2OrchestratorState *orch, PTO2SharedMemoryHandle *sm_handle, void *gm_heap, uint64_t heap_size,
    int32_t dep_pool_capacity
) {
    *orch = PTO2OrchestratorState{};

    orch->sm_handle = sm_handle;
    orch->gm_heap_base = gm_heap;
    orch->gm_heap_size = heap_size * PTO2_MAX_RING_DEPTH;
    orch->fatal = false;

    // Initialize per-ring resources
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        void *ring_heap_base = reinterpret_cast<char *>(gm_heap) + r * heap_size;
        auto &fc = sm_handle->header->rings[r].fc;

        // Initialize unified task allocator
        orch->rings[r].task_allocator.init(
            sm_handle->task_descriptors[r], sm_handle->header->rings[r].task_window_size, &fc.current_task_index,
            &fc.last_task_alive, ring_heap_base, heap_size, &sm_handle->header->orch_error_code
        );

        // Allocate and initialize dependency list pool (per-ring)
        PTO2DepListEntry *dep_entries =
            reinterpret_cast<PTO2DepListEntry *>(calloc(dep_pool_capacity, sizeof(PTO2DepListEntry)));
        if (!dep_entries) {
            // Cleanup previously allocated rings
            for (int j = 0; j < r; j++) {
                free(orch->rings[j].dep_pool.base);
            }
            return false;
        }
        orch->rings[r].dep_pool.init(dep_entries, dep_pool_capacity, &sm_handle->header->orch_error_code);
    }

    // Initialize TensorMap with per-ring task window sizes
    int32_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = sm_handle->header->rings[r].task_window_size;
    }
    if (!orch->tensor_map.init_default(task_window_sizes)) {
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            free(orch->rings[r].dep_pool.base);
        }
        return false;
    }
    orch->tensor_map.orch = orch;

    // Initialize scope stack: one flat buffer for task IDs + one array for begin offsets
    uint64_t max_depth = PTO2_MAX_SCOPE_DEPTH;
    int32_t init_cap = PTO2_SCOPE_TASKS_INIT_CAP;
    orch->scope_tasks = reinterpret_cast<PTO2TaskSlotState **>(malloc(init_cap * sizeof(PTO2TaskSlotState *)));
    orch->scope_begins = reinterpret_cast<int32_t *>(malloc(max_depth * sizeof(int32_t)));
    if (!orch->scope_tasks || !orch->scope_begins) {
        free(orch->scope_tasks);
        free(orch->scope_begins);
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            free(orch->rings[r].dep_pool.base);
        }
        orch->tensor_map.destroy();
        return false;
    }
    orch->scope_tasks_size = 0;
    orch->scope_tasks_capacity = init_cap;
    orch->scope_stack_top = -1;
    orch->scope_stack_capacity = max_depth;

    return true;
}

void pto2_orchestrator_destroy(PTO2OrchestratorState *orch) {
    orch->tensor_map.destroy();

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        free(orch->rings[r].dep_pool.base);
        orch->rings[r].dep_pool.base = NULL;
    }

    free(orch->scope_tasks);
    orch->scope_tasks = NULL;
    free(orch->scope_begins);
    orch->scope_begins = NULL;
}

void pto2_orchestrator_set_scheduler(PTO2OrchestratorState *orch, PTO2SchedulerState *scheduler) {
    orch->scheduler = scheduler;
}

// =============================================================================
// Scope Management
// =============================================================================

static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state) {
    if (orch->scope_tasks_size >= orch->scope_tasks_capacity) {
        int32_t new_cap = orch->scope_tasks_capacity * 2;
        PTO2TaskSlotState **new_buf =
            reinterpret_cast<PTO2TaskSlotState **>(realloc(orch->scope_tasks, new_cap * sizeof(PTO2TaskSlotState *)));
        assert(new_buf && "Failed to grow scope task buffer");
        orch->scope_tasks = new_buf;
        orch->scope_tasks_capacity = new_cap;
    }
    orch->scope_tasks[orch->scope_tasks_size++] = task_slot_state;
}

void pto2_scope_begin(PTO2OrchestratorState *orch) {
    if (orch->fatal) {
        return;
    }
    assert(orch->scope_stack_top < static_cast<int32_t>(orch->scope_stack_capacity - 1) && "Scope stack overflow");

    ++orch->scope_stack_top;
    orch->scope_begins[orch->scope_stack_top] = orch->scope_tasks_size;
}

void pto2_scope_end(PTO2OrchestratorState *orch) {
    if (orch->fatal) {
        return;
    }
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

#if PTO2_ORCH_PROFILING
    uint64_t _se0 = get_sys_cnt_aicpu();
#endif

    int32_t begin = orch->scope_begins[orch->scope_stack_top--];
    int32_t count = orch->scope_tasks_size - begin;

    if (orch->scheduler && count > 0) {
        orch->scheduler->on_scope_end(&orch->scope_tasks[begin], count);
    }

    // Rewind the task buffer — these entries are no longer needed
    orch->scope_tasks_size = begin;

#if PTO2_ORCH_PROFILING
    uint64_t _se1 = get_sys_cnt_aicpu();
    g_orch_scope_end_cycle += (_se1 - _se0);
    // perf_aicpu_record_orch_phase(AicpuPhaseId::ORCH_SCOPE_END, _se0, _se1, g_orch_submit_idx, -1);
#endif
}

// =============================================================================
// Task Submission
// =============================================================================
TaskOutputTensors
pto2_submit_mixed_task(PTO2OrchestratorState *orch, const MixedKernels &mixed_kernels, const Arg &args) {
    CYCLE_COUNT_START();

    TaskOutputTensors result;

    // Fast path after fatal error — all subsequent submits are no-ops
    if (orch->fatal) {
        return result;
    }

    // Validate Arg construction (errors recorded by add_input/add_output/etc.)
    if (args.has_error) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg Detected!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg ? args.error_msg : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("This is a bug in the orchestration code.");
        LOG_ERROR("========================================");
        orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_INVALID_ARGS, std::memory_order_release);
        orch->fatal = true;
        return result;
    }

    // Determine which ring this task belongs to
    uint8_t ring_id = orch->current_ring_id();
    auto &allocator = orch->rings[ring_id].task_allocator;
    PTO2SchedulerState *sched = orch->scheduler;
    PTO2RingFlowControl &fc = orch->sm_handle->header->rings[ring_id].fc;

    // === Validate submit inputs ===
    uint8_t active_mask = pto2_mixed_kernels_to_active_mask(mixed_kernels);
    always_assert(active_mask != 0 && "MixedKernels must have at least one active slot");

    int16_t block_num = args.launch_spec.block_num();
    always_assert(block_num >= 1 && "block_num must be >= 1");

    // Normalize single-AIV tasks: if only aiv1 is set (no aic, no aiv0), move
    // it to the aiv0 slot.  This guarantees the dispatch path can always use
    // PTO2SubtaskSlot::AIV0 for single-AIV shapes without inspecting active_mask.
    // Mixed tasks (AIC+AIV) keep their original AIV identity so the correct
    // hardware channel (AIV0→AIC vs AIV1→AIC) is used at dispatch time.
    MixedKernels normalized = mixed_kernels;
    bool has_aic = (active_mask & PTO2_SUBTASK_MASK_AIC) != 0;
    bool has_aiv0 = (active_mask & PTO2_SUBTASK_MASK_AIV0) != 0;
    bool has_aiv1 = (active_mask & PTO2_SUBTASK_MASK_AIV1) != 0;
    if (!has_aic && has_aiv1 && !has_aiv0) {
        normalized.aiv0_kernel_id = normalized.aiv1_kernel_id;
        normalized.aiv1_kernel_id = INVALID_KERNEL_ID;
        active_mask = pto2_mixed_kernels_to_active_mask(normalized);
    }

    // Encode require_sync_start into active_mask bit 3 (only meaningful for tasks with block_num > 1)
    if (block_num > 1 && args.launch_spec.require_sync_start()) {
        // Deadlock check: block_num >= total available slots of the required type.
        // For MIX/AIC: limit is total_cluster_count (one AIC per cluster).
        // For AIV:     limit is total_aiv_count.
        PTO2ResourceShape shape = pto2_active_mask_to_shape(active_mask);
        int32_t limit = (shape == PTO2ResourceShape::AIV) ? orch->total_aiv_count : orch->total_cluster_count;
        if (limit > 0 && block_num > limit) {
            LOG_ERROR("FATAL: require_sync_start block_num=%d > limit=%d (deadlock guaranteed)", block_num, limit);
            orch->fatal = true;
            return TaskOutputTensors{};
        }
        active_mask |= PTO2_SUBTASK_FLAG_SYNC_START;
    }

    // Submission without an open scope is illegal
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");

    // === Scope deadlock pre-check ===
    // Tasks within a scope hold a fanout_count reference released only at scope_end.
    // If scope task count >= window_size, no slots can ever be reclaimed → deadlock.
    {
        int32_t scope_task_count = orch->scope_tasks_size - orch->scope_begins[orch->scope_stack_top];
        if (scope_task_count >= allocator.window_size() - 1) {
            int32_t active_count = allocator.active_count();

            LOG_ERROR("========================================");
            LOG_ERROR("FATAL: Scope Deadlock Detected! (ring %d)", ring_id);
            LOG_ERROR("========================================");
            LOG_ERROR(
                "Tasks in current scope (%d) >= task_window_size (%d).", scope_task_count, allocator.window_size()
            );
            LOG_ERROR("  scope_depth:        %d", orch->scope_stack_top + 1);
            LOG_ERROR("  ring_id:            %d", ring_id);
            LOG_ERROR("  scope_task_count:   %d", scope_task_count);
            LOG_ERROR("  active_tasks:       %d / %d", active_count, allocator.window_size());
            LOG_ERROR("Root Cause:");
            LOG_ERROR("  Tasks within a scope hold a fanout_count reference that is only");
            LOG_ERROR("  released at scope_end. When scope task count >= window_size,");
            LOG_ERROR("  no slots can be reclaimed -> deadlock.");
            LOG_ERROR("Solution:");
            LOG_ERROR("  1. Reduce tasks per scope (use batching/unroll)");
            LOG_ERROR("  2. Increase task window (current: %d)", allocator.window_size());
            LOG_ERROR("     Compile-time: PTO2_TASK_WINDOW_SIZE in pto_runtime2_types.h");
            LOG_ERROR("     Runtime env:  PTO2_RING_TASK_WINDOW=<power-of-2>");
            LOG_ERROR("  3. Split work across multiple scopes");
            LOG_ERROR("========================================");
            orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_SCOPE_DEADLOCK, std::memory_order_release);
            orch->fatal = true;
            return result;
        }
    }

    // === Calculate output size (from runtime-created OUTPUT args) ===
    uint64_t offsets[MAX_TENSOR_ARGS] = {};
    uint64_t buffer_sizes[MAX_TENSOR_ARGS] = {};
    int32_t total_output_size = 0;
    for (int i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) == TensorArgType::OUTPUT) {
            offsets[i] = total_output_size;
            buffer_sizes[i] = PTO2_ALIGN_UP(args.tensor(i).create_info->buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
            total_output_size += buffer_sizes[i];
        }
    }

    // === STEP 1: Unified alloc — task slot + packed output buffer (blocks until available) ===
    PTO2TaskAllocResult alloc_result = allocator.alloc(total_output_size);
    if (alloc_result.failed()) {
        orch->fatal = true;
        return result;
    }

    int32_t local_id = alloc_result.task_id;
    int32_t slot = alloc_result.slot;
    PTO2TaskId task_id = PTO2TaskId::make(ring_id, static_cast<uint32_t>(local_id));

    PTO2TaskDescriptor &task = allocator.task_by_slot(slot);
    PTO2TaskPayload *payload = &orch->sm_handle->task_payloads[ring_id][slot];

    // Early write-prefetch payload GM cache lines to issue RFO in background.
    // ~130 lines of computation (lookup, insert) follow before
    // param_copy writes, giving ample time for prefetch to complete.
    // Use locality=3 (PSTL1KEEP) so prefetched CLs survive lookup/insert eviction.
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        __builtin_prefetch(&payload->tensors[i], 1, 3);
        __builtin_prefetch(reinterpret_cast<char *>(&payload->tensors[i]) + 64, 1, 3);
    }
    for (int32_t i = 0; i < args.scalar_count(); i += 8) {
        __builtin_prefetch(&payload->scalars[i], 1, 3);
    }
    __builtin_prefetch(payload, 1, 3);
    __builtin_prefetch(reinterpret_cast<char *>(payload) + 64, 1, 3);
    __builtin_prefetch(reinterpret_cast<char *>(payload) + 128, 1, 3);

    // Initialize slot state (scheduler-private)
    if (sched) {
        auto &rs = sched->ring_sched_states[ring_id];
        PTO2TaskSlotState &slot_state = rs.get_slot_state_by_slot(slot);
        slot_state.fanin_count = 0;
        slot_state.fanout_head = nullptr;
        slot_state.fanout_lock.store(0, std::memory_order_relaxed);
        // Initial fanout_count = 1 (the owning scope holds one reference)
        slot_state.fanout_count = 1;
        slot_state.fanout_refcount.store(0, std::memory_order_release);
        slot_state.fanin_refcount.store(0, std::memory_order_release);
        slot_state.payload = payload;
        slot_state.task = &task;
        slot_state.active_mask = active_mask;
        slot_state.subtask_done_mask.store(0, std::memory_order_relaxed);
        slot_state.ring_id = ring_id;
        scope_tasks_push(orch, &slot_state);
    } else {
        scope_tasks_push(orch, nullptr);
    }

    // Temporary storage for fanin (cached slot state pointers, avoids repeated ring/slot lookups)
    PTO2TaskSlotState *fanin_states[PTO2_MAX_INPUTS];
    int32_t fanin_count = 0;

    CYCLE_COUNT_LAP_RECORD(g_orch_alloc_cycle, AicpuPhaseId::ORCH_ALLOC, task_id.raw);

#if PTO2_PROFILING
    if (total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += total_output_size;
    }
#endif

    // === STEP 2: Sync TensorMap validity and optional cleanup ===
    // Read current last_task_alive from shared memory for this ring
    int32_t sm_last_task_alive = fc.last_task_alive.load(std::memory_order_acquire);

    orch->tensor_map.sync_tensormap(ring_id, sm_last_task_alive);

    if (sched) {
        orch->rings[ring_id].dep_pool.reclaim(*sched, ring_id, sm_last_task_alive);
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_sync_cycle, AicpuPhaseId::ORCH_SYNC, task_id.raw);

    // === STEP 3: Lookup inputs + materialize runtime-created outputs ===
    for (int i = 0; i < args.tensor_count(); i++) {
        TensorArgType ptype = args.tag(i);
        if (ptype == TensorArgType::OUTPUT) {
            // Runtime-created OUTPUT tensors are not looked up in the TensorMap since they have no dependencies.
            continue;
        }

        const Tensor *tensor = args.tensor(i).ptr;

        // Step A: creator retention — all existing tensors extend their creator lifetime.
        PTO2TaskId owner = tensor->owner_task_id;
        if (owner.is_valid() && sched != nullptr) {
            PTO2TaskSlotState *prod_state =
                &sched->ring_sched_states[owner.ring()].get_slot_state_by_task_id(owner.local());
            if (!pto2_append_fanin_or_fail(
                    orch, task_id, i, ptype, prod_state, fanin_states, &fanin_count, "creator retention"
                )) {
                return result;
            }
        }

        // Step B: only INPUT/INOUT need modifier dependency lookup.
        if (ptype != TensorArgType::INPUT && ptype != TensorArgType::INOUT) {
            continue;
        }
        if (tensor->manual_dep) {
            continue;
        }

        PTO2LookupResult lookup_result;
        orch->tensor_map.lookup(*tensor, lookup_result);

        for (int r = 0; r < lookup_result.count; r++) {
            PTO2TensorMapEntry &entry = *lookup_result.entries[r].entry;
            auto overlap_status = lookup_result.entries[r].overlap_status;
            auto prod_ring = entry.producer_task_id.ring();
            auto prod_local = entry.producer_task_id.local();
            PTO2TaskSlotState *prod_state = &sched->ring_sched_states[prod_ring].get_slot_state_by_task_id(prod_local);
            if (!pto2_append_fanin_or_fail(
                    orch, task_id, i, ptype, prod_state, fanin_states, &fanin_count, "overlap lookup"
                )) {
                return result;
            }
            if (ptype == TensorArgType::INOUT && overlap_status == OverlapStatus::COVERED) {
                orch->tensor_map.remove_entry(entry);
            }
        }
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_lookup_cycle, AicpuPhaseId::ORCH_LOOKUP, task_id.raw);

    // === STEP 4: Register outputs/inouts in TensorMap (must be separate from lookup) ===
    {
        for (int i = 0; i < args.tensor_count(); i++) {
            TensorArgType ptype = args.tag(i);
            if (ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING) {
                if (!args.tensor(i).ptr->manual_dep) {
                    orch->tensor_map.insert(*args.tensor(i).ptr, task_id);
                }
            }
        }
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_insert_cycle, AicpuPhaseId::ORCH_INSERT, task_id.raw);

    // === STEP 5: Batch-write to GM (single cache line burst) ===
    // Deferred from allocation phase to avoid scattered GM writes that get
    // evicted by TensorMap lookup/insert cache pressure.
    __builtin_prefetch(&task, 1, 1);
    task.task_id = task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = normalized.aic_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = normalized.aiv0_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = normalized.aiv1_kernel_id;
    task.packed_buffer_base = alloc_result.packed_base;
    task.packed_buffer_end = alloc_result.packed_end;

    // Prefetch producer slot_states and cur_slot_state (written at init but likely
    // evicted by lookup/insert/heap). param_copy below provides hide time.
    if (sched) {
        auto &rs = sched->ring_sched_states[ring_id];
        __builtin_prefetch(&rs.get_slot_state_by_slot(slot), 1, 0);
        for (int i = 0; i < fanin_count; i++) {
            __builtin_prefetch(fanin_states[i], 1, 0);
        }
    }

    payload->init(args, result, alloc_result.packed_base, offsets, buffer_sizes);

    // Write owner_task_id into materialized OUTPUT tensors so creator-only dependency
    // tracking remains available even when manual_dep skips OverlapMap publication.
    for (int i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) == TensorArgType::OUTPUT) {
            payload->tensors[i].owner_task_id = task_id;
        }
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_args_cycle, AicpuPhaseId::ORCH_PARAMS, task_id.raw);
#if PTO2_ORCH_PROFILING
    g_orch_args_atomic_count += 2;  // fanout_lock.store + fanout_count.store
#endif

    // === STEP 6: Finalize fanin list ===
    // First build the fanin list
    if (sched) {
        auto &rs = sched->ring_sched_states[ring_id];
        PTO2TaskSlotState &cur_slot_state = rs.get_slot_state_by_slot(slot);
        // Initialize scheduler state BEFORE adding to producer fanout lists,
        // so concurrent on_mixed_task_complete can safely access task_state/fanout_refcount.
        cur_slot_state.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
        cur_slot_state.fanout_refcount.store(0, std::memory_order_relaxed);
        cur_slot_state.completed_subtasks.store(0, std::memory_order_relaxed);
        cur_slot_state.total_required_subtasks =
            static_cast<int16_t>(block_num * __builtin_popcount(pto2_core_mask(active_mask)));
        cur_slot_state.block_num = block_num;
        cur_slot_state.next_block_idx = 0;

        auto &dep_pool = orch->rings[ring_id].dep_pool;
        // Ensure dep pool has space: fanin_count entries + 1 pre-alloc
        dep_pool.ensure_space(*sched, fc, ring_id, fanin_count + 1);

        int32_t early_finished = 0;
        cur_slot_state.fanin_count = fanin_count + 1;  // +1 redundance for not being ready too early
        payload->fanin_actual_count = fanin_count;
        for (int i = 0; i < fanin_count; i++) {
            payload->fanin_slot_states[i] = fanin_states[i];
        }
        for (int i = 0; i < fanin_count; i++) {
            PTO2TaskSlotState &producer_slot_state = *fanin_states[i];
#if PTO2_ORCH_PROFILING
            pto2_fanout_lock(producer_slot_state, g_orch_fanin_atomic_count, g_orch_fanin_wait_cycle);
#else
            pto2_fanout_lock(producer_slot_state);
#endif
            // Normal path: prepend consumer to producer's fanout list
            producer_slot_state.fanout_count += 1;
            int32_t prod_state = producer_slot_state.task_state.load(std::memory_order_acquire);
            if (prod_state >= PTO2_TASK_COMPLETED) {
                // Early return optimization: if producer already completed, we can skip adding dependency and directly
                // decrement fanin_count
                early_finished++;
            } else {
                producer_slot_state.fanout_head = dep_pool.prepend(producer_slot_state.fanout_head, &cur_slot_state);
            }
            pto2_fanout_unlock(producer_slot_state);
        }
        // Combined release: merge early_finished batch with the +1 init release
        // into a single atomic fetch_add (saves one acq_rel cache-line bounce per task).
        int32_t initial_refcount = early_finished + 1;  // +1 for the init release
        int32_t new_rc =
            cur_slot_state.fanin_refcount.fetch_add(initial_refcount, std::memory_order_acq_rel) + initial_refcount;
        if (new_rc >= fanin_count + 1) {
            PTO2ResourceShape shape = pto2_active_mask_to_shape(active_mask);
            sched->ready_queues[static_cast<int32_t>(shape)].push(&cur_slot_state);
        }
        // Record dep pool watermark in local slot state (used by tail reclamation)
        cur_slot_state.dep_pool_mark = orch->rings[ring_id].dep_pool.top;
#if PTO2_ORCH_PROFILING
        // Per producer: fetch_add(fanout_count) + load(task_state) + store(unlock) = 3 atomics
        // Lock atomics (loads + CAS) are counted inside pto2_fanout_lock
        g_orch_fanin_atomic_count += fanin_count * 3;
        if (early_finished > 0) {
            g_orch_fanin_atomic_count += 1;  // fanin_refcount.fetch_add
        }
#endif
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_fanin_cycle, AicpuPhaseId::ORCH_FANIN, task_id.raw);

#if PTO2_PROFILING
    orch->tasks_submitted++;
#if PTO2_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif
    return result;
}

// =============================================================================
// Flow Control
// =============================================================================

void pto2_orchestrator_done(PTO2OrchestratorState *orch) {
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        int32_t total_tasks = orch->rings[r].task_allocator.active_count();
        if (total_tasks > 0) {
            LOG_INFO("=== [Orchestrator] ring %d: total_tasks=%d ===", r, total_tasks);
        }
        auto &pool = orch->rings[r].dep_pool;
        if (pool.top > 0) {
            LOG_INFO(
                "=== [DepPool %d] top=%d tail=%d used=%d high_water=%d capacity=%d ===", r, pool.top, pool.tail,
                pool.top - pool.tail, pool.high_water, pool.capacity
            );
        }
    }
    orch->sm_handle->header->orchestrator_done.store(1, std::memory_order_release);
#if !PTO2_ORCH_PROFILING && PTO2_PROFILING
    g_orch_submit_idx = 0;
#endif
}

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_orchestrator_print_stats(PTO2OrchestratorState *orch) {
    LOG_INFO("=== Orchestrator Statistics ===");
#if PTO2_PROFILING
    LOG_INFO("Tasks submitted:     %" PRId64, orch->tasks_submitted);
    LOG_INFO("Buffers allocated:   %" PRId64, orch->buffers_allocated);
    LOG_INFO("Bytes allocated:     %" PRId64, orch->bytes_allocated);
#endif
    LOG_INFO("Current scope depth: %d", orch->scope_stack_top + 1);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        int32_t active = orch->rings[r].task_allocator.active_count();
        if (active > 0) {
            LOG_INFO("Ring %d task active:  %d", r, active);
            LOG_INFO(
                "Ring %d heap used:    %" PRIu64 " / %" PRIu64, r, orch->rings[r].task_allocator.heap_top(),
                orch->rings[r].task_allocator.heap_capacity()
            );
            LOG_INFO(
                "Ring %d dep pool:     %d / %d", r, orch->rings[r].dep_pool.used(), orch->rings[r].dep_pool.capacity
            );
        }
    }
    LOG_INFO("TensorMap valid:     %d", orch->tensor_map.valid_count());
    LOG_INFO("===============================");
}

void pto2_orchestrator_print_scope_stack(PTO2OrchestratorState *orch) {
    LOG_INFO("=== Scope Stack ===");
    LOG_INFO("Depth: %d", orch->scope_stack_top + 1);

    for (int i = 0; i <= orch->scope_stack_top; i++) {
        int32_t begin = orch->scope_begins[i];
        int32_t end = (i < orch->scope_stack_top) ? orch->scope_begins[i + 1] : orch->scope_tasks_size;
        LOG_INFO("  [%d] tasks_owned = %d", i, end - begin);
    }

    LOG_INFO("==================");
}

#if PTO2_ORCH_PROFILING
PTO2OrchProfilingData pto2_orchestrator_get_profiling() {
    PTO2OrchProfilingData d;
    d.sync_cycle = g_orch_sync_cycle;
    d.alloc_cycle = g_orch_alloc_cycle;
    d.args_cycle = g_orch_args_cycle;
    d.lookup_cycle = g_orch_lookup_cycle;
    d.insert_cycle = g_orch_insert_cycle;
    d.fanin_cycle = g_orch_fanin_cycle;
    d.scope_end_cycle = g_orch_scope_end_cycle;
    d.submit_count = g_orch_submit_count;
    d.alloc_wait_cycle = g_orch_alloc_wait_cycle;
    d.fanin_wait_cycle = g_orch_fanin_wait_cycle;
    d.alloc_atomic_count = g_orch_alloc_atomic_count;
    d.args_atomic_count = g_orch_args_atomic_count;
    d.fanin_atomic_count = g_orch_fanin_atomic_count;
    d.finalize_atomic_count = g_orch_finalize_atomic_count;
    d.scope_end_atomic_count = g_orch_scope_end_atomic_count;

    // Reset
    g_orch_sync_cycle = g_orch_alloc_cycle = g_orch_args_cycle = 0;
    g_orch_lookup_cycle = g_orch_insert_cycle = 0;
    g_orch_fanin_cycle = g_orch_scope_end_cycle = 0;
    g_orch_submit_count = 0;
    g_orch_submit_idx = 0;
    g_orch_alloc_wait_cycle = 0;
    g_orch_fanin_wait_cycle = 0;
    g_orch_alloc_atomic_count = 0;
    g_orch_args_atomic_count = 0;
    g_orch_fanin_atomic_count = 0;
    g_orch_finalize_atomic_count = 0;
    g_orch_scope_end_atomic_count = 0;
    return d;
}
#endif
