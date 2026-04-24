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
 * PTO Runtime2 - Orchestrator Implementation (Explicit Dependency Variant)
 *
 * Implements orchestrator state management, scope handling, task submission
 * with explicit dependencies, and scope-end batch publish.
 *
 * Key differences from tensormap_and_ringbuffer:
 * - No TensorMap: submit_task is a 3-step process (alloc, heap, write)
 * - add_dependency: explicitly wires producer -> consumer edges
 * - scope_end: batch-publishes all tasks (releases +1 fanin redundance)
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
#include "pto_types.h"
#include "tensor.h"

// =============================================================================
// Orchestrator Profiling (compile-time toggle)
// =============================================================================
#if PTO2_ORCH_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/l2_perf_collector_aicpu.h"
// Weak fallback for builds that don't link device_time.cpp (e.g. host).
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }
__attribute__((weak, visibility("hidden"))) void
l2_perf_aicpu_record_orch_phase(AicpuPhaseId, uint64_t, uint64_t, uint32_t, uint64_t) {}
static uint64_t g_orch_alloc_cycle = 0;
static uint64_t g_orch_args_cycle = 0;
static uint64_t g_orch_heap_cycle = 0;
static uint64_t g_orch_fanin_cycle = 0;
static uint64_t g_orch_scope_end_cycle = 0;
static int64_t g_orch_submit_count = 0;
static uint32_t g_orch_submit_idx = 0;
uint64_t g_orch_alloc_wait_cycle = 0;
uint64_t g_orch_heap_wait_cycle = 0;
uint64_t g_orch_fanin_wait_cycle = 0;
uint64_t g_orch_alloc_atomic_count = 0;
uint64_t g_orch_args_atomic_count = 0;
uint64_t g_orch_heap_atomic_count = 0;
uint64_t g_orch_fanin_atomic_count = 0;
uint64_t g_orch_scope_end_atomic_count = 0;
#define CYCLE_COUNT_START() uint64_t _t0 = get_sys_cnt_aicpu(), _t1
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)                                       \
    do {                                                                                 \
        _t1 = get_sys_cnt_aicpu();                                                       \
        acc += (_t1 - _t0);                                                              \
        l2_perf_aicpu_record_orch_phase((phase_id), _t0, _t1, g_orch_submit_idx, (tid)); \
        _t0 = _t1;                                                                       \
    } while (0)
#elif PTO2_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/l2_perf_collector_aicpu.h"
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }
__attribute__((weak, visibility("hidden"))) void
l2_perf_aicpu_record_orch_phase(AicpuPhaseId, uint64_t, uint64_t, uint32_t, uint64_t) {}
static uint32_t g_orch_submit_idx = 0;
#define CYCLE_COUNT_START()                       \
    bool _prof_active = orch->enable_l2_swimlane; \
    uint64_t _t0 = _prof_active ? get_sys_cnt_aicpu() : 0, _t1 = 0
#define CYCLE_COUNT_LAP(acc) \
    do {                     \
    } while (0)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)                                           \
    do {                                                                                     \
        if (_prof_active) {                                                                  \
            _t1 = get_sys_cnt_aicpu();                                                       \
            l2_perf_aicpu_record_orch_phase((phase_id), _t0, _t1, g_orch_submit_idx, (tid)); \
            _t0 = _t1;                                                                       \
        }                                                                                    \
    } while (0)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)
#endif

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

        pto2_heap_ring_init(&orch->rings[r].heap_ring, ring_heap_base, heap_size, &fc.heap_tail, &fc.heap_top);
        orch->rings[r].heap_ring.error_code_ptr = &sm_handle->header->orch_error_code;

        pto2_task_ring_init(
            &orch->rings[r].task_ring, sm_handle->task_descriptors[r], sm_handle->header->rings[r].task_window_size,
            &fc.last_task_alive, &fc.current_task_index
        );
        orch->rings[r].task_ring.error_code_ptr = &sm_handle->header->orch_error_code;

        PTO2DepListEntry *dep_entries =
            reinterpret_cast<PTO2DepListEntry *>(calloc(dep_pool_capacity, sizeof(PTO2DepListEntry)));
        if (!dep_entries) {
            for (int j = 0; j < r; j++) {
                free(orch->rings[j].dep_pool.base);
            }
            return false;
        }
        orch->rings[r].dep_pool.init(dep_entries, dep_pool_capacity, &sm_handle->header->orch_error_code);
    }

    // Initialize scope stack
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
        return false;
    }
    orch->scope_tasks_size = 0;
    orch->scope_tasks_capacity = init_cap;
    orch->scope_stack_top = -1;
    orch->scope_stack_capacity = max_depth;

    return true;
}

void pto2_orchestrator_destroy(PTO2OrchestratorState *orch) {
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
        PTO2TaskSlotState **tasks = &orch->scope_tasks[begin];

        // Batch publish: release the "+1 redundance" in fanin for each task.
        // Tasks whose fanin is fully satisfied become READY and are pushed
        // to the scheduler's ready queues.
        for (int32_t i = 0; i < count; i++) {
            PTO2TaskSlotState *slot = tasks[i];
            if (!slot) continue;

            // task_state is already PENDING from submit_task (defensive store)
            slot->task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);

            // Release the +1 fanin redundance
            int32_t new_rc = slot->fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (new_rc >= slot->fanin_count) {
                PTO2ResourceShape shape = pto2_active_mask_to_shape(slot->active_mask);
                orch->scheduler->ready_queues[static_cast<int32_t>(shape)].push(slot);
            }
        }

        // Release the scope's fanout reference on each task (enables CONSUMED transition)
        orch->scheduler->on_scope_end(tasks, count);
    }

    // Rewind the task buffer
    orch->scope_tasks_size = begin;

#if PTO2_ORCH_PROFILING
    uint64_t _se1 = get_sys_cnt_aicpu();
    g_orch_scope_end_cycle += (_se1 - _se0);
#endif
}

// =============================================================================
// Task Submission (3-step: alloc, heap, write — no TensorMap)
// =============================================================================
SubmitResult pto2_submit_mixed_task(PTO2OrchestratorState *orch, const MixedKernels &mixed_kernels, const Arg &args) {
    CYCLE_COUNT_START();

    SubmitResult result;

    if (orch->fatal) {
        return result;
    }

    // Validate Arg
    if (args.has_error) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg Detected!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg ? args.error_msg : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("========================================");
        orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_INVALID_ARGS, std::memory_order_release);
        orch->fatal = true;
        return result;
    }

    uint8_t ring_id = orch->current_ring_id();
    auto &task_ring = orch->rings[ring_id].task_ring;
    PTO2SchedulerState *sched = orch->scheduler;

    // Validate submit inputs
    uint8_t active_mask = pto2_mixed_kernels_to_active_mask(mixed_kernels);
    always_assert(active_mask != 0 && "MixedKernels must have at least one active slot");

    // Normalize single-AIV tasks
    MixedKernels normalized = mixed_kernels;
    bool has_aiv0 = (active_mask & PTO2_SUBTASK_MASK_AIV0) != 0;
    bool has_aiv1 = (active_mask & PTO2_SUBTASK_MASK_AIV1) != 0;
    if (has_aiv1 && !has_aiv0) {
        normalized.aiv0_kernel_id = normalized.aiv1_kernel_id;
        normalized.aiv1_kernel_id = INVALID_KERNEL_ID;
        active_mask = pto2_mixed_kernels_to_active_mask(normalized);
    }

    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");

    // Scope deadlock pre-check
    {
        int32_t scope_task_count = orch->scope_tasks_size - orch->scope_begins[orch->scope_stack_top];
        if (scope_task_count >= task_ring.window_size - 1) {
            int32_t total_submitted = task_ring.current_index_ptr->load(std::memory_order_acquire);
            int32_t last_alive = task_ring.last_alive_ptr->load(std::memory_order_acquire);
            int32_t active_count = total_submitted - last_alive;

            LOG_ERROR("========================================");
            LOG_ERROR("FATAL: Scope Deadlock Detected! (ring %d)", ring_id);
            LOG_ERROR("========================================");
            LOG_ERROR("Tasks in current scope (%d) >= task_window_size (%d).", scope_task_count, task_ring.window_size);
            LOG_ERROR("  scope_depth:        %d", orch->scope_stack_top + 1);
            LOG_ERROR("  ring_id:            %d", ring_id);
            LOG_ERROR("  scope_task_count:   %d", scope_task_count);
            LOG_ERROR("  total_submitted:    %d", total_submitted);
            LOG_ERROR("  last_task_alive:    %d", last_alive);
            LOG_ERROR("  active_tasks:       %d / %d", active_count, task_ring.window_size);
            LOG_ERROR("========================================");
            orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_SCOPE_DEADLOCK, std::memory_order_release);
            orch->fatal = true;
            return result;
        }
    }

    // === STEP 1: Allocate task slot from Task Ring ===
    int32_t local_id = task_ring.pto2_task_ring_alloc();
    if (local_id < 0) {
        orch->fatal = true;
        return result;
    }
    int32_t slot = task_ring.get_task_slot(local_id);
    PTO2TaskId task_id = pto2_make_task_id(ring_id, static_cast<uint32_t>(local_id));

    PTO2TaskDescriptor &task = task_ring.get_task_by_slot(slot);
    PTO2TaskPayload *payload = &orch->sm_handle->task_payloads[ring_id][slot];

    // Prefetch payload cache lines for write
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

    // Initialize slot state
    if (sched) {
        auto &rs = sched->ring_sched_states[ring_id];
        PTO2TaskSlotState &slot_state = rs.get_slot_state_by_slot(slot);
        // fanin_count starts at 1: the "+1 redundance" released at scope_end
        slot_state.fanin_count = 1;
        slot_state.fanout_head = nullptr;
        slot_state.fanout_lock.store(0, std::memory_order_relaxed);
        // fanout_count = 1 (owning scope holds one reference)
        slot_state.fanout_count = 1;
        slot_state.fanout_refcount.store(0, std::memory_order_release);
        slot_state.fanin_refcount.store(0, std::memory_order_release);
        slot_state.payload = payload;
        slot_state.task = &task;
        slot_state.active_mask = active_mask;
        slot_state.subtask_done_mask.store(0, std::memory_order_relaxed);
        slot_state.ring_id = ring_id;
        // Reset task_state so add_dependency doesn't see stale COMPLETED/CONSUMED
        // from a previously-reused slot. The scheduler won't act on PENDING tasks
        // until they're pushed to a ready queue at scope_end.
        slot_state.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
        scope_tasks_push(orch, &slot_state);
    } else {
        scope_tasks_push(orch, nullptr);
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_alloc_cycle, AicpuPhaseId::ORCH_ALLOC, task_id.raw);

    // === STEP 2: Heap allocation for OUTPUT tensors ===
    int32_t total_output_size = 0;
    for (int i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) == TensorArgType::OUTPUT) {
            total_output_size +=
                PTO2_ALIGN_UP(args.tensor(i).create_info.buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
        }
    }

    void *local_packed_base = nullptr;
    void *local_packed_end = nullptr;
    if (total_output_size > 0) {
        local_packed_base = orch->pto2_alloc_packed_buffer(total_output_size);
        if (!local_packed_base) {
            orch->fatal = true;
            return result;
        }
        local_packed_end = reinterpret_cast<char *>(local_packed_base) + total_output_size;
    }

    // Materialize OUTPUT tensors into TaskOutputTensors
    int32_t offset = 0;
    for (int i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) == TensorArgType::OUTPUT) {
            const TensorCreateInfo &ci = args.tensor(i).create_info;
            uint64_t buffer_size = ci.buffer_size_bytes();
            uint64_t alloc_addr = reinterpret_cast<uint64_t>(reinterpret_cast<char *>(local_packed_base) + offset);
            offset += PTO2_ALIGN_UP(buffer_size, PTO2_PACKED_OUTPUT_ALIGN);
            result.outputs.materialize_output(ci, reinterpret_cast<void *>(alloc_addr), /*version=*/0);
        }
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_heap_cycle, AicpuPhaseId::ORCH_HEAP, task_id.raw);

    // Periodically reclaim dep_pool entries from retired tasks
    if (sched) {
        int32_t sm_last_task_alive = task_ring.last_alive_ptr->load(std::memory_order_acquire);
        orch->rings[ring_id].dep_pool.reclaim(*sched, ring_id, sm_last_task_alive);
    }

    // === STEP 3: Write task descriptor and payload ===
    __builtin_prefetch(&task, 1, 1);
    task.task_id = task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = normalized.aic_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = normalized.aiv0_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = normalized.aiv1_kernel_id;
    task.packed_buffer_base = local_packed_base;
    task.packed_buffer_end = local_packed_end;

    payload->fanin_actual_count = 0;
    payload->init(args, result.outputs);

    CYCLE_COUNT_LAP_RECORD(g_orch_args_cycle, AicpuPhaseId::ORCH_PARAMS, task_id.raw);

    // Record dep pool watermark
    if (sched) {
        auto &rs = sched->ring_sched_states[ring_id];
        PTO2TaskSlotState &slot_state = rs.get_slot_state_by_slot(slot);
        slot_state.dep_pool_mark = orch->rings[ring_id].dep_pool.top;
    }

#if PTO2_PROFILING
    orch->tasks_submitted++;
#if PTO2_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif

    result.task_id = task_id;
    return result;
}

// =============================================================================
// Explicit Dependency Management
// =============================================================================

void pto2_add_dependency(PTO2OrchestratorState *orch, PTO2TaskId producer_id, PTO2TaskId consumer_id) {
    if (orch->fatal) return;

    PTO2SchedulerState *sched = orch->scheduler;
    if (!sched) return;

    uint8_t prod_ring = producer_id.ring();
    uint32_t prod_local = producer_id.local();
    uint8_t cons_ring = consumer_id.ring();
    uint32_t cons_local = consumer_id.local();

    auto &prod_rs = sched->ring_sched_states[prod_ring];
    auto &cons_rs = sched->ring_sched_states[cons_ring];

    PTO2TaskSlotState &prod_state = prod_rs.get_slot_state_by_task_id(prod_local);
    PTO2TaskSlotState &cons_state = cons_rs.get_slot_state_by_task_id(cons_local);

    // Increment consumer's fanin_count (+1 for this dependency)
    cons_state.fanin_count += 1;

    // Record producer in consumer's payload for DFX/debugging
    PTO2TaskPayload *cons_payload = cons_state.payload;
    if (cons_payload->fanin_actual_count < PTO2_MAX_INPUTS) {
        cons_payload->fanin_slot_states[cons_payload->fanin_actual_count] = &prod_state;
        cons_payload->fanin_actual_count++;
    }
#if PTO2_PROFILING
    // Resolve producer task_id now while slot_state is still valid.
    if (cons_payload->fanin_resolved_count < PTO2TaskPayload::kMaxFaninResolved) {
        cons_payload->fanin_resolved_ids[cons_payload->fanin_resolved_count] = prod_state.task->task_id.raw;
        cons_payload->fanin_resolved_count++;
    }
#endif

    // Wire the fanout edge from producer to consumer.
    // Always use fanout_lock: the producer may be from a previous scope
    // and already visible to the scheduler.
    auto &dep_pool = orch->rings[cons_ring].dep_pool;

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    pto2_fanout_lock(prod_state, g_orch_fanin_atomic_count, g_orch_fanin_wait_cycle);
#else
    pto2_fanout_lock(prod_state);
#endif

    prod_state.fanout_count += 1;
    int32_t prod_task_state = prod_state.task_state.load(std::memory_order_acquire);

    if (prod_task_state >= PTO2_TASK_COMPLETED) {
        // Producer already completed — count as early finish
        cons_state.fanin_refcount.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Producer not yet completed — add consumer to producer's fanout list
        prod_state.fanout_head = dep_pool.prepend(prod_state.fanout_head, &cons_state);
    }

    pto2_fanout_unlock(prod_state);

#if PTO2_ORCH_PROFILING
    g_orch_fanin_atomic_count += 3;  // lock CAS + load(task_state) + unlock store
#endif
}

// =============================================================================
// Flow Control
// =============================================================================

void pto2_orchestrator_done(PTO2OrchestratorState *orch) {
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        int32_t total_tasks = orch->rings[r].task_ring.current_index_ptr->load(std::memory_order_acquire);
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
        int32_t active = pto2_task_ring_active_count(&orch->rings[r].task_ring);
        if (active > 0) {
            LOG_INFO("Ring %d task active:  %d", r, active);
            LOG_INFO(
                "Ring %d heap used:    %" PRIu64 " / %" PRIu64, r,
                orch->rings[r].heap_ring.top_ptr->load(std::memory_order_relaxed), orch->rings[r].heap_ring.size
            );
            LOG_INFO(
                "Ring %d dep pool:     %d / %d", r, orch->rings[r].dep_pool.used(), orch->rings[r].dep_pool.capacity
            );
        }
    }
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
    d.alloc_cycle = g_orch_alloc_cycle;
    d.args_cycle = g_orch_args_cycle;
    d.heap_cycle = g_orch_heap_cycle;
    d.fanin_cycle = g_orch_fanin_cycle;
    d.scope_end_cycle = g_orch_scope_end_cycle;
    d.submit_count = g_orch_submit_count;
    d.alloc_wait_cycle = g_orch_alloc_wait_cycle;
    d.heap_wait_cycle = g_orch_heap_wait_cycle;
    d.fanin_wait_cycle = g_orch_fanin_wait_cycle;
    d.alloc_atomic_count = g_orch_alloc_atomic_count;
    d.args_atomic_count = g_orch_args_atomic_count;
    d.heap_atomic_count = g_orch_heap_atomic_count;
    d.fanin_atomic_count = g_orch_fanin_atomic_count;
    d.scope_end_atomic_count = g_orch_scope_end_atomic_count;

    // Reset
    g_orch_alloc_cycle = g_orch_args_cycle = 0;
    g_orch_heap_cycle = g_orch_fanin_cycle = 0;
    g_orch_scope_end_cycle = 0;
    g_orch_submit_count = 0;
    g_orch_submit_idx = 0;
    g_orch_alloc_wait_cycle = 0;
    g_orch_heap_wait_cycle = 0;
    g_orch_fanin_wait_cycle = 0;
    g_orch_alloc_atomic_count = 0;
    g_orch_args_atomic_count = 0;
    g_orch_heap_atomic_count = 0;
    g_orch_fanin_atomic_count = 0;
    g_orch_scope_end_atomic_count = 0;
    return d;
}
#endif
