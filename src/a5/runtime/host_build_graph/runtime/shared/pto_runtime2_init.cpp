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
 * Host/AICPU shared runtime-arena layout, init_data and wire implementations.
 *
 * Lives under runtime/shared/ so it is included in both the host_runtime.so
 * build (host pre-populates the prebuilt arena image) and the aicpu_runtime
 * build (AICPU runs wire_arena_pointers + destroy after attach). The
 * device-only parts of pto_runtime2.cpp / pto_orchestrator.cpp / pto_scheduler.cpp
 * (ops table, scope/submit/dispatch business logic, profiling) stay in their
 * original files and the aicpu build only.
 */

#include <stdlib.h>
#include <string.h>

#include "pto_orchestrator.h"
#include "pto_runtime2.h"
#include "pto_ring_buffer.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "scheduler/pto_scheduler.h"

// =============================================================================
// Ready queue
// =============================================================================

size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity) {
    // Align the slots[] base to a full cache line so MPMC CAS traffic on the
    // first slot cannot false-share with whatever region sits in front of us
    // (e.g. orchestrator tensormap heads written by the orch thread).
    return arena.reserve(capacity * sizeof(PTO2ReadyQueueSlot), PTO2_ALIGN_SIZE);
}

// Initialize the queue header only. slots[] carries the sequence ramp push
// compares against, but it lives past the uploaded range and is seeded on the
// device by PTO2SchedulerState::seed_queue_slots(), so nothing writes it here.
void ready_queue_init_data_from_layout(PTO2ReadyQueue *queue, uint64_t capacity) {
    queue->capacity = capacity;
    queue->mask = capacity - 1;
    queue->enqueue_pos.store(0, std::memory_order_relaxed);
    queue->dequeue_pos.store(0, std::memory_order_relaxed);
    queue->max_occupancy.store(0, std::memory_order_relaxed);
}

void ready_queue_wire_arena_pointers(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off) {
    queue->slots = static_cast<PTO2ReadyQueueSlot *>(arena.region_ptr(slots_off));
}

void ready_queue_destroy(PTO2ReadyQueue *queue) {
    // Arena owns the slots[] buffer; just forget the pointer.
    queue->slots = nullptr;
}

// =============================================================================
// Scheduler
// =============================================================================

bool PTO2SchedulerState::RingSchedState::init_data_from_layout(void *sm_dev_base) {
    // ring stores the device address of the SM ring header — pure offset
    // arithmetic, no SM load.
    ring = pto2_sm_layout::ring_header_addr(sm_dev_base);
    advance_lock.store(0, std::memory_order_relaxed);

    // Per-slot SM-side initialization (reset_for_reuse + active_mask, and clearing
    // the completion flag) happens init-on-write in orch::prepare_task as each slot
    // is claimed; host prebuilt-arena init skips SM access here.

    return true;
}

void PTO2SchedulerState::RingSchedState::destroy() { ring = nullptr; }

PTO2SchedulerLayout PTO2SchedulerState::reserve_layout(DeviceArena &arena) {
    PTO2SchedulerLayout layout{};
    layout.ready_queue_capacity = PTO2_READY_QUEUE_SIZE;

    // Fixed-capacity early-dispatch queues first, then the PTO2_READY_QUEUE_SIZE
    // ones. The big nine are the arena's last reservations so that the bytes bind
    // uploads stay one contiguous range no matter how much of them is in use.
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        layout.off_early_dispatch_queue_slots[i] = ready_queue_reserve_layout(arena, PTO2_EARLY_DISPATCH_QUEUE_SIZE);
    }
    layout.off_early_sync_start_queue_slots = ready_queue_reserve_layout(arena, PTO2_EARLY_DISPATCH_QUEUE_SIZE);
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        layout.off_ready_queue_slots[i] = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    }
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        layout.off_ready_sync_queue_slots[i] = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    }
    layout.off_dummy_ready_queue_slots = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    layout.off_graph_ready_queue_slots = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    layout.off_graph_prepare_queue_slots = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    // Polling: no dep_pool arena region — producer dependencies are inline ids on
    // the payload and readiness is via completion_flags.
    return layout;
}

bool PTO2SchedulerState::init_data_from_layout(
    const PTO2SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base
) {
    PTO2SchedulerState *sched = this;
    sched->sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);
#if SIMPLER_SCHED_PROFILING
    sched->tasks_completed.store(0, std::memory_order_relaxed);
    sched->tasks_consumed.store(0, std::memory_order_relaxed);
#endif

    if (!sched->ring_sched_state.init_data_from_layout(sm_dev_base)) {
        return false;
    }

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_init_data_from_layout(&sched->ready_queues[i], layout.ready_queue_capacity);
    }
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_init_data_from_layout(&sched->ready_sync_queues[i], layout.ready_queue_capacity);
    }
    ready_queue_init_data_from_layout(&sched->dummy_ready_queue, layout.ready_queue_capacity);
    ready_queue_init_data_from_layout(&sched->graph_ready_queue, layout.ready_queue_capacity);
    ready_queue_init_data_from_layout(&sched->graph_prepare_queue, layout.ready_queue_capacity);
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_init_data_from_layout(&sched->early_dispatch_queues[i], PTO2_EARLY_DISPATCH_QUEUE_SIZE);
    }
    ready_queue_init_data_from_layout(&sched->early_sync_start_queue, PTO2_EARLY_DISPATCH_QUEUE_SIZE);

    // Polling: no dep_pool arena region to initialize.
    (void)arena;
    (void)layout;
    return true;
}

// Device-only: establish every queue's empty ramp. Mirrors the enumeration in
// wire_arena_pointers / destroy — a queue whose slots[] is never seeded accepts
// one push and then reports full, since push claims a slot only when its
// sequence already equals the position being claimed.
void PTO2SchedulerState::seed_queue_slots() {
    PTO2SchedulerState *sched = this;
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        sched->ready_queues[i].seed_slots();
    }
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        sched->ready_sync_queues[i].seed_slots();
    }
    sched->dummy_ready_queue.seed_slots();
    sched->graph_ready_queue.seed_slots();
    sched->graph_prepare_queue.seed_slots();
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        sched->early_dispatch_queues[i].seed_slots();
    }
    sched->early_sync_start_queue.seed_slots();
}

void PTO2SchedulerState::wire_arena_pointers(const PTO2SchedulerLayout &layout, DeviceArena &arena) {
    PTO2SchedulerState *sched = this;
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_wire_arena_pointers(&sched->ready_queues[i], arena, layout.off_ready_queue_slots[i]);
    }
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_wire_arena_pointers(&sched->ready_sync_queues[i], arena, layout.off_ready_sync_queue_slots[i]);
    }
    ready_queue_wire_arena_pointers(&sched->dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots);
    ready_queue_wire_arena_pointers(&sched->graph_ready_queue, arena, layout.off_graph_ready_queue_slots);
    ready_queue_wire_arena_pointers(&sched->graph_prepare_queue, arena, layout.off_graph_prepare_queue_slots);
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_wire_arena_pointers(
            &sched->early_dispatch_queues[i], arena, layout.off_early_dispatch_queue_slots[i]
        );
    }
    ready_queue_wire_arena_pointers(&sched->early_sync_start_queue, arena, layout.off_early_sync_start_queue_slots);
}

void PTO2SchedulerState::destroy() {
    PTO2SchedulerState *sched = this;
    sched->ring_sched_state.destroy();
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_destroy(&sched->ready_queues[i]);
    }
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_destroy(&sched->ready_sync_queues[i]);
    }
    ready_queue_destroy(&sched->dummy_ready_queue);
    ready_queue_destroy(&sched->graph_ready_queue);
    ready_queue_destroy(&sched->graph_prepare_queue);
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_destroy(&sched->early_dispatch_queues[i]);
    }
    ready_queue_destroy(&sched->early_sync_start_queue);
}

// =============================================================================
// Orchestrator
// =============================================================================

PTO2OrchestratorLayout PTO2OrchestratorState::reserve_layout(DeviceArena &arena, int32_t task_capacity) {
    PTO2OrchestratorLayout layout{};
    // scope_tasks holds every task in the open scope, so its cap is the real
    // in-flight budget = the runtime task capacity.
    always_assert(task_capacity > 0);
    layout.scope_tasks_cap = task_capacity;
    layout.scope_stack_capacity = PTO2_MAX_SCOPE_DEPTH;

    // Polling: no fanin-spill pool — producer ids are inline on the payload.
    always_assert(task_capacity > 0 && (task_capacity & (task_capacity - 1)) == 0);
    const size_t seen_epoch_bytes =
        PTO2_ALIGN_UP(static_cast<size_t>(task_capacity) * sizeof(uint32_t), PTO2_ALIGN_SIZE);
    layout.off_fanin_seen_epoch = arena.reserve(seen_epoch_bytes, PTO2_ALIGN_SIZE);

    layout.off_scope_tasks =
        arena.reserve(static_cast<size_t>(layout.scope_tasks_cap) * sizeof(uintptr_t), alignof(PTO2TaskSlotState *));
    layout.off_scope_begins =
        arena.reserve(static_cast<size_t>(layout.scope_stack_capacity) * sizeof(int32_t), alignof(int32_t));
    layout.tensor_map = PTO2TensorMap::reserve_layout_default(arena, task_capacity);
    return layout;
}

bool PTO2OrchestratorState::init_data_from_layout(
    const PTO2OrchestratorLayout &layout, DeviceArena &arena, void *sm_dev_base, void *gm_heap, uint64_t heap_size,
    uint64_t task_capacity
) {
    auto *orch = this;
    *orch = PTO2OrchestratorState{};

    orch->sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);
    orch->gm_heap_base = gm_heap;
    orch->gm_heap_size = heap_size;
    orch->fatal = false;

    auto *orch_err = pto2_sm_layout::orch_error_code_addr(sm_dev_base);
    auto *cur_idx_dev = pto2_sm_layout::ring_current_task_index_addr(sm_dev_base);

    orch->task_allocator.init(static_cast<int32_t>(task_capacity), cur_idx_dev, gm_heap, heap_size, orch_err);

    const size_t seen_epoch_bytes =
        PTO2_ALIGN_UP(static_cast<size_t>(layout.tensor_map.task_capacity) * sizeof(uint32_t), PTO2_ALIGN_SIZE);
    auto *seen_epoch = static_cast<uint32_t *>(arena.region_ptr(layout.off_fanin_seen_epoch));
    memset(seen_epoch, 0, seen_epoch_bytes);
    orch->fanin_seen_epoch = seen_epoch;

    if (!orch->tensor_map.init_data_from_layout(layout.tensor_map, arena)) {
        return false;
    }

    orch->scope_tasks_size = 0;
    orch->scope_tasks_capacity = layout.scope_tasks_cap;
    orch->scope_stack_top = -1;
    orch->scope_stack_capacity = layout.scope_stack_capacity;
    orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;

    return true;
}

void PTO2OrchestratorState::wire_arena_pointers(
    const PTO2OrchestratorLayout &layout, DeviceArena &arena, PTO2SchedulerState *scheduler_arg
) {
    auto *orch = this;
    orch->fanin_seen_epoch = static_cast<uint32_t *>(arena.region_ptr(layout.off_fanin_seen_epoch));
    orch->tensor_map.wire_arena_pointers(layout.tensor_map, arena);
    orch->scope_tasks = static_cast<PTO2TaskSlotState **>(arena.region_ptr(layout.off_scope_tasks));
    orch->scope_begins = static_cast<int32_t *>(arena.region_ptr(layout.off_scope_begins));
    orch->scheduler = scheduler_arg;
}

void PTO2OrchestratorState::destroy() {
    auto *orch = this;
    orch->tensor_map.destroy();
    orch->fanin_seen_epoch = nullptr;
    orch->scope_tasks = nullptr;
    orch->scope_begins = nullptr;
}

void PTO2OrchestratorState::set_scheduler(PTO2SchedulerState *scheduler) { this->scheduler = scheduler; }

// =============================================================================
// Top-level runtime arena
// =============================================================================

PTO2RuntimeArenaLayout runtime_reserve_layout(DeviceArena &arena, uint64_t task_capacity) {
    PTO2RuntimeArenaLayout layout{};
    layout.task_capacity = task_capacity;

    // Reservation order is the zone partition (see PTO2RuntimeArenaLayout):
    // everything the device initializes itself, then the one copied range. Each
    // zone is contiguous, so bind is a single copy and no consumer has to infer a
    // boundary from what happens to come first.
    //
    // The copied zone comes last of the two so the device's shared-memory tail can
    // begin where it ends, making the two adjacent and the upload one copy.
    layout.off_sm_handle = arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
    layout.off_mailbox = arena.reserve(sizeof(AICoreCompletionMailbox), alignof(AICoreCompletionMailbox));
    layout.off_scheduler = arena.reserve(sizeof(PTO2SchedulerState), alignof(PTO2SchedulerState));
    layout.sched = PTO2SchedulerState::reserve_layout(arena);

    layout.off_copied_begin = arena.total_size();
    // Padded to a PTO2_ALIGN_SIZE boundary: the shared-memory image starts at
    // off_copied_end on the device and its segment offsets are aligned from there.
    layout.off_runtime = arena.reserve(PTO2_ALIGN_UP(sizeof(PTO2Runtime), PTO2_ALIGN_SIZE), PTO2_ALIGN_SIZE);
    layout.off_copied_end = arena.total_size();

    layout.device_bytes = arena.total_size();
    layout.orch = PTO2OrchestratorState::reserve_layout(arena, static_cast<int32_t>(task_capacity));

    layout.arena_size = arena.total_size();
    return layout;
}

/**
 * Populate the prebuilt runtime-arena image in place (host build path).
 *
 * Zeroes the PTO2Runtime header at layout.off_runtime, records the GM heap,
 * and initializes the scheduler (ready / sync / dummy / graph queues) against
 * the device SM. The orchestrator is deliberately left zeroed: the host-orch
 * path (run_host_orchestration) initializes it against the host SM once that
 * buffer exists. Initializing it here would be dead work — overwritten by that
 * re-init, and the orchestrator arena block is never uploaded to the device. Caller must follow up with
 * runtime_wire_arena_pointers. Returns the arena-resident PTO2Runtime*, or
 * nullptr on failure.
 */
PTO2Runtime *runtime_init_data_from_layout(
    DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2RuntimeMode mode, void *sm_dev_base,
    uint64_t /*sm_size*/, void *gm_heap_dev_base, uint64_t heap_size
) {
    PTO2Runtime *rt = static_cast<PTO2Runtime *>(arena.region_ptr(layout.off_runtime));
    memset(rt, 0, sizeof(*rt));

    // rt->ops is filled by the AICPU at boot.
    rt->mode = mode;
    rt->gm_heap = gm_heap_dev_base;
    rt->gm_heap_size = heap_size;
    rt->gm_heap_owned = false;
    rt->total_cycles = 0;
    rt->active_callable_hash = 0;

    // Two components are deliberately not initialized here.
    //
    // The orchestrator is initialized by the host-orch path
    // (run_host_orchestration) against the host SM once it is allocated. Doing it
    // here would be dead work: its arena content (tensormap + seen_epoch memset)
    // is immediately overwritten by that re-init, and the orchestrator block is
    // host-only anyway.
    //
    // The scheduler and sm_handle live in the device-only zone, so their bytes
    // never travel; the AICPU initializes them at boot. Writing them here would
    // be writing an initialization pattern that nothing reads.
    (void)sm_dev_base;

    return rt;
}

void runtime_wire_arena_pointers(DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2Runtime *rt) {
    rt->sm_handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(layout.off_sm_handle));
    rt->aicore_mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.off_mailbox));
    rt->scheduler = static_cast<PTO2SchedulerState *>(arena.region_ptr(layout.off_scheduler));
    rt->scheduler->wire_arena_pointers(layout.sched, arena);
}

void runtime_wire_host_only_pointers(DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2Runtime *rt) {
    rt->orchestrator.wire_arena_pointers(layout.orch, arena, rt->scheduler);
}

void runtime_clear_host_only_pointers(PTO2Runtime *rt) { rt->orchestrator.destroy(); }

void runtime_destroy(PTO2Runtime *rt, DeviceArena & /*arena*/) {
    // Arena buffer is pooled across runs by DeviceRunner — never freed here.
    if (!rt) return;
    rt->scheduler->destroy();
    rt->orchestrator.destroy();
    rt->aicore_mailbox = nullptr;
    rt->sm_handle = nullptr;
}
