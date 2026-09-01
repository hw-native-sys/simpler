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
 * Lives under host_build_graph/shared/ so it is included in both the
 * host_runtime.so build (host pre-populates the prebuilt arena image) and the
 * aicpu_runtime build (AICPU runs wire_arena_pointers + destroy after attach).
 * The device-only parts of runtime_core.cpp / orchestrator.cpp / scheduler.cpp
 * (ops table, scope/submit/dispatch business logic, profiling) stay in their
 * original files and the aicpu build only.
 */

#include <new>
#include <stdlib.h>
#include <string.h>

#include "host_build_graph/orchestrator.h"
#include "host_build_graph/runtime_core.h"
#include "host_build_graph/task_allocator.h"
#include "host_build_graph/shared_memory.h"
#include "host_build_graph/tensormap.h"
#include "scheduler/scheduler.h"

// =============================================================================
// Ready queue
// =============================================================================

size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity) {
    // Align the slots[] base to a full cache line so MPMC CAS traffic on the
    // first slot cannot false-share with whatever region sits in front of us.
    return arena.reserve(capacity * sizeof(ChipReadyQueueSlot), CHIP_ALIGN_SIZE);
}

// Initialize the queue header only. slots[] carries the sequence ramp push
// compares against, but it lives past the uploaded range and is seeded on the
// device by SchedulerState::seed_queue_slots(), so nothing writes it here.
void ready_queue_init_data_from_layout(ChipReadyQueue *queue, uint64_t capacity) {
    queue->capacity = capacity;
    queue->mask = capacity - 1;
    queue->enqueue_pos.store(0, std::memory_order_relaxed);
    queue->dequeue_pos.store(0, std::memory_order_relaxed);
    queue->max_occupancy.store(0, std::memory_order_relaxed);
}

void ready_queue_wire_arena_pointers(ChipReadyQueue *queue, DeviceArena &arena, size_t slots_off) {
    queue->slots = static_cast<ChipReadyQueueSlot *>(arena.region_ptr(slots_off));
}

void ready_queue_destroy(ChipReadyQueue *queue) {
    // Arena owns the slots[] buffer; just forget the pointer.
    queue->slots = nullptr;
}

// =============================================================================
// Scheduler
// =============================================================================

bool SchedulerState::TaskHeaderView::init_data_from_layout(void *sm_dev_base) {
    // `tasks` is the device address of the SM task header — pure offset
    // arithmetic, no SM load.
    tasks = sm_layout::task_header_addr(sm_dev_base);

    // Per-slot SM-side initialization (reset_for_reuse + active_mask, and clearing
    // the completion flag) happens init-on-write in orch::prepare_task as each slot
    // is claimed; host prebuilt-arena init skips SM access here.

    return true;
}

void SchedulerState::TaskHeaderView::destroy() { tasks = nullptr; }

SchedulerLayout SchedulerState::reserve_layout(DeviceArena &arena) {
    SchedulerLayout layout{};
    for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i) {
        layout.capacities.ready[i] = READY_QUEUE_CAPACITY_LIMIT;
        layout.capacities.ready_sync[i] = READY_QUEUE_CAPACITY_LIMIT;
    }
    layout.capacities.dummy = READY_QUEUE_CAPACITY_LIMIT;
    layout.capacities.graph_ready = READY_QUEUE_CAPACITY_LIMIT;
    layout.capacities.graph_prepare = READY_QUEUE_CAPACITY_LIMIT;

    // Fixed-capacity early-dispatch queues first, then the configurable queues.
    // The big nine are the arena's last reservations so that the bytes bind
    // uploads stay one contiguous range no matter how much of them is in use.
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        layout.off_early_dispatch_queue_slots[i] = ready_queue_reserve_layout(arena, CHIP_EARLY_DISPATCH_QUEUE_SIZE);
    }
    layout.off_early_sync_start_queue_slots = ready_queue_reserve_layout(arena, CHIP_EARLY_DISPATCH_QUEUE_SIZE);
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        layout.off_ready_queue_slots[i] = ready_queue_reserve_layout(arena, READY_QUEUE_CAPACITY_LIMIT);
    }
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        layout.off_ready_sync_queue_slots[i] = ready_queue_reserve_layout(arena, READY_QUEUE_CAPACITY_LIMIT);
    }
    layout.off_dummy_ready_queue_slots = ready_queue_reserve_layout(arena, READY_QUEUE_CAPACITY_LIMIT);
    layout.off_graph_ready_queue_slots = ready_queue_reserve_layout(arena, READY_QUEUE_CAPACITY_LIMIT);
    layout.off_graph_prepare_queue_slots = ready_queue_reserve_layout(arena, READY_QUEUE_CAPACITY_LIMIT);
    // Polling: no dep_pool arena region — producer dependencies are inline ids on
    // the payload and readiness is via completion_flags.
    return layout;
}

bool SchedulerState::init_data_from_layout(const SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base) {
    SchedulerState *sched = this;
    sched->sm_header = reinterpret_cast<SharedMemoryHeader *>(sm_dev_base);
#if SIMPLER_SCHED_PROFILING
    sched->tasks_completed.store(0, std::memory_order_relaxed);
    sched->tasks_consumed.store(0, std::memory_order_relaxed);
#endif

    if (!sched->task_view.init_data_from_layout(sm_dev_base)) {
        return false;
    }

    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        ready_queue_init_data_from_layout(&sched->ready_queues[i], layout.capacities.ready[i]);
    }
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        ready_queue_init_data_from_layout(&sched->ready_sync_queues[i], layout.capacities.ready_sync[i]);
    }
    ready_queue_init_data_from_layout(&sched->dummy_ready_queue, layout.capacities.dummy);
    ready_queue_init_data_from_layout(&sched->graph_ready_queue, layout.capacities.graph_ready);
    ready_queue_init_data_from_layout(&sched->graph_prepare_queue, layout.capacities.graph_prepare);
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        ready_queue_init_data_from_layout(&sched->early_dispatch_queues[i], CHIP_EARLY_DISPATCH_QUEUE_SIZE);
    }
    ready_queue_init_data_from_layout(&sched->early_sync_start_queue, CHIP_EARLY_DISPATCH_QUEUE_SIZE);

    // Polling: no dep_pool arena region to initialize.
    (void)arena;
    (void)layout;
    return true;
}

// Device-only: establish every queue's empty ramp. Mirrors the enumeration in
// wire_arena_pointers / destroy — a queue whose slots[] is never seeded accepts
// one push and then reports full, since push claims a slot only when its
// sequence already equals the position being claimed.
void SchedulerState::seed_queue_slots() {
    SchedulerState *sched = this;
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        sched->ready_queues[i].seed_slots();
    }
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        sched->ready_sync_queues[i].seed_slots();
    }
    sched->dummy_ready_queue.seed_slots();
    sched->graph_ready_queue.seed_slots();
    sched->graph_prepare_queue.seed_slots();
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        sched->early_dispatch_queues[i].seed_slots();
    }
    sched->early_sync_start_queue.seed_slots();
}

void SchedulerState::wire_arena_pointers(const SchedulerLayout &layout, DeviceArena &arena) {
    SchedulerState *sched = this;
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        ready_queue_wire_arena_pointers(&sched->ready_queues[i], arena, layout.off_ready_queue_slots[i]);
    }
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        ready_queue_wire_arena_pointers(&sched->ready_sync_queues[i], arena, layout.off_ready_sync_queue_slots[i]);
    }
    ready_queue_wire_arena_pointers(&sched->dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots);
    ready_queue_wire_arena_pointers(&sched->graph_ready_queue, arena, layout.off_graph_ready_queue_slots);
    ready_queue_wire_arena_pointers(&sched->graph_prepare_queue, arena, layout.off_graph_prepare_queue_slots);
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        ready_queue_wire_arena_pointers(
            &sched->early_dispatch_queues[i], arena, layout.off_early_dispatch_queue_slots[i]
        );
    }
    ready_queue_wire_arena_pointers(&sched->early_sync_start_queue, arena, layout.off_early_sync_start_queue_slots);
}

void SchedulerState::destroy() {
    SchedulerState *sched = this;
    sched->task_view.destroy();
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        ready_queue_destroy(&sched->ready_queues[i]);
    }
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        ready_queue_destroy(&sched->ready_sync_queues[i]);
    }
    ready_queue_destroy(&sched->dummy_ready_queue);
    ready_queue_destroy(&sched->graph_ready_queue);
    ready_queue_destroy(&sched->graph_prepare_queue);
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        ready_queue_destroy(&sched->early_dispatch_queues[i]);
    }
    ready_queue_destroy(&sched->early_sync_start_queue);
}

// =============================================================================
// Orchestrator
// =============================================================================

bool OrchestratorState::init(
    void *sm_base, void *gm_heap, uint64_t heap_size, uint64_t max_tasks, SchedulerState *scheduler_arg
) {
    // Reset in place rather than by move-assignment: fatal_code is a std::atomic,
    // which is neither copy- nor move-assignable, and a re-init has to clear every
    // field the previous pass left behind (the pool cursors below rely on it).
    this->~OrchestratorState();
    auto *orch = new (static_cast<void *>(this)) OrchestratorState{};

    always_assert(max_tasks > 0);

    orch->sm_header = reinterpret_cast<SharedMemoryHeader *>(sm_base);
    orch->scheduler = scheduler_arg;

    orch->task_allocator.init(static_cast<int32_t>(max_tasks), gm_heap, heap_size, &orch->fatal_code);

    // The mirror's argument pools. Offset arithmetic on the same base as sm_header,
    // so it holds for whichever SM this orchestrator was pointed at. The cursors
    // reset with the rest of the state above.
    auto *sm_bytes = static_cast<char *>(sm_base);
    const auto pools = sm_layout::segment_offsets(sm_layout::mirror_extents(max_tasks));
    orch->fanin_pool = reinterpret_cast<int32_t *>(sm_bytes + pools.fanin_pool);
    orch->tensor_pool = reinterpret_cast<simpler::hbg::Tensor *>(sm_bytes + pools.tensor_pool);
    orch->scalar_pool = reinterpret_cast<uint64_t *>(sm_bytes + pools.scalar_pool);

    // Polling: no fanin-spill pool — producer ids are inline on the payload.
    const auto slots = static_cast<size_t>(max_tasks);
    orch->fanin_seen_epoch.reset(new (std::nothrow) uint32_t[slots]);
    if (orch->fanin_seen_epoch == nullptr) {
        LOG_ERROR("Orchestrator scratch allocation failed (max_tasks=%" PRIu64 ")", max_tasks);
        return false;
    }
    memset(orch->fanin_seen_epoch.get(), 0, slots * sizeof(uint32_t));

    // One ancestor word per task slot: 8 B each, 128 KiB at the default 16,384-task
    // table, linear in runtime_env.ring_task_window. Zeroed here so a slot claimed
    // by a submit that fails before publishing its own entry reads as "no known
    // ancestors", which keeps reduction conservative rather than wrong.
    orch->fanin_reach.reset(new (std::nothrow) uint64_t[slots]);
    if (orch->fanin_reach == nullptr) {
        LOG_ERROR("Orchestrator scratch allocation failed (max_tasks=%" PRIu64 ")", max_tasks);
        return false;
    }
    memset(orch->fanin_reach.get(), 0, slots * sizeof(uint64_t));

    if (!orch->tensor_map.init_default(static_cast<int32_t>(max_tasks))) {
        return false;
    }

    orch->scope_stack_top = -1;
    orch->manual_begin_depth = CHIP_MAX_SCOPE_DEPTH;

    return true;
}

// =============================================================================
// Top-level runtime arena
// =============================================================================

RuntimeArenaLayout runtime_reserve_layout(DeviceArena &arena, uint64_t task_capacity) {
    RuntimeArenaLayout layout{};

    layout.task_capacity = task_capacity;

    // Reservation order is the zone partition (see RuntimeArenaLayout):
    // everything the device initializes itself, then the one copied range. Each
    // zone is contiguous, so bind is a single copy and no consumer has to infer a
    // boundary from what happens to come first.
    //
    // The copied zone comes last of the two so the device's shared-memory tail can
    // begin where it ends, making the two adjacent and the upload one copy.
    layout.off_sm_handle = arena.reserve(sizeof(SharedMemoryHandle), alignof(SharedMemoryHandle));
    layout.off_mailbox = arena.reserve(sizeof(AICoreCompletionMailbox), alignof(AICoreCompletionMailbox));
    layout.off_scheduler = arena.reserve(sizeof(SchedulerState), alignof(SchedulerState));
    layout.sched = SchedulerState::reserve_layout(arena);

    layout.off_copied_begin = arena.total_size();
    // Padded to a CHIP_ALIGN_SIZE boundary: the shared-memory image starts at
    // off_copied_end on the device and its segment offsets are aligned from there.
    layout.off_runtime = arena.reserve(CHIP_ALIGN_UP(sizeof(RuntimeContext), CHIP_ALIGN_SIZE), CHIP_ALIGN_SIZE);
    layout.off_copied_end = arena.total_size();

    layout.arena_size = arena.total_size();
    return layout;
}

/**
 * Populate the prebuilt runtime-arena image in place (host build path).
 *
 * Zeroes the RuntimeContext header at layout.off_runtime and initializes the
 * scheduler (ready / sync / dummy / graph queues) against the device SM.
 * rt->orchestrator is left null: the
 * orchestrator is a host-owned object the host-orch path
 * (run_host_orchestration) stands up against the host SM once that buffer
 * exists, and it is never uploaded to the device. Caller must follow up with
 * runtime_wire_arena_pointers. Returns the arena-resident RuntimeContext*, or
 * nullptr on failure.
 */
RuntimeContext *runtime_init_data_from_layout(
    DeviceArena &arena, const RuntimeArenaLayout &layout, RuntimeMode mode, void *sm_dev_base, uint64_t /*sm_size*/
) {
    RuntimeContext *rt = static_cast<RuntimeContext *>(arena.region_ptr(layout.off_runtime));
    memset(rt, 0, sizeof(*rt));

    // rt->ops is filled by the AICPU at boot.
    rt->mode = mode;
    rt->total_cycles = 0;
    rt->active_callable_hash = 0;

    // Two components are deliberately not initialized here.
    //
    // The orchestrator is not in this arena at all: it is a host-owned object the
    // host-orch path (run_host_orchestration) stands up against the host SM once
    // that buffer is allocated, and rt->orchestrator only points at it for the
    // duration of that pass.
    //
    // The scheduler and sm_handle live in the device-only zone, so their bytes
    // never travel; the AICPU initializes them at boot. Writing them here would
    // be writing an initialization pattern that nothing reads.
    (void)sm_dev_base;

    return rt;
}

void runtime_wire_arena_pointers(DeviceArena &arena, const RuntimeArenaLayout &layout, RuntimeContext *rt) {
    rt->sm_handle = static_cast<SharedMemoryHandle *>(arena.region_ptr(layout.off_sm_handle));
    rt->aicore_mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.off_mailbox));
    rt->scheduler = static_cast<SchedulerState *>(arena.region_ptr(layout.off_scheduler));
    rt->scheduler->wire_arena_pointers(layout.sched, arena);
}

void runtime_destroy(RuntimeContext *rt, DeviceArena & /*arena*/) {
    // Arena buffer is pooled across runs by DeviceRunner — never freed here.
    if (!rt) return;
    rt->scheduler->destroy();
    rt->aicore_mailbox = nullptr;
    rt->sm_handle = nullptr;
}
