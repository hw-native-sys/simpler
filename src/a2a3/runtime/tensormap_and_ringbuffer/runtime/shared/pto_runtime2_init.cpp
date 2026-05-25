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

bool ready_queue_init_data_from_layout(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off, uint64_t capacity) {
    // Address the slots region for data writes without storing the pointer in
    // queue->slots — that field is set by ready_queue_wire_arena_pointers.
    auto *slots_arena = static_cast<PTO2ReadyQueueSlot *>(arena.region_ptr(slots_off));
    queue->capacity = capacity;
    queue->mask = capacity - 1;
    queue->enqueue_pos.store(0, std::memory_order_relaxed);
    queue->dequeue_pos.store(0, std::memory_order_relaxed);

    for (uint64_t i = 0; i < capacity; i++) {
        slots_arena[i].sequence.store((int64_t)i, std::memory_order_relaxed);
        slots_arena[i].slot_state = nullptr;
    }

    return true;
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

bool PTO2SchedulerState::RingSchedState::init_data_from_layout(void *sm_dev_base, int32_t ring_id) {
    // ring stores the device address of the SM ring header — pure offset
    // arithmetic, no SM load.
    ring = pto2_sm_layout::ring_header_addr(sm_dev_base, ring_id);
    last_task_alive = 0;
    advance_lock.store(0, std::memory_order_relaxed);

    // Per-slot SM-side initialization (bind_ring + reset_for_reuse +
    // fanin_count/active_mask zero) lives in PTO2SharedMemoryHandle::
    // init_header_per_ring so the AICPU performs it during SM reset; host
    // prebuilt-arena init skips SM access here.

    return true;
}

void PTO2SchedulerState::RingSchedState::destroy() { ring = nullptr; }

PTO2SchedulerLayout PTO2SchedulerState::reserve_layout(DeviceArena &arena, int32_t dep_pool_capacity) {
    PTO2SchedulerLayout layout{};
    layout.ready_queue_capacity = PTO2_READY_QUEUE_SIZE;
    layout.spsc_capacity = PTO2_WRIRING_QUEUE_SIZE;
    layout.dep_pool_capacity = dep_pool_capacity;

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        layout.off_ready_queue_slots[i] = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    }
    layout.off_dummy_ready_queue_slots = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        // Force a cache-line base so writes from scheduler thread 0 (sole
        // writer of this ring's dep_pool) do not invalidate adjacent
        // multi-threaded regions like ready_queue.slots.
        layout.off_dep_pool_entries[r] =
            arena.reserve(static_cast<size_t>(dep_pool_capacity) * sizeof(PTO2DepListEntry), PTO2_ALIGN_SIZE);
    }
    layout.off_wiring_spsc_buffer = PTO2SpscQueue::reserve_layout(arena, PTO2_WRIRING_QUEUE_SIZE);
    return layout;
}

bool PTO2SchedulerState::init_data_from_layout(
    const PTO2SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base
) {
    PTO2SchedulerState *sched = this;
    sched->sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);
#if PTO2_SCHED_PROFILING
    sched->tasks_completed.store(0, std::memory_order_relaxed);
    sched->tasks_consumed.store(0, std::memory_order_relaxed);
#endif

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        if (!sched->ring_sched_states[r].init_data_from_layout(sm_dev_base, r)) {
            return false;
        }
    }

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        if (!ready_queue_init_data_from_layout(
                &sched->ready_queues[i], arena, layout.off_ready_queue_slots[i], layout.ready_queue_capacity
            )) {
            return false;
        }
    }
    if (!ready_queue_init_data_from_layout(
            &sched->dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots, layout.ready_queue_capacity
        )) {
        return false;
    }

    auto *orch_err = pto2_sm_layout::orch_error_code_addr(sm_dev_base);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        auto *dep_entries = static_cast<PTO2DepListEntry *>(arena.region_ptr(layout.off_dep_pool_entries[r]));
        memset(dep_entries, 0, static_cast<size_t>(layout.dep_pool_capacity) * sizeof(PTO2DepListEntry));
        sched->ring_sched_states[r].dep_pool.init(dep_entries, layout.dep_pool_capacity, orch_err);
    }

    if (!sched->wiring.queue.init_data_from_layout(arena, layout.off_wiring_spsc_buffer, layout.spsc_capacity)) {
        return false;
    }
    sched->wiring.batch_count = 0;
    sched->wiring.batch_index = 0;
    sched->wiring.backoff_counter = 0;

    return true;
}

void PTO2SchedulerState::wire_arena_pointers(const PTO2SchedulerLayout &layout, DeviceArena &arena) {
    PTO2SchedulerState *sched = this;
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_wire_arena_pointers(&sched->ready_queues[i], arena, layout.off_ready_queue_slots[i]);
    }
    ready_queue_wire_arena_pointers(&sched->dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        sched->ring_sched_states[r].dep_pool.base =
            static_cast<PTO2DepListEntry *>(arena.region_ptr(layout.off_dep_pool_entries[r]));
    }
    sched->wiring.queue.wire_arena_pointers(arena, layout.off_wiring_spsc_buffer);
}

void PTO2SchedulerState::destroy() {
    PTO2SchedulerState *sched = this;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        sched->ring_sched_states[r].destroy();
        sched->ring_sched_states[r].dep_pool.base = nullptr;
    }
    sched->wiring.queue.destroy();
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_destroy(&sched->ready_queues[i]);
    }
    ready_queue_destroy(&sched->dummy_ready_queue);
}

// =============================================================================
// Orchestrator
// =============================================================================

PTO2OrchestratorLayout PTO2OrchestratorState::reserve_layout(
    DeviceArena &arena, const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH], int32_t dep_pool_capacity
) {
    PTO2OrchestratorLayout layout{};
    layout.dep_pool_capacity = dep_pool_capacity;
    layout.scope_tasks_cap = PTO2_SCOPE_TASKS_CAP;
    layout.scope_stack_capacity = PTO2_MAX_SCOPE_DEPTH;

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        const size_t fanin_pool_bytes =
            PTO2_ALIGN_UP(static_cast<size_t>(dep_pool_capacity) * sizeof(PTO2FaninSpillEntry), PTO2_ALIGN_SIZE);
        layout.off_fanin_pool[r] = arena.reserve(fanin_pool_bytes, PTO2_ALIGN_SIZE);
    }
    layout.off_scope_tasks = arena.reserve(
        static_cast<size_t>(layout.scope_tasks_cap) * sizeof(PTO2TaskSlotState *), alignof(PTO2TaskSlotState *)
    );
    layout.off_scope_begins =
        arena.reserve(static_cast<size_t>(layout.scope_stack_capacity) * sizeof(int32_t), alignof(int32_t));
    layout.tensor_map = PTO2TensorMap::reserve_layout_default(arena, task_window_sizes);
    return layout;
}

bool PTO2OrchestratorState::init_data_from_layout(
    const PTO2OrchestratorLayout &layout, DeviceArena &arena, void *sm_dev_base, void *gm_heap, uint64_t heap_size,
    uint64_t task_window_size
) {
    auto *orch = this;
    *orch = PTO2OrchestratorState{};

    orch->sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);
    orch->gm_heap_base = gm_heap;
    orch->gm_heap_size = heap_size * PTO2_MAX_RING_DEPTH;
    orch->fatal = false;

    // Mirror the SM API's per-ring window-size shape so a future per-ring
    // SM layout cannot silently disagree with the addresses we compute here.
    uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        task_window_sizes[r] = task_window_size;

    auto *orch_err = pto2_sm_layout::orch_error_code_addr(sm_dev_base);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        void *ring_heap_base = reinterpret_cast<char *>(gm_heap) + r * heap_size;
        auto *task_descs_dev = pto2_sm_layout::ring_task_descriptors_addr(sm_dev_base, task_window_sizes, r);
        auto *cur_idx_dev = pto2_sm_layout::ring_current_task_index_addr(sm_dev_base, r);
        auto *last_alive_dev = pto2_sm_layout::ring_last_task_alive_addr(sm_dev_base, r);

        orch->rings[r].task_allocator.init(
            task_descs_dev, static_cast<int32_t>(task_window_size), cur_idx_dev, last_alive_dev, ring_heap_base,
            heap_size, orch_err
        );

        const size_t fanin_pool_bytes =
            PTO2_ALIGN_UP(static_cast<size_t>(layout.dep_pool_capacity) * sizeof(PTO2FaninSpillEntry), PTO2_ALIGN_SIZE);
        auto *fanin_entries = static_cast<PTO2FaninSpillEntry *>(arena.region_ptr(layout.off_fanin_pool[r]));
        memset(fanin_entries, 0, fanin_pool_bytes);
        orch->rings[r].fanin_pool.init(fanin_entries, layout.dep_pool_capacity, orch_err);
    }

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
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        orch->rings[r].fanin_pool.base = static_cast<PTO2FaninSpillEntry *>(arena.region_ptr(layout.off_fanin_pool[r]));
    }
    orch->tensor_map.wire_arena_pointers(layout.tensor_map, arena);
    orch->scope_tasks = static_cast<PTO2TaskSlotState **>(arena.region_ptr(layout.off_scope_tasks));
    orch->scope_begins = static_cast<int32_t *>(arena.region_ptr(layout.off_scope_begins));
    orch->scheduler = scheduler_arg;
}

void PTO2OrchestratorState::destroy() {
    auto *orch = this;
    orch->tensor_map.destroy();
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        orch->rings[r].fanin_pool.base = nullptr;
    }
    orch->scope_tasks = nullptr;
    orch->scope_begins = nullptr;
}

void PTO2OrchestratorState::set_scheduler(PTO2SchedulerState *scheduler) { this->scheduler = scheduler; }

// =============================================================================
// Top-level runtime arena
// =============================================================================

PTO2RuntimeArenaLayout
runtime_reserve_layout(DeviceArena &arena, uint64_t task_window_size, int32_t dep_pool_capacity) {
    PTO2RuntimeArenaLayout layout{};
    layout.task_window_size = task_window_size;
    layout.dep_pool_capacity = dep_pool_capacity;

    int32_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = static_cast<int32_t>(task_window_size);
    }

    layout.off_sm_handle = arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
    layout.orch = PTO2OrchestratorState::reserve_layout(arena, task_window_sizes, dep_pool_capacity);
    layout.sched = PTO2SchedulerState::reserve_layout(arena, dep_pool_capacity);
    layout.off_runtime = arena.reserve(sizeof(PTO2Runtime), PTO2_ALIGN_SIZE);
    layout.off_mailbox = arena.reserve(sizeof(AICoreCompletionMailbox), alignof(AICoreCompletionMailbox));

    layout.arena_size = arena.total_size();
    return layout;
}

PTO2Runtime *runtime_init_data_from_layout(
    DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2RuntimeMode mode, void *sm_dev_base,
    uint64_t /*sm_size*/, void *gm_heap_dev_base, uint64_t heap_size
) {
    PTO2Runtime *rt = static_cast<PTO2Runtime *>(arena.region_ptr(layout.off_runtime));
    memset(rt, 0, sizeof(*rt));

    auto *sm_wrap = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(layout.off_sm_handle));
    memset(sm_wrap, 0, sizeof(*sm_wrap));

    // rt->ops is filled by the AICPU at boot.
    rt->mode = mode;
    rt->gm_heap = gm_heap_dev_base;
    rt->gm_heap_size = heap_size > 0 ? heap_size * PTO2_MAX_RING_DEPTH : 0;
    rt->gm_heap_owned = false;
    rt->total_cycles = 0;

    if (!rt->orchestrator.init_data_from_layout(
            layout.orch, arena, sm_dev_base, gm_heap_dev_base, heap_size, layout.task_window_size
        )) {
        return nullptr;
    }
    if (!rt->scheduler.init_data_from_layout(layout.sched, arena, sm_dev_base)) {
        return nullptr;
    }

    auto *mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.off_mailbox));
    memset(mailbox, 0, sizeof(*mailbox));

    return rt;
}

void runtime_wire_arena_pointers(DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2Runtime *rt) {
    rt->sm_handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(layout.off_sm_handle));
    rt->aicore_mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.off_mailbox));
    rt->orchestrator.wire_arena_pointers(layout.orch, arena, &rt->scheduler);
    rt->scheduler.wire_arena_pointers(layout.sched, arena);
}

void runtime_destroy(PTO2Runtime *rt, DeviceArena & /*arena*/) {
    // Arena buffer is pooled across runs by DeviceRunner — never freed here.
    if (!rt) return;
    rt->scheduler.destroy();
    rt->orchestrator.destroy();
    rt->aicore_mailbox = nullptr;
    rt->sm_handle = nullptr;
}
