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
 * PTO Runtime2 - cold-path layout/init/wire/reset for the orchestrator and
 * scheduler. Compiled into both the host and AICPU runtimes (runtime/shared),
 * so host-side arena setup and AICPU-side execution share one definition.
 */

#include <limits>

#include "pto_orchestrator.h"
#include "pto_runtime2.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "scheduler/pto_scheduler.h"

PTO2OrchestratorLayout PTO2OrchestratorState::reserve_layout(
    DeviceArena &arena, const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH], int32_t dep_pool_capacity
) {
    PTO2OrchestratorLayout layout{};
    layout.dep_pool_capacity = dep_pool_capacity;
    // scope_tasks holds every task in the open scope across all rings, so its cap
    // is the real in-flight budget = sum of the (runtime) per-ring windows.
    // Accumulate in int64; each window is validated <= INT32_MAX individually but
    // their sum can exceed it. See upstream #1192.
    int64_t scope_tasks_cap = 0;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        always_assert(task_window_sizes[r] > 0);
        scope_tasks_cap += task_window_sizes[r];
    }
    always_assert(scope_tasks_cap <= std::numeric_limits<int32_t>::max());
    layout.scope_tasks_cap = static_cast<int32_t>(scope_tasks_cap);
    layout.scope_stack_capacity = PTO2_MAX_SCOPE_DEPTH;

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
    }

    if (!orch->tensor_map.init_data_from_layout(layout.tensor_map, arena)) return false;

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
    orch->tensor_map.wire_arena_pointers(layout.tensor_map, arena);
    orch->scope_tasks = static_cast<PTO2TaskSlotState **>(arena.region_ptr(layout.off_scope_tasks));
    orch->scope_begins = static_cast<int32_t *>(arena.region_ptr(layout.off_scope_begins));
    orch->scheduler = scheduler_arg;
}

// Surgical reset for the arena-reuse path (#1234). Only touches state that
// mutates across runs — leaves the arena-internal pointers wired by
// wire_arena_pointers alone, and skips the O(pool_size + num_buckets)
// tensor_map re-init in favour of an epoch bump (bucket_epochs and
// task_entry_head_epochs are compared against current_epoch on every
// lookup; a bump invalidates all stale entries in O(1)).
void PTO2OrchestratorState::reset_for_reuse() {
    auto *orch = this;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        orch->rings[r].task_allocator.reset_for_reuse();
    }
    orch->tensor_map.reset_for_reuse();
    orch->scope_tasks_size = 0;
    orch->scope_stack_top = -1;
    orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
    orch->fatal = false;
    orch->inline_completed_tasks = 0;
    orch->fanin_seen_current_epoch++;
    if (orch->fanin_seen_current_epoch == 0) orch->fanin_seen_current_epoch = 1;
}

// Forget pointers; arena owns the backing buffers.
void PTO2OrchestratorState::destroy() {
    auto *orch = this;
    orch->tensor_map.destroy();
    orch->scope_tasks = nullptr;
    orch->scope_begins = nullptr;
}

void PTO2OrchestratorState::set_scheduler(PTO2SchedulerState *scheduler) { this->scheduler = scheduler; }
size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity) {
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

bool PTO2SchedulerState::RingSchedState::init_data_from_layout(void *sm_dev_base, int32_t ring_id) {
    ring = pto2_sm_layout::ring_header_addr(sm_dev_base, ring_id);
    last_task_alive = 0;
    advance_lock.store(0, std::memory_order_relaxed);
    return true;
}

void PTO2SchedulerState::RingSchedState::destroy() { ring = nullptr; }

PTO2SchedulerLayout PTO2SchedulerState::reserve_layout(DeviceArena &arena, int32_t) {
    PTO2SchedulerLayout layout{};
    layout.ready_queue_capacity = PTO2_READY_QUEUE_SIZE;
    layout.spsc_capacity = PTO2_WRIRING_QUEUE_SIZE;

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++)
        layout.off_ready_queue_slots[i] = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    layout.off_dummy_ready_queue_slots = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    layout.off_pending_spsc_buffer = PTO2SpscQueue::reserve_layout(arena, PTO2_WRIRING_QUEUE_SIZE);
    return layout;
}

bool PTO2SchedulerState::init_data_from_layout(
    const PTO2SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base
) {
    PTO2SchedulerState *sched = this;
    sched->sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        if (!sched->ring_sched_states[r].init_data_from_layout(sm_dev_base, r)) return false;

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++)
        if (!ready_queue_init_data_from_layout(
                &sched->ready_queues[i], arena, layout.off_ready_queue_slots[i], layout.ready_queue_capacity
            ))
            return false;
    if (!ready_queue_init_data_from_layout(
            &sched->dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots, layout.ready_queue_capacity
        ))
        return false;

    if (!sched->wiring.queue.init_data_from_layout(arena, layout.off_pending_spsc_buffer, layout.spsc_capacity))
        return false;

    sched->wiring.backoff_counter = 0;

    return true;
}

void PTO2SchedulerState::wire_arena_pointers(const PTO2SchedulerLayout &layout, DeviceArena &arena) {
    PTO2SchedulerState *sched = this;
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++)
        ready_queue_wire_arena_pointers(&sched->ready_queues[i], arena, layout.off_ready_queue_slots[i]);
    ready_queue_wire_arena_pointers(&sched->dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots);
    sched->wiring.queue.wire_arena_pointers(arena, layout.off_pending_spsc_buffer);
}

void PTO2SchedulerState::destroy() {
    PTO2SchedulerState *sched = this;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        sched->ring_sched_states[r].destroy();
    sched->wiring.queue.destroy();
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++)
        ready_queue_destroy(&sched->ready_queues[i]);
    ready_queue_destroy(&sched->dummy_ready_queue);
}

void PTO2SchedulerState::reset_for_reuse(void *sm_dev_base) {
    PTO2SchedulerState *sched = this;
    sched->sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        sched->ring_sched_states[r].ring = pto2_sm_layout::ring_header_addr(sm_dev_base, r);
        sched->ring_sched_states[r].last_task_alive = 0;
        sched->ring_sched_states[r].advance_lock.store(0, std::memory_order_relaxed);
    }
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++)
        sched->ready_queues[i].reset_for_reuse();
    sched->dummy_ready_queue.reset_for_reuse();
    sched->wiring.queue.reset_for_reuse();
    sched->wiring.backoff_counter = 0;
    sched->wiring.orch_needs_drain.store(false, std::memory_order_relaxed);
    sched->async_wait_list.reset_for_reuse();
}
void runtime_wire_arena_pointers(DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2Runtime *rt) {
    rt->sm_handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(layout.offsets.off_sm_handle));
    rt->aicore_mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.offsets.off_mailbox));
    rt->orchestrator.wire_arena_pointers(layout.offsets.orch, arena, &rt->scheduler);
    rt->scheduler.wire_arena_pointers(layout.offsets.sched, arena);
}

bool runtime_reset_for_reuse(DeviceArena &, const PTO2RuntimeArenaLayout &, PTO2Runtime *rt) {
    if (rt == nullptr) return false;

    rt->pending_scope_mode = PTO2ScopeMode::AUTO;
    rt->total_cycles = 0;
    rt->gm_heap_owned = false;

    void *sm_dev_base = rt->sm_handle ? rt->sm_handle->sm_base : nullptr;
    if (sm_dev_base == nullptr) return false;

    rt->orchestrator.reset_for_reuse();
    rt->scheduler.reset_for_reuse(sm_dev_base);

    return true;
}
