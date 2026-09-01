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
 * build (AICPU runs wire_arena_pointers + reset_for_reuse after attach). The
 * device-only parts of runtime_core.cpp / orchestrator.cpp / scheduler.cpp
 * (ops table, scope/submit/dispatch business logic, profiling) stay in their
 * original files and the aicpu build only.
 */

#include <stdlib.h>
#include <string.h>

#include <limits>

#include "orchestrator.h"
#include "runtime_core.h"
#include "ring_buffer.h"
#include "shared_memory.h"
#include "tensormap.h"
#include "scheduler/scheduler.h"

static bool sum_ring_heap_sizes(const uint64_t heap_sizes[CHIP_MAX_RING_DEPTH], uint64_t *total) {
    uint64_t sum = 0;
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        if (heap_sizes[r] > std::numeric_limits<uint64_t>::max() - sum) {
            LOG_ERROR("Total ring heap size overflows uint64_t");
            return false;
        }
        sum += heap_sizes[r];
    }
    *total = sum;
    return true;
}

// =============================================================================
// Ready queue
// =============================================================================

size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity) {
    // Align the slots[] base to a full cache line so MPMC CAS traffic on the
    // first slot cannot false-share with whatever region sits in front of us
    // (e.g. orchestrator tensormap heads written by the orch thread).
    return arena.reserve(capacity * sizeof(ChipReadyQueueSlot), CHIP_ALIGN_SIZE);
}

bool ready_queue_init_data_from_layout(ChipReadyQueue *queue, DeviceArena &arena, size_t slots_off, uint64_t capacity) {
    // Address the slots region for data writes without storing the pointer in
    // queue->slots — that field is set by ready_queue_wire_arena_pointers.
    auto *slots_arena = static_cast<ChipReadyQueueSlot *>(arena.region_ptr(slots_off));
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

bool SchedulerState::RingSchedState::init_data_from_layout(void *sm_dev_base, int32_t ring_id) {
    // ring stores the device address of the SM ring header — pure offset
    // arithmetic, no SM load.
    ring = sm_layout::ring_header_addr(sm_dev_base, ring_id);
    last_task_alive = 0;
    last_published_to_sm = 0;
    publication_batching_enabled = false;
    advance_lock.store(0, std::memory_order_relaxed);
#if SIMPLER_DFX
    dep_pool_snapshot_tail.store(1, std::memory_order_relaxed);
    dep_pool_snapshot_top.store(1, std::memory_order_relaxed);
#endif

    // Per-slot SM-side initialization (bind_ring + reset_for_reuse +
    // fanin_count/active_mask zero) lives in SharedMemoryHandle::
    // init_header_per_ring so the AICPU performs it during SM reset; host
    // prebuilt-arena init skips SM access here.

    return true;
}

void SchedulerState::RingSchedState::reset_for_reuse(
    void *sm_dev_base, int32_t ring_id, std::atomic<int32_t> *orch_err
) {
    ring = sm_layout::ring_header_addr(sm_dev_base, ring_id);
    last_task_alive = 0;
    last_published_to_sm = 0;
    publication_batching_enabled = false;
    advance_lock.store(0, std::memory_order_relaxed);
    dep_pool.reset_for_reuse(orch_err);
#if SIMPLER_DFX
    dep_pool_snapshot_tail.store(1, std::memory_order_relaxed);
    dep_pool_snapshot_top.store(1, std::memory_order_relaxed);
#endif
}

void SchedulerState::RingSchedState::destroy() {
    publication_batching_enabled = false;
    ring = nullptr;
}

SchedulerLayout SchedulerState::reserve_layout(DeviceArena &arena, int32_t dep_pool_capacity) {
    int32_t dep_pool_capacities[CHIP_MAX_RING_DEPTH];
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        dep_pool_capacities[r] = dep_pool_capacity;
    }
    return reserve_layout(arena, dep_pool_capacities);
}

SchedulerLayout
SchedulerState::reserve_layout(DeviceArena &arena, const int32_t dep_pool_capacities[CHIP_MAX_RING_DEPTH]) {
    SchedulerLayout layout{};
    layout.ready_queue_capacity = CHIP_READY_QUEUE_SIZE;
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        layout.dep_pool_capacities[r] = dep_pool_capacities[r];
    }

    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        layout.off_ready_queue_slots[i] = ready_queue_reserve_layout(arena, CHIP_READY_QUEUE_SIZE);
    }
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        layout.off_ready_sync_queue_slots[i] = ready_queue_reserve_layout(arena, CHIP_READY_QUEUE_SIZE);
    }
    layout.off_dummy_ready_queue_slots = ready_queue_reserve_layout(arena, CHIP_READY_QUEUE_SIZE);
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        layout.off_early_dispatch_queue_slots[i] = ready_queue_reserve_layout(arena, CHIP_EARLY_DISPATCH_QUEUE_SIZE);
    }
    layout.off_early_sync_start_queue_slots = ready_queue_reserve_layout(arena, CHIP_EARLY_DISPATCH_QUEUE_SIZE);
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        // Force a cache-line base so Orch-side dep_pool writes do not invalidate
        // adjacent multi-threaded regions like ready_queue.slots.
        layout.off_dep_pool_entries[r] =
            arena.reserve(static_cast<size_t>(dep_pool_capacities[r]) * sizeof(DepListEntry), CHIP_ALIGN_SIZE);
    }
    return layout;
}

bool SchedulerState::init_data_from_layout(const SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base) {
    SchedulerState *sched = this;
    sched->sm_header = reinterpret_cast<SharedMemoryHeader *>(sm_dev_base);
    sched->advance_pending_mask.store(0, std::memory_order_relaxed);
    sched->publication_request_mask.store(0, std::memory_order_relaxed);
    sched->publication_ack_mask.store(0, std::memory_order_relaxed);
#if SIMPLER_SCHED_PROFILING
    sched->tasks_completed.store(0, std::memory_order_relaxed);
    sched->tasks_consumed.store(0, std::memory_order_relaxed);
#endif

    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        if (!sched->ring_sched_states[r].init_data_from_layout(sm_dev_base, r)) {
            return false;
        }
    }

    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        if (!ready_queue_init_data_from_layout(
                &sched->ready_queues[i], arena, layout.off_ready_queue_slots[i], layout.ready_queue_capacity
            )) {
            return false;
        }
    }
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        if (!ready_queue_init_data_from_layout(
                &sched->ready_sync_queues[i], arena, layout.off_ready_sync_queue_slots[i], layout.ready_queue_capacity
            )) {
            return false;
        }
    }
    if (!ready_queue_init_data_from_layout(
            &sched->dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots, layout.ready_queue_capacity
        )) {
        return false;
    }
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        if (!ready_queue_init_data_from_layout(
                &sched->early_dispatch_queues[i], arena, layout.off_early_dispatch_queue_slots[i],
                CHIP_EARLY_DISPATCH_QUEUE_SIZE
            )) {
            return false;
        }
    }
    if (!ready_queue_init_data_from_layout(
            &sched->early_sync_start_queue, arena, layout.off_early_sync_start_queue_slots,
            CHIP_EARLY_DISPATCH_QUEUE_SIZE
        )) {
        return false;
    }

    auto *orch_err = sm_layout::orch_error_code_addr(sm_dev_base);
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        auto *dep_entries = static_cast<DepListEntry *>(arena.region_ptr(layout.off_dep_pool_entries[r]));
        memset(dep_entries, 0, static_cast<size_t>(layout.dep_pool_capacities[r]) * sizeof(DepListEntry));
        sched->ring_sched_states[r].dep_pool.init(dep_entries, layout.dep_pool_capacities[r], orch_err);
        sched->ring_sched_states[r].dep_pool.set_reclaim_publication_request(
            &sched->publication_request_mask, &sched->publication_ack_mask, static_cast<uint8_t>(r)
        );
    }

    return true;
}

void SchedulerState::reset_for_reuse(const SchedulerLayout &layout, void *sm_dev_base) {
    SchedulerState *sched = this;
    sched->sm_header = reinterpret_cast<SharedMemoryHeader *>(sm_dev_base);
    sched->advance_pending_mask.store(0, std::memory_order_relaxed);
    sched->publication_request_mask.store(0, std::memory_order_relaxed);
    sched->publication_ack_mask.store(0, std::memory_order_relaxed);
#if SIMPLER_SCHED_PROFILING
    sched->tasks_completed.store(0, std::memory_order_relaxed);
    sched->tasks_consumed.store(0, std::memory_order_relaxed);
#endif

    auto *orch_err = sm_layout::orch_error_code_addr(sm_dev_base);
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        sched->ring_sched_states[r].reset_for_reuse(sm_dev_base, r, orch_err);
    }

    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        sched->ready_queues[i].reset_for_reuse();
    }
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        sched->ready_sync_queues[i].reset_for_reuse();
    }
    sched->dummy_ready_queue.reset_for_reuse();
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        sched->early_dispatch_queues[i].reset_for_reuse();
    }
    sched->early_sync_start_queue.reset_for_reuse();

    sched->async_wait_list.reset_for_reuse();
    (void)layout;
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
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        ready_queue_wire_arena_pointers(
            &sched->early_dispatch_queues[i], arena, layout.off_early_dispatch_queue_slots[i]
        );
    }
    ready_queue_wire_arena_pointers(&sched->early_sync_start_queue, arena, layout.off_early_sync_start_queue_slots);
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        auto &dep_pool = sched->ring_sched_states[r].dep_pool;
        dep_pool.base = static_cast<DepListEntry *>(arena.region_ptr(layout.off_dep_pool_entries[r]));
        dep_pool.set_reclaim_publication_request(
            &sched->publication_request_mask, &sched->publication_ack_mask, static_cast<uint8_t>(r)
        );
    }
}

void SchedulerState::destroy() {
    SchedulerState *sched = this;
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        sched->ring_sched_states[r].destroy();
        sched->ring_sched_states[r].dep_pool.base = nullptr;
    }
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        ready_queue_destroy(&sched->ready_queues[i]);
    }
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        ready_queue_destroy(&sched->ready_sync_queues[i]);
    }
    ready_queue_destroy(&sched->dummy_ready_queue);
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        ready_queue_destroy(&sched->early_dispatch_queues[i]);
    }
    ready_queue_destroy(&sched->early_sync_start_queue);
}

// =============================================================================
// Orchestrator
// =============================================================================

OrchestratorLayout OrchestratorState::reserve_layout(
    DeviceArena &arena, const int32_t task_window_sizes[CHIP_MAX_RING_DEPTH], int32_t dep_pool_capacity
) {
    int32_t dep_pool_capacities[CHIP_MAX_RING_DEPTH];
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        dep_pool_capacities[r] = dep_pool_capacity;
    }
    return reserve_layout(arena, task_window_sizes, dep_pool_capacities);
}

OrchestratorLayout OrchestratorState::reserve_layout(
    DeviceArena &arena, const int32_t task_window_sizes[CHIP_MAX_RING_DEPTH],
    const int32_t dep_pool_capacities[CHIP_MAX_RING_DEPTH]
) {
    OrchestratorLayout layout{};
    // scope_tasks holds every task in the open scope across all rings, so its cap
    // is the real in-flight budget = sum of the (runtime) per-ring windows. Using
    // the compile-time CHIP_SCOPE_TASKS_CAP instead under-sized the buffer when
    // ring_task_window was enlarged past the default (premature SCOPE_TASKS_OVERFLOW)
    // and over-allocated it when shrunk. See issue #1188.
    //
    // Accumulate in int64: each window is validated <= INT32_MAX individually, but
    // the sum of CHIP_MAX_RING_DEPTH windows can exceed it — a bare int32 sum would
    // wrap to a negative/undersized cap. Bound the result before narrowing.
    int64_t scope_tasks_cap = 0;
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        always_assert(task_window_sizes[r] > 0);
        scope_tasks_cap += task_window_sizes[r];
    }
    always_assert(scope_tasks_cap <= std::numeric_limits<int32_t>::max());
    layout.scope_tasks_cap = static_cast<int32_t>(scope_tasks_cap);
    layout.scope_stack_capacity = CHIP_MAX_SCOPE_DEPTH;
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        layout.dep_pool_capacities[r] = dep_pool_capacities[r];
    }

    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        const size_t fanin_pool_bytes =
            CHIP_ALIGN_UP(static_cast<size_t>(dep_pool_capacities[r]) * sizeof(FaninSpillEntry), CHIP_ALIGN_SIZE);
        layout.off_fanin_pool[r] = arena.reserve(fanin_pool_bytes, CHIP_ALIGN_SIZE);

        always_assert(task_window_sizes[r] > 0 && (task_window_sizes[r] & (task_window_sizes[r] - 1)) == 0);
        const size_t seen_epoch_bytes =
            CHIP_ALIGN_UP(static_cast<size_t>(task_window_sizes[r]) * sizeof(uint32_t), CHIP_ALIGN_SIZE);
        layout.off_fanin_seen_epoch[r] = arena.reserve(seen_epoch_bytes, CHIP_ALIGN_SIZE);

        const size_t wait_reach_bytes =
            CHIP_ALIGN_UP(static_cast<size_t>(task_window_sizes[r]) * sizeof(WaitReachEntry), CHIP_ALIGN_SIZE);
        layout.off_wait_reach[r] = arena.reserve(wait_reach_bytes, CHIP_ALIGN_SIZE);
    }
    layout.off_scope_tasks =
        arena.reserve(static_cast<size_t>(layout.scope_tasks_cap) * sizeof(uintptr_t), alignof(ChipTaskSlotState *));
    layout.off_scope_begins =
        arena.reserve(static_cast<size_t>(layout.scope_stack_capacity) * sizeof(int32_t), alignof(int32_t));
    layout.tensor_map = ChipTensorMap::reserve_layout_default(arena, task_window_sizes);
    return layout;
}

bool OrchestratorState::init_data_from_layout(
    const OrchestratorLayout &layout, DeviceArena &arena, void *sm_dev_base, void *gm_heap, uint64_t heap_size,
    uint64_t task_window_size
) {
    uint64_t heap_sizes[CHIP_MAX_RING_DEPTH];
    uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH];
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        heap_sizes[r] = heap_size;
        task_window_sizes[r] = task_window_size;
    }
    return init_data_from_layout(layout, arena, sm_dev_base, gm_heap, heap_sizes, task_window_sizes);
}

bool OrchestratorState::init_data_from_layout(
    const OrchestratorLayout &layout, DeviceArena &arena, void *sm_dev_base, void *gm_heap,
    const uint64_t heap_sizes[CHIP_MAX_RING_DEPTH], const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH]
) {
    auto *orch = this;
    *orch = OrchestratorState{};

    orch->sm_header = reinterpret_cast<SharedMemoryHeader *>(sm_dev_base);
    orch->gm_heap_base = gm_heap;
    uint64_t total_heap_size = 0;
    if (!sum_ring_heap_sizes(heap_sizes, &total_heap_size)) {
        return false;
    }
    orch->gm_heap_size = total_heap_size;
    orch->fatal = false;

    auto *orch_err = sm_layout::orch_error_code_addr(sm_dev_base);
    uint64_t heap_offset = 0;
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        void *ring_heap_base = reinterpret_cast<char *>(gm_heap) + heap_offset;
        auto *task_descs_dev = sm_layout::ring_task_descriptors_addr(sm_dev_base, task_window_sizes, r);
        auto *slot_states_dev = sm_layout::ring_slot_states_addr(sm_dev_base, task_window_sizes, r);
        auto *cur_idx_dev = sm_layout::ring_current_task_index_addr(sm_dev_base, r);
        auto *last_alive_dev = sm_layout::ring_last_task_alive_addr(sm_dev_base, r);

        orch->rings[r].task_allocator.init(
            task_descs_dev, static_cast<int32_t>(task_window_sizes[r]), cur_idx_dev, last_alive_dev, ring_heap_base,
            heap_sizes[r], orch_err, slot_states_dev, 0, static_cast<uint8_t>(r)
        );
        heap_offset += heap_sizes[r];

        const size_t fanin_pool_bytes = CHIP_ALIGN_UP(
            static_cast<size_t>(layout.dep_pool_capacities[r]) * sizeof(FaninSpillEntry), CHIP_ALIGN_SIZE
        );
        auto *fanin_entries = static_cast<FaninSpillEntry *>(arena.region_ptr(layout.off_fanin_pool[r]));
        memset(fanin_entries, 0, fanin_pool_bytes);
        orch->rings[r].fanin_pool.init(fanin_entries, layout.dep_pool_capacities[r], orch_err);

        const size_t seen_epoch_bytes = CHIP_ALIGN_UP(
            static_cast<size_t>(layout.tensor_map.task_window_sizes[r]) * sizeof(uint32_t), CHIP_ALIGN_SIZE
        );
        auto *seen_epoch = static_cast<uint32_t *>(arena.region_ptr(layout.off_fanin_seen_epoch[r]));
        memset(seen_epoch, 0, seen_epoch_bytes);
        orch->fanin_seen_epoch[r] = seen_epoch;

        const size_t wait_reach_bytes = CHIP_ALIGN_UP(
            static_cast<size_t>(layout.tensor_map.task_window_sizes[r]) * sizeof(WaitReachEntry), CHIP_ALIGN_SIZE
        );
        auto *wait_reach = static_cast<WaitReachEntry *>(arena.region_ptr(layout.off_wait_reach[r]));
        memset(wait_reach, 0, wait_reach_bytes);
        orch->wait_reach[r] = wait_reach;
    }

    if (!orch->tensor_map.init_data_from_layout(layout.tensor_map, arena)) {
        return false;
    }

    orch->scope_tasks_size = 0;
    orch->scope_tasks_capacity = layout.scope_tasks_cap;
    orch->scope_stack_top = -1;
    orch->scope_stack_capacity = layout.scope_stack_capacity;
    orch->manual_begin_depth = CHIP_MAX_SCOPE_DEPTH;

    return true;
}

bool OrchestratorState::reset_for_reuse(
    const OrchestratorLayout &layout, void *sm_dev_base, void *gm_heap, const uint64_t heap_sizes[CHIP_MAX_RING_DEPTH],
    const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH]
) {
    auto *orch = this;
    orch->sm_header = reinterpret_cast<SharedMemoryHeader *>(sm_dev_base);
    orch->gm_heap_base = gm_heap;
    uint64_t total_heap_size = 0;
    if (!sum_ring_heap_sizes(heap_sizes, &total_heap_size)) {
        return false;
    }
    orch->gm_heap_size = total_heap_size;
    orch->fatal = false;
    orch->inline_completed_tasks = 0;
    // The wait_reach entries are not cleared here: every slot publishes its
    // bitmap before any submit can read it as a producer, so entries left from
    // a previous run are unreachable.
    orch->submit_seq = 0;

    uint32_t next_epoch = orch->fanin_seen_current_epoch + 1;
    if (next_epoch == 0) {
        next_epoch = 1;
        for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
            memset(
                orch->fanin_seen_epoch[r], 0,
                static_cast<size_t>(layout.tensor_map.task_window_sizes[r]) * sizeof(uint32_t)
            );
        }
    }
    orch->fanin_seen_current_epoch = next_epoch;

    auto *orch_err = sm_layout::orch_error_code_addr(sm_dev_base);
    uint64_t heap_offset = 0;
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        void *ring_heap_base = reinterpret_cast<char *>(gm_heap) + heap_offset;
        auto *task_descs_dev = sm_layout::ring_task_descriptors_addr(sm_dev_base, task_window_sizes, r);
        auto *slot_states_dev = sm_layout::ring_slot_states_addr(sm_dev_base, task_window_sizes, r);
        auto *cur_idx_dev = sm_layout::ring_current_task_index_addr(sm_dev_base, r);
        auto *last_alive_dev = sm_layout::ring_last_task_alive_addr(sm_dev_base, r);

        orch->rings[r].task_allocator.init(
            task_descs_dev, static_cast<int32_t>(task_window_sizes[r]), cur_idx_dev, last_alive_dev, ring_heap_base,
            heap_sizes[r], orch_err, slot_states_dev, 0, static_cast<uint8_t>(r)
        );
        if (orch->scheduler != nullptr) {
            orch->rings[r].task_allocator.set_reclaim_publication_request(
                &orch->scheduler->publication_request_mask, &orch->scheduler->publication_ack_mask
            );
        }
        heap_offset += heap_sizes[r];
        orch->rings[r].fanin_pool.reset_for_reuse(orch_err);
    }

    orch->tensor_map.reset_for_reuse(layout.tensor_map);
    orch->scope_tasks_size = 0;
    orch->scope_tasks_capacity = layout.scope_tasks_cap;
    orch->scope_stack_top = -1;
    orch->scope_stack_capacity = layout.scope_stack_capacity;
    orch->manual_begin_depth = CHIP_MAX_SCOPE_DEPTH;
    orch->total_cluster_count = 0;
    orch->total_aiv_count = 0;
#if SIMPLER_DFX
    orch->tasks_submitted = 0;
    orch->buffers_allocated = 0;
    orch->bytes_allocated = 0;
#endif
    return true;
}

void OrchestratorState::wire_arena_pointers(
    const OrchestratorLayout &layout, DeviceArena &arena, SchedulerState *scheduler_arg
) {
    auto *orch = this;
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        orch->rings[r].fanin_pool.base = static_cast<FaninSpillEntry *>(arena.region_ptr(layout.off_fanin_pool[r]));
        orch->fanin_seen_epoch[r] = static_cast<uint32_t *>(arena.region_ptr(layout.off_fanin_seen_epoch[r]));
        orch->wait_reach[r] = static_cast<WaitReachEntry *>(arena.region_ptr(layout.off_wait_reach[r]));
    }
    orch->tensor_map.wire_arena_pointers(layout.tensor_map, arena);
    orch->scope_tasks = static_cast<ChipTaskSlotState **>(arena.region_ptr(layout.off_scope_tasks));
    orch->scope_begins = static_cast<int32_t *>(arena.region_ptr(layout.off_scope_begins));
    orch->set_scheduler(scheduler_arg);
}

void OrchestratorState::destroy() {
    auto *orch = this;
    orch->tensor_map.destroy();
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        orch->rings[r].fanin_pool.base = nullptr;
        orch->fanin_seen_epoch[r] = nullptr;
        orch->wait_reach[r] = nullptr;
    }
    orch->scope_tasks = nullptr;
    orch->scope_begins = nullptr;
}

void OrchestratorState::set_scheduler(SchedulerState *scheduler_arg) {
    scheduler = scheduler_arg;
    if (scheduler == nullptr) return;
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        rings[r].task_allocator.set_reclaim_publication_request(
            &scheduler->publication_request_mask, &scheduler->publication_ack_mask
        );
        rings[r].fanin_pool.set_reclaim_publication_request(
            &scheduler->publication_request_mask, &scheduler->publication_ack_mask, static_cast<uint8_t>(r)
        );
    }
}

// =============================================================================
// Top-level runtime arena
// =============================================================================

RuntimeArenaLayout runtime_reserve_layout(DeviceArena &arena, uint64_t task_window_size, int32_t dep_pool_capacity) {
    uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH];
    uint64_t heap_sizes[CHIP_MAX_RING_DEPTH];
    int32_t dep_pool_capacities[CHIP_MAX_RING_DEPTH];
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = task_window_size;
        heap_sizes[r] = 0;
        dep_pool_capacities[r] = dep_pool_capacity;
    }
    return runtime_reserve_layout(arena, task_window_sizes, heap_sizes, dep_pool_capacities);
}

RuntimeArenaLayout runtime_reserve_layout(
    DeviceArena &arena, const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH],
    const uint64_t heap_sizes[CHIP_MAX_RING_DEPTH], const int32_t dep_pool_capacities[CHIP_MAX_RING_DEPTH]
) {
    RuntimeArenaLayout layout{};

    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        layout.sizing.task_window_sizes[r] = task_window_sizes[r];
        layout.sizing.heap_sizes[r] = heap_sizes[r];
        layout.sizing.dep_pool_capacities[r] = dep_pool_capacities[r];
    }

    layout.offsets.off_sm_handle = arena.reserve(sizeof(SharedMemoryHandle), alignof(SharedMemoryHandle));
    int32_t task_window_sizes_i32[CHIP_MAX_RING_DEPTH];
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        task_window_sizes_i32[r] = static_cast<int32_t>(task_window_sizes[r]);
    }
    layout.offsets.orch = OrchestratorState::reserve_layout(arena, task_window_sizes_i32, dep_pool_capacities);
    layout.offsets.sched = SchedulerState::reserve_layout(arena, dep_pool_capacities);
    layout.offsets.off_runtime = arena.reserve(sizeof(RuntimeContext), CHIP_ALIGN_SIZE);
    layout.offsets.off_mailbox = arena.reserve(sizeof(AICoreCompletionMailbox), alignof(AICoreCompletionMailbox));

    layout.offsets.arena_size = arena.total_size();
    return layout;
}

RuntimeContext *runtime_init_data_from_layout(
    DeviceArena &arena, const RuntimeArenaLayout &layout, RuntimeMode mode, void *sm_dev_base, uint64_t /*sm_size*/,
    void *gm_heap_dev_base, uint64_t heap_size
) {
    uint64_t heap_sizes[CHIP_MAX_RING_DEPTH];
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        heap_sizes[r] = heap_size;
    }
    return runtime_init_data_from_layout(arena, layout, mode, sm_dev_base, 0, gm_heap_dev_base, heap_sizes);
}

RuntimeContext *runtime_init_data_from_layout(
    DeviceArena &arena, const RuntimeArenaLayout &layout, RuntimeMode mode, void *sm_dev_base, uint64_t /*sm_size*/,
    void *gm_heap_dev_base, const uint64_t heap_sizes[CHIP_MAX_RING_DEPTH]
) {
    RuntimeContext *rt = static_cast<RuntimeContext *>(arena.region_ptr(layout.offsets.off_runtime));
    memset(rt, 0, sizeof(*rt));

    auto *sm_wrap = static_cast<SharedMemoryHandle *>(arena.region_ptr(layout.offsets.off_sm_handle));
    memset(sm_wrap, 0, sizeof(*sm_wrap));

    // rt->ops is filled by the AICPU at boot.
    rt->mode = mode;
    rt->gm_heap = gm_heap_dev_base;
    uint64_t total_heap_size = 0;
    if (!sum_ring_heap_sizes(heap_sizes, &total_heap_size)) {
        return nullptr;
    }
    rt->gm_heap_size = total_heap_size;
    rt->gm_heap_owned = false;
    rt->total_cycles = 0;

    if (!rt->orchestrator.init_data_from_layout(
            layout.offsets.orch, arena, sm_dev_base, gm_heap_dev_base, heap_sizes, layout.sizing.task_window_sizes
        )) {
        return nullptr;
    }
    if (!rt->scheduler.init_data_from_layout(layout.offsets.sched, arena, sm_dev_base)) {
        return nullptr;
    }

    auto *mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.offsets.off_mailbox));
    memset(mailbox, 0, sizeof(*mailbox));

    return rt;
}

static bool reclaim_publication_wiring_is_complete(const RuntimeContext *rt) {
    if (rt == nullptr || rt->orchestrator.scheduler != &rt->scheduler) return false;

    const auto *request_mask = &rt->scheduler.publication_request_mask;
    const auto *ack_mask = &rt->scheduler.publication_ack_mask;
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        uint8_t ring_id = static_cast<uint8_t>(r);
        if (!rt->orchestrator.rings[r].task_allocator.reclaim_publication_is_wired_to(
                request_mask, ack_mask, ring_id
            ) ||
            !rt->orchestrator.rings[r].fanin_pool.reclaim_publication_is_wired_to(request_mask, ack_mask, ring_id) ||
            !rt->scheduler.ring_sched_states[r].dep_pool.reclaim_publication_is_wired_to(
                request_mask, ack_mask, ring_id
            )) {
            return false;
        }
    }
    return true;
}

static void enable_publication_batching_if_safe(RuntimeContext *rt) {
    rt->scheduler.set_publication_batching_enabled(reclaim_publication_wiring_is_complete(rt));
}

void runtime_wire_arena_pointers(DeviceArena &arena, const RuntimeArenaLayout &layout, RuntimeContext *rt) {
    rt->scheduler.set_publication_batching_enabled(false);
    rt->sm_handle = static_cast<SharedMemoryHandle *>(arena.region_ptr(layout.offsets.off_sm_handle));
    rt->aicore_mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.offsets.off_mailbox));
    rt->orchestrator.wire_arena_pointers(layout.offsets.orch, arena, &rt->scheduler);
    rt->scheduler.wire_arena_pointers(layout.offsets.sched, arena);
    enable_publication_batching_if_safe(rt);
}

bool runtime_reset_for_reuse(DeviceArena &arena, const RuntimeArenaLayout &layout, RuntimeContext *rt) {
    (void)arena;
    if (rt == nullptr) {
        return false;
    }

    rt->scheduler.set_publication_batching_enabled(false);

    rt->pending_scope_mode = ScopeMode::AUTO;
    rt->total_cycles = 0;
    rt->gm_heap_owned = false;

    uint64_t total_heap_size = 0;
    if (!sum_ring_heap_sizes(layout.sizing.heap_sizes, &total_heap_size)) {
        return false;
    }
    rt->gm_heap_size = total_heap_size;

    if (!rt->orchestrator.reset_for_reuse(
            layout.offsets.orch, rt->sm_handle->sm_base, rt->gm_heap, layout.sizing.heap_sizes,
            layout.sizing.task_window_sizes
        )) {
        return false;
    }
    rt->scheduler.reset_for_reuse(layout.offsets.sched, rt->sm_handle->sm_base);
    enable_publication_batching_if_safe(rt);
    return true;
}

void runtime_destroy(RuntimeContext *rt, DeviceArena & /*arena*/) {
    // Arena buffer is pooled across runs by DeviceRunner — never freed here.
    if (!rt) return;
    rt->scheduler.destroy();
    rt->orchestrator.destroy();
    rt->aicore_mailbox = nullptr;
    rt->sm_handle = nullptr;
}
