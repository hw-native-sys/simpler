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
 * Host/AICPU shared runtime-arena layout, initialization, and pointer wiring.
 *
 * The host prebuilds the runtime image. The AICPU rewires it, resets it between
 * invocations, and uses two task/heap arenas for graph-level pipelining.
 */

#include <stdlib.h>
#include <string.h>

#include <new>

#include "pto_orchestrator.h"
#include "pto_runtime2.h"
#include "pto_ring_buffer.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "scheduler/pto_scheduler.h"

namespace {

void reset_graph_pipeline(PTO2Runtime *rt) {
    PTO2ReplayGraphPipelineState &pipeline = rt->graph_pipeline;
    pipeline.all_done.store(0, std::memory_order_relaxed);
    pipeline.published_task_count.store(0, std::memory_order_relaxed);
    pipeline.active_buffer = 0;
    pipeline.graph_count = 0;
    pipeline.current_graph_epoch = 0;
    for (int32_t i = 0; i < PTO2_REPLAY_GRAPH_BUFFER_COUNT; i++) {
        PTO2ReplayGraphBufferControl &buffer = pipeline.buffers[i];
        buffer.state.store(
            i == 0 ? PTO2ReplayGraphBufferState::BUILDING : PTO2ReplayGraphBufferState::FREE, std::memory_order_relaxed
        );
        buffer.exec_done.store(i == 0 ? 0 : 1, std::memory_order_relaxed);
        buffer.dep_closed.store(i == 0 ? 0 : 1, std::memory_order_relaxed);
        buffer.completed_count.store(0, std::memory_order_relaxed);
        buffer.buffer_id = i;
        buffer.graph_epoch = 0;
        buffer.task_begin = 0;
        buffer.task_count = 0;
    }
}

}  // namespace

// =============================================================================
// Ready queue
// =============================================================================

size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity) {
    return arena.reserve(capacity * sizeof(PTO2ReadyQueueSlot), PTO2_ALIGN_SIZE);
}

bool ready_queue_init_data_from_layout(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off, uint64_t capacity) {
    auto *slots = static_cast<PTO2ReadyQueueSlot *>(arena.region_ptr(slots_off));
    queue->capacity = capacity;
    queue->mask = capacity - 1;
    queue->enqueue_pos.store(0, std::memory_order_relaxed);
    queue->dequeue_pos.store(0, std::memory_order_relaxed);
    for (uint64_t i = 0; i < capacity; i++) {
        slots[i].sequence.store(static_cast<int64_t>(i), std::memory_order_relaxed);
        slots[i].slot_state = nullptr;
        slots[i].task_id_snapshot = 0;
    }
    return true;
}

void ready_queue_wire_arena_pointers(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off) {
    queue->slots = static_cast<PTO2ReadyQueueSlot *>(arena.region_ptr(slots_off));
}

void ready_queue_destroy(PTO2ReadyQueue *queue) { queue->slots = nullptr; }

// =============================================================================
// Scheduler
// =============================================================================

PTO2SchedulerLayout PTO2SchedulerState::reserve_layout(DeviceArena &arena) {
    PTO2SchedulerLayout layout{};
    layout.ready_queue_capacity = PTO2_READY_QUEUE_SIZE;
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        layout.off_ready_queue_slots[i] = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
        layout.off_ready_sync_queue_slots[i] = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
        layout.off_early_dispatch_queue_slots[i] = ready_queue_reserve_layout(arena, PTO2_EARLY_DISPATCH_QUEUE_SIZE);
    }
    layout.off_dummy_ready_queue_slots = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    layout.off_graph_ready_queue_slots = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    layout.off_graph_prepare_queue_slots = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    layout.off_early_sync_start_queue_slots = ready_queue_reserve_layout(arena, PTO2_EARLY_DISPATCH_QUEUE_SIZE);
    return layout;
}

bool PTO2SchedulerState::init_data_from_layout(
    const PTO2SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base
) {
    sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);
    graph_pipeline = nullptr;
#if SIMPLER_SCHED_PROFILING
    tasks_completed.store(0, std::memory_order_relaxed);
#endif

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        if (!ready_queue_init_data_from_layout(
                &ready_queues[i], arena, layout.off_ready_queue_slots[i], layout.ready_queue_capacity
            ) ||
            !ready_queue_init_data_from_layout(
                &ready_sync_queues[i], arena, layout.off_ready_sync_queue_slots[i], layout.ready_queue_capacity
            ) ||
            !ready_queue_init_data_from_layout(
                &early_dispatch_queues[i], arena, layout.off_early_dispatch_queue_slots[i],
                PTO2_EARLY_DISPATCH_QUEUE_SIZE
            )) {
            return false;
        }
    }
    if (!ready_queue_init_data_from_layout(
            &dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots, layout.ready_queue_capacity
        ) ||
        !ready_queue_init_data_from_layout(
            &graph_ready_queue, arena, layout.off_graph_ready_queue_slots, layout.ready_queue_capacity
        ) ||
        !ready_queue_init_data_from_layout(
            &graph_prepare_queue, arena, layout.off_graph_prepare_queue_slots, layout.ready_queue_capacity
        ) ||
        !ready_queue_init_data_from_layout(
            &early_sync_start_queue, arena, layout.off_early_sync_start_queue_slots, PTO2_EARLY_DISPATCH_QUEUE_SIZE
        )) {
        return false;
    }
    return true;
}

void PTO2SchedulerState::reset_for_reuse(const PTO2SchedulerLayout &layout, void *sm_dev_base) {
    sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);
    graph_pipeline = nullptr;
#if SIMPLER_SCHED_PROFILING
    tasks_completed.store(0, std::memory_order_relaxed);
#endif
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queues[i].reset_for_reuse();
        ready_sync_queues[i].reset_for_reuse();
        early_dispatch_queues[i].reset_for_reuse();
    }
    dummy_ready_queue.reset_for_reuse();
    graph_ready_queue.reset_for_reuse();
    graph_prepare_queue.reset_for_reuse();
    early_sync_start_queue.reset_for_reuse();
    async_wait_list.reset_for_reuse();
    (void)layout;
}

void PTO2SchedulerState::wire_arena_pointers(const PTO2SchedulerLayout &layout, DeviceArena &arena) {
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_wire_arena_pointers(&ready_queues[i], arena, layout.off_ready_queue_slots[i]);
        ready_queue_wire_arena_pointers(&ready_sync_queues[i], arena, layout.off_ready_sync_queue_slots[i]);
        ready_queue_wire_arena_pointers(&early_dispatch_queues[i], arena, layout.off_early_dispatch_queue_slots[i]);
    }
    ready_queue_wire_arena_pointers(&dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots);
    ready_queue_wire_arena_pointers(&graph_ready_queue, arena, layout.off_graph_ready_queue_slots);
    ready_queue_wire_arena_pointers(&graph_prepare_queue, arena, layout.off_graph_prepare_queue_slots);
    ready_queue_wire_arena_pointers(&early_sync_start_queue, arena, layout.off_early_sync_start_queue_slots);
}

void PTO2SchedulerState::destroy() {
    sm_header = nullptr;
    graph_pipeline = nullptr;
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_destroy(&ready_queues[i]);
        ready_queue_destroy(&ready_sync_queues[i]);
        ready_queue_destroy(&early_dispatch_queues[i]);
    }
    ready_queue_destroy(&dummy_ready_queue);
    ready_queue_destroy(&graph_ready_queue);
    ready_queue_destroy(&graph_prepare_queue);
    ready_queue_destroy(&early_sync_start_queue);
}

// =============================================================================
// Orchestrator
// =============================================================================

PTO2OrchestratorLayout
PTO2OrchestratorState::reserve_layout(DeviceArena &arena, int32_t task_window_size, int32_t dep_pool_capacity) {
    always_assert(task_window_size > 0);
    PTO2OrchestratorLayout layout{};
    layout.scope_stack_capacity = PTO2_MAX_SCOPE_DEPTH;
    layout.dep_pool_capacity = dep_pool_capacity;
    layout.off_dep_pool_entries =
        arena.reserve(static_cast<size_t>(dep_pool_capacity) * sizeof(PTO2DepListEntry), PTO2_ALIGN_SIZE);
    layout.off_fanin_seen_epoch =
        arena.reserve(static_cast<size_t>(task_window_size) * sizeof(uint32_t), PTO2_ALIGN_SIZE);
    layout.tensor_map = PTO2TensorMap::reserve_layout_default(arena);
    return layout;
}

bool PTO2OrchestratorState::init_data_from_layout(
    const PTO2OrchestratorLayout &layout, DeviceArena &arena, void *sm_dev_base, void *gm_heap, uint64_t heap_size,
    uint64_t task_window_size
) {
    new (this) PTO2OrchestratorState{};
    sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);
    gm_heap_base = gm_heap;
    gm_heap_size = heap_size;
    fatal = false;

    auto *orch_err = pto2_sm_layout::orch_error_code_addr(sm_dev_base);
    task_allocator.init(
        static_cast<int32_t>(task_window_size), pto2_sm_layout::task_count_addr(sm_dev_base),
        pto2_sm_layout::task_slot_map_addr(sm_dev_base, task_window_size), gm_heap, heap_size, orch_err
    );

    auto *dep_entries = static_cast<PTO2DepListEntry *>(arena.region_ptr(layout.off_dep_pool_entries));
    memset(dep_entries, 0, static_cast<size_t>(layout.dep_pool_capacity) * sizeof(PTO2DepListEntry));
    dep_pool.init(dep_entries, layout.dep_pool_capacity, orch_err);

    auto *seen_epoch = static_cast<uint32_t *>(arena.region_ptr(layout.off_fanin_seen_epoch));
    memset(seen_epoch, 0, static_cast<size_t>(task_window_size) * sizeof(uint32_t));
    fanin_seen_epoch = seen_epoch;

    if (!tensor_map.init_data_from_layout(layout.tensor_map, arena)) return false;

    scope_stack_top = -1;
    scope_stack_capacity = layout.scope_stack_capacity;
    manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
    return true;
}

bool PTO2OrchestratorState::reset_for_reuse(
    const PTO2OrchestratorLayout &layout, void *sm_dev_base, void *gm_heap, uint64_t heap_size,
    uint64_t task_window_size
) {
    sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);
    gm_heap_base = gm_heap;
    gm_heap_size = heap_size;
    fatal = false;
    inline_completed_tasks = 0;
    for (int32_t i = 0; i < PTO2_REPLAY_GRAPH_BUFFER_COUNT; ++i)
        inline_completed_by_buffer[i] = 0;

    auto *orch_err = pto2_sm_layout::orch_error_code_addr(sm_dev_base);
    task_allocator.init(
        static_cast<int32_t>(task_window_size), pto2_sm_layout::task_count_addr(sm_dev_base),
        pto2_sm_layout::task_slot_map_addr(sm_dev_base, task_window_size), gm_heap, heap_size, orch_err
    );
    dep_pool.init(dep_pool.base, layout.dep_pool_capacity, orch_err);

    uint32_t next_epoch = fanin_seen_current_epoch + 1;
    if (next_epoch == 0) {
        next_epoch = 1;
        memset(fanin_seen_epoch, 0, static_cast<size_t>(task_window_size) * sizeof(uint32_t));
    }
    fanin_seen_current_epoch = next_epoch;

    tensor_map.reset_for_reuse(layout.tensor_map);
    scope_stack_top = -1;
    scope_stack_capacity = layout.scope_stack_capacity;
    manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
    total_cluster_count = 0;
    total_aiv_count = 0;
#if SIMPLER_DFX
    tasks_submitted = 0;
    buffers_allocated = 0;
    bytes_allocated = 0;
#endif
    return true;
}

void PTO2OrchestratorState::wire_arena_pointers(
    const PTO2OrchestratorLayout &layout, DeviceArena &arena, PTO2SchedulerState *scheduler_arg
) {
    dep_pool.base = static_cast<PTO2DepListEntry *>(arena.region_ptr(layout.off_dep_pool_entries));
    fanin_seen_epoch = static_cast<uint32_t *>(arena.region_ptr(layout.off_fanin_seen_epoch));
    tensor_map.wire_arena_pointers(layout.tensor_map, arena);
    scheduler = scheduler_arg;
}

void PTO2OrchestratorState::destroy() {
    tensor_map.destroy();
    dep_pool.base = nullptr;
    fanin_seen_epoch = nullptr;
    scheduler = nullptr;
}

void PTO2OrchestratorState::set_scheduler(PTO2SchedulerState *scheduler_arg) { scheduler = scheduler_arg; }

// =============================================================================
// Top-level runtime arena
// =============================================================================

PTO2RuntimeArenaLayout
runtime_reserve_layout(DeviceArena &arena, uint64_t task_window_size, int32_t dep_pool_capacity) {
    return runtime_reserve_layout(arena, task_window_size, 0, dep_pool_capacity);
}

PTO2RuntimeArenaLayout
runtime_reserve_layout(DeviceArena &arena, uint64_t task_window_size, uint64_t heap_size, int32_t dep_pool_capacity) {
    PTO2RuntimeArenaLayout layout{};
    layout.sizing.task_window_size = task_window_size;
    layout.sizing.heap_size = heap_size;
    layout.sizing.dep_pool_capacity = dep_pool_capacity;
    layout.offsets.off_sm_handle = arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
    layout.offsets.orch =
        PTO2OrchestratorState::reserve_layout(arena, static_cast<int32_t>(task_window_size), dep_pool_capacity);
    layout.offsets.sched = PTO2SchedulerState::reserve_layout(arena);
    layout.offsets.off_runtime = arena.reserve(sizeof(PTO2Runtime), PTO2_ALIGN_SIZE);
    layout.offsets.off_mailbox = arena.reserve(sizeof(AICoreCompletionMailbox), alignof(AICoreCompletionMailbox));
    layout.offsets.arena_size = arena.total_size();
    return layout;
}

PTO2Runtime *runtime_init_data_from_layout(
    DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2RuntimeMode mode, void *sm_dev_base,
    uint64_t /*sm_size*/, void *gm_heap_dev_base, uint64_t heap_size
) {
    PTO2Runtime *rt = static_cast<PTO2Runtime *>(arena.region_ptr(layout.offsets.off_runtime));
    new (rt) PTO2Runtime{};

    auto *sm_handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(layout.offsets.off_sm_handle));
    memset(sm_handle, 0, sizeof(*sm_handle));

    rt->mode = mode;
    rt->gm_heap = gm_heap_dev_base;
    rt->gm_heap_size = heap_size;
    rt->gm_heap_owned = false;
    rt->graph_cache_enabled = false;
    rt->active_callable_hash = 0;
    rt->graph_cache_stats = PTO2GraphCacheStats{};
    reset_graph_pipeline(rt);
    if (!rt->orchestrator.init_data_from_layout(
            layout.offsets.orch, arena, sm_dev_base, gm_heap_dev_base, heap_size, layout.sizing.task_window_size
        ) ||
        !rt->scheduler.init_data_from_layout(layout.offsets.sched, arena, sm_dev_base)) {
        return nullptr;
    }

    auto *mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.offsets.off_mailbox));
    new (mailbox) AICoreCompletionMailbox{};
    return rt;
}

void runtime_wire_arena_pointers(DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2Runtime *rt) {
    rt->sm_handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(layout.offsets.off_sm_handle));
    rt->aicore_mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.offsets.off_mailbox));
    rt->orchestrator.wire_arena_pointers(layout.offsets.orch, arena, &rt->scheduler);
    rt->scheduler.wire_arena_pointers(layout.offsets.sched, arena);
}

bool runtime_reset_for_reuse(DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2Runtime *rt) {
    (void)arena;
    if (rt == nullptr) return false;
    rt->pending_scope_mode = PTO2ScopeMode::AUTO;
    rt->total_cycles = 0;
    rt->graph_cache_stats = PTO2GraphCacheStats{};
    reset_graph_pipeline(rt);
    rt->gm_heap_size = layout.sizing.heap_size;
    rt->gm_heap_owned = false;
    if (!rt->orchestrator.reset_for_reuse(
            layout.offsets.orch, rt->sm_handle->sm_base, rt->gm_heap, layout.sizing.heap_size,
            layout.sizing.task_window_size
        )) {
        return false;
    }
    rt->scheduler.reset_for_reuse(layout.offsets.sched, rt->sm_handle->sm_base);
    return true;
}

void runtime_destroy(PTO2Runtime *rt, DeviceArena & /*arena*/) {
    if (!rt) return;
    rt->scheduler.destroy();
    rt->orchestrator.destroy();
    rt->aicore_mailbox = nullptr;
    rt->sm_handle = nullptr;
}
