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

#include "host_build_graph/shared_arena_layout.h"
#include <atomic>
#include <cstddef>
#include "common/tagged_ptr.h"
#include "common/lane_demand_counter.h"
#include "utils/device_arena.h"
#include "host_build_graph/runtime_types.h"
#include "host_build_graph/shared_memory.h"
#include "async_wait.h"
#include "scheduler_graph.h"
#include "profiling_config.h"

static_assert(sizeof(ChipTaskStorage) == SCHEDULER_GRAPH_TASK_STORAGE_STRIDE);
static_assert(offsetof(ChipTaskStorage, task) == SCHEDULER_GRAPH_DESCRIPTOR_OFFSET);
static_assert(offsetof(ChipTaskStorage, payload) == SCHEDULER_GRAPH_PAYLOAD_OFFSET);
static_assert(offsetof(TaskPayload, predicate) == SCHEDULER_GRAPH_PREDICATE_OFFSET);

// Layout declarations consumed by common HBG Host arena initialization.
// WSE execution uses per-PE queues in pe_executor.cpp.
using CapturedReadySlot = TaggedPtr<ChipTaskSlotState, LaneMask, SUBTASK_SLOT_COUNT>;
struct ChipReadyQueueSlot {
    std::atomic<int64_t> sequence;
    CapturedReadySlot slot_value;
    TaskId task_id_snapshot;
};

struct alignas(64) ChipReadyQueue {
    ChipReadyQueueSlot *slots;
    uint64_t capacity;
    uint64_t mask;  // capacity - 1
    LaneDemandCounter mixed_lanes;
    char _pad0[64 - 32];  // Pad to own cache line

    std::atomic<uint64_t> enqueue_pos;
    char _pad1[64 - sizeof(std::atomic<uint64_t>)];  // Own cache line

    std::atomic<uint64_t> dequeue_pos;
    std::atomic<uint64_t> max_occupancy;
    char _pad2[64 - 2 * sizeof(std::atomic<uint64_t>)];  // Own cache line

    void seed_slots() {
        for (uint64_t i = 0; i < capacity; i++) {
            slots[i].sequence.store(static_cast<int64_t>(i), std::memory_order_relaxed);
            slots[i].slot_value = CapturedReadySlot{};
        }
    }
};
size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity);
// Writes the header fields only: capacity, mask, positions, occupancy counter.
// The slot array is neither addressed nor written — it lives past the uploaded
// range and ChipReadyQueue::seed_slots() fills it on the device. Call
// `ready_queue_wire_arena_pointers` to set the `slots` pointer itself.
void ready_queue_init_data_from_layout(ChipReadyQueue *queue, uint64_t capacity);
// Stores queue->slots = arena.region_ptr(slots_off). Idempotent.
void ready_queue_wire_arena_pointers(ChipReadyQueue *queue, DeviceArena &arena, size_t slots_off);
void ready_queue_destroy(ChipReadyQueue *queue);

struct SchedulerState {
    SharedMemoryHeader *sm_header;

    struct alignas(64) TaskHeaderView {
        SharedMemoryTaskHeader *tasks;

        bool init_data_from_layout(void *sm_dev_base);
        void destroy();
    } task_view;

    ChipReadyQueue ready_queues[NUM_RESOURCE_SHAPES];

    ChipReadyQueue ready_sync_queues[NUM_RESOURCE_SHAPES];

    ChipReadyQueue dummy_ready_queue;

    ChipReadyQueue graph_prepare_queue;

    alignas(64) AsyncWaitList async_wait_list;

#if SIMPLER_SCHED_PROFILING
    alignas(64) std::atomic<int64_t> tasks_completed;
    std::atomic<int64_t> tasks_consumed;
#endif
    ChipReadyQueue early_dispatch_queues[NUM_RESOURCE_SHAPES];
    ChipReadyQueue early_sync_start_queue;
    ChipReadyQueue ed_publish_drain_queue;
    static SchedulerLayout reserve_layout(DeviceArena &arena);
    bool init_data_from_layout(const SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base);
    void wire_arena_pointers(const SchedulerLayout &layout, DeviceArena &arena);
    void seed_queue_slots();
    void destroy();
};
