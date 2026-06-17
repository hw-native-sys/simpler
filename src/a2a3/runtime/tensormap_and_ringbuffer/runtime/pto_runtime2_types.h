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

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_RUNTIME2_TYPES_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_RUNTIME2_TYPES_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <atomic>

#include "pto_constants.h"
#include "pto_runtime_status.h"
#include "pto2_dispatch_payload.h"
#include "aicore_completion_mailbox.h"
#include "pto_submit_types.h"
#include "pto_task_id.h"
#include "pto_types.h"

#if __has_include("spin_hint.h")
#include "spin_hint.h"
#else
#define SPIN_WAIT_HINT() ((void)0)
#endif

#define PTO2_TASK_WINDOW_SIZE 16384  // Default per-ring task window size (power of 2)

// Step 1 of static-N migration: single-ring layout. All scopes map to ring 0.
#define PTO2_MAX_RING_DEPTH 1

// Memory pools (per-ring defaults; total = value × PTO2_MAX_RING_DEPTH)
#define PTO2_HEAP_SIZE (256 * 1024 * 1024)  // 256MB per ring (1GB total)
#define PTO2_DEP_LIST_POOL_SIZE 16384       // Per-ring dependency list pool entries
#define PTO2_TENSORMAP_POOL_SIZE (65536)    // TensorMap entry pool
#define PTO2_TENSORMAP_NUM_BUCKETS 4096     // Power of 2 for fast hash (4096×8B=32KB fits L1)

// Scope management
#define PTO2_MAX_SCOPE_DEPTH 64  // Maximum nesting depth
#define PTO2_SCOPE_TASKS_CAP (PTO2_TASK_WINDOW_SIZE * PTO2_MAX_RING_DEPTH)

// Ready queue
#define PTO2_READY_QUEUE_SIZE 65536  // Per-shape queue size

// Wiring queue
#define PTO2_WRIRING_QUEUE_SIZE 1024  // Per-shape queue size

// Fanin storage — absolute max number of unique fanin dependencies per task.
#define PTO2_MAX_FANIN 16

// TensorMap cleanup interval
#define PTO2_TENSORMAP_CLEANUP_INTERVAL 64  // Cleanup every N retired tasks
#define PTO2_DEP_POOL_CLEANUP_INTERVAL 64   // Cleanup every N retired tasks

// get_tensor_data/set_tensor_data spin wait timeout in cycles.
// ~10s on hardware (1.5 GHz counter), ~10s on simulation (chrono-based).
constexpr uint64_t PTO2_TENSOR_DATA_TIMEOUT_CYCLES = 15 * 1000 * 1000 * 1000ULL;

typedef enum
{
    PTO2_TASK_PENDING = 0,    // Submitted; awaiting fanin, queued, or dispatched
    PTO2_TASK_COMPLETED = 1,  // Execution finished, output may still be in use
    PTO2_TASK_CONSUMED = 2    // Output fully consumed, buffers can be released
} PTO2TaskState;

struct PTO2TaskAllocResult
{
    int32_t task_id;    // Absolute task ID (not wrapped)
    int32_t slot;       // task_id & (window_size - 1)
    void *packed_base;  // Heap allocation result (nullptr if failure)
    void *packed_end;   // packed_base + aligned output_size

    bool failed() const
    {
        return task_id < 0;
    }
};

struct PTO2OutputLayout
{
    uint64_t offsets[MAX_TENSOR_ARGS] = {};
    uint64_t buffer_sizes[MAX_TENSOR_ARGS] = {};
    int32_t total_output_size = 0;
};

struct PTO2TaskSlotState;  // Forward declaration

struct PTO2TaskDescriptor
{
    // Mixed-task identification (encodes ring_id in upper 32 bits)
    PTO2TaskId task_id;  // raw: (ring_id << 32) | local_id

    // Per-slot kernel IDs (INVALID_KERNEL_ID = inactive)
    int32_t kernel_id[PTO2_SUBTASK_SLOT_COUNT];

    // Packed output buffer (all outputs packed into single contiguous buffer)
    void *packed_buffer_base;  // Start of packed buffer in GM Heap
    void *packed_buffer_end;   // End of packed buffer (for heap reclamation)
};

struct PTO2TaskPayload
{
    // === Cache lines 0-2 (192B) — metadata + fanin ===
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    int32_t fanin_count{0};  // Number of valid entries in fanin_slot_states
    PTO2TaskSlotState *fanin_slot_states[PTO2_MAX_FANIN];
    // === Tensors (Tensor is alignas(64); array is naturally aligned) ===
    Tensor tensors[MAX_TENSOR_ARGS];
    // === Scalars ===
    uint64_t scalars[MAX_SCALAR_ARGS];

    static_assert(sizeof(Tensor) == 128, "Tensor must be 2 cache lines");
    static_assert(MAX_SCALAR_ARGS * sizeof(uint64_t) == MAX_SCALAR_ARGS * 8, "scalar region size matches MAX_SCALAR_ARGS");

    void init(const Arg &args, TaskOutputTensors &result, PTO2TaskAllocResult &alloc_result, PTO2OutputLayout &layout)
    {
        tensor_count = args.tensor_count();
        scalar_count = args.scalar_count();

        // int32_t out_idx = 0;
        for (int32_t i = 0; i < args.tensor_count(); i++)
        {
            if (args.tag(i) != TensorArgType::OUTPUT)
            {
                tensors[i].copy(*args.tensor(i).ptr);
            }
            else
            {
                tensors[i].init_from_create_info(*args.tensor(i).create_info, reinterpret_cast<void *>(reinterpret_cast<char *>(alloc_result.packed_base) + layout.offsets[i]), layout.buffer_sizes[i]);
                tensors[i].owner_task_id = result.task_id();
                result.materialize_output(tensors[i]);
            }
        }
        // Round up to cache line boundary. Both arrays are 1024B so no overrun.
        // Eliminates branches; extra bytes within the same CL have zero additional cost.
        memcpy(scalars, args.scalars(), PTO2_ALIGN_UP(args.scalar_count() * sizeof(uint64_t), 64));
    }
};

// PTO2TaskPayload layout verification (offsetof requires complete type).
static_assert(offsetof(PTO2TaskPayload, fanin_slot_states) == 16, "fanin array must follow metadata words");
static_assert(offsetof(PTO2TaskPayload, scalars) == offsetof(PTO2TaskPayload, tensors) + MAX_TENSOR_ARGS * sizeof(Tensor), "scalars must immediately follow tensors");
static_assert(sizeof(PTO2TaskPayload) == offsetof(PTO2TaskPayload, scalars) + MAX_SCALAR_ARGS * sizeof(uint64_t), "no trailing padding after scalars");

struct alignas(64) PTO2TaskSlotState
{
    // Fanout: tracks producer->CONSUMED transition. Incremented by the
    // orchestrator (+1 sentinel and once per consumer of this slot) and
    // matched by release_producer in on_task_release.
    int32_t fanout_count;
    std::atomic<int32_t> fanout_refcount;

    // Task state (PENDING/COMPLETED/CONSUMED). Polling readiness reads
    // task_state on producer slots.
    std::atomic<PTO2TaskState> task_state;

    PTO2TaskPayload *payload;
    PTO2TaskDescriptor *task;

    // Intrusive linkage for the thread-0 pending-readiness queue.
    PTO2TaskSlotState *next_pending{nullptr};

    // --- Set per-submit (depend on task inputs) ---
    ActiveMask active_mask;  // Bitmask of active subtask slots (set once)
    uint8_t ring_id;         // Ring layer (immutable after init)
    std::atomic<bool> any_subtask_deferred{false};
    uint8_t _async_pad{0};

    std::atomic<int16_t> completed_subtasks{0};  // Each core completion increments by 1
    int16_t total_required_subtasks{0};          // = logical_block_num * popcount(active_mask)
    int16_t logical_block_num{1};                // Total logical blocks (set by orchestrator)
    int16_t next_block_idx{0};                   // Next block to dispatch (scheduler state)

    void bind_ring(uint8_t rid)
    {
        ring_id = rid;
    }

    void bind_buffers(PTO2TaskPayload *p, PTO2TaskDescriptor *t)
    {
        payload = p;
        task = t;
    }

    void reset_for_reuse()
    {
        fanout_count = 1;
        fanout_refcount.store(0, std::memory_order_relaxed);
        completed_subtasks.store(0, std::memory_order_relaxed);
        next_block_idx = 0;
        any_subtask_deferred.store(false, std::memory_order_relaxed);
        next_pending = nullptr;
    }
};

static_assert(sizeof(PTO2TaskSlotState) <= 128, "slot state should fit in two cache lines");

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_RUNTIME2_TYPES_H_
