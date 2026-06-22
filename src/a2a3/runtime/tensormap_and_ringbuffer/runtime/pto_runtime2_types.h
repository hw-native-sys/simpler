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

#include "profiling_config.h"
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

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
#include "aicpu/device_time.h"
#endif

// =============================================================================
// Configuration Constants
// =============================================================================

// Task management
// NOTE: PTO2_TASK_WINDOW_SIZE is now a per-ring default value.
// Actual window size is passed at runtime to runtime_create_from_sm().
// Use pto2_task_slot(sched, task_id) for slot calculation.
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

// Cross-thread early-dispatch work queue (power of two)
#define PTO2_EARLY_DISPATCH_QUEUE_SIZE 64

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
    PTO2_TASK_PENDING = 0,   // Submitted; awaiting fanin, queued, or dispatched
    PTO2_TASK_COMPLETED = 1  // Execution finished; per-ring completed_watermark
                             // advances past this slot's last_consumer_local_id
                             // to make its heap chunk reclaimable.
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

// =============================================================================
// Per-Slot Scheduling State
// =============================================================================

/**
 * Task payload data (cold path - only accessed during orchestration and dispatch)
 *
 * Layout: metadata + flat fanin_local_ids[] in the first 2 cache lines,
 * followed by bulk tensor and scalar data.
 */

struct PTO2TaskPayload {
    // === Cache lines 0-2 (192B) — metadata + fanin (wireless model) ===
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    // wireless: flat fanin_local_ids[] populated at submit. The thread-0
    // pending poll indexes a compact ring-level completion_flags byte array
    // via these ids — avoids a pointer chase per fanin into a 128B-aligned
    // slot_state.
    int32_t fanin_count{0};
    int32_t fanin_local_ids[PTO2_MAX_FANIN];
    // === Tensors (Tensor is alignas(64); array is naturally aligned) ===
    Tensor tensors[MAX_TENSOR_ARGS];
    // === Scalars ===
    uint64_t scalars[MAX_SCALAR_ARGS];

    static_assert(sizeof(Tensor) == 128, "Tensor must be 2 cache lines");
    static_assert(MAX_SCALAR_ARGS * sizeof(uint64_t) == MAX_SCALAR_ARGS * 8, "scalar region size matches MAX_SCALAR_ARGS");

    /**
     * Prefetch (for write) the regions init() is about to fill so the stores land
     * in warm cache. tensor_count/scalar_count come from the Arg — the payload's
     * own counts are not set until init(). Warms the early-dispatch spec block at
     * offset 536 (cache line 8) too. A member fn lowers to the same prefetch
     * instructions as a free function (`this` is just a register), no cache impact.
     */
    void prefetch(int32_t tensor_count, int32_t scalar_count) const {
        for (int32_t i = 0; i < tensor_count; i++) {
            __builtin_prefetch(&tensors[i], 1, 3);
            __builtin_prefetch(reinterpret_cast<const char *>(&tensors[i]) + 64, 1, 3);
        }
        for (int32_t i = 0; i < scalar_count; i += 8) {
            __builtin_prefetch(&scalars[i], 1, 3);
        }
        __builtin_prefetch(this, 1, 3);
        __builtin_prefetch(reinterpret_cast<const char *>(this) + 64, 1, 3);
        __builtin_prefetch(reinterpret_cast<const char *>(this) + 128, 1, 3);
    }

    /**
     * Initialize payload: copy tensors, store scalars.
     *
     * For each param slot, the tensor source is determined by TensorArgType:
     * - OUTPUT -> use materialized_outputs.output_ptr(out_idx++)
     * - INPUT / INOUT -> use refs[i].tensor
     *
     * @param args                Task arguments (tensors + scalars)
     * @param result  Materialized output tensors (from TensorCreateInfo path)
     */
    void init(const Arg &args, TaskOutputTensors &result, PTO2TaskAllocResult &alloc_result, PTO2OutputLayout &layout) {
        tensor_count = args.tensor_count();
        scalar_count = args.scalar_count();

        // int32_t out_idx = 0;
        for (int32_t i = 0; i < args.tensor_count(); i++)
        {
            if (args.tag(i) != TensorArgType::OUTPUT)
            {
                tensors[i].copy(*args.tensor(i).ptr);
            } else {
                init_tensor_from_create_info(
                    tensors[i], *args.tensor(i).create_info,
                    reinterpret_cast<void *>(reinterpret_cast<char *>(alloc_result.packed_base) + layout.offsets[i]),
                    layout.buffer_sizes[i]
                );
                tensors[i].owner_task_id = result.task_id();
                result.materialize_output(tensors[i]);
            }
        }
        // Round up to cache line boundary. Both arrays are 128B so no overrun.
        // Eliminates branches; extra bytes within the same CL have zero additional cost.
        memcpy(scalars, args.scalars(), PTO2_ALIGN_UP(args.scalar_count() * sizeof(uint64_t), 64));
    }
};

// PTO2TaskPayload layout verification (offsetof requires complete type).
static_assert(offsetof(PTO2TaskPayload, fanin_local_ids) == 12, "fanin array must follow metadata words");
static_assert(offsetof(PTO2TaskPayload, scalars) == offsetof(PTO2TaskPayload, tensors) + MAX_TENSOR_ARGS * sizeof(Tensor), "scalars must immediately follow tensors");
static_assert(sizeof(PTO2TaskPayload) == offsetof(PTO2TaskPayload, scalars) + MAX_SCALAR_ARGS * sizeof(uint64_t), "no trailing padding after scalars");

struct alignas(64) PTO2TaskSlotState
{
    // Highest local task id among this slot's consumers. Set to this slot's
    // own local_id in prepare_task; bumped via max() in submit_task_common for
    // each consumer that has this slot as a fanin. The slot's heap chunk is
    // safe to reclaim when the per-ring completed_watermark reaches at least
    // this id (i.e. every task up to and including the last consumer has
    // transitioned to COMPLETED). Single-writer (orchestrator) at submit time.
    int32_t last_consumer_local_id;

    PTO2TaskPayload *payload;
    PTO2TaskDescriptor *task;

    // --- (e) Wake-list: lightweight last-fanin notification ---
    // When a pending consumer's fanin scan finds exactly ONE unmet fanin,
    // it registers itself on the producer's wake list (CAS push). On producer
    // completion, the producer atomic-exchanges wake_list_head to the
    // SENTINEL value and pushes every waiter to the ready queues. Consumers
    // that observe SENTINEL during registration push themselves directly
    // (producer already completed). Reset to nullptr on slot reuse.
    std::atomic<PTO2TaskSlotState *> wake_list_head{nullptr};
    PTO2TaskSlotState *next_in_wake_list{nullptr};

    // --- Set per-submit (depend on task inputs) ---
    ActiveMask active_mask;  // Bitmask of active subtask slots (set once)
    uint8_t ring_id;         // Ring layer (immutable after init)
    std::atomic<bool> any_subtask_deferred{false};
    uint8_t _async_pad{0};

    std::atomic<int16_t> completed_subtasks{0};  // Each core completion increments by 1
    int16_t total_required_subtasks{0};          // = logical_block_num * popcount(active_mask)
    int16_t logical_block_num{1};                // Total logical blocks (set by orchestrator)
    // Next block to dispatch. Atomic so concurrent speculative stagers can each
    // claim a distinct block via CAS; normal dispatch (ready-queue serialized)
    // uses plain relaxed load/store. The two phases never overlap in time (staging
    // happens before release; normal dispatch of the remainder happens after).
    std::atomic<int16_t> next_block_idx{0};

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
        completed_subtasks.store(0, std::memory_order_relaxed);
        next_block_idx.store(0, std::memory_order_relaxed);
        any_subtask_deferred.store(false, std::memory_order_relaxed);
        // (e) Wake list: clear for the next incarnation. Previous incarnation
        // left it at WAKE_LIST_SENTINEL (set by its on_mixed_task_complete).
        wake_list_head.store(nullptr, std::memory_order_relaxed);
        next_in_wake_list = nullptr;
        // last_consumer_local_id is reset in prepare_task once the task_id is known.
    }
};

// (e) Sentinel marking a wake list as "owner already completed; no more
// registrations accepted". Distinct from any real slot_state pointer.
inline PTO2TaskSlotState *const WAKE_LIST_SENTINEL = reinterpret_cast<PTO2TaskSlotState *>(uintptr_t{1});

static_assert(sizeof(PTO2TaskSlotState) <= 128, "slot state should fit in two cache lines");

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_RUNTIME2_TYPES_H_
