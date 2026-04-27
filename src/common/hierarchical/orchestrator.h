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
 * Orchestrator — DAG builder.
 *
 * Public API (called by the user's orch fn during Worker::run):
 *   - submit_next_level(callable, TaskArgs, CallConfig)
 *   - submit_next_level_group(callable, vector<TaskArgs>, CallConfig)
 *   - submit_sub(callable_id, TaskArgs)
 *   - submit_sub_group(callable_id, vector<TaskArgs>)
 *   - alloc(shape, dtype) — runtime-owned intermediate buffer
 *
 * Each TaskArgs carries per-tensor TensorArgType tags. The Orchestrator
 * walks those tags to drive dependency inference and — for OUTPUT tags with
 * a null data pointer — automatically assigns a slab from the HeapRing
 * (see docs/orchestrator.md §8b).
 *
 * Internal:
 *   - scope_begin / scope_end / drain — invoked only by Worker::run
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "../task_interface/call_config.h"
#include "../task_interface/data_type.h"
#include "../task_interface/task_args.h"
#include "../task_interface/tensor_arg.h"
#include "ring.h"
#include "scope.h"
#include "tensormap.h"
#include "types.h"

class WorkerManager;

// ---------------------------------------------------------------------------
// SubmitResult — just the slot id
// ---------------------------------------------------------------------------
//
// Downstream consumers reference outputs by their own tensor pointers (the
// tensors live in the HeapRing allocated by the Worker), and tensormap.lookup
// finds the producer slot from the data pointer. No outputs[] field needed.

struct SubmitResult {
    TaskSlot task_slot{INVALID_SLOT};
};

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

class Orchestrator {
public:
    // Strict-4: the engine keeps one ReadyQueue per WorkerType so a
    // saturated sub pool cannot head-of-line-block chip dispatch (and vice
    // versa). Submit routes to the queue matching the task's worker_type;
    // the Scheduler's dispatch_ready walks each queue independently.
    void init(
        TensorMap *tensormap, Ring *allocator, Scope *scope, ReadyQueue *ready_next_level_queue,
        ReadyQueue *ready_sub_queue, WorkerManager *manager = nullptr
    );

    // Allocate an intermediate buffer from the Worker's HeapRing (MAP_SHARED,
    // visible to forked child workers). Returns a ContinuousTensor whose
    // `.data` points into the ring.
    //
    // Lifetime: aligned with a synthetic task slot. The buffer is reclaimed
    // (FIFO, via last_alive) once every downstream consumer tagging the
    // pointer has reached CONSUMED and scope_end has released the scope ref.
    ContinuousTensor alloc(const std::vector<uint32_t> &shape, DataType dtype);

    // Memory management on a specific next-level worker. Thread-safe:
    // can be called from the orch thread while the target worker is
    // running a task (MemoryAllocator is mutex-protected).
    uint64_t malloc(int worker_id, size_t size);
    void free(int worker_id, uint64_t ptr);
    void copy_to(int worker_id, uint64_t dst, uint64_t src, size_t size);
    void copy_from(int worker_id, uint64_t dst, uint64_t src, size_t size);

    // Submit a NEXT_LEVEL task. `callable` is the chip callable buffer pointer
    // (uint64_t handle from Python — typically ChipCallable.buffer_ptr()).
    // Tags inside `args` drive dependency inference; OUTPUT tensors with null
    // data are auto-allocated from the HeapRing.
    // `worker`: logical worker id for affinity (-1 = unconstrained).
    SubmitResult
    submit_next_level(uint64_t callable, const TaskArgs &args, const CallConfig &config, int8_t worker = -1);

    // Submit a group of NEXT_LEVEL tasks: N args -> N workers, 1 DAG node.
    // `workers`: per-args affinity (empty = all unconstrained).
    SubmitResult submit_next_level_group(
        uint64_t callable, const std::vector<TaskArgs> &args_list, const CallConfig &config,
        const std::vector<int8_t> &workers = {}
    );

    // Submit a SUB task by registered callable id.
    SubmitResult submit_sub(int32_t callable_id, const TaskArgs &args);

    // Submit a group of SUB tasks: N args -> N workers, 1 DAG node.
    SubmitResult submit_sub_group(int32_t callable_id, const std::vector<TaskArgs> &args_list);

    // Open a nested scope. Every task submitted between this call and the
    // matching `scope_end()` picks a heap ring based on the current scope
    // depth (`min(depth, MAX_RING_DEPTH - 1)`) so its slab reclaims
    // independently of the outer scope's slabs (Strict-1). `Worker::run`
    // opens the outermost scope automatically; user orch fns may nest up
    // to `MAX_SCOPE_DEPTH` additional scopes.
    //
    // Non-blocking: `scope_end` walks the scope's tasks and releases one
    // ref per task, returning immediately. Actual CONSUMED transitions
    // happen asynchronously as each task's consumer count reaches
    // threshold (mirrors L2's `pto2_scope_end`). Callers that need a
    // synchronous wait must call `drain()` separately.
    void scope_begin();
    void scope_end();

    // Block until every submitted task has reached CONSUMED. Invoked by
    // Worker::run after scope_end; not part of the user-facing orch-fn API.
    // Rethrows the first exception reported by any WorkerThread during
    // dispatch (fail-fast): the wait itself is unaffected — every in-flight
    // task is allowed to finish so ring slots aren't leaked.
    void drain();

    // Clear any stored dispatch error so the next Worker::run() starts
    // from a clean slate. Called by Worker::run before scope_begin.
    void clear_error();

    // Called by Scheduler (via Worker) when a task becomes CONSUMED:
    // erases TensorMap entries, releases the allocator slot (and implicitly
    // the slot's heap slab via last_alive).
    // Returns true iff this call performed the COMPLETED -> CONSUMED transition.
    // Idempotent: concurrent callers (release_ref vs try_consume) race on a
    // CAS — only the winner returns true and runs cleanup; losers return false.
    bool on_consumed(TaskSlot slot);

private:
    TensorMap *tensormap_ = nullptr;
    Ring *allocator_ = nullptr;
    Scope *scope_ = nullptr;
    WorkerManager *manager_ = nullptr;
    // Strict-4 per-worker-type ready queues. Each queue handles tasks of
    // exactly one WorkerType so the Scheduler can dispatch from an idle pool
    // without being blocked by another pool's saturation.
    ReadyQueue *ready_next_level_queue_ = nullptr;
    ReadyQueue *ready_sub_queue_ = nullptr;

    // Returns the ready queue that owns tasks of the given worker type.
    // The method itself does not mutate the Orchestrator (hence `const`);
    // the returned pointer is non-const because callers push into the queue.
    ReadyQueue *ready_queue_for(WorkerType t) const {
        return t == WorkerType::NEXT_LEVEL ? ready_next_level_queue_ : ready_sub_queue_;
    }

    // --- Drain support (owned here, not on Worker) ---
    std::atomic<int32_t> active_tasks_{0};
    std::mutex drain_mu_;
    std::condition_variable drain_cv_;

    // Slot state lives in the Ring; the pointer stays stable for the
    // slot's lifetime. Throws if the id is out of range — callers that
    // hold a recently-allocated slot id should always get a valid pointer.
    TaskSlotState &slot_state(TaskSlot s);

    // Shared submit machinery. Takes `args_list` by value so the Orchestrator
    // can patch `tensor.data` on OUTPUT tensors flagged for auto-allocation.
    SubmitResult submit_impl(
        WorkerType worker_type, uint64_t callable_ptr, int32_t callable_id, const CallConfig &config,
        std::vector<TaskArgs> args_list, std::vector<int8_t> affinities = {}
    );

    // Size, in aligned bytes, an OUTPUT tensor should occupy in the HeapRing.
    static uint64_t output_alloc_bytes(const ContinuousTensor &t);

    // Rewrite any OUTPUT tensors with a null data pointer to point into a
    // freshly-allocated HeapRing slab. Returns the total aligned byte span
    // consumed, and populates `slot` / `heap_ptr` / `heap_end_offset` via the
    // output params (reused for book-keeping on the slot state). Throws on
    // back-pressure timeout.
    AllocResult reserve_outputs_and_slot(std::vector<TaskArgs> &args_list);

    // Walk the tags of each TaskArgs in `args_list`, accumulating producer
    // slots (for INPUT/INOUT tags) and registering outputs in the tensormap
    // (for OUTPUT/INOUT/OUTPUT_EXISTING tags). NO_DEP tags are skipped.
    // `affinities` maps args_list[i] → worker id for TensorKey construction.
    void infer_deps(
        TaskSlot slot, const std::vector<TaskArgs> &args_list, const std::vector<int8_t> &affinities,
        std::vector<TaskSlot> &producers, std::vector<TensorKey> &output_keys
    );

    // Release one fanout reference on 'slot'.
    // If all references are released → transition to CONSUMED.
    void release_ref(TaskSlot slot);
};
