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
 *   - submit_next_level(CallableIdentity, TaskArgs, CallConfig)
 *   - submit_next_level_group(CallableIdentity, vector<TaskArgs>, CallConfig)
 *   - submit_sub(CallableIdentity, TaskArgs)
 *   - submit_sub_group(CallableIdentity, vector<TaskArgs>)
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
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "../task_interface/call_config.h"
#include "../task_interface/data_type.h"
#include "../task_interface/task_args.h"
#include "../task_interface/tensor.h"
#include "ring.h"
#include "scope.h"
#include "tensormap.h"
#include "types.h"

class WorkerManager;

// ---------------------------------------------------------------------------
// SubmitResult — C++ internal slot id
// ---------------------------------------------------------------------------
//
// Downstream consumers reference outputs by their own tensor pointers (the
// tensors live in the HeapRing allocated by the Worker), and tensormap.lookup
// finds the producer slot from the data pointer. No outputs[] field needed.
// This is intentionally not exposed through the Python facade.

struct SubmitResult {
    TaskSlot task_slot{INVALID_SLOT};
};

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

class Orchestrator {
public:
    // NEXT_LEVEL singles use one FIFO per stable worker id. NEXT_LEVEL groups
    // and SUB tasks retain their shared queues in this stage of the stack.
    void init(
        TensorMap *tensormap, Ring *allocator, Scope *scope, ReadyQueue *ready_next_level_queue,
        ReadyQueue *ready_sub_queue, PerWorkerReadyQueues *ready_next_level_single_queues,
        WorkerManager *manager = nullptr, std::function<void()> ready_notify_cb = {}
    );

    // Allocate an intermediate buffer from the Worker's HeapRing (MAP_SHARED,
    // visible to forked child workers). Returns a contiguous Tensor whose
    // `.buffer.addr` points into the ring.
    //
    // Lifetime: aligned with a synthetic task slot. The buffer is reclaimed
    // (FIFO, via last_alive) once every downstream consumer tagging the
    // pointer has reached CONSUMED and scope_end has released the scope ref.
    Tensor alloc(const std::vector<uint32_t> &shape, DataType dtype);

    // Memory management on a specific next-level worker. Thread-safe:
    // can be called from the orch thread while the target worker is
    // running a task (MemoryAllocator is mutex-protected).
    uint64_t malloc(int worker_id, size_t size);
    void free(int worker_id, uint64_t ptr);
    void copy_to(int worker_id, uint64_t dst, uint64_t src, size_t size);
    void copy_from(int worker_id, uint64_t dst, uint64_t src, size_t size);

    // Submit a NEXT_LEVEL task. `callable` is the stable identity returned
    // by Worker.register(); the child resolves its digest to a private slot.
    // Tags inside `args` drive dependency inference; OUTPUT tensors with
    // null data are auto-allocated from the HeapRing.
    // `worker_id`: exact stable NEXT_LEVEL worker id that runs this task.
    SubmitResult submit_next_level(
        const CallableIdentity &callable, const TaskArgs &args, const CallConfig &config, int32_t worker_id,
        const std::vector<int32_t> &eligible_worker_ids = {}, const RemoteTaskArgsSidecar &remote_sidecar = {}
    );

    // Submit a group of NEXT_LEVEL tasks: N args -> N worker selections, 1 DAG node.
    // `worker_ids`: one exact stable NEXT_LEVEL worker id per member.
    SubmitResult submit_next_level_group(
        const CallableIdentity &callable, const std::vector<TaskArgs> &args_list, const CallConfig &config,
        const std::vector<int32_t> &worker_ids, const std::vector<std::vector<int32_t>> &eligible_worker_ids = {},
        const std::vector<RemoteTaskArgsSidecar> &remote_sidecars = {}
    );

    // Submit a SUB task by registered callable identity.
    SubmitResult submit_sub(const CallableIdentity &callable, const TaskArgs &args);

    // Submit a group of SUB tasks: N args -> N workers, 1 DAG node.
    SubmitResult submit_sub_group(const CallableIdentity &callable, const std::vector<TaskArgs> &args_list);

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

    // Wire the Scheduler's loop mutex so drain() can hold it across
    // reset_to_empty(), preventing the scheduler thread from touching slot
    // state mid-teardown (heap-use-after-free). Set once by Worker after the
    // Scheduler is constructed.
    void set_scheduler_loop_mutex(std::mutex *m) { sched_loop_mu_ = m; }

    // Clear any stored dispatch error so the next Worker::run() starts
    // from a clean slate. Called by Worker::run before scope_begin.
    void clear_error();

    // Called by Scheduler (via Worker) when a task becomes CONSUMED:
    // erases TensorMap entries, releases the allocator slot (and implicitly
    // the slot's heap slab via last_alive).
    // Returns true iff this call performed the COMPLETED/FAILED -> CONSUMED transition.
    // Idempotent: concurrent callers (release_ref vs try_consume) race on a
    // CAS — only the winner returns true and runs cleanup; losers return false.
    bool on_consumed(TaskSlot slot);

    // Route a slot whose state is already READY to the queue that owns it.
    // Scheduler uses the same path after releasing the final dependency.
    void enqueue_ready(TaskSlot slot);

private:
    TensorMap *tensormap_ = nullptr;
    Ring *allocator_ = nullptr;
    Scope *scope_ = nullptr;
    WorkerManager *manager_ = nullptr;
    std::function<void()> ready_notify_cb_;
    // Strict-4 per-worker-type ready queues. Each queue handles tasks of
    // exactly one WorkerType so the Scheduler can dispatch from an idle pool
    // without being blocked by another pool's saturation.
    ReadyQueue *ready_next_level_queue_ = nullptr;
    ReadyQueue *ready_sub_queue_ = nullptr;
    PerWorkerReadyQueues *ready_next_level_single_queues_ = nullptr;

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
    // Scheduler's loop mutex (not owned). Held across reset_to_empty() in
    // drain() so the scheduler can't be mid-on_task_complete during teardown.
    std::mutex *sched_loop_mu_{nullptr};

    // Slot state lives in the Ring; the pointer stays stable for the
    // slot's lifetime. Throws if the id is out of range — callers that
    // hold a recently-allocated slot id should always get a valid pointer.
    TaskSlotState &slot_state(TaskSlot s);

    // Shared submit machinery. Takes `args_list` by value so the Orchestrator
    // can patch `tensor.data` on OUTPUT tensors flagged for auto-allocation.
    SubmitResult submit_impl(
        WorkerType worker_type, const CallableIdentity &callable, const CallConfig &config,
        std::vector<TaskArgs> args_list, std::vector<int32_t> affinities = {},
        std::vector<std::vector<int32_t>> eligible_worker_ids = {},
        std::vector<RemoteTaskArgsSidecar> remote_sidecars = {}
    );

    // Size, in aligned bytes, an OUTPUT tensor should occupy in the HeapRing.
    static uint64_t output_alloc_bytes(const Tensor &t);

    // Rewrite any OUTPUT tensors with a null data pointer to point into a
    // freshly-allocated HeapRing slab. Returns the total aligned byte span
    // consumed, and populates `slot` / `heap_ptr` / `heap_end_offset` via the
    // output params (reused for book-keeping on the slot state). Throws on
    // back-pressure timeout.
    AllocResult reserve_outputs_and_slot(
        std::vector<TaskArgs> &args_list, const std::vector<RemoteTaskArgsSidecar> &remote_sidecars
    );

    // Walk the tags of each TaskArgs in `args_list`, accumulating producer
    // slots (for INPUT/INOUT tags) and registering outputs in the tensormap
    // (for OUTPUT/INOUT/OUTPUT_EXISTING tags). NO_DEP tags are skipped.
    // `affinities` maps args_list[i] to worker id for TensorKey construction.
    void infer_deps(
        TaskSlot slot, const std::vector<TaskArgs> &args_list, const std::vector<int32_t> &affinities,
        const std::vector<RemoteTaskArgsSidecar> &remote_sidecars, std::vector<TaskSlot> &producers,
        std::vector<TensorKey> &output_keys
    );
    void validate_worker_eligibility(
        WorkerType worker_type, size_t args_count, const std::vector<int32_t> &affinities,
        const std::vector<std::vector<int32_t>> &eligible_worker_ids
    ) const;
    void validate_remote_sidecars(
        const std::vector<TaskArgs> &args_list, const std::vector<RemoteTaskArgsSidecar> &remote_sidecars,
        const std::vector<std::vector<int32_t>> &eligible_worker_ids
    ) const;

    // Release one fanout reference on 'slot'.
    // If all references are released → transition to CONSUMED.
    void release_ref(TaskSlot slot);
    void try_consume(TaskSlot slot);
};
