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
 * Distributed runtime — shared types and IWorker interface.
 *
 * Every level in the hierarchy (L3 HostWorker, L4, L5, …) runs the same
 * scheduling engine.  This header defines:
 *   - WorkerType / TaskState enumerations
 *   - TaskSlotState: per-task scheduling bookkeeping (stores TaskArgs
 *                        directly — no separate dispatch carrier struct)
 *   - ReadyQueue: Orch→Scheduler notification channel
 *   - IWorker: abstract interface implemented by ChipWorker, SubWorker,
 *              and Worker itself (recursive composition)
 *
 * IWorker::run takes (callable, TaskArgsView, ChipCallConfig) directly.
 * THREAD-mode dispatch builds the view via `slot.args_view(i)` from the
 * slot's stored TaskArgs; PROCESS-mode dispatch encodes the TaskArgs into
 * the per-WorkerThread shm mailbox via write_blob() and the child rebuilds
 * the view with read_blob().
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <vector>

#include "../task_interface/chip_call_config.h"
#include "../task_interface/task_args.h"

// =============================================================================
// Constants
// =============================================================================

// User-visible scope-nesting cap. Matches L2 PTO2_MAX_SCOPE_DEPTH.
static constexpr int32_t MAX_SCOPE_DEPTH = 64;

// Number of independent HeapRing layers inside Ring. Scope depth maps
// to ring index via `min(depth, MAX_RING_DEPTH - 1)` (L2-style);
// scopes deeper than MAX_RING_DEPTH share the innermost ring.
// Matches L2's PTO2_MAX_RING_DEPTH (Strict-1).
static constexpr int32_t MAX_RING_DEPTH = 4;

static constexpr int32_t INVALID_SLOT = -1;

// =============================================================================
// Task slot index type
// =============================================================================

using TaskSlot = int32_t;

// =============================================================================
// WorkerType
// =============================================================================

enum class WorkerType : int32_t {
    NEXT_LEVEL = 0,  // Next-level Worker (L3→ChipWorker, L4→Worker(L3), …)
    SUB = 1,         // SubWorker: fork/shm Python function
};

// =============================================================================
// TaskState
// =============================================================================

enum class TaskState : int32_t {
    FREE = 0,       // slot not in use
    PENDING = 1,    // waiting for fanin dependencies
    READY = 2,      // all fanins satisfied, in ready queue
    RUNNING = 3,    // dispatched to a worker
    COMPLETED = 4,  // worker finished, outputs may still be referenced
    CONSUMED = 5,   // all references released, slot may be reused
};

// =============================================================================
// TaskSlotState — per-task scheduling bookkeeping
// =============================================================================
//
// Stores the submitted TaskArgs directly. Dispatch builds a TaskArgsView on
// demand via `args_view(i)` (THREAD mode) or write_blob → read_blob
// (PROCESS mode). There is no separate dispatch carrier struct; the old
// WorkerPayload was removed in PR-C.

struct TaskSlotState {
    std::atomic<TaskState> state{TaskState::FREE};

    // --- Fanin (orch writes once; scheduler reads atomically) ---
    int32_t fanin_count{0};
    std::atomic<int32_t> fanin_released{0};  // incremented by each completing producer

    // --- Fanout (protected by fanout_mu) ---
    // orch adds consumers; scheduler traverses on completion
    std::mutex fanout_mu;
    std::vector<TaskSlot> fanout_consumers;
    int32_t fanout_total{0};                  // 1 (scope ref) + fanout_consumers.size()
    std::atomic<int32_t> fanout_released{0};  // incremented as each ref is released

    // --- TensorMap keys registered by this task (for cleanup on CONSUMED) ---
    std::vector<uint64_t> output_keys;

    // --- Producer tasks this task depends on (for deferred release) ---
    // When this task reaches COMPLETED, the Scheduler releases one fanout ref
    // on each producer — mirroring L2's "deferred release: walk fanin" step.
    std::vector<TaskSlot> fanin_producers;

    // --- Task data (stored on parent heap, lives until slot CONSUMED) ---
    WorkerType worker_type{WorkerType::NEXT_LEVEL};
    uint64_t callable{0};     // NEXT_LEVEL: ChipCallable buffer ptr; SUB: unused
    int32_t callable_id{-1};  // SUB: registered callable id
    ChipCallConfig config{};  // NEXT_LEVEL config (block_dim, aicpu_thread_num, enable_profiling)

    // Unified task-args storage: `task_args` is the single-task builder;
    // when `is_group_` is true, `task_args_list` carries one TaskArgs per
    // worker (N-SPMD group, L3-flavoured — each member has its own distinct
    // tensors/scalars, unlike L2's SPMD single-payload). `task_args` stays
    // empty for groups.
    TaskArgs task_args;
    std::vector<TaskArgs> task_args_list;
    bool is_group_{false};

    // Runtime-owned OUTPUT slabs live in the Worker's HeapRing and are
    // reclaimed implicitly by Ring::release(slot) — no per-slot
    // munmap is needed. See docs/orchestrator.md §8b.

    // --- HeapRing layer membership (Strict-1 per-scope rings) ---
    // Set by Ring::alloc from the caller's scope depth. ring_idx picks
    // which of the MAX_RING_DEPTH heaps holds this slot's slab;
    // ring_slot_idx is the slot's position within that ring's FIFO order
    // and indexes the ring's per-slot released/heap_end vectors.
    int32_t ring_idx{0};
    int32_t ring_slot_idx{0};

    // --- Group bookkeeping ---
    std::atomic<int32_t> sub_complete_count{0};

    bool is_group() const { return is_group_; }
    int32_t group_size() const { return is_group_ ? static_cast<int32_t>(task_args_list.size()) : 1; }

    // Zero-copy view over the i-th worker's args (THREAD-mode dispatch).
    // `i` must be 0 for non-group slots; 0..group_size()-1 for groups.
    TaskArgsView args_view(int32_t i) const {
        return is_group_ ? make_view(task_args_list[static_cast<size_t>(i)]) : make_view(task_args);
    }

    TaskSlotState() = default;
    TaskSlotState(const TaskSlotState &) = delete;
    TaskSlotState &operator=(const TaskSlotState &) = delete;

    void reset();
};

// =============================================================================
// ReadyQueue — Orch pushes, Scheduler pops
// =============================================================================

class ReadyQueue {
public:
    void push(TaskSlot slot);

    // Non-blocking: returns false immediately if empty.
    bool try_pop(TaskSlot &out);

    // Blocking: waits until a slot is available or shutdown() is called.
    // Returns false only when shutdown and queue is empty.
    bool wait_pop(TaskSlot &out);

    void shutdown();

private:
    std::queue<TaskSlot> q_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool shutdown_{false};
};

// =============================================================================
// IWorker — abstract interface
// =============================================================================

class IWorker {
public:
    virtual ~IWorker() = default;

    // Execute one task synchronously. Called in the worker's own thread,
    // blocking until the task is complete.
    //
    // Each implementation interprets `callable` per its semantics:
    //   - ChipWorker: uint64 holding a ChipCallable buffer ptr; builds a
    //     ChipStorageTaskArgs POD from `args` and calls pto2_run_runtime.
    //   - ChipProcess / SubWorker: dispatch proxies — forward
    //     callable / config / args through the shm mailbox to the forked
    //     child, which invokes the actual IWorker in its own address space.
    //   - Worker (L4+): `callable` decodes to an orch-fn handle;
    //     placeholder until PR-F.
    //
    // slot_id is not a parameter — completion routing is owned by
    // WorkerThread / Scheduler at a higher layer.
    virtual void run(uint64_t callable, TaskArgsView args, const ChipCallConfig &config) = 0;
};
