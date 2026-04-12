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
 *   - WorkerPayload: the data dispatched to an IWorker
 *   - DistTaskSlotState: per-task scheduling bookkeeping
 *   - DistReadyQueue: Orch→Scheduler notification channel
 *   - IWorker: abstract interface implemented by ChipWorker, SubWorker,
 *              and DistWorker itself (recursive composition)
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <vector>

// =============================================================================
// Constants
// =============================================================================

static constexpr int32_t DIST_TASK_WINDOW_SIZE = 128;  // slots per engine instance
static constexpr int32_t DIST_MAX_SCOPE_DEPTH = 64;
static constexpr int32_t DIST_INVALID_SLOT = -1;

// =============================================================================
// Task slot index type
// =============================================================================

using DistTaskSlot = int32_t;

struct DistTensorKey {
    // base_ptr alone is not a stable distributed identity: different workers may
    // legally expose the same virtual address. Include worker_index so TensorMap
    // can distinguish external buffers coming from different chip/sub-worker
    // address spaces.
    int32_t worker_index{-1};
    uint64_t base_ptr{0};

    bool operator==(const DistTensorKey &other) const {
        return worker_index == other.worker_index && base_ptr == other.base_ptr;
    }
};

// =============================================================================
// WorkerType
// =============================================================================

enum class WorkerType : int32_t {
    CHIP = 0,  // ChipWorker: L2 hardware device
    SUB = 1,   // SubWorker:  fork/shm Python function
    DIST = 2,  // DistWorker: lower-level node (L4+)
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
// WorkerPayload — dispatched from Scheduler to IWorker
// =============================================================================

struct WorkerPayload {
    DistTaskSlot task_slot = DIST_INVALID_SLOT;
    WorkerType worker_type = WorkerType::CHIP;

    // --- ChipWorker fields (set in PR 2-2) ---
    const void *callable = nullptr;  // ChipCallable buffer ptr
    const void *args = nullptr;      // ChipStorageTaskArgs*
    int32_t block_dim = 1;
    int32_t aicpu_thread_num = 3;
    bool enable_profiling = false;

    // --- SubWorker fields ---
    int32_t callable_id = -1;
    // 'args' pointer above is reused as shm args addr for SubWorker
};

// =============================================================================
// DistTaskSlotState — per-task scheduling bookkeeping
// =============================================================================

struct DistTaskSlotState {
    std::atomic<TaskState> state{TaskState::FREE};

    // --- Fanin (orch writes once; scheduler reads atomically) ---
    int32_t fanin_count{0};
    std::atomic<int32_t> fanin_released{0};  // incremented by each completing producer

    // --- Fanout (protected by fanout_mu) ---
    // orch adds consumers; scheduler traverses on completion
    std::mutex fanout_mu;
    std::vector<DistTaskSlot> fanout_consumers;
    int32_t fanout_total{0};                  // 1 (scope ref) + fanout_consumers.size()
    std::atomic<int32_t> fanout_released{0};  // incremented as each ref is released

    // --- Output buffers (malloced by orch, freed when CONSUMED) ---
    std::vector<void *> output_bufs;  // one entry per output
    std::vector<size_t> output_sizes;
    std::vector<int32_t> output_ownerships;  // DistOutputOwnership per output_bufs[i]; controls reset()-time free

    // --- TensorMap keys registered by this task (for cleanup on CONSUMED) ---
    std::vector<DistTensorKey> output_keys;

    // --- Producer tasks this task depends on (for deferred release) ---
    // When this task reaches COMPLETED, the Scheduler releases one fanout ref
    // on each producer — mirroring L2's "deferred release: walk fanin" step.
    std::vector<DistTaskSlot> fanin_producers;

    // --- Dispatch payload (stored for scheduler dispatch) ---
    WorkerPayload payload;

    // --- Group task (N workers on 1 DAG node) ---
    // args_list stores per-worker args pointers.  size()==1 for normal tasks.
    // Scheduler dispatches worker[i] with args_list[i].
    std::vector<const void *> args_list;
    std::atomic<int32_t> sub_complete_count{0};

    bool is_group() const { return args_list.size() > 1; }
    int32_t group_size() const { return static_cast<int32_t>(args_list.size()); }

    DistTaskSlotState() = default;
    DistTaskSlotState(const DistTaskSlotState &) = delete;
    DistTaskSlotState &operator=(const DistTaskSlotState &) = delete;

    void reset();
};

// =============================================================================
// DistReadyQueue — Orch pushes, Scheduler pops
// =============================================================================

class DistReadyQueue {
public:
    void push(DistTaskSlot slot);

    // Non-blocking: returns false immediately if empty.
    bool try_pop(DistTaskSlot &out);

    // Blocking: waits until a slot is available or shutdown() is called.
    // Returns false only when shutdown and queue is empty.
    bool wait_pop(DistTaskSlot &out);

    void shutdown();

private:
    std::queue<DistTaskSlot> q_;
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

    // Execute one task synchronously.  Called in the worker's own thread.
    // Blocks until the task is complete (mirroring ChipWorker::run()).
    virtual void run(const WorkerPayload &payload) = 0;
};
