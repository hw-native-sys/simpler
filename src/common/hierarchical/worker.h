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
 * Worker — top-level distributed worker node.
 *
 * Worker is the implementation of one level in the hierarchy (L3, L4, …).
 * It contains the full scheduling engine (TensorMap, Allocator, Scope,
 * Orchestrator, Scheduler) and a set of sub-workers (each a forked Python
 * child reachable via a shared-memory mailbox) it dispatches to.
 *
 * Public surface:
 *   - add_worker(type, mailbox)    — register a sub-worker (before init).
 *                                    `mailbox` is a MAILBOX_SIZE-byte
 *                                    MAP_SHARED region; the real IWorker
 *                                    lives in the forked child.
 *   - init() / close()             — lifecycle
 *   - get_orchestrator()           — accessor used by the Python facade
 *                                    (scope_begin / drain / scope_end live
 *                                     on the Orchestrator, not here)
 *
 * Worker holds no submit / scope / drain / active-task bookkeeping — those
 * concepts belong to Orchestrator.
 *
 * Construction is separated from `init()` so Python callers can mmap the
 * HeapRing in the parent process *before* forking children (children see the
 * MAP_SHARED region at the same virtual address). Start the scheduler and
 * WorkerThreads with `init()` only after forks have happened.
 */

#pragma once

#include <cstdint>
#include <memory>

#include "ring.h"
#include "orchestrator.h"
#include "scheduler.h"
#include "scope.h"
#include "tensormap.h"
#include "types.h"
#include "worker_manager.h"

class Worker {
public:
    // Construct a Worker for hierarchy `level`. `heap_ring_size` is the
    // MAP_SHARED|MAP_ANONYMOUS region handed out by the Orchestrator for
    // auto-allocated OUTPUT tensors and `orch.alloc()` buffers.
    //
    // The heap is mmap'd here (before any fork) so forked child workers
    // inherit the same mapping. Thread-hostile hygiene (setenv of
    // OMP/MKL/BLIS/OPENBLAS thread-count knobs and the pthread_atfork
    // installation) also runs in the ctor, still in the parent, before
    // child forks.
    explicit Worker(int32_t level, uint64_t heap_ring_size = DEFAULT_HEAP_RING_SIZE);
    ~Worker();

    Worker(const Worker &) = delete;
    Worker &operator=(const Worker &) = delete;

    // Register a sub-worker before calling init(). `mailbox` is a
    // MAILBOX_SIZE-byte MAP_SHARED region; the real IWorker lives in the
    // forked child and consumes the mailbox via the Python child loop.
    void add_worker(WorkerType type, void *mailbox);

    // Start the scheduler thread. Must be called AFTER the parent has forked
    // any child workers — init() spins up threads in the parent that would
    // otherwise be accidentally inherited across fork.
    void init();

    // Shut down the Scheduler thread and release resources.
    void close();

    // Accessor: the Orchestrator handle used by the user's orch fn. Valid
    // only between init() and close().
    Orchestrator &get_orchestrator() { return orchestrator_; }

private:
    int32_t level_;
    bool initialized_{false};

    // --- Scheduling engine components ---
    // Per-task slot state lives inside `allocator_` (Ring) — Orchestrator
    // and Scheduler access it via `allocator_.slot_state(id)`. The slot
    // itself is the dispatch state; no separate fixed-size slots array.
    TensorMap tensormap_;
    Ring allocator_;
    Scope scope_;
    // One ready queue per WorkerType. Submit routes by s.worker_type;
    // the Scheduler drains each queue independently so saturation of
    // one pool cannot head-of-line-block the other.
    ReadyQueue ready_next_level_queue_;
    ReadyQueue ready_sub_queue_;
    Orchestrator orchestrator_;
    Scheduler scheduler_;
    WorkerManager manager_;
};
