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
 * DistWorker — top-level distributed worker node.
 *
 * DistWorker is the implementation of one level in the hierarchy (L3, L4, …).
 * From the level above it looks like an IWorker; internally it contains the
 * full scheduling engine (TensorMap, Allocator, Scope, Orchestrator, Scheduler)
 * and a set of sub-IWorkers it dispatches to.
 *
 * Public surface:
 *   - add_worker(type, IWorker*)  — register sub-workers (before init)
 *   - init() / close()             — lifecycle
 *   - get_orchestrator()           — accessor used by Worker::run / Python facade
 *                                    (scope_begin / drain / scope_end live on the
 *                                     Orchestrator, not here)
 *   - run(payload)                 — IWorker entry (placeholder for L4+ recursion)
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

#include "dist_ring.h"
#include "dist_orchestrator.h"
#include "dist_scheduler.h"
#include "dist_scope.h"
#include "dist_tensormap.h"
#include "dist_types.h"
#include "dist_worker_manager.h"

class DistWorker : public IWorker {
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
    explicit DistWorker(int32_t level, uint64_t heap_ring_size = DIST_DEFAULT_HEAP_RING_SIZE);
    ~DistWorker() override;

    DistWorker(const DistWorker &) = delete;
    DistWorker &operator=(const DistWorker &) = delete;

    // Register sub-workers before calling init().
    // THREAD mode — parent calls worker->run() directly.
    void add_worker(WorkerType type, IWorker *worker);

    // PROCESS mode — parent writes the unified mailbox; a pre-forked child
    // process reads it and runs the real IWorker in its own address space.
    // `mailbox` must point to a DIST_MAILBOX_SIZE-byte MAP_SHARED region.
    void add_process_worker(WorkerType type, void *mailbox);

    // Start the scheduler thread. Must be called AFTER the parent has forked
    // any child workers — init() spins up threads in the parent that would
    // otherwise be accidentally inherited across fork.
    void init();

    // Shut down the Scheduler thread and release resources.
    void close();

    // Accessor: the Orchestrator handle used by the user's orch fn. Valid
    // only between init() and close().
    DistOrchestrator &get_orchestrator() { return orchestrator_; }

    // IWorker — used when this DistWorker is itself a sub-worker of L4+.
    // In THREAD mode, the parent's WorkerThread calls run() directly;
    // run() invokes run_callback_ which acquires the GIL and delegates
    // to the Python Worker._run_as_child method (approach (b): Python
    // callback — simpler than full C++ registry lookup).
    void run(uint64_t callable, TaskArgsView args, const ChipCallConfig &config) override;

    using RunCallback = std::function<void(uint64_t, TaskArgsView, const ChipCallConfig &)>;
    void set_run_callback(RunCallback cb) { run_callback_ = std::move(cb); }

private:
    RunCallback run_callback_;
    int32_t level_;
    bool initialized_{false};

    // --- Scheduling engine components ---
    // Per-task slot state lives inside `allocator_` (DistRing) — Orchestrator
    // and Scheduler access it via `allocator_.slot_state(id)`. No separate
    // fixed-size slots array at L3 (see plan Allowed Exception #6).
    DistTensorMap tensormap_;
    DistRing allocator_;
    DistScope scope_;
    // Strict-4: one ready queue per WorkerType. Submit routes by
    // s.worker_type; the Scheduler drains each queue independently so
    // saturation of one pool cannot head-of-line-block the other.
    DistReadyQueue ready_next_level_queue_;
    DistReadyQueue ready_sub_queue_;
    DistOrchestrator orchestrator_;
    DistScheduler scheduler_;
    DistWorkerManager manager_;
};
