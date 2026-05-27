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
 *                                    MAP_SHARED region; the real worker
 *                                    (a `ChipWorker` for NEXT_LEVEL, a
 *                                    Python callable for SUB) lives in
 *                                    the forked child.
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
#include <string>
#include <vector>

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
    // MAILBOX_SIZE-byte MAP_SHARED region; the real worker (a `ChipWorker`
    // for NEXT_LEVEL, a Python callable for SUB) lives in the forked
    // child and consumes the mailbox via the Python child loop.
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

    // Forward CTRL_PREPARE to a specific NEXT_LEVEL worker (prewarm path
    // used by the Python facade at end of _start_hierarchical).
    void control_prepare(int worker_id, int32_t cid) { manager_.control_prepare(worker_id, cid); }

    // Drive a single chip child through one CommDomain alloc / release.  The
    // Python orch facade is expected to call this on every participating chip
    // in parallel (one thread per chip) so the child-side `file_barrier` can
    // converge.  Blocks per-chip until CONTROL_DONE; raises on child error.
    void control_alloc_domain(int worker_id, const std::string &request_shm_name, const std::string &reply_shm_name) {
        manager_.control_alloc_domain(worker_id, request_shm_name.c_str(), reply_shm_name.c_str());
    }
    void control_release_domain(int worker_id, const std::string &request_shm_name) {
        manager_.control_release_domain(worker_id, request_shm_name.c_str());
    }
    void control_comm_init(int worker_id, const std::string &request_shm_name) {
        manager_.control_comm_init(worker_id, request_shm_name.c_str());
    }
    uint64_t control_open_mapped_region(int worker_id, uint64_t data_bytes, uint32_t signal_count, uint32_t flags) {
        return manager_.control_open_mapped_region(worker_id, data_bytes, signal_count, flags);
    }
    void control_close_mapped_region(int worker_id, uint64_t handle) {
        manager_.control_close_mapped_region(worker_id, handle);
    }
    void control_mapped_region_payload(int worker_id, uint64_t sub_cmd, const std::string &shm_name) {
        manager_.control_mapped_region_payload(worker_id, sub_cmd, shm_name.c_str());
    }
    void control_mapped_region_notify(int worker_id, uint64_t handle, uint32_t signal_id, uint32_t value) {
        manager_.control_mapped_region_notify(worker_id, handle, signal_id, value);
    }
    void control_mapped_region_wait(
        int worker_id, uint64_t handle, uint32_t signal_id, uint32_t target, uint32_t timeout_us
    ) {
        manager_.control_mapped_region_wait(worker_id, handle, signal_id, target, timeout_us);
    }

    // Broadcast CTRL_REGISTER / CTRL_UNREGISTER for a ChipCallable cid to
    // every NEXT_LEVEL child in parallel. `blob_ptr`/`blob_size` describe
    // the contiguous ChipCallable bytes (see PyChipCallable::buffer_ptr /
    // buffer_size). Throws on any child error for register; unregister is
    // best-effort and returns the per-child error list.
    void broadcast_register_all(int32_t cid, uint64_t blob_ptr, uint64_t blob_size) {
        manager_.broadcast_register_all(cid, reinterpret_cast<const void *>(blob_ptr), static_cast<size_t>(blob_size));
    }
    std::vector<std::string> broadcast_unregister_all(int32_t cid) { return manager_.broadcast_unregister_all(cid); }
    std::vector<ControlResult> broadcast_control_all(
        WorkerType type, uint64_t sub_cmd, int32_t cid, const void *payload, size_t payload_size, double timeout_s
    ) {
        return manager_.broadcast_control_all(type, sub_cmd, cid, payload, payload_size, timeout_s);
    }

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
