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
    void add_next_level_worker(int32_t worker_id, void *mailbox);

    // Register a REMOTE_L3 endpoint only after its session runner completed
    // prestart and reported HELLO READY on the command lane.
    void add_remote_l3_socket(
        int32_t worker_id, uint64_t session_id, const std::string &transport_name, const std::string &host,
        uint16_t port, const std::string &health_host, uint16_t health_port, double timeout_s
    );

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
    void control_prepare(int worker_id, const uint8_t *digest) { manager_.control_prepare(worker_id, digest); }

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
    void control_l3_l2_orch_comm_init(int worker_id, const std::string &control_shm_name) {
        manager_.control_l3_l2_orch_comm_init(worker_id, control_shm_name.c_str());
    }
    void
    control_l3_l2_region_create(int worker_id, const std::string &request_shm_name, const std::string &reply_shm_name) {
        manager_.control_l3_l2_region_create(worker_id, request_shm_name.c_str(), reply_shm_name.c_str());
    }
    void control_l3_l2_region_release(int worker_id, uint64_t region_id) {
        manager_.control_l3_l2_region_release(worker_id, region_id);
    }

    ControlResult
    control_digest_only(WorkerType type, int worker_id, uint64_t sub_cmd, const uint8_t *digest, double timeout_s) {
        return manager_.control_digest_only(type, worker_id, sub_cmd, digest, timeout_s);
    }
    ControlResult remote_prepare_register(
        int worker_id, remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const void *payload,
        size_t payload_size, const uint8_t *digest
    ) {
        return manager_.control_remote_prepare_register(
            worker_id, target_registry, callable_kind, payload, payload_size, digest
        );
    }
    ControlResult remote_commit_register(
        int worker_id, remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind,
        const uint8_t *digest
    ) {
        return manager_.control_remote_commit_register(worker_id, target_registry, callable_kind, digest);
    }
    ControlResult remote_abort_register(
        int worker_id, remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind,
        const uint8_t *digest
    ) {
        return manager_.control_remote_abort_register(worker_id, target_registry, callable_kind, digest);
    }
    ControlResult remote_unregister(
        int worker_id, remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind,
        const uint8_t *digest
    ) {
        return manager_.control_remote_unregister(worker_id, target_registry, callable_kind, digest);
    }
    RemoteBufferHandle remote_malloc(int worker_id, size_t size) {
        return manager_.control_remote_malloc(worker_id, size);
    }
    void remote_free(const RemoteBufferHandle &handle) { manager_.control_remote_free(handle); }
    void remote_copy_to(const RemoteBufferHandle &handle, uint64_t offset, const void *src, size_t size) {
        manager_.control_remote_copy_to(handle, offset, src, size);
    }
    void remote_copy_from(void *dst, const RemoteBufferHandle &handle, uint64_t offset, size_t size) {
        manager_.control_remote_copy_from(dst, handle, offset, size);
    }
    RemoteBufferExport remote_export(
        const RemoteBufferHandle &handle, uint64_t offset, uint64_t size, uint32_t access_flags,
        const std::string &transport_profile
    ) {
        return manager_.control_remote_export(handle, offset, size, access_flags, transport_profile);
    }
    RemoteBufferHandle
    remote_import(int32_t importer_worker_id, const RemoteBufferExport &export_desc, uint32_t requested_access_flags) {
        return manager_.control_remote_import(importer_worker_id, export_desc, requested_access_flags);
    }
    void remote_release_import(const RemoteBufferHandle &handle) { manager_.control_remote_release_import(handle); }

    // Broadcast CTRL_REGISTER / CTRL_UNREGISTER for a ChipCallable digest to
    // every NEXT_LEVEL child in parallel. `blob_ptr`/`blob_size` describe
    // the contiguous ChipCallable bytes (see PyChipCallable::buffer_ptr /
    // buffer_size). Register returns per-child status so the facade can
    // reverse only confirmed installs; unregister remains best-effort.
    std::vector<ControlResult> broadcast_register_all(uint64_t blob_ptr, uint64_t blob_size, const uint8_t *digest) {
        return manager_.broadcast_register_all(
            reinterpret_cast<const void *>(blob_ptr), static_cast<size_t>(blob_size), digest
        );
    }
    std::vector<std::string> broadcast_unregister_all(const uint8_t *digest) {
        return manager_.broadcast_unregister_all(digest);
    }
    std::vector<ControlResult> broadcast_control_all(
        WorkerType type, uint64_t sub_cmd, const void *payload, size_t payload_size, const uint8_t *digest,
        double timeout_s
    ) {
        return manager_.broadcast_control_all(type, sub_cmd, payload, payload_size, digest, timeout_s);
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
