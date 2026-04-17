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

#include "worker.h"

#include <pthread.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Fork hygiene
// ---------------------------------------------------------------------------
//
// Thread-pool libraries linked transitively into the Python process (OpenMP,
// OpenBLAS, MKL, BLIS, KMP) spin up worker threads the first time they are
// touched, and those threads do not survive `fork()` cleanly. Pin every
// library we know about to a single thread before Worker.init() spawns its
// own pool, and let KMP tolerate duplicate libomp loads on macOS where
// multiple shared libraries link against their own copy.
//
// pthread_atfork installs handlers once per process that take the locks we
// own in a fixed order on the parent side and release them in reverse on
// both parent and child. The order is coarse-to-fine so nested locking in
// normal paths cannot deadlock the atfork handler. The set of locks is
// intentionally narrow: only those we own. Foreign locks (malloc, GIL) are
// out of scope and handled by the other libraries' own atfork plumbing.
//
// The handler list is process-global (pthread_atfork cannot be unregistered)
// and idempotent — installing it more than once is harmless because the
// actual state it touches is per-Worker. For PR-H we keep the handler empty
// until we have a second Worker-owned lock worth guarding (Allocator::mu_
// is the only one so far, and its lifetime is tied to a Worker that
// always fork-then-init). Registering an empty handler is still valuable as
// a diagnostic hook for fork misuse and as a landing pad for locks added
// in PR-C / PR-D (per-scope rings, per-worker-type queues, callable registry).

namespace {

std::once_flag g_fork_hygiene_once;

void apply_env_defaults_once() {
    // setenv with overwrite=0 leaves user-supplied values intact.
    setenv("OMP_NUM_THREADS", "1", 0);
    setenv("OPENBLAS_NUM_THREADS", "1", 0);
    setenv("MKL_NUM_THREADS", "1", 0);
    setenv("BLIS_NUM_THREADS", "1", 0);
#if defined(__APPLE__)
    setenv("KMP_DUPLICATE_LIB_OK", "TRUE", 0);
#endif
}

void atfork_prepare() {
    // Reserved for locks we own. Acquisition order (coarse-to-fine):
    //   1. callable_registry.mu_   (owned by Python Worker today; future
    //      PR-E will move this to C++)
    //   2. worker_manager.pool_mu_ (PR-D)
    //   3. worker_thread.queue_mu_ (PR-D, per thread)
    //   4. scheduler.completion_mu_
    //   5. allocator.mu_
    //   6. tensormap.mu_
    // Locks are taken in this order and released in reverse on prepare/parent
    // handlers so the handlers never themselves deadlock. Today none of our
    // locks are held across potential fork points, so the handler is empty;
    // keep the landing pad so subsequent PRs can add locks without revisiting
    // the atfork bookkeeping.
}

void atfork_parent() {}

void atfork_child() {}

void install_atfork_once() {
    // Registered once per process; the handlers close over file-scope statics
    // so multiple Worker instances share a single registration.
    static std::once_flag s_atfork;
    std::call_once(s_atfork, []() {
        pthread_atfork(atfork_prepare, atfork_parent, atfork_child);
    });
}

void fork_hygiene_once() {
    std::call_once(g_fork_hygiene_once, []() {
        apply_env_defaults_once();
        install_atfork_once();
    });
}

}  // namespace

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

Worker::Worker(int32_t level, uint64_t heap_ring_size) :
    level_(level) {
    // Fork hygiene runs before the HeapRing mmap so the env-var defaults
    // apply to any thread-pool library that observes them at library init.
    fork_hygiene_once();

    // mmap the HeapRing region here, in the ctor, so Python callers can
    // construct the Worker before fork()-ing children. The children
    // inherit the MAP_SHARED region at the same virtual address.
    allocator_.init(heap_ring_size, ALLOC_TIMEOUT_MS);
}

Worker::~Worker() {
    if (initialized_) close();
}

void Worker::add_worker(WorkerType type, IWorker *worker) {
    if (initialized_) throw std::runtime_error("Worker: add_worker after init");
    if (type == WorkerType::NEXT_LEVEL) manager_.add_next_level(worker);
    else manager_.add_sub(worker);
}

void Worker::add_process_worker(WorkerType type, void *mailbox) {
    if (initialized_) throw std::runtime_error("Worker: add_process_worker after init");
    if (type == WorkerType::NEXT_LEVEL) manager_.add_next_level_process(mailbox);
    else manager_.add_sub_process(mailbox);
}

void Worker::init() {
    if (initialized_) throw std::runtime_error("Worker: already initialized");

    orchestrator_.init(&tensormap_, &allocator_, &scope_, &ready_next_level_queue_, &ready_sub_queue_);

    // Start WorkerManager first — creates WorkerThreads.
    // The on_complete callback routes through the Scheduler's worker_done().
    manager_.start(&allocator_, [this](TaskSlot slot) {
        scheduler_.worker_done(slot);
    });

    Scheduler::Config cfg;
    cfg.ring = &allocator_;
    cfg.ready_next_level_queue = &ready_next_level_queue_;
    cfg.ready_sub_queue = &ready_sub_queue_;
    cfg.manager = &manager_;
    cfg.on_consumed_cb = [this](TaskSlot slot) {
        orchestrator_.on_consumed(slot);
    };

    scheduler_.start(cfg);
    initialized_ = true;
}

void Worker::close() {
    if (!initialized_) return;
    scheduler_.stop();
    manager_.stop();
    allocator_.shutdown();
    initialized_ = false;
}

// =============================================================================
// IWorker::run() — Worker as sub-worker of a higher level (THREAD mode)
// =============================================================================

void Worker::run(uint64_t callable, TaskArgsView args, const ChipCallConfig &config) {
    if (run_callback_) {
        run_callback_(callable, args, config);
    }
}
