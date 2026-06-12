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

#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "remote_endpoint.h"

// ---------------------------------------------------------------------------
// Fork hygiene
// ---------------------------------------------------------------------------
//
// Thread-pool libraries linked transitively into the Python process (OpenMP,
// OpenBLAS, MKL, BLIS, KMP) spin up worker threads on first use, and those
// threads do not survive `fork()` cleanly. Pin each library to a single
// thread before Worker children are forked, and let KMP tolerate duplicate
// libomp loads on macOS where multiple shared libraries link against their
// own copy.

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

void fork_hygiene_once() { std::call_once(g_fork_hygiene_once, apply_env_defaults_once); }

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

void Worker::add_worker(WorkerType type, void *mailbox) {
    if (initialized_) throw std::runtime_error("Worker: add_worker after init");
    if (type == WorkerType::NEXT_LEVEL) manager_.add_next_level(mailbox);
    else manager_.add_sub(mailbox);
}

void Worker::add_remote_l3_socket(
    int32_t endpoint_id, uint64_t session_id, const std::string &transport_name, const std::string &host, uint16_t port,
    const std::string &health_host, uint16_t health_port, double timeout_s
) {
    if (initialized_) throw std::runtime_error("Worker: add_remote_l3_socket after init");
    auto transport = std::make_unique<RemoteL3SocketTransport>(host, port, health_host, health_port, timeout_s);
    transport->expect_hello_ready(session_id, endpoint_id, transport_name);
    manager_.add_next_level_endpoint(
        std::make_unique<RemoteL3Endpoint>(endpoint_id, session_id, transport_name, std::move(transport))
    );
}

void Worker::init() {
    if (initialized_) throw std::runtime_error("Worker: already initialized");

    orchestrator_.init(&tensormap_, &allocator_, &scope_, &ready_next_level_queue_, &ready_sub_queue_, &manager_);

    // Start WorkerManager first — creates WorkerThreads.
    // The on_complete callback routes through the Scheduler's worker_done().
    manager_.start(&allocator_, [this](WorkerCompletion completion) {
        scheduler_.worker_done(std::move(completion));
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
    // Let drain() hold the scheduler's loop mutex across ring teardown so slots
    // aren't freed while the scheduler thread is mid-on_task_complete.
    orchestrator_.set_scheduler_loop_mutex(&scheduler_.loop_mutex());
    initialized_ = true;
}

void Worker::close() {
    if (!initialized_) return;
    scheduler_.stop();
    manager_.stop();
    allocator_.shutdown();
    initialized_ = false;
}
