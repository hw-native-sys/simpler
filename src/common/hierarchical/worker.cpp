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

#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "remote_endpoint.h"
#include "mpi_direct_transport.h"

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

// Appends into a NUL-terminated buffer, truncating rather than overflowing.
// snprintf reports the length it wanted, not the length it wrote, so a
// would-be-longer result saturates `len` at the last writable index.
template <typename... Args>
size_t append_truncating(char *buf, size_t cap, size_t len, const char *fmt, Args... args) {
    if (len + 1 >= cap) return cap - 1;
    const int wanted = std::snprintf(buf + len, cap - len, fmt, args...);
    if (wanted < 0) return len;
    return static_cast<size_t>(wanted) >= cap - len ? cap - 1 : len + static_cast<size_t>(wanted);
}

void report_reservation_stall(void *, const Scheduler::ReservationStallDiagnostic &diagnostic) noexcept {
    // Formatted into automatic storage and emitted with one write(2): the sink
    // is noexcept and runs on the scheduler dispatch path, so it allocates
    // nothing (a throwing allocation here would terminate the process), takes
    // no stdio lock a forked Worker child could inherit held, and leaves
    // nothing running for process exit to race. A message longer than the
    // buffer loses its tail, which for a diagnostic beats any of those.
    char message[512];
    size_t len = append_truncating(
        message, sizeof(message), 0, "[WARN] NEXT_LEVEL group reservation stalled: group_slot=%d busy_target_ids=[",
        diagnostic.group_slot
    );
    for (size_t i = 0; i < diagnostic.busy_target_count; ++i) {
        len = append_truncating(
            message, sizeof(message), len, "%s%d", i == 0 ? "" : ",", diagnostic.busy_target_worker_ids[i]
        );
    }
    len = append_truncating(message, sizeof(message), len, "] idle_targets_with_queued_singles=[");
    for (size_t i = 0; i < diagnostic.idle_queued_target_count; ++i) {
        len = append_truncating(
            message, sizeof(message), len, "%s%d:head_slot=%d", i == 0 ? "" : ",",
            diagnostic.idle_queued_target_worker_ids[i], diagnostic.idle_queued_single_head_slots[i]
        );
    }
    len = append_truncating(message, sizeof(message), len, "]\n");
    // A truncated tail still has to end the line, or this diagnostic runs into
    // whatever writes to stderr next.
    if (len > 0 && message[len - 1] != '\n') message[len - 1] = '\n';
    ssize_t written = ::write(STDERR_FILENO, message, len);
    (void)written;
}

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

void Worker::add_worker(WorkerType type, void *mailbox, int child_pid, uint32_t task_frame_count) {
    if (initialized_) throw std::runtime_error("Worker: add_worker after init");
    if (type == WorkerType::NEXT_LEVEL) {
        manager_.add_next_level(mailbox, child_pid, task_frame_count);
    } else manager_.add_sub(mailbox, child_pid);
}

void Worker::add_next_level_worker(int32_t worker_id, void *mailbox, int child_pid, uint32_t task_frame_count) {
    if (initialized_) throw std::runtime_error("Worker: add_next_level_worker after init");
    manager_.add_next_level_at(worker_id, mailbox, child_pid, task_frame_count);
}

void Worker::add_remote_l3_socket(
    int32_t worker_id, uint64_t session_id, const std::string &transport_name, const std::string &host, uint16_t port,
    const std::string &health_host, uint16_t health_port, double attach_timeout_s, double runtime_timeout_s
) {
    if (initialized_) throw std::runtime_error("Worker: add_remote_l3_socket after init");
    auto transport = std::make_unique<RemoteL3SocketTransport>(
        host, port, health_host, health_port, attach_timeout_s, runtime_timeout_s
    );
    transport->expect_hello_ready(session_id, worker_id, transport_name);
    manager_.add_next_level_endpoint(
        std::make_unique<RemoteL3Endpoint>(worker_id, session_id, transport_name, std::move(transport))
    );
}

void Worker::add_mpi_group_mailbox(
    const std::vector<int32_t> &worker_ids, const std::vector<uint64_t> &session_ids, void *mailbox,
    size_t mailbox_bytes, int mpirun_pid, double runtime_timeout_s
) {
    if (initialized_) throw std::runtime_error("Worker: add_mpi_group_mailbox after init");
    if (worker_ids.empty() || worker_ids.size() != session_ids.size()) {
        throw std::invalid_argument("Worker: MPI group worker_ids and session_ids must have the same non-zero size");
    }
    auto channel = std::make_shared<MpiGroupMailboxChannel>(
        mailbox, mailbox_bytes, static_cast<int32_t>(worker_ids.size()), mpirun_pid, runtime_timeout_s
    );
    for (size_t rank = 0; rank < worker_ids.size(); ++rank) {
        manager_.add_next_level_endpoint(
            std::make_unique<MpiGroupMailboxEndpoint>(
                worker_ids[rank], session_ids[rank], static_cast<int32_t>(rank), channel
            )
        );
    }
}

void Worker::add_remote_l3_mpi(
    int32_t worker_id, uint64_t session_id, const std::string &transport_name,
    const std::shared_ptr<MpiDirectTransportHub> &hub, double attach_timeout_s, double runtime_timeout_s
) {
    if (initialized_) throw std::runtime_error("Worker: add_remote_l3_mpi after init");
    auto transport = std::make_unique<MpiDirectTransport>(hub, worker_id, attach_timeout_s, runtime_timeout_s);
    transport->expect_hello_ready();
    manager_.add_next_level_endpoint(
        std::make_unique<RemoteL3Endpoint>(worker_id, session_id, transport_name, std::move(transport))
    );
}

void Worker::init() {
    if (initialized_) throw std::runtime_error("Worker: already initialized");

    // Start WorkerManager first — creates endpoint lanes.
    // The on_complete callback routes through the Scheduler's worker_done().
    manager_.start(
        &allocator_,
        [this](WorkerCompletion completion) {
            scheduler_.worker_done(std::move(completion));
        },
        [this](WorkerDispatch dispatch) {
            orchestrator_.mark_task_accepted(dispatch.task_slot);
        }
    );
    ready_next_level_queues_.reset(manager_.next_level_worker_ids());
    orchestrator_.init(
        &tensormap_, &allocator_, &scope_, &ready_sub_queue_, &ready_next_level_queues_, &manager_, [this] {
            scheduler_.notify_ready();
        }
    );

    Scheduler::Config cfg;
    cfg.ring = &allocator_;
    cfg.ready_sub_queue = &ready_sub_queue_;
    cfg.ready_next_level_queues = &ready_next_level_queues_;
    cfg.manager = &manager_;
    cfg.enqueue_ready_cb = [this](TaskSlot slot) {
        orchestrator_.enqueue_ready(slot);
    };
    cfg.active_run_cb = [this] {
        return orchestrator_.dispatchable_run_id();
    };
    cfg.preparable_run_cb = [this] {
        return orchestrator_.preparable_run_id();
    };
    cfg.on_consumed_cb = [this](TaskSlot slot) {
        orchestrator_.on_consumed(slot);
    };
    cfg.on_task_failed_cb = [this](TaskSlot slot, const std::string &message) {
        orchestrator_.report_task_error(slot, message);
    };
    cfg.reservation_stall_sink = report_reservation_stall;

    scheduler_.start(cfg);
    // Allocator compaction and scheduler slot access share this mutex.
    orchestrator_.set_scheduler_loop_mutex(&scheduler_.loop_mutex());
    initialized_ = true;
}

void Worker::close() {
    if (!initialized_) return;
    scheduler_.request_stop();
    scheduler_.stop();
    manager_.stop();
    allocator_.shutdown();
    initialized_ = false;
}
