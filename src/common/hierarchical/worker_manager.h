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
 * WorkerManager — worker pool lifecycle and dispatch.
 *
 * Owns WorkerThread instances (one per registered worker).
 * Provides idle-worker selection and dispatch to the Scheduler.
 * The Scheduler drives the DAG; the Manager drives the workers.
 *
 * Each WorkerThread encodes `(callable digest, config, args_blob)` into a
 * pre-forked child's shared-memory mailbox, signals TASK_READY, and
 * spin-polls TASK_DONE. The child process loop (Python) reads the
 * digest, resolves it to a child-local slot, and runs that slot on its
 * `ChipWorker` (NEXT_LEVEL) or registered Python callable (SUB) in its
 * own address space.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "../task_interface/call_config.h"
#include "types.h"

class Ring;  // forward decl — owns the slot state pool
class WorkerManager;

// =============================================================================
// Unified mailbox layout (PROCESS mode)
// =============================================================================
//
// One layout for both NEXT_LEVEL (chip) and SUB workers. SUB children
// read the callable hash digest from MAILBOX_OFF_ARGS, resolve it to a
// private slot, and decode the TaskArgs blob after the digest prefix.

enum class MailboxState : int32_t {
    IDLE = 0,
    TASK_READY = 1,
    TASK_DONE = 2,
    SHUTDOWN = 3,
    CONTROL_REQUEST = 4,
    CONTROL_DONE = 5,
    // Child writes this after its expensive init (ChipWorker::init / inner
    // Worker::init) completes. Parent's _start_hierarchical spin-waits for
    // EVERY chip child to reach INIT_DONE before any dispatch (CTRL_PREPARE
    // or TASK_READY) goes out. This aligns the host-side stream-sync windows
    // across distributed ranks so cross-rank init skew never charges against
    // the per-rank PLATFORM_STREAM_SYNC_TIMEOUT_MS budget (issue #897).
    INIT_DONE = 6,
};

// Sized so the args region can hold any TaskArgs the runtime itself accepts
// (CHIP_MAX_TENSOR_ARGS tensors + CHIP_MAX_SCALAR_ARGS scalars; see the
// static_assert after MAILBOX_ARGS_CAPACITY). 4096 was too tight for composed
// child kernels with many tensor args (issue #1024).
static constexpr size_t MAILBOX_SIZE = 16384;

// Error message region lives at the mailbox tail. 256 B of headroom is
// enough for `<ExceptionType>: <short message>` produced by the child-side
// Python loops; anything longer is truncated + NUL-terminated.
static constexpr size_t MAILBOX_ERROR_MSG_SIZE = 256;

// CallConfig is written/read as a single packed POD block (see call_config.h).
// Both ends transfer it with one memcpy — no per-field offsets to keep in sync.
//
// MAILBOX_OFF_ARGS is derived: round up CallConfig's end to 8 bytes so the
// args blob's first ContinuousTensor.data (uint64_t at OFF_ARGS+8) is 8-byte
// aligned, avoiding SIGBUS on strict-alignment platforms (aarch64 atomics,
// some ARM cores). The control region (CTRL_OFF_ARG0..CTRL_OFF_RESULT) lives
// inside the CallConfig byte range — that's safe because control commands
// and task dispatch are mutually exclusive in time.
static constexpr ptrdiff_t MAILBOX_OFF_STATE = 0;
static constexpr ptrdiff_t MAILBOX_OFF_ERROR = 4;
static constexpr ptrdiff_t MAILBOX_OFF_CALLABLE = 8;  // also: control sub-command (uint64)
static constexpr ptrdiff_t MAILBOX_OFF_CONFIG = 16;
static constexpr ptrdiff_t MAILBOX_OFF_ARGS =
    (MAILBOX_OFF_CONFIG + static_cast<ptrdiff_t>(sizeof(CallConfig)) + 7) & ~ptrdiff_t{7};
static_assert(MAILBOX_OFF_ARGS % 8 == 0, "MAILBOX_OFF_ARGS must be 8-aligned for ContinuousTensor.data");
static_assert(
    MAILBOX_OFF_CONFIG + static_cast<ptrdiff_t>(sizeof(CallConfig)) <= MAILBOX_OFF_ARGS,
    "CallConfig overflows reserved config region"
);
static constexpr ptrdiff_t MAILBOX_OFF_ERROR_MSG =
    static_cast<ptrdiff_t>(MAILBOX_SIZE) - static_cast<ptrdiff_t>(MAILBOX_ERROR_MSG_SIZE);
static constexpr ptrdiff_t MAILBOX_OFF_TASK_CALLABLE_HASH = MAILBOX_OFF_ARGS;
static constexpr ptrdiff_t MAILBOX_OFF_TASK_ARGS_BLOB =
    MAILBOX_OFF_TASK_CALLABLE_HASH + static_cast<ptrdiff_t>(CALLABLE_HASH_DIGEST_SIZE);
static constexpr size_t CTRL_SHM_NAME_BYTES = 32;
static constexpr ptrdiff_t MAILBOX_OFF_CONTROL_CALLABLE_HASH =
    MAILBOX_OFF_ARGS + static_cast<ptrdiff_t>(CTRL_SHM_NAME_BYTES);
static constexpr size_t MAILBOX_ARGS_CAPACITY =
    MAILBOX_SIZE - static_cast<size_t>(MAILBOX_OFF_TASK_ARGS_BLOB) - MAILBOX_ERROR_MSG_SIZE;
static_assert(
    MAILBOX_ARGS_CAPACITY >= TASK_ARGS_BLOB_HEADER_SIZE +
                                 static_cast<size_t>(CHIP_MAX_TENSOR_ARGS) * sizeof(ContinuousTensor) +
                                 static_cast<size_t>(CHIP_MAX_SCALAR_ARGS) * sizeof(uint64_t),
    "mailbox args region must hold the largest TaskArgs blob the runtime accepts (issue #1024)"
);

// Control sub-commands (written at MAILBOX_OFF_CALLABLE when state == CONTROL_*)
static constexpr uint64_t CTRL_MALLOC = 0;
static constexpr uint64_t CTRL_FREE = 1;
static constexpr uint64_t CTRL_COPY_TO = 2;
static constexpr uint64_t CTRL_COPY_FROM = 3;
// Pre-warm a chip child by callable digest; issued at end of init() so the
// first run_prepared does not pay the H2D cost.
static constexpr uint64_t CTRL_PREPARE = 4;
// Dynamic post-init register/unregister of a callable identity. CTRL_REGISTER
// carries (shm_name, blob_size, digest) with bytes staged in POSIX shm by
// the parent; CTRL_UNREGISTER carries digest only.
static constexpr uint64_t CTRL_REGISTER = 5;
static constexpr uint64_t CTRL_UNREGISTER = 6;
// Dynamic per-orch CommDomain allocation/release.  Both carry a pair of
// NUL-terminated POSIX shm names at MAILBOX_OFF_ARGS — first the request shm
// (parent → child: header + rank_ids + buffer_nbytes), second the reply shm
// (child → parent: device_ctx + local_window_base + buffer_ptrs).  RELEASE
// uses only the request shm; no payload comes back beyond status.
static constexpr uint64_t CTRL_ALLOC_DOMAIN = 7;
static constexpr uint64_t CTRL_RELEASE_DOMAIN = 8;
// Establish the base HCCL/sim communicator on this chip.  Driven lazily by
// the orch facade on the first ``allocate_domain`` call.  Payload is a single
// NUL-terminated shm name at MAILBOX_OFF_ARGS that points to a small request
// shm containing (rank: u32, nranks: u32, rootinfo_path: NUL-terminated).
// Caches the comm handle on the chip's ChipWorker so subsequent
// CTRL_ALLOC_DOMAIN calls can find it.
static constexpr uint64_t CTRL_COMM_INIT = 9;
static constexpr uint64_t CTRL_PY_REGISTER = 10;
static constexpr uint64_t CTRL_PY_UNREGISTER = 11;

// Control args reuse the task mailbox region (mutually exclusive with task dispatch):
//   offset 16: uint64 arg0 (size for malloc/register; ptr for free; dst for copy)
//   offset 24: uint64 arg1 (src for copy)
//   offset 32: uint64 arg2 (nbytes for copy)
//   offset 40: uint64 result (returned ptr from malloc)
static constexpr ptrdiff_t CTRL_OFF_ARG0 = 16;
static constexpr ptrdiff_t CTRL_OFF_ARG1 = 24;
static constexpr ptrdiff_t CTRL_OFF_ARG2 = 32;
static constexpr ptrdiff_t CTRL_OFF_RESULT = 40;

// CTRL_REGISTER puts the NUL-terminated POSIX shm name at MAILBOX_OFF_ARGS,
// the exact staged blob size at CTRL_OFF_ARG0, and the callable digest
// immediately after the shm-name slot.
// Fixed-width so the wire layout stays simple; well above the encoded length
// of "simpler-cb-<pid>-<counter>" with pid < 32-bit max.

struct ControlResult {
    std::string worker_type;
    int32_t worker_index{0};
    bool ok{false};
    std::string error_message;
};

// =============================================================================
// WorkerDispatch — per-dispatch handle handed to a WorkerThread.
// =============================================================================
//
// `task_slot` is the slot id; `group_index` is 0 for single tasks and
// 0..group_size-1 for group members. The thread resolves callable / args /
// config by reading `ring->slot_state(task_slot)`.

struct WorkerDispatch {
    TaskSlot task_slot{INVALID_SLOT};
    int32_t group_index{0};
};

// =============================================================================
// WorkerThread — one worker, one std::thread, mailbox-IPC dispatch.
// =============================================================================

class WorkerThread {
public:
    WorkerThread() = default;
    ~WorkerThread() { stop(); }
    WorkerThread(const WorkerThread &) = delete;
    WorkerThread &operator=(const WorkerThread &) = delete;

    // Start the worker thread.
    //
    // `mailbox` points to a MAILBOX_SIZE-byte MAP_SHARED region managed
    // by the Python facade — the real worker (a `ChipWorker` for
    // NEXT_LEVEL, a Python callable for SUB) lives in the forked child
    // and consumes the mailbox via `_chip_process_loop` / `_sub_worker_loop`.
    //
    // `ring` is a borrowed pointer to the engine's slot-state pool —
    // the thread reads callable/args/config from
    // `ring->slot_state(task_slot)` on each dispatch.
    // on_complete(slot) is called (in the WorkerThread) after each run().
    // `manager` is a borrowed pointer used to report dispatch failures
    // (exception_ptr routed out of the worker thread to the orch thread).
    void start(Ring *ring, WorkerManager *manager, const std::function<void(TaskSlot)> &on_complete, void *mailbox);

    // Enqueue a dispatch for the worker. Non-blocking.
    void dispatch(WorkerDispatch d);

    // True if the worker has no active task.
    bool idle() const { return idle_.load(std::memory_order_acquire); }

    void stop();

    // Write SHUTDOWN to the mailbox so the child process exits its loop.
    // Does NOT waitpid — the Python facade owns the child PID.
    void shutdown_child();

    // Memory control — callable from the orch thread while the worker
    // thread may be running a task. Issues a control command via the
    // mailbox and blocks until the child responds.
    //
    // The mailbox is a single shared region; dispatch_process and the
    // control_* methods both write its state field. They serialize on
    // `mailbox_mu_` so a control request issued mid-dispatch waits for
    // TASK_DONE before claiming the mailbox.
    uint64_t control_malloc(size_t size);
    void control_free(uint64_t ptr);
    void control_copy_to(uint64_t dst, uint64_t src, size_t size);
    void control_copy_from(uint64_t dst, uint64_t src, size_t size);

    // Pre-warm a chip child by triggering prepare_callable for the digest's
    // target-local slot via CTRL_PREPARE.
    void control_prepare(const uint8_t *digest);

    // Dynamic post-init register/unregister of a ChipCallable identity.
    // `shm_name` is the (NUL-terminated, ≤ CTRL_SHM_NAME_BYTES-1) POSIX shm
    // name where the ChipCallable bytes are staged; `blob_size` is the exact
    // byte span to read from that shm. Both methods hold mailbox_mu_, so a
    // CTRL_REGISTER concurrent with dispatch_process waits for the in-flight
    // TASK_DONE before claiming the mailbox.
    void control_register(const char *shm_name, size_t blob_size, const uint8_t *digest);
    void control_unregister(const uint8_t *digest);
    void control_generic(uint64_t sub_cmd, const char *shm_name, double timeout_s, const uint8_t *digest);

    // Dynamic CommDomain allocate / release.  `request_shm_name` carries the
    // request payload (header + rank_ids + buffer_nbytes); for alloc the child
    // writes its (device_ctx, local_window_base, buffer_ptrs) into
    // `reply_shm_name`.  Both names are NUL-terminated and ≤
    // CTRL_SHM_NAME_BYTES-1.  Holds mailbox_mu_ so it serialises with task
    // dispatch on the same chip mailbox.
    void control_alloc_domain(const char *request_shm_name, const char *reply_shm_name);
    void control_release_domain(const char *request_shm_name);

    // Lazy comm_init driver — payload shm carries (rank, nranks, rootinfo_path).
    // Caller dispatches in parallel to every chip; child runs cw.comm_init.
    void control_comm_init(const char *request_shm_name);

private:
    Ring *ring_{nullptr};
    WorkerManager *manager_{nullptr};
    void *mailbox_{nullptr};
    std::function<void(TaskSlot)> on_complete_;

    std::thread thread_;
    std::queue<WorkerDispatch> queue_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool shutdown_{false};
    std::atomic<bool> idle_{true};

    // Serializes parent-side mailbox access between this WorkerThread's
    // dispatch loop and the orch-thread control_* path. Per-WorkerThread,
    // so different workers can dispatch in parallel.
    std::mutex mailbox_mu_;
    bool mailbox_control_timed_out_{false};

    void loop();
    void dispatch_process(TaskSlotState &s, int32_t group_index);

    // Common tail for the four control_* methods. Caller writes the args
    // region and holds `mailbox_mu_`; this helper signals the child,
    // spin-polls CONTROL_DONE, and throws on a non-zero child error code.
    void run_control_command(const char *op_name, double timeout_s = -1.0);

    char *mbox() const { return static_cast<char *>(mailbox_); }
    MailboxState read_mailbox_state() const;
    void write_mailbox_state(MailboxState s);
};

// =============================================================================
// WorkerManager — worker pool lifecycle and dispatch
// =============================================================================

class WorkerManager {
public:
    using OnCompleteFn = std::function<void(TaskSlot)>;

    // Register a worker. `mailbox` is a MAILBOX_SIZE-byte MAP_SHARED
    // region; the real worker (a `ChipWorker` for NEXT_LEVEL, a Python
    // callable for SUB) lives in the forked child.
    void add_next_level(void *mailbox);
    void add_sub(void *mailbox);

    void start(Ring *ring, const OnCompleteFn &on_complete);
    void stop();

    WorkerThread *pick_idle(WorkerType type) const;
    std::vector<WorkerThread *> pick_n_idle(WorkerType type, int n) const;

    // Direct index into the worker pool by logical id (0-based).
    WorkerThread *get_worker(WorkerType type, int logical_id) const;

    // Pick one idle worker NOT in `exclude`. Returns nullptr if none available.
    WorkerThread *pick_idle_excluding(WorkerType type, const std::vector<WorkerThread *> &exclude) const;

    bool any_busy() const;

    // Forward CTRL_PREPARE to a specific NEXT_LEVEL worker. Thin wrapper
    // over WorkerThread::control_prepare; exposed at manager level so the
    // Python facade can prewarm without reaching into individual WorkerThreads.
    void control_prepare(int worker_id, const uint8_t *digest);

    // Forward CTRL_ALLOC_DOMAIN / CTRL_RELEASE_DOMAIN to a specific NEXT_LEVEL
    // worker.  Used by the Python orch facade to drive collective domain
    // allocation across a subset of chips — caller dispatches to each
    // participating chip and joins on completion.
    void control_alloc_domain(int worker_id, const char *request_shm_name, const char *reply_shm_name);
    void control_release_domain(int worker_id, const char *request_shm_name);
    void control_comm_init(int worker_id, const char *request_shm_name);
    ControlResult
    control_digest_only(WorkerType type, int worker_id, uint64_t sub_cmd, const uint8_t *digest, double timeout_s);

    // Broadcast CTRL_REGISTER for `digest` to every NEXT_LEVEL worker in
    // parallel. Stages `blob_size` bytes from `blob_ptr` into a per-call
    // POSIX shm under name "simpler-cb-<pid>-<counter>", spawns one
    // std::thread per WorkerThread, and joins. Returns one ControlResult per
    // target so the Python facade can clean up only targets that confirmed
    // install/refcount increment on a partial failure.
    std::vector<ControlResult> broadcast_register_all(const void *blob_ptr, size_t blob_size, const uint8_t *digest);

    // Best-effort: broadcast CTRL_UNREGISTER for `digest` to every NEXT_LEVEL
    // worker in parallel. Returns a vector of per-worker error strings
    // (empty on full success). Caller decides whether to log / surface.
    std::vector<std::string> broadcast_unregister_all(const uint8_t *digest);
    std::vector<ControlResult> broadcast_control_all(
        WorkerType type, uint64_t sub_cmd, const void *payload, size_t payload_size, const uint8_t *digest,
        double timeout_s
    );

    // Write SHUTDOWN to every registered mailbox.
    void shutdown_children();

    // Error propagation: first dispatch failure from any WorkerThread wins.
    // The orch thread inspects via `has_error()` / `take_error()` and
    // clears between Worker.run() invocations via `clear_error()`.
    void report_error(std::exception_ptr e);
    bool has_error() const { return has_error_.load(std::memory_order_acquire); }
    std::exception_ptr take_error();
    void clear_error();

private:
    std::vector<void *> next_level_entries_;
    std::vector<void *> sub_entries_;

    std::vector<std::unique_ptr<WorkerThread>> next_level_threads_;
    std::vector<std::unique_ptr<WorkerThread>> sub_threads_;

    // First-error-wins exception slot. Written under err_mu_ by
    // WorkerThread::loop() catch handlers; read by the orch thread at
    // submit_*/drain boundaries.
    std::atomic<bool> has_error_{false};
    mutable std::mutex err_mu_;
    std::exception_ptr first_error_;
};
