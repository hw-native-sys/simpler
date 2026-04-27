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
 * Each WorkerThread operates in one of two modes:
 *
 *   THREAD  — calls `worker_->run(callable, view, config)` directly in
 *             the parent process.
 *   PROCESS — encodes `(callable, config, args_blob)` into a pre-forked
 *             child's shared-memory mailbox, signals TASK_READY, and
 *             spin-polls TASK_DONE. The child process loop (Python) reads
 *             the mailbox and calls the appropriate IWorker / Python
 *             callable in its own address space.
 *
 * PROCESS mode absorbs the logic that used to live in the standalone
 * `ChipProcess` and `SubWorker` classes (deleted in PR-D-2).
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
// read `callable` as a uint64 encoding the callable_id and ignore
// config + args_blob. Matches the former ChipProcess layout at the
// byte level so the chip child loop in Python needs no offset changes.

enum class MailboxState : int32_t {
    IDLE = 0,
    TASK_READY = 1,
    TASK_DONE = 2,
    SHUTDOWN = 3,
    CONTROL_REQUEST = 4,
    CONTROL_DONE = 5,
};

static constexpr size_t MAILBOX_SIZE = 4096;

// Error message region lives at the mailbox tail. 256 B of headroom is
// enough for `<ExceptionType>: <short message>` produced by the child-side
// Python loops; anything longer is truncated + NUL-terminated.
static constexpr size_t MAILBOX_ERROR_MSG_SIZE = 256;

// CallConfig is written/read as a single packed POD block (see call_config.h).
// Both ends transfer it with one memcpy — no per-field offsets to keep in sync.
//
// MAILBOX_OFF_ARGS is held at 64 (rather than packed flush against CONFIG) for
// two reasons: (a) the args blob is `[T:i32][S:i32][ContinuousTensor[T]][u64[S]]`,
// so the first ContinuousTensor's `uint64_t data` field lands at OFF_ARGS+8 —
// it must be 8-byte aligned to avoid SIGBUS on strict-alignment platforms
// (aarch64 atomics, some ARM cores); (b) the control region (CTRL_OFF_ARG0..
// CTRL_OFF_RESULT) overlaps offsets 16–47 and is mutually exclusive with task
// dispatch, but using offset 64 keeps the two regions visually disjoint.
static constexpr ptrdiff_t MAILBOX_OFF_STATE = 0;
static constexpr ptrdiff_t MAILBOX_OFF_ERROR = 4;
static constexpr ptrdiff_t MAILBOX_OFF_CALLABLE = 8;  // also: control sub-command (uint64)
static constexpr ptrdiff_t MAILBOX_OFF_CONFIG = 16;
static constexpr ptrdiff_t MAILBOX_OFF_ARGS = 64;
static_assert(
    MAILBOX_OFF_CONFIG + static_cast<ptrdiff_t>(sizeof(CallConfig)) <= MAILBOX_OFF_ARGS,
    "CallConfig overflows reserved config region"
);
static_assert(MAILBOX_OFF_ARGS % 8 == 0, "MAILBOX_OFF_ARGS must be 8-aligned for ContinuousTensor.data");
static constexpr ptrdiff_t MAILBOX_OFF_ERROR_MSG =
    static_cast<ptrdiff_t>(MAILBOX_SIZE) - static_cast<ptrdiff_t>(MAILBOX_ERROR_MSG_SIZE);
static constexpr size_t MAILBOX_ARGS_CAPACITY =
    MAILBOX_SIZE - static_cast<size_t>(MAILBOX_OFF_ARGS) - MAILBOX_ERROR_MSG_SIZE;

// Control sub-commands (written at MAILBOX_OFF_CALLABLE when state == CONTROL_*)
static constexpr uint64_t CTRL_MALLOC = 0;
static constexpr uint64_t CTRL_FREE = 1;
static constexpr uint64_t CTRL_COPY_TO = 2;
static constexpr uint64_t CTRL_COPY_FROM = 3;

// Control args reuse the task mailbox region (mutually exclusive with task dispatch):
//   offset 16: uint64 arg0 (size for malloc; ptr for free; dst for copy)
//   offset 24: uint64 arg1 (src for copy)
//   offset 32: uint64 arg2 (nbytes for copy)
//   offset 40: uint64 result (returned ptr from malloc)
static constexpr ptrdiff_t CTRL_OFF_ARG0 = 16;
static constexpr ptrdiff_t CTRL_OFF_ARG1 = 24;
static constexpr ptrdiff_t CTRL_OFF_ARG2 = 32;
static constexpr ptrdiff_t CTRL_OFF_RESULT = 40;

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
// WorkerThread — one worker, one std::thread, two execution modes.
// =============================================================================

class WorkerThread {
public:
    enum class Mode { THREAD, PROCESS };

    WorkerThread() = default;
    ~WorkerThread() { stop(); }
    WorkerThread(const WorkerThread &) = delete;
    WorkerThread &operator=(const WorkerThread &) = delete;

    // Start the worker thread.
    //
    // THREAD mode: `worker` is called directly via `worker->run(...)`.
    //   `mailbox` must be nullptr.
    //
    // PROCESS mode: `worker` is nullptr (the real IWorker lives in the
    //   forked child). `mailbox` points to a MAILBOX_SIZE-byte
    //   MAP_SHARED region managed by the Python facade.
    //
    // `ring` is a borrowed pointer to the engine's slot-state pool —
    // the thread reads callable/args/config from
    // `ring->slot_state(task_slot)` on each dispatch.
    // on_complete(slot) is called (in the WorkerThread) after each run().
    // `manager` is a borrowed pointer used to report dispatch failures
    // (exception_ptr routed out of the worker thread to the orch thread).
    void start(
        Mode mode, IWorker *worker, Ring *ring, WorkerManager *manager,
        const std::function<void(TaskSlot)> &on_complete, void *mailbox = nullptr
    );

    // Enqueue a dispatch for the worker. Non-blocking.
    void dispatch(WorkerDispatch d);

    // True if the worker has no active task.
    bool idle() const { return idle_.load(std::memory_order_acquire); }

    void stop();

    // PROCESS mode only: write SHUTDOWN to the mailbox so the child
    // process exits its loop. No-op in THREAD mode. Does NOT waitpid —
    // the Python facade owns the child PID.
    void shutdown_child();

    // Memory control — callable from the orch thread while the worker
    // thread may be running a task (MemoryAllocator is mutex-protected).
    // THREAD mode: direct call on the ChipWorker.
    // PROCESS mode: control command via mailbox (blocks until child responds).
    uint64_t control_malloc(size_t size);
    void control_free(uint64_t ptr);
    void control_copy_to(uint64_t dst, uint64_t src, size_t size);
    void control_copy_from(uint64_t dst, uint64_t src, size_t size);

private:
    Mode mode_{Mode::THREAD};
    IWorker *worker_{nullptr};
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

    void loop();
    void dispatch_thread(TaskSlotState &s, int32_t group_index);
    void dispatch_process(TaskSlotState &s, int32_t group_index);

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

    // THREAD mode: worker is called directly.
    void add_next_level(IWorker *worker);
    void add_sub(IWorker *worker);

    // PROCESS mode: mailbox is a MAILBOX_SIZE-byte MAP_SHARED region.
    // Worker is nullptr (child has its own).
    void add_next_level_process(void *mailbox);
    void add_sub_process(void *mailbox);

    void start(Ring *ring, const OnCompleteFn &on_complete);
    void stop();

    WorkerThread *pick_idle(WorkerType type) const;
    std::vector<WorkerThread *> pick_n_idle(WorkerType type, int n) const;

    // Direct index into the worker pool by logical id (0-based).
    WorkerThread *get_worker(WorkerType type, int logical_id) const;

    // Pick one idle worker NOT in `exclude`. Returns nullptr if none available.
    WorkerThread *pick_idle_excluding(WorkerType type, const std::vector<WorkerThread *> &exclude) const;

    bool any_busy() const;

    // Write SHUTDOWN to every PROCESS-mode mailbox.
    void shutdown_children();

    // Error propagation: first dispatch failure from any WorkerThread wins.
    // The orch thread inspects via `has_error()` / `take_error()` and
    // clears between Worker.run() invocations via `clear_error()`.
    void report_error(std::exception_ptr e);
    bool has_error() const { return has_error_.load(std::memory_order_acquire); }
    std::exception_ptr take_error();
    void clear_error();

private:
    struct WorkerEntry {
        IWorker *worker;  // nullptr for PROCESS mode
        WorkerThread::Mode mode;
        void *mailbox;  // nullptr for THREAD mode
    };

    std::vector<WorkerEntry> next_level_entries_;
    std::vector<WorkerEntry> sub_entries_;

    std::vector<std::unique_ptr<WorkerThread>> next_level_threads_;
    std::vector<std::unique_ptr<WorkerThread>> sub_threads_;

    // First-error-wins exception slot. Written under err_mu_ by
    // WorkerThread::loop() catch handlers; read by the orch thread at
    // submit_*/drain boundaries.
    std::atomic<bool> has_error_{false};
    mutable std::mutex err_mu_;
    std::exception_ptr first_error_;
};
