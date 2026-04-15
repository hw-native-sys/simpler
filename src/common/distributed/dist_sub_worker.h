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
 * DistSubWorker — C++ side of the fork/shm SubWorker.
 *
 * Each SubWorker corresponds to one forked Python child process.  The fork and
 * the Python callable loop are managed from Python (HostWorker.__init__).  This
 * class implements IWorker so the Scheduler's WorkerThread can call run() and
 * block until the forked process signals TASK_DONE.
 *
 * run() flow (executes in WorkerThread's own thread, not the Scheduler thread):
 *   1. Write callable_id to mailbox
 *   2. write_state(TASK_READY)   — release store: child sees consistent mailbox
 *   3. Spin-poll until read_state() == TASK_DONE  — blocking in WorkerThread
 *   4. write_state(IDLE)         — reset for next task
 *   5. return  →  WorkerThread pushes to completion_queue + notifies Scheduler
 *
 * Mailbox layout (DIST_SUB_MAILBOX_SIZE bytes):
 *   offset  0  int32  state         IDLE=0, TASK_READY=1, TASK_DONE=2, SHUTDOWN=3
 *   offset  4  int32  callable_id
 *   offset 24  int32  error_code    0=ok
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>

#include "dist_types.h"

static constexpr size_t DIST_SUB_MAILBOX_SIZE = 256;  // 4 cache lines

enum class SubMailboxState : int32_t {
    IDLE = 0,
    TASK_READY = 1,
    TASK_DONE = 2,
    SHUTDOWN = 3,
};

class DistSubWorker : public IWorker {
public:
    // mailbox_ptr must point to DIST_SUB_MAILBOX_SIZE bytes of shared memory
    // (allocated from Python before fork).
    explicit DistSubWorker(void *mailbox_ptr);

    // IWorker: write mailbox → spin-poll TASK_DONE → reset IDLE. `callable`
    // carries the registered callable id (uint64; low 32 bits = int32 cid).
    // `args` is currently ignored — the child's py registry receives no
    // args yet; PR-E threads the TaskArgsView into the Python call.
    // Blocks in the caller's thread (WorkerThread), not the Scheduler thread.
    void run(uint64_t callable, TaskArgsView args, const ChipCallConfig &config) override;

    // Signal the child process to exit (SHUTDOWN state).
    void shutdown();

private:
    void *mailbox_;

    volatile int32_t *state_ptr() const;
    volatile int32_t *callable_id_ptr() const;

    SubMailboxState read_state() const;
    void write_state(SubMailboxState s);
};
