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
 * DistChipProcess — C++ side of the fork/shm ChipWorker.
 *
 * Each DistChipProcess corresponds to one forked child process that loads
 * host_runtime.so in its own address space (full process isolation).
 * The fork and ChipWorker init are managed from Python (Worker.__init__).
 *
 * run() flow (executes in WorkerThread's own thread, not the Scheduler thread):
 *   1. Write callable_ptr, config fields to mailbox
 *   2. memcpy ChipStorageTaskArgs into mailbox at ARGS_OFFSET
 *   3. write_state(TASK_READY)  — release store
 *   4. Spin-poll until read_state() == TASK_DONE  — blocking in WorkerThread
 *   5. write_state(IDLE)        — reset for next task
 *   6. return  → WorkerThread pushes to completion_queue
 *
 * Mailbox layout (DIST_CHIP_MAILBOX_SIZE bytes):
 *   offset  0  int32   state              IDLE=0, TASK_READY=1, TASK_DONE=2, SHUTDOWN=3
 *   offset  4  int32   error_code         0=ok
 *   offset  8  uint64  callable_ptr       ChipCallable buffer address (COW)
 *   offset 16  int32   block_dim
 *   offset 20  int32   aicpu_thread_num
 *   offset 24  int32   enable_profiling
 *   offset 64  [bytes] ChipStorageTaskArgs (memcpy'd, read in-place by child)
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>

#include "dist_types.h"

static constexpr size_t DIST_CHIP_MAILBOX_SIZE = 4096;
static constexpr size_t DIST_CHIP_ARGS_CAPACITY = DIST_CHIP_MAILBOX_SIZE - 64;

enum class ChipMailboxState : int32_t {
    IDLE = 0,
    TASK_READY = 1,
    TASK_DONE = 2,
    SHUTDOWN = 3,
};

class DistChipProcess : public IWorker {
public:
    explicit DistChipProcess(void *mailbox_ptr, size_t args_size);

    // IWorker: write payload to mailbox → spin-poll TASK_DONE → reset IDLE.
    void run(const WorkerPayload &payload) override;

    void shutdown();

private:
    void *mailbox_;
    size_t args_size_;

    static constexpr ptrdiff_t OFF_STATE = 0;
    static constexpr ptrdiff_t OFF_ERROR = 4;
    static constexpr ptrdiff_t OFF_CALLABLE = 8;
    static constexpr ptrdiff_t OFF_BLOCK_DIM = 16;
    static constexpr ptrdiff_t OFF_AICPU_THREAD_NUM = 20;
    static constexpr ptrdiff_t OFF_ENABLE_PROFILING = 24;
    static constexpr ptrdiff_t OFF_ARGS = 64;

    char *base() const { return static_cast<char *>(mailbox_); }

    ChipMailboxState read_state() const;
    void write_state(ChipMailboxState s);
};
