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

#include "dist_sub_worker.h"

#include <cstdint>
#include <stdexcept>

// Mailbox byte offsets (must match Python layout in test_hostsub_fork_shm.py)
static constexpr ptrdiff_t OFF_STATE = 0;
static constexpr ptrdiff_t OFF_CALLABLE_ID = 4;

DistSubWorker::DistSubWorker(void *mailbox_ptr) :
    mailbox_(mailbox_ptr) {
    if (!mailbox_ptr) throw std::invalid_argument("DistSubWorker: null mailbox_ptr");
}

volatile int32_t *DistSubWorker::state_ptr() const {
    return reinterpret_cast<volatile int32_t *>(static_cast<char *>(mailbox_) + OFF_STATE);
}

volatile int32_t *DistSubWorker::callable_id_ptr() const {
    return reinterpret_cast<volatile int32_t *>(static_cast<char *>(mailbox_) + OFF_CALLABLE_ID);
}

SubMailboxState DistSubWorker::read_state() const {
    int32_t v;
#if defined(__aarch64__)
    __asm__ volatile("ldar %w0, [%1]" : "=r"(v) : "r"(state_ptr()) : "memory");
#elif defined(__x86_64__)
    v = *state_ptr();
    __asm__ volatile("" ::: "memory");
#else
    __atomic_load(state_ptr(), &v, __ATOMIC_ACQUIRE);
#endif
    return static_cast<SubMailboxState>(v);
}

void DistSubWorker::write_state(SubMailboxState s) {
    int32_t v = static_cast<int32_t>(s);
#if defined(__aarch64__)
    __asm__ volatile("stlr %w0, [%1]" : : "r"(v), "r"(state_ptr()) : "memory");
#elif defined(__x86_64__)
    __asm__ volatile("" ::: "memory");
    *state_ptr() = v;
#else
    __atomic_store(state_ptr(), &v, __ATOMIC_RELEASE);
#endif
}

// =============================================================================
// IWorker::run() — blocks in the WorkerThread's own thread
// =============================================================================

void DistSubWorker::run(uint64_t callable, TaskArgsView /*args*/, const ChipCallConfig & /*config*/) {
    // `callable` encodes the registered callable id as uint64. Write the
    // low 32 bits — matches the Python-side mailbox layout.
    *callable_id_ptr() = static_cast<int32_t>(callable);
    write_state(SubMailboxState::TASK_READY);

    // Self-poll until child signals TASK_DONE.
    // This blocks in the WorkerThread, not in the Scheduler thread.
    while (read_state() != SubMailboxState::TASK_DONE) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    write_state(SubMailboxState::IDLE);
}

void DistSubWorker::shutdown() { write_state(SubMailboxState::SHUTDOWN); }
