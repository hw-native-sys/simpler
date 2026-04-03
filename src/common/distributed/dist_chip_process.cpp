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

#include "dist_chip_process.h"

#include <stdexcept>

DistChipProcess::DistChipProcess(void *mailbox_ptr, size_t args_size) :
    mailbox_(mailbox_ptr),
    args_size_(args_size) {
    if (!mailbox_ptr) throw std::invalid_argument("DistChipProcess: null mailbox_ptr");
    if (args_size > DIST_CHIP_ARGS_CAPACITY) {
        throw std::invalid_argument("DistChipProcess: args_size exceeds mailbox capacity");
    }
}

ChipMailboxState DistChipProcess::read_state() const {
    volatile int32_t *ptr = reinterpret_cast<volatile int32_t *>(base() + OFF_STATE);
    int32_t v;
#if defined(__aarch64__)
    __asm__ volatile("ldar %w0, [%1]" : "=r"(v) : "r"(ptr) : "memory");
#elif defined(__x86_64__)
    v = *ptr;
    __asm__ volatile("" ::: "memory");
#else
    __atomic_load(ptr, &v, __ATOMIC_ACQUIRE);
#endif
    return static_cast<ChipMailboxState>(v);
}

void DistChipProcess::write_state(ChipMailboxState s) {
    volatile int32_t *ptr = reinterpret_cast<volatile int32_t *>(base() + OFF_STATE);
    int32_t v = static_cast<int32_t>(s);
#if defined(__aarch64__)
    __asm__ volatile("stlr %w0, [%1]" : : "r"(v), "r"(ptr) : "memory");
#elif defined(__x86_64__)
    __asm__ volatile("" ::: "memory");
    *ptr = v;
#else
    __atomic_store(ptr, &v, __ATOMIC_RELEASE);
#endif
}

void DistChipProcess::run(const WorkerPayload &payload) {
    // Write callable pointer
    uint64_t callable_val = reinterpret_cast<uint64_t>(payload.callable);
    std::memcpy(base() + OFF_CALLABLE, &callable_val, sizeof(uint64_t));

    // Write config fields
    int32_t block_dim = payload.block_dim;
    int32_t aicpu_tn = payload.aicpu_thread_num;
    int32_t profiling = payload.enable_profiling ? 1 : 0;
    std::memcpy(base() + OFF_BLOCK_DIM, &block_dim, sizeof(int32_t));
    std::memcpy(base() + OFF_AICPU_THREAD_NUM, &aicpu_tn, sizeof(int32_t));
    std::memcpy(base() + OFF_ENABLE_PROFILING, &profiling, sizeof(int32_t));

    // Copy args into mailbox (child reads from mailbox address)
    if (payload.args != nullptr && args_size_ > 0) {
        std::memcpy(base() + OFF_ARGS, payload.args, args_size_);
    }

    // Signal child process
    write_state(ChipMailboxState::TASK_READY);

    // Spin-poll until child signals TASK_DONE
    while (read_state() != ChipMailboxState::TASK_DONE) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    write_state(ChipMailboxState::IDLE);
}

void DistChipProcess::shutdown() { write_state(ChipMailboxState::SHUTDOWN); }
