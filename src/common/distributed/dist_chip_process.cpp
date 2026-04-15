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

DistChipProcess::DistChipProcess(void *mailbox_ptr, size_t args_capacity) :
    mailbox_(mailbox_ptr),
    args_capacity_(args_capacity) {
    if (!mailbox_ptr) throw std::invalid_argument("DistChipProcess: null mailbox_ptr");
    if (args_capacity > DIST_CHIP_ARGS_CAPACITY) {
        throw std::invalid_argument("DistChipProcess: args_capacity exceeds mailbox capacity");
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

void DistChipProcess::run(uint64_t callable, TaskArgsView args, const ChipCallConfig &config) {
    // Write callable pointer (child dereferences this via fork-COW).
    std::memcpy(base() + OFF_CALLABLE, &callable, sizeof(uint64_t));

    // Write config fields.
    int32_t block_dim = config.block_dim;
    int32_t aicpu_tn = config.aicpu_thread_num;
    int32_t profiling = config.perf_level;
    std::memcpy(base() + OFF_BLOCK_DIM, &block_dim, sizeof(int32_t));
    std::memcpy(base() + OFF_AICPU_THREAD_NUM, &aicpu_tn, sizeof(int32_t));
    std::memcpy(base() + OFF_ENABLE_PROFILING, &profiling, sizeof(int32_t));

    // Write length-prefixed TaskArgs blob: [T][S][tensors][scalars]. The
    // child reads it with read_blob() and assembles a ChipStorageTaskArgs
    // POD on its own heap before invoking pto2_run_runtime.
    size_t blob_bytes = TASK_ARGS_BLOB_HEADER_SIZE + static_cast<size_t>(args.tensor_count) * sizeof(ContinuousTensor) +
                        static_cast<size_t>(args.scalar_count) * sizeof(uint64_t);
    if (blob_bytes > args_capacity_) {
        throw std::runtime_error("DistChipProcess::run: args blob exceeds mailbox capacity");
    }
    uint8_t *d = reinterpret_cast<uint8_t *>(base() + OFF_ARGS);
    std::memcpy(d + 0, &args.tensor_count, sizeof(int32_t));
    std::memcpy(d + 4, &args.scalar_count, sizeof(int32_t));
    if (args.tensor_count > 0) {
        std::memcpy(
            d + TASK_ARGS_BLOB_HEADER_SIZE, args.tensors,
            static_cast<size_t>(args.tensor_count) * sizeof(ContinuousTensor)
        );
    }
    if (args.scalar_count > 0) {
        std::memcpy(
            d + TASK_ARGS_BLOB_HEADER_SIZE + static_cast<size_t>(args.tensor_count) * sizeof(ContinuousTensor),
            args.scalars, static_cast<size_t>(args.scalar_count) * sizeof(uint64_t)
        );
    }

    // Signal child process.
    write_state(ChipMailboxState::TASK_READY);

    // Spin-poll until child signals TASK_DONE.
    while (read_state() != ChipMailboxState::TASK_DONE) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    write_state(ChipMailboxState::IDLE);
}

void DistChipProcess::shutdown() { write_state(ChipMailboxState::SHUTDOWN); }
