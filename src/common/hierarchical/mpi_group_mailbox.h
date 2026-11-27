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

#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>

#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <ctime>
#else
#include <chrono>
#endif

namespace mpi_group_mailbox {

inline constexpr uint8_t MAGIC[8] = {'S', 'M', 'P', 'I', 'B', 'O', 'X', '\0'};
inline constexpr size_t HEADER_BYTES = 256;
inline constexpr size_t PAYLOAD_BYTES = 16U * 1024U * 1024U;
inline constexpr size_t ERROR_BYTES = 64U * 1024U;
inline constexpr size_t REQUEST_OFFSET = HEADER_BYTES;
inline constexpr size_t RESPONSE_OFFSET = REQUEST_OFFSET + PAYLOAD_BYTES;
inline constexpr size_t ERROR_OFFSET = RESPONSE_OFFSET + PAYLOAD_BYTES;
inline constexpr size_t MAILBOX_BYTES = ERROR_OFFSET + ERROR_BYTES;

inline constexpr size_t OFF_MAGIC = 0;
inline constexpr size_t OFF_HEADER_BYTES = 12;
inline constexpr size_t OFF_MAILBOX_BYTES = 16;
inline constexpr size_t OFF_WORLD_SIZE = 24;
inline constexpr size_t OFF_GROUP_STATE = 28;
inline constexpr size_t OFF_REQUEST_STATE = 32;
inline constexpr size_t OFF_SEQUENCE_ID = 40;
inline constexpr size_t OFF_OPCODE = 48;
inline constexpr size_t OFF_TARGET = 52;
inline constexpr size_t OFF_TARGET_RANK = 56;
inline constexpr size_t OFF_REQUEST_COUNT = 60;
inline constexpr size_t OFF_REQUEST_BYTES = 64;
inline constexpr size_t OFF_RESPONSE_COUNT = 68;
inline constexpr size_t OFF_RESPONSE_BYTES = 72;
inline constexpr size_t OFF_ERROR_BYTES = 76;
// Header bytes [RESERVED_OFFSET, HEADER_BYTES) are reserved: the creator zeroes
// them and every attach path rejects a mailbox whose reserved bytes are
// non-zero, so a later field can claim them. The word at offset 8 is likewise
// unclaimed and left zero.
inline constexpr size_t OFF_CANCEL_SEQUENCE = 80;
inline constexpr size_t OFF_FIRST_ERROR_BYTES = 88;
inline constexpr size_t RESERVED_OFFSET = 92;

// The request-state word doubles as a shared futex: the mailbox is mapped by
// more than one process, so the wake must not use FUTEX_PRIVATE_FLAG. Both
// helpers may return spuriously; callers re-check the word.
inline void wake_word(int32_t *addr) {
#if defined(__linux__)
    (void)::syscall(SYS_futex, addr, FUTEX_WAKE, INT32_MAX, nullptr, nullptr, 0);
#else
    (void)addr;
#endif
}

enum class WaitWordResult : int32_t {
    NO_WAIT = 0,
    VALUE_MISMATCH = 1,
    WOKEN = 2,
    TIMED_OUT = 3,
    INTERRUPTED = 4,
    ERROR = 5,
};

// Blocks until the word at `addr` differs from `expected`, a wake arrives, or
// `timeout_s` elapses. Non-Linux builds have no futex and poll the word until
// the timeout instead.
inline WaitWordResult wait_word(int32_t *addr, int32_t expected, double timeout_s) {
    if (!(timeout_s > 0.0)) return WaitWordResult::NO_WAIT;
#if defined(__linux__)
    struct timespec ts;
    const int64_t timeout_ns = static_cast<int64_t>(timeout_s * 1e9);
    ts.tv_sec = static_cast<time_t>(timeout_ns / 1000000000);
    ts.tv_nsec = static_cast<long>(timeout_ns % 1000000000);
    long rc = ::syscall(SYS_futex, addr, FUTEX_WAIT, expected, &ts, nullptr, 0);
    if (rc == 0) return WaitWordResult::WOKEN;
    if (errno == EAGAIN) return WaitWordResult::VALUE_MISMATCH;
    if (errno == ETIMEDOUT) return WaitWordResult::TIMED_OUT;
    if (errno == EINTR) return WaitWordResult::INTERRUPTED;
    return WaitWordResult::ERROR;
#else
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
    int32_t value = expected;
    while (std::chrono::steady_clock::now() < deadline) {
        __atomic_load(addr, &value, __ATOMIC_ACQUIRE);
        if (value != expected) return WaitWordResult::VALUE_MISMATCH;
    }
    return WaitWordResult::TIMED_OUT;
#endif
}

enum class GroupState : int32_t {
    INITIALIZING = 0,
    READY = 1,
    TERMINAL = 2,
    CLOSED = 3,
};

enum class RequestState : int32_t {
    IDLE = 0,
    REQUEST_READY = 1,
    TASK_ACCEPTED = 2,
    TASK_DONE = 3,
    TASK_FAILED = 4,
    SHUTDOWN_READY = 5,
    SHUTDOWN_DONE = 6,
};

enum class Opcode : uint32_t {
    TASK = 1,
    CONTROL = 2,
    PING = 3,
    SHUTDOWN = 4,
};

enum class Target : uint32_t {
    GROUP = 1,
    RANK = 2,
    PER_RANK = 3,
};

}  // namespace mpi_group_mailbox
