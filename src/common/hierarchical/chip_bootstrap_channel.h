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
 * ChipBootstrapChannel — one-shot cross-process mailbox for per-chip bootstrap.
 *
 * Lifecycle: parent allocates a CHIP_BOOTSTRAP_MAILBOX_SIZE shared-memory region,
 * child writes SUCCESS/ERROR once, parent polls state() until done.
 * Not a general-purpose mailbox — independent of the task-mailbox protocol.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

static constexpr size_t CHIP_BOOTSTRAP_MAILBOX_SIZE = 4096;
static constexpr size_t CHIP_BOOTSTRAP_HEADER_SIZE = 64;
static constexpr size_t CHIP_BOOTSTRAP_ERROR_MSG_SIZE = 1024;
static constexpr size_t CHIP_BOOTSTRAP_PTR_CAPACITY =
    (CHIP_BOOTSTRAP_MAILBOX_SIZE - CHIP_BOOTSTRAP_HEADER_SIZE - CHIP_BOOTSTRAP_ERROR_MSG_SIZE) / sizeof(uint64_t);

// Fixed offsets within the mailbox region.
static constexpr ptrdiff_t CHIP_BOOTSTRAP_OFF_STATE = 0;
static constexpr ptrdiff_t CHIP_BOOTSTRAP_OFF_ERROR_CODE = 4;
static constexpr ptrdiff_t CHIP_BOOTSTRAP_OFF_BUFFER_COUNT = 8;
// 4 bytes implicit padding for uint64 alignment
static constexpr ptrdiff_t CHIP_BOOTSTRAP_OFF_DEVICE_CTX = 16;
static constexpr ptrdiff_t CHIP_BOOTSTRAP_OFF_LOCAL_WINDOW_BASE = 24;
static constexpr ptrdiff_t CHIP_BOOTSTRAP_OFF_ACTUAL_WINDOW_SIZE = 32;
static constexpr ptrdiff_t CHIP_BOOTSTRAP_OFF_BUFFER_PTRS = 64;
static constexpr ptrdiff_t CHIP_BOOTSTRAP_OFF_ERROR_MSG =
    CHIP_BOOTSTRAP_OFF_BUFFER_PTRS + static_cast<ptrdiff_t>(CHIP_BOOTSTRAP_PTR_CAPACITY * sizeof(uint64_t));

static_assert(
    CHIP_BOOTSTRAP_OFF_ERROR_MSG + static_cast<ptrdiff_t>(CHIP_BOOTSTRAP_ERROR_MSG_SIZE) ==
        static_cast<ptrdiff_t>(CHIP_BOOTSTRAP_MAILBOX_SIZE),
    "mailbox layout must sum to 4096"
);

enum class ChipBootstrapMailboxState : int32_t {
    IDLE = 0,
    SUCCESS = 1,
    ERROR = 2,
};

class ChipBootstrapChannel {
public:
    ChipBootstrapChannel(void *mailbox, size_t max_buffer_count);

    // Write side (child process).
    void reset();
    void write_success(
        uint64_t device_ctx, uint64_t local_window_base, uint64_t actual_window_size,
        const std::vector<uint64_t> &buffer_ptrs
    );
    void write_error(int32_t error_code, const std::string &message);

    // Read side (parent process).
    ChipBootstrapMailboxState state() const;
    int32_t error_code() const;
    uint64_t device_ctx() const;
    uint64_t local_window_base() const;
    uint64_t actual_window_size() const;
    std::vector<uint64_t> buffer_ptrs() const;
    std::string error_message() const;

private:
    void *mailbox_;
    size_t max_buffer_count_;
};
