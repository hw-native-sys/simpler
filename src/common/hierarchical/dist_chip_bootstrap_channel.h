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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

static constexpr size_t DIST_CHIP_BOOTSTRAP_MAILBOX_SIZE = 4096;
static constexpr size_t DIST_CHIP_BOOTSTRAP_HEADER_SIZE = 64;
static constexpr size_t DIST_CHIP_BOOTSTRAP_ERROR_MSG_SIZE = 1024;
static constexpr size_t DIST_CHIP_BOOTSTRAP_PTR_CAPACITY =
    (DIST_CHIP_BOOTSTRAP_MAILBOX_SIZE - DIST_CHIP_BOOTSTRAP_HEADER_SIZE - DIST_CHIP_BOOTSTRAP_ERROR_MSG_SIZE) /
    sizeof(uint64_t);

enum class ChipBootstrapMailboxState : int32_t {
    IDLE = 0,
    SUCCESS = 1,
    ERROR = 2,
};

class DistChipBootstrapChannel {
public:
    DistChipBootstrapChannel(void *mailbox_ptr, size_t max_buffer_count);

    void reset();
    void write_success(uint64_t device_ctx, uint64_t local_window_base, uint64_t actual_window_size,
                       const std::vector<uint64_t> &buffer_ptrs);
    void write_error(int32_t error_code, const std::string &message);

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

    static constexpr ptrdiff_t OFF_STATE = 0;
    static constexpr ptrdiff_t OFF_ERROR_CODE = 4;
    static constexpr ptrdiff_t OFF_BUFFER_COUNT = 8;
    static constexpr ptrdiff_t OFF_DEVICE_CTX = 16;
    static constexpr ptrdiff_t OFF_LOCAL_WINDOW_BASE = 24;
    static constexpr ptrdiff_t OFF_ACTUAL_WINDOW_SIZE = 32;
    static constexpr ptrdiff_t OFF_BUFFER_PTRS = 64;

    char *base() const { return static_cast<char *>(mailbox_); }
    ptrdiff_t error_msg_offset() const { return OFF_BUFFER_PTRS + static_cast<ptrdiff_t>(max_buffer_count_ * 8); }
    size_t error_msg_capacity() const {
        return DIST_CHIP_BOOTSTRAP_MAILBOX_SIZE - static_cast<size_t>(error_msg_offset());
    }

    ChipBootstrapMailboxState read_state() const;
    void write_state(ChipBootstrapMailboxState state);
};
