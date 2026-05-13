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
static constexpr size_t CHIP_BOOTSTRAP_MAX_DOMAINS = 8;
static constexpr size_t CHIP_BOOTSTRAP_DOMAIN_NAME_SIZE = 32;  // includes trailing '\0'
static constexpr size_t CHIP_BOOTSTRAP_DOMAIN_RECORD_SIZE = 80;

// Fixed offsets within the mailbox region.
static constexpr ptrdiff_t CHIP_BOOTSTRAP_OFF_STATE = 0;
static constexpr ptrdiff_t CHIP_BOOTSTRAP_OFF_ERROR_CODE = 4;
static constexpr ptrdiff_t CHIP_BOOTSTRAP_OFF_DOMAIN_COUNT = 8;
static constexpr ptrdiff_t CHIP_BOOTSTRAP_OFF_DOMAIN_RECORDS = CHIP_BOOTSTRAP_HEADER_SIZE;
static constexpr ptrdiff_t CHIP_BOOTSTRAP_OFF_BUFFER_PTRS =
    CHIP_BOOTSTRAP_OFF_DOMAIN_RECORDS +
    static_cast<ptrdiff_t>(CHIP_BOOTSTRAP_MAX_DOMAINS * CHIP_BOOTSTRAP_DOMAIN_RECORD_SIZE);
static constexpr ptrdiff_t CHIP_BOOTSTRAP_OFF_ERROR_MSG =
    static_cast<ptrdiff_t>(CHIP_BOOTSTRAP_MAILBOX_SIZE - CHIP_BOOTSTRAP_ERROR_MSG_SIZE);
static constexpr size_t CHIP_BOOTSTRAP_PTR_CAPACITY =
    (CHIP_BOOTSTRAP_OFF_ERROR_MSG - CHIP_BOOTSTRAP_OFF_BUFFER_PTRS) / sizeof(uint64_t);

// Backward-compatible alias for the pre-domain single-result layout.
static constexpr ptrdiff_t CHIP_BOOTSTRAP_OFF_BUFFER_COUNT = CHIP_BOOTSTRAP_OFF_DOMAIN_COUNT;

static_assert(
    CHIP_BOOTSTRAP_OFF_BUFFER_PTRS <= CHIP_BOOTSTRAP_OFF_ERROR_MSG, "domain records must fit before error message"
);
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

struct ChipBootstrapDomainResult {
    std::string name;
    int32_t domain_rank = 0;
    int32_t domain_size = 0;
    uint64_t device_ctx = 0;
    uint64_t local_window_base = 0;
    uint64_t actual_window_size = 0;
    std::vector<uint64_t> buffer_ptrs;

    ChipBootstrapDomainResult() = default;
    ChipBootstrapDomainResult(
        std::string domain_name, int32_t rank, int32_t size, uint64_t ctx, uint64_t window_base, uint64_t window_size,
        std::vector<uint64_t> ptrs
    );
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
    void write_success_domains(const std::vector<ChipBootstrapDomainResult> &domains);
    void write_error(int32_t error_code, const std::string &message);

    // Read side (parent process).
    ChipBootstrapMailboxState state() const;
    int32_t error_code() const;
    int32_t domain_count() const;
    std::vector<ChipBootstrapDomainResult> domains() const;
    ChipBootstrapDomainResult domain(const std::string &name) const;
    uint64_t device_ctx() const;
    uint64_t local_window_base() const;
    uint64_t actual_window_size() const;
    std::vector<uint64_t> buffer_ptrs() const;
    std::string error_message() const;

private:
    void *mailbox_;
    size_t max_buffer_count_;
};
