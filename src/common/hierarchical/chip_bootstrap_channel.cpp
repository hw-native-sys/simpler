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

#include "chip_bootstrap_channel.h"

#include <cstring>
#include <stdexcept>

// =============================================================================
// Internal helpers
// =============================================================================

namespace {

void write_state(void *mailbox, ChipBootstrapMailboxState s) {
    auto *ptr = reinterpret_cast<volatile int32_t *>(static_cast<char *>(mailbox) + CHIP_BOOTSTRAP_OFF_STATE);
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

ChipBootstrapMailboxState read_state(void *mailbox) {
    auto *ptr = reinterpret_cast<volatile int32_t *>(static_cast<char *>(mailbox) + CHIP_BOOTSTRAP_OFF_STATE);
    int32_t v;
#if defined(__aarch64__)
    __asm__ volatile("ldar %w0, [%1]" : "=r"(v) : "r"(ptr) : "memory");
#elif defined(__x86_64__)
    v = *ptr;
    __asm__ volatile("" ::: "memory");
#else
    __atomic_load(ptr, &v, __ATOMIC_ACQUIRE);
#endif
    return static_cast<ChipBootstrapMailboxState>(v);
}

}  // namespace

// =============================================================================
// ChipBootstrapChannel
// =============================================================================

ChipBootstrapChannel::ChipBootstrapChannel(void *mailbox, size_t max_buffer_count) :
    mailbox_(mailbox),
    max_buffer_count_(max_buffer_count) {
    if (mailbox_ == nullptr) {
        throw std::invalid_argument("mailbox must not be null");
    }
    if (max_buffer_count_ > CHIP_BOOTSTRAP_PTR_CAPACITY) {
        throw std::invalid_argument("max_buffer_count exceeds CHIP_BOOTSTRAP_PTR_CAPACITY");
    }
}

void ChipBootstrapChannel::reset() {
    std::memset(mailbox_, 0, CHIP_BOOTSTRAP_MAILBOX_SIZE);
    write_state(mailbox_, ChipBootstrapMailboxState::IDLE);
}

void ChipBootstrapChannel::write_success(
    uint64_t device_ctx, uint64_t local_window_base, uint64_t actual_window_size,
    const std::vector<uint64_t> &buffer_ptrs
) {
    if (buffer_ptrs.size() > max_buffer_count_) {
        throw std::invalid_argument("buffer_ptrs exceeds max_buffer_count");
    }

    auto *base = static_cast<char *>(mailbox_);

    int32_t count = static_cast<int32_t>(buffer_ptrs.size());
    std::memcpy(base + CHIP_BOOTSTRAP_OFF_BUFFER_COUNT, &count, sizeof(count));
    std::memcpy(base + CHIP_BOOTSTRAP_OFF_DEVICE_CTX, &device_ctx, sizeof(device_ctx));
    std::memcpy(base + CHIP_BOOTSTRAP_OFF_LOCAL_WINDOW_BASE, &local_window_base, sizeof(local_window_base));
    std::memcpy(base + CHIP_BOOTSTRAP_OFF_ACTUAL_WINDOW_SIZE, &actual_window_size, sizeof(actual_window_size));

    if (!buffer_ptrs.empty()) {
        std::memcpy(base + CHIP_BOOTSTRAP_OFF_BUFFER_PTRS, buffer_ptrs.data(), buffer_ptrs.size() * sizeof(uint64_t));
    }

    write_state(mailbox_, ChipBootstrapMailboxState::SUCCESS);
}

void ChipBootstrapChannel::write_error(int32_t error_code, const std::string &message) {
    auto *base = static_cast<char *>(mailbox_);

    std::memcpy(base + CHIP_BOOTSTRAP_OFF_ERROR_CODE, &error_code, sizeof(error_code));

    size_t max_len = CHIP_BOOTSTRAP_ERROR_MSG_SIZE - 1;
    size_t copy_len = message.size() < max_len ? message.size() : max_len;
    std::memcpy(base + CHIP_BOOTSTRAP_OFF_ERROR_MSG, message.data(), copy_len);
    base[CHIP_BOOTSTRAP_OFF_ERROR_MSG + copy_len] = '\0';

    write_state(mailbox_, ChipBootstrapMailboxState::ERROR);
}

ChipBootstrapMailboxState ChipBootstrapChannel::state() const { return read_state(mailbox_); }

int32_t ChipBootstrapChannel::error_code() const {
    auto *base = static_cast<const char *>(mailbox_);
    int32_t v;
    std::memcpy(&v, base + CHIP_BOOTSTRAP_OFF_ERROR_CODE, sizeof(v));
    return v;
}

uint64_t ChipBootstrapChannel::device_ctx() const {
    auto *base = static_cast<const char *>(mailbox_);
    uint64_t v;
    std::memcpy(&v, base + CHIP_BOOTSTRAP_OFF_DEVICE_CTX, sizeof(v));
    return v;
}

uint64_t ChipBootstrapChannel::local_window_base() const {
    auto *base = static_cast<const char *>(mailbox_);
    uint64_t v;
    std::memcpy(&v, base + CHIP_BOOTSTRAP_OFF_LOCAL_WINDOW_BASE, sizeof(v));
    return v;
}

uint64_t ChipBootstrapChannel::actual_window_size() const {
    auto *base = static_cast<const char *>(mailbox_);
    uint64_t v;
    std::memcpy(&v, base + CHIP_BOOTSTRAP_OFF_ACTUAL_WINDOW_SIZE, sizeof(v));
    return v;
}

std::vector<uint64_t> ChipBootstrapChannel::buffer_ptrs() const {
    auto *base = static_cast<const char *>(mailbox_);
    int32_t raw_count;
    std::memcpy(&raw_count, base + CHIP_BOOTSTRAP_OFF_BUFFER_COUNT, sizeof(raw_count));

    // Ctor guarantees max_buffer_count_ <= CHIP_BOOTSTRAP_PTR_CAPACITY, so clamping
    // count against max_buffer_count_ alone is sufficient to keep the read bounded.
    size_t count =
        raw_count <= 0 ?
            0 :
            (static_cast<size_t>(raw_count) < max_buffer_count_ ? static_cast<size_t>(raw_count) : max_buffer_count_);

    std::vector<uint64_t> ptrs(count);
    if (count > 0) {
        std::memcpy(ptrs.data(), base + CHIP_BOOTSTRAP_OFF_BUFFER_PTRS, count * sizeof(uint64_t));
    }
    return ptrs;
}

std::string ChipBootstrapChannel::error_message() const {
    auto *base = static_cast<const char *>(mailbox_);
    const char *msg_ptr = base + CHIP_BOOTSTRAP_OFF_ERROR_MSG;
    // Bound the read against the layout size so a missing null-terminator in
    // shared memory (corrupt producer, premature read) can't walk off the page.
    size_t len = strnlen(msg_ptr, CHIP_BOOTSTRAP_ERROR_MSG_SIZE);
    return std::string(msg_ptr, len);
}
