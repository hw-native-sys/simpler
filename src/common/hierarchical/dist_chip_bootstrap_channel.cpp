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

#include "dist_chip_bootstrap_channel.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

DistChipBootstrapChannel::DistChipBootstrapChannel(void *mailbox_ptr, size_t max_buffer_count) :
    mailbox_(mailbox_ptr),
    max_buffer_count_(max_buffer_count) {
    if (mailbox_ptr == nullptr) throw std::invalid_argument("DistChipBootstrapChannel: null mailbox_ptr");
    if (max_buffer_count > DIST_CHIP_BOOTSTRAP_PTR_CAPACITY) {
        throw std::invalid_argument("DistChipBootstrapChannel: buffer count exceeds mailbox capacity");
    }
}

ChipBootstrapMailboxState DistChipBootstrapChannel::read_state() const {
    volatile int32_t *ptr = reinterpret_cast<volatile int32_t *>(base() + OFF_STATE);
    int32_t value = 0;
#if defined(__aarch64__)
    __asm__ volatile("ldar %w0, [%1]" : "=r"(value) : "r"(ptr) : "memory");
#elif defined(__x86_64__)
    value = *ptr;
    __asm__ volatile("" ::: "memory");
#else
    __atomic_load(ptr, &value, __ATOMIC_ACQUIRE);
#endif
    return static_cast<ChipBootstrapMailboxState>(value);
}

void DistChipBootstrapChannel::write_state(ChipBootstrapMailboxState state) {
    volatile int32_t *ptr = reinterpret_cast<volatile int32_t *>(base() + OFF_STATE);
    int32_t value = static_cast<int32_t>(state);
#if defined(__aarch64__)
    __asm__ volatile("stlr %w0, [%1]" : : "r"(value), "r"(ptr) : "memory");
#elif defined(__x86_64__)
    __asm__ volatile("" ::: "memory");
    *ptr = value;
#else
    __atomic_store(ptr, &value, __ATOMIC_RELEASE);
#endif
}

void DistChipBootstrapChannel::reset() {
    std::memset(base(), 0, DIST_CHIP_BOOTSTRAP_MAILBOX_SIZE);
    write_state(ChipBootstrapMailboxState::IDLE);
}

void DistChipBootstrapChannel::write_success(
    uint64_t device_ctx, uint64_t local_window_base, uint64_t actual_window_size, const std::vector<uint64_t> &buffer_ptrs
) {
    if (buffer_ptrs.size() > max_buffer_count_) {
        throw std::invalid_argument("DistChipBootstrapChannel::write_success: buffer count exceeds configured capacity");
    }
    reset();
    int32_t error_code = 0;
    int32_t buffer_count = static_cast<int32_t>(buffer_ptrs.size());
    std::memcpy(base() + OFF_ERROR_CODE, &error_code, sizeof(error_code));
    std::memcpy(base() + OFF_BUFFER_COUNT, &buffer_count, sizeof(buffer_count));
    std::memcpy(base() + OFF_DEVICE_CTX, &device_ctx, sizeof(device_ctx));
    std::memcpy(base() + OFF_LOCAL_WINDOW_BASE, &local_window_base, sizeof(local_window_base));
    std::memcpy(base() + OFF_ACTUAL_WINDOW_SIZE, &actual_window_size, sizeof(actual_window_size));
    if (!buffer_ptrs.empty()) {
        std::memcpy(base() + OFF_BUFFER_PTRS, buffer_ptrs.data(), buffer_ptrs.size() * sizeof(uint64_t));
    }
    write_state(ChipBootstrapMailboxState::SUCCESS);
}

void DistChipBootstrapChannel::write_error(int32_t error_code, const std::string &message) {
    reset();
    std::memcpy(base() + OFF_ERROR_CODE, &error_code, sizeof(error_code));
    const size_t max_copy = std::max<size_t>(1, error_msg_capacity()) - 1;
    const size_t copy_size = std::min(max_copy, message.size());
    std::memcpy(base() + error_msg_offset(), message.data(), copy_size);
    base()[error_msg_offset() + static_cast<ptrdiff_t>(copy_size)] = '\0';
    write_state(ChipBootstrapMailboxState::ERROR);
}

ChipBootstrapMailboxState DistChipBootstrapChannel::state() const { return read_state(); }

int32_t DistChipBootstrapChannel::error_code() const {
    int32_t value = 0;
    std::memcpy(&value, base() + OFF_ERROR_CODE, sizeof(value));
    return value;
}

uint64_t DistChipBootstrapChannel::device_ctx() const {
    uint64_t value = 0;
    std::memcpy(&value, base() + OFF_DEVICE_CTX, sizeof(value));
    return value;
}

uint64_t DistChipBootstrapChannel::local_window_base() const {
    uint64_t value = 0;
    std::memcpy(&value, base() + OFF_LOCAL_WINDOW_BASE, sizeof(value));
    return value;
}

uint64_t DistChipBootstrapChannel::actual_window_size() const {
    uint64_t value = 0;
    std::memcpy(&value, base() + OFF_ACTUAL_WINDOW_SIZE, sizeof(value));
    return value;
}

std::vector<uint64_t> DistChipBootstrapChannel::buffer_ptrs() const {
    int32_t count = 0;
    std::memcpy(&count, base() + OFF_BUFFER_COUNT, sizeof(count));
    if (count < 0 || static_cast<size_t>(count) > max_buffer_count_) {
        throw std::runtime_error("DistChipBootstrapChannel: invalid buffer count in mailbox");
    }
    std::vector<uint64_t> values(static_cast<size_t>(count));
    if (!values.empty()) {
        std::memcpy(values.data(), base() + OFF_BUFFER_PTRS, values.size() * sizeof(uint64_t));
    }
    return values;
}

std::string DistChipBootstrapChannel::error_message() const {
    const char *msg = base() + error_msg_offset();
    return std::string(msg, strnlen(msg, error_msg_capacity()));
}
