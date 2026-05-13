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

#include <climits>
#include <cstring>
#include <stdexcept>
#include <unordered_set>
#include <utility>

// =============================================================================
// Internal helpers
// =============================================================================

namespace {

static constexpr ptrdiff_t DOMAIN_OFF_NAME = 0;
static constexpr ptrdiff_t DOMAIN_OFF_DOMAIN_RANK = 32;
static constexpr ptrdiff_t DOMAIN_OFF_DOMAIN_SIZE = 36;
static constexpr ptrdiff_t DOMAIN_OFF_BUFFER_COUNT = 40;
static constexpr ptrdiff_t DOMAIN_OFF_PTR_OFFSET = 44;
static constexpr ptrdiff_t DOMAIN_OFF_DEVICE_CTX = 48;
static constexpr ptrdiff_t DOMAIN_OFF_LOCAL_WINDOW_BASE = 56;
static constexpr ptrdiff_t DOMAIN_OFF_ACTUAL_WINDOW_SIZE = 64;

static_assert(
    DOMAIN_OFF_ACTUAL_WINDOW_SIZE + static_cast<ptrdiff_t>(sizeof(uint64_t)) <=
        static_cast<ptrdiff_t>(CHIP_BOOTSTRAP_DOMAIN_RECORD_SIZE),
    "domain record layout exceeds CHIP_BOOTSTRAP_DOMAIN_RECORD_SIZE"
);

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

char *domain_record(char *base, size_t index) {
    return base + CHIP_BOOTSTRAP_OFF_DOMAIN_RECORDS + static_cast<ptrdiff_t>(index * CHIP_BOOTSTRAP_DOMAIN_RECORD_SIZE);
}

const char *domain_record(const char *base, size_t index) {
    return base + CHIP_BOOTSTRAP_OFF_DOMAIN_RECORDS + static_cast<ptrdiff_t>(index * CHIP_BOOTSTRAP_DOMAIN_RECORD_SIZE);
}

int32_t read_i32(const char *addr) {
    int32_t v;
    std::memcpy(&v, addr, sizeof(v));
    return v;
}

uint64_t read_u64(const char *addr) {
    uint64_t v;
    std::memcpy(&v, addr, sizeof(v));
    return v;
}

void write_i32(char *addr, int32_t v) { std::memcpy(addr, &v, sizeof(v)); }

void write_u64(char *addr, uint64_t v) { std::memcpy(addr, &v, sizeof(v)); }

size_t checked_buffer_count(int32_t raw_count, size_t max_buffer_count) {
    if (raw_count < 0) {
        throw std::runtime_error("bootstrap domain has negative buffer count");
    }
    size_t count = static_cast<size_t>(raw_count);
    if (count > max_buffer_count) {
        throw std::runtime_error("bootstrap domain buffer count exceeds max_buffer_count");
    }
    return count;
}

}  // namespace

// =============================================================================
// ChipBootstrapChannel
// =============================================================================

ChipBootstrapDomainResult::ChipBootstrapDomainResult(
    std::string domain_name, int32_t rank, int32_t size, uint64_t ctx, uint64_t window_base, uint64_t window_size,
    std::vector<uint64_t> ptrs
) :
    name(std::move(domain_name)),
    domain_rank(rank),
    domain_size(size),
    device_ctx(ctx),
    local_window_base(window_base),
    actual_window_size(window_size),
    buffer_ptrs(std::move(ptrs)) {}

ChipBootstrapChannel::ChipBootstrapChannel(void *mailbox, size_t max_buffer_count) :
    mailbox_(mailbox),
    max_buffer_count_(max_buffer_count) {
    if (mailbox_ == nullptr) {
        throw std::invalid_argument("mailbox must not be null");
    }
    if (max_buffer_count_ > static_cast<size_t>(INT32_MAX)) {
        throw std::invalid_argument("max_buffer_count exceeds int32 capacity");
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
    ChipBootstrapDomainResult domain(
        "default", 0, 1, device_ctx, local_window_base, actual_window_size, std::vector<uint64_t>(buffer_ptrs)
    );
    write_success_domains({domain});
}

void ChipBootstrapChannel::write_success_domains(const std::vector<ChipBootstrapDomainResult> &domains) {
    if (domains.size() > CHIP_BOOTSTRAP_MAX_DOMAINS) {
        throw std::invalid_argument("domain count exceeds CHIP_BOOTSTRAP_MAX_DOMAINS");
    }

    size_t total_buffer_count = 0;
    std::unordered_set<std::string> names;
    names.reserve(domains.size());

    for (const auto &domain : domains) {
        if (domain.name.empty()) {
            throw std::invalid_argument("domain name must not be empty");
        }
        if (domain.name.size() >= CHIP_BOOTSTRAP_DOMAIN_NAME_SIZE) {
            throw std::invalid_argument("domain name exceeds CHIP_BOOTSTRAP_DOMAIN_NAME_SIZE");
        }
        if (!names.insert(domain.name).second) {
            throw std::invalid_argument("duplicate domain name in bootstrap success payload");
        }
        if (domain.domain_rank < 0) {
            throw std::invalid_argument("domain_rank must be non-negative");
        }
        if (domain.domain_size <= 0) {
            throw std::invalid_argument("domain_size must be positive");
        }
        if (domain.domain_rank >= domain.domain_size) {
            throw std::invalid_argument("domain_rank must be smaller than domain_size");
        }
        if (domain.buffer_ptrs.size() > max_buffer_count_) {
            throw std::invalid_argument("domain buffer_ptrs exceeds max_buffer_count");
        }
        if (domain.buffer_ptrs.size() > static_cast<size_t>(INT32_MAX)) {
            throw std::invalid_argument("domain buffer_ptrs count exceeds int32 capacity");
        }
        if (domain.buffer_ptrs.size() > CHIP_BOOTSTRAP_PTR_CAPACITY ||
            total_buffer_count > CHIP_BOOTSTRAP_PTR_CAPACITY - domain.buffer_ptrs.size()) {
            throw std::invalid_argument("bootstrap success payload exceeds CHIP_BOOTSTRAP_PTR_CAPACITY");
        }
        total_buffer_count += domain.buffer_ptrs.size();
    }

    auto *base = static_cast<char *>(mailbox_);

    std::memset(
        base + CHIP_BOOTSTRAP_OFF_DOMAIN_COUNT, 0, CHIP_BOOTSTRAP_OFF_ERROR_MSG - CHIP_BOOTSTRAP_OFF_DOMAIN_COUNT
    );

    int32_t count = static_cast<int32_t>(domains.size());
    write_i32(base + CHIP_BOOTSTRAP_OFF_DOMAIN_COUNT, count);

    size_t ptr_offset = 0;
    for (size_t i = 0; i < domains.size(); ++i) {
        const auto &domain = domains[i];
        char *record = domain_record(base, i);

        std::memcpy(record + DOMAIN_OFF_NAME, domain.name.data(), domain.name.size());
        record[DOMAIN_OFF_NAME + domain.name.size()] = '\0';

        write_i32(record + DOMAIN_OFF_DOMAIN_RANK, domain.domain_rank);
        write_i32(record + DOMAIN_OFF_DOMAIN_SIZE, domain.domain_size);
        write_i32(record + DOMAIN_OFF_BUFFER_COUNT, static_cast<int32_t>(domain.buffer_ptrs.size()));
        write_i32(record + DOMAIN_OFF_PTR_OFFSET, static_cast<int32_t>(ptr_offset));
        write_u64(record + DOMAIN_OFF_DEVICE_CTX, domain.device_ctx);
        write_u64(record + DOMAIN_OFF_LOCAL_WINDOW_BASE, domain.local_window_base);
        write_u64(record + DOMAIN_OFF_ACTUAL_WINDOW_SIZE, domain.actual_window_size);

        if (!domain.buffer_ptrs.empty()) {
            std::memcpy(
                base + CHIP_BOOTSTRAP_OFF_BUFFER_PTRS + static_cast<ptrdiff_t>(ptr_offset * sizeof(uint64_t)),
                domain.buffer_ptrs.data(), domain.buffer_ptrs.size() * sizeof(uint64_t)
            );
            ptr_offset += domain.buffer_ptrs.size();
        }
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
    return read_i32(base + CHIP_BOOTSTRAP_OFF_ERROR_CODE);
}

int32_t ChipBootstrapChannel::domain_count() const {
    auto *base = static_cast<const char *>(mailbox_);
    int32_t count = read_i32(base + CHIP_BOOTSTRAP_OFF_DOMAIN_COUNT);
    if (count < 0 || count > static_cast<int32_t>(CHIP_BOOTSTRAP_MAX_DOMAINS)) {
        throw std::runtime_error("bootstrap domain count exceeds CHIP_BOOTSTRAP_MAX_DOMAINS");
    }
    return count;
}

std::vector<ChipBootstrapDomainResult> ChipBootstrapChannel::domains() const {
    auto *base = static_cast<const char *>(mailbox_);
    int32_t count = domain_count();
    std::vector<ChipBootstrapDomainResult> results;
    results.reserve(static_cast<size_t>(count));
    std::unordered_set<std::string> names;
    names.reserve(static_cast<size_t>(count));

    for (int32_t i = 0; i < count; ++i) {
        const char *record = domain_record(base, static_cast<size_t>(i));
        const char *name_ptr = record + DOMAIN_OFF_NAME;
        size_t name_len = strnlen(name_ptr, CHIP_BOOTSTRAP_DOMAIN_NAME_SIZE);
        if (name_len == 0) {
            throw std::runtime_error("bootstrap domain name must not be empty");
        }
        if (name_len == CHIP_BOOTSTRAP_DOMAIN_NAME_SIZE) {
            throw std::runtime_error("bootstrap domain name is not null-terminated");
        }
        std::string name(name_ptr, name_len);
        if (!names.insert(name).second) {
            throw std::runtime_error("duplicate domain name in bootstrap success payload");
        }

        int32_t buffer_count_raw = read_i32(record + DOMAIN_OFF_BUFFER_COUNT);
        int32_t ptr_offset_raw = read_i32(record + DOMAIN_OFF_PTR_OFFSET);
        size_t buffer_count = checked_buffer_count(buffer_count_raw, max_buffer_count_);
        if (ptr_offset_raw < 0) {
            throw std::runtime_error("bootstrap domain has negative buffer pointer offset");
        }
        size_t ptr_offset = static_cast<size_t>(ptr_offset_raw);
        if (ptr_offset > CHIP_BOOTSTRAP_PTR_CAPACITY || buffer_count > CHIP_BOOTSTRAP_PTR_CAPACITY - ptr_offset) {
            throw std::runtime_error("bootstrap domain buffer pointer range exceeds mailbox capacity");
        }
        std::vector<uint64_t> ptrs(buffer_count);
        if (buffer_count > 0) {
            std::memcpy(
                ptrs.data(),
                base + CHIP_BOOTSTRAP_OFF_BUFFER_PTRS + static_cast<ptrdiff_t>(ptr_offset * sizeof(uint64_t)),
                buffer_count * sizeof(uint64_t)
            );
        }

        ChipBootstrapDomainResult result(
            name, read_i32(record + DOMAIN_OFF_DOMAIN_RANK), read_i32(record + DOMAIN_OFF_DOMAIN_SIZE),
            read_u64(record + DOMAIN_OFF_DEVICE_CTX), read_u64(record + DOMAIN_OFF_LOCAL_WINDOW_BASE),
            read_u64(record + DOMAIN_OFF_ACTUAL_WINDOW_SIZE), std::move(ptrs)
        );
        if (result.domain_rank < 0 || result.domain_size <= 0 || result.domain_rank >= result.domain_size) {
            throw std::runtime_error("bootstrap domain rank/size is invalid");
        }
        results.emplace_back(std::move(result));
    }

    return results;
}

ChipBootstrapDomainResult ChipBootstrapChannel::domain(const std::string &name) const {
    for (auto &domain : domains()) {
        if (domain.name == name) {
            return domain;
        }
    }
    throw std::out_of_range("bootstrap domain not found: " + name);
}

uint64_t ChipBootstrapChannel::device_ctx() const { return domain("default").device_ctx; }

uint64_t ChipBootstrapChannel::local_window_base() const { return domain("default").local_window_base; }

uint64_t ChipBootstrapChannel::actual_window_size() const { return domain("default").actual_window_size; }

std::vector<uint64_t> ChipBootstrapChannel::buffer_ptrs() const { return domain("default").buffer_ptrs; }

std::string ChipBootstrapChannel::error_message() const {
    auto *base = static_cast<const char *>(mailbox_);
    const char *msg_ptr = base + CHIP_BOOTSTRAP_OFF_ERROR_MSG;
    // Bound the read against the layout size so a missing null-terminator in
    // shared memory (corrupt producer, premature read) can't walk off the page.
    size_t len = strnlen(msg_ptr, CHIP_BOOTSTRAP_ERROR_MSG_SIZE);
    return std::string(msg_ptr, len);
}
