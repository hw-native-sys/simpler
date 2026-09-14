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

#include <algorithm>
#include <cstring>
#include <vector>

#include "chip_callable_layout.h"
#include "callable_protocol.h"
#include "runtime_c_api.h"
#include "kernel_entry_validation.h"

struct KernelCallableResidency {
    int32_t callable_id{-1};
    uint64_t device_address{0};
    size_t bytes{0};
};

// The caller serializes prepare/resolve/close. Entries and device addresses
// remain immutable until external quiescence permits context close.
class KernelCallableCache {
public:
    static constexpr size_t kByteLimit = 2ULL * 1024 * 1024 * 1024;
    static constexpr size_t kBlockSize = 2ULL * 1024 * 1024;

    explicit KernelCallableCache(size_t byte_limit = kByteLimit) :
        byte_limit_(std::min(byte_limit, kByteLimit)) {}

    struct Ops {
        void *context;
        void *(*allocate)(void *, size_t);
        int (*copy)(void *, void *, const void *, size_t);
    };

    int stage(const ChipCallable *callable, size_t bytes, const Ops &ops, int32_t &out_callable_id) {
        out_callable_id = -1;
        if (bytes > byte_limit_) return PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED;
        int rc = validate_image(callable, bytes);
        if (rc != 0) return rc;
        const auto layout = compute_chip_callable_layout(callable);
        if (!entries_.empty() && !entries_.back().ready) return PTO_RUNTIME_ERR_INVALID_STATE;
        if (entries_.size() >= MAX_REGISTERED_CALLABLE_IDS) return PTO_RUNTIME_ERR_CALLABLE_COUNT_EXCEEDED;
        const size_t padding = (CALLABLE_ALIGN - bytes % CALLABLE_ALIGN) % CALLABLE_ALIGN;
        if (bytes > byte_limit_ - used_ || padding > byte_limit_ - used_ - bytes)
            return PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED;
        const size_t charged = bytes + padding;
        size_t block_index = blocks_.size();
        for (size_t i = 0; i < blocks_.size(); ++i) {
            const auto &block = blocks_[i];
            if (charged <= block.capacity - block.used) {
                block_index = i;
                break;
            }
        }
        const size_t allocation_size = std::max(charged, std::min(kBlockSize, byte_limit_));
        if (block_index == blocks_.size() && allocation_size > byte_limit_ - allocated_)
            return PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED;
        const auto id = static_cast<int32_t>(entries_.size());
        Entry candidate;
        candidate.residency = {id, 0, bytes};
        candidate.image = std::vector<uint8_t>(
            reinterpret_cast<const uint8_t *>(callable), reinterpret_cast<const uint8_t *>(callable) + bytes
        );
        candidate.charged = charged;
        entries_.push_back(std::move(candidate));
        try {
            auto &entry = entries_.back();
            if (block_index == blocks_.size()) {
                blocks_.push_back({nullptr, allocation_size, 0});
                try {
                    blocks_.back().address = ops.allocate(ops.context, allocation_size);
                } catch (...) {
                    blocks_.pop_back();
                    throw;
                }
                if (!blocks_.back().address) {
                    blocks_.pop_back();
                    entries_.pop_back();
                    return PTO_RUNTIME_ERR_INTERNAL;
                }
                allocated_ += allocation_size;
            }
            auto &block = blocks_[block_index];
            entry.block_index = block_index;
            entry.residency.device_address = reinterpret_cast<uint64_t>(block.address) + block.used;
            std::vector<uint8_t> scratch(entry.image);
            patch_chip_callable_scratch_for_device(callable, layout, entry.residency.device_address, scratch.data());
            rc = ops.copy(ops.context, reinterpret_cast<void *>(entry.residency.device_address), scratch.data(), bytes);
            if (rc != 0) {
                entries_.pop_back();
                return rc;
            }
            block.used += charged;
            used_ += charged;
            out_callable_id = id;
        } catch (...) {
            entries_.pop_back();
            throw;
        }
        return 0;
    }

    void commit(int32_t id) {
        if (!entries_.empty() && entries_.back().residency.callable_id == id) entries_.back().ready = true;
    }
    void rollback(int32_t id) {
        if (!entries_.empty() && !entries_.back().ready && entries_.back().residency.callable_id == id) {
            blocks_[entries_.back().block_index].used -= entries_.back().charged;
            used_ -= entries_.back().charged;
            entries_.pop_back();
        }
    }
    int resolve(int32_t id, KernelCallableResidency &out) const {
        out = {};
        if (id < 0) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        if (id >= MAX_REGISTERED_CALLABLE_IDS) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        if (static_cast<size_t>(id) < entries_.size() && entries_[id].ready) {
            out = entries_[id].residency;
            return 0;
        }
        return PTO_RUNTIME_ERR_CALLABLE_NOT_RESIDENT;
    }
    // The serialized registration bridge consumes only this admission's upload.
    // Committed residents are never candidates for a later registration.
    uint64_t pending_uploaded_address() const {
        if (entries_.empty() || entries_.back().ready) return 0;
        return entries_.back().residency.device_address;
    }
    size_t resident_bytes() const { return used_; }
    size_t allocated_bytes() const { return allocated_; }
    size_t resident_count() const {
        return std::count_if(entries_.begin(), entries_.end(), [](const Entry &entry) {
            return entry.ready;
        });
    }
    size_t host_bytes() const { return used_ - padding_bytes(); }
    // MemoryAllocator owns the device blocks; this operation only drops host metadata.
    void clear() {
        entries_.clear();
        blocks_.clear();
        allocated_ = 0;
        used_ = 0;
    }

    static int validate_image(const ChipCallable *callable, size_t bytes) {
        if (!callable || bytes < sizeof(ChipCallable) || reinterpret_cast<uintptr_t>(callable) % alignof(ChipCallable))
            return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        const int rc = validate_kernel_callable_image(callable, bytes);
        if (rc != 0) return rc;
        int32_t scalars = 0;
        for (int32_t i = 0; i < callable->sig_count_; ++i) {
            const auto direction = callable->signature_[i];
            if (direction < ArgDirection::SCALAR || direction > ArgDirection::INOUT)
                return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
            scalars += direction == ArgDirection::SCALAR;
        }
        return scalars <= CHIP_MAX_SCALAR_ARGS ? 0 : PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    }

private:
    struct Block {
        void *address;
        size_t capacity;
        size_t used;
    };
    struct Entry {
        KernelCallableResidency residency;
        std::vector<uint8_t> image;
        size_t charged{0};
        size_t block_index{0};
        bool ready{false};
    };
    size_t padding_bytes() const {
        size_t padding = 0;
        for (const auto &entry : entries_)
            if (entry.charged) padding += entry.charged - entry.image.size();
        return padding;
    }
    size_t byte_limit_;
    size_t used_{0};
    size_t allocated_{0};
    std::vector<Block> blocks_;
    std::vector<Entry> entries_;
};
