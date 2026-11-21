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
#include <functional>
#include <memory>
#include <mutex>

#include "runtime_c_api.h"
#include "utils/device_arena.h"

// One fixed allocation per ordered execution context. Leases pin its address,
// not its contents; the invocation ordering contract governs scratch reuse.
class ScratchPool {
public:
    using AllocFn = DeviceArena::AllocFn;
    using FreeFn = DeviceArena::FreeFn;
    using RetireFn = std::function<void(void *raw)>;
    struct Lease {
        void *base{nullptr};
        size_t capacity{0};
        void *raw{nullptr};
        bool valid() const noexcept { return base != nullptr; }
    };
    ScratchPool(AllocFn alloc, FreeFn free_fn, void *ctx, RetireFn retire = {}) :
        arena_(alloc, free_fn, ctx),
        retire_(std::move(retire)) {}
    ~ScratchPool() { reset(); }
    ScratchPool(const ScratchPool &) = delete;
    ScratchPool &operator=(const ScratchPool &) = delete;

    // The first successful allocation fixes capacity for the context lifetime.
    // Later calls must supply the same configured capacity, even with no leases.
    int acquire(size_t capacity, Lease *out) {
        if (out == nullptr || capacity == 0) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> lock(mutex_);
        if (lease_.valid() && capacity != lease_.capacity) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        if (!lease_.valid()) {
            try {
                arena_.reserve(capacity, DeviceArena::kDefaultBaseAlign);
                void *base = arena_.commit(DeviceArena::kDefaultBaseAlign);
                if (base == nullptr) {
                    arena_.release();
                    return PTO_RUNTIME_ERR_INTERNAL;
                }
                lease_ = {base, capacity, arena_.raw_backing()};
            } catch (...) {
                arena_.release();
                return PTO_RUNTIME_ERR_INTERNAL;
            }
        }
        ++leases_;
        *out = lease_;
        return 0;
    }
    void release(const Lease &lease) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (lease.valid() && lease.base == lease_.base && leases_ != 0) --leases_;
    }
    bool covers(size_t bytes) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return lease_.valid() && lease_.capacity == bytes;
    }
    size_t capacity() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return lease_.capacity;
    }
    void *current_base() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return lease_.base;
    }
    size_t live_leases() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return leases_;
    }
    // Only context teardown may release backing, after all device users finish.
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (lease_.valid() && retire_) retire_(lease_.raw);
        arena_.release();
        lease_ = {};
        leases_ = 0;
    }
    void abandon() {
        std::lock_guard<std::mutex> lock(mutex_);
        arena_.abandon_after_device_failure();
        lease_ = {};
        leases_ = 0;
    }

private:
    DeviceArena arena_;
    RetireFn retire_;
    mutable std::mutex mutex_;
    Lease lease_;
    size_t leases_{0};
};
