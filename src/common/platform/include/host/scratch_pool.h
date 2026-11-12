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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "runtime_c_api.h"
#include "utils/device_arena.h"

/**
 * Shared device scratch with explicit generations and leases.
 *
 * The current generation serves every acquire whose size it covers. A larger
 * acquire publishes a new generation; the previous one stays allocated until
 * its last lease is released. A lease is the only ownership record: no caller
 * decides reclamation from a shared_ptr count, and a generation with a live
 * lease is never freed. Callers order device execution before reusing its contents.
 *
 * The pool owns device memory, not the workspace ledger. The injected allocator
 * is the caller's, so the runner can route each generation through its ledger
 * and the unit test can use plain malloc.
 */
class ScratchPool {
public:
    using AllocFn = DeviceArena::AllocFn;
    using FreeFn = DeviceArena::FreeFn;

    struct Lease {
        uint64_t generation{0};
        void *base{nullptr};
        size_t capacity{0};
        // The allocator's block behind `base`; the ledger owns it, not the
        // arena that forwards an aligned address inside it.
        void *raw{nullptr};

        bool valid() const noexcept { return generation != 0; }
    };

    // Called when a retired generation's last lease ends, before its device
    // arena is released, so the runner can hand the block back to its ledger.
    using RetireFn = std::function<void(uint64_t generation, void *raw)>;

    ScratchPool(AllocFn alloc, FreeFn free_fn, void *ctx, RetireFn retire = {}) :
        alloc_(alloc),
        free_(free_fn),
        ctx_(ctx),
        retire_(std::move(retire)) {}
    ~ScratchPool() { reset(); }

    ScratchPool(const ScratchPool &) = delete;
    ScratchPool &operator=(const ScratchPool &) = delete;

    /**
     * Lease at least `bytes`, reusing the current generation when it fits.
     *
     * On success `out` names the generation and its base; on failure it is left
     * untouched and every previously acquired lease remains valid.
     */
    int acquire(size_t bytes, Lease *out) {
        if (out == nullptr || bytes == 0) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_ != nullptr && current_->capacity >= bytes) {
            ++current_->leases;
            *out = lease_for(*current_);
            return 0;
        }

        auto next = std::make_unique<Generation>();
        next->arena = std::make_unique<DeviceArena>(alloc_, free_, ctx_);
        next->arena->reserve(bytes, DeviceArena::kDefaultBaseAlign);
        void *base = nullptr;
        try {
            base = next->arena->commit(DeviceArena::kDefaultBaseAlign);
        } catch (...) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        if (base == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        next->id = next_generation_;
        next->base = base;
        next->capacity = bytes;
        next->raw = next->arena->raw_backing();
        next->leases = 1;
        next->current = true;
        // Container growth can throw. Publish only after it owns the new arena.
        generations_.push_back(std::move(next));
        Generation *previous = current_;
        current_ = generations_.back().get();
        ++next_generation_;
        if (previous != nullptr) {
            previous->current = false;
            // An idle predecessor has no later release to trigger reclamation.
            if (previous->leases == 0) {
                if (retire_) retire_(previous->id, previous->raw);
                const auto it = std::find_if(generations_.begin(), generations_.end(), [&](const auto &generation) {
                    return generation.get() == previous;
                });
                generations_.erase(it);
            }
        }
        *out = lease_for(*current_);
        return 0;
    }

    /** Release one lease; the last lease of a retired generation frees it. */
    void release(const Lease &lease) {
        if (lease.generation == 0) return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = std::find_if(generations_.begin(), generations_.end(), [&](const auto &generation) {
            return generation->id == lease.generation;
        });
        if (it == generations_.end() || (*it)->leases == 0) return;
        Generation &generation = **it;
        --generation.leases;
        if (generation.leases == 0 && !generation.current) {
            if (retire_) retire_(generation.id, generation.raw);
            generations_.erase(it);
        }
    }

    /** Whether the current generation covers `bytes` without allocating. */
    bool covers(size_t bytes) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_ != nullptr && current_->capacity >= bytes;
    }

    /** The base of the current generation, or nullptr when none exists. */
    void *current_base() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_ != nullptr ? current_->base : nullptr;
    }

    /** Live generations, including a current generation with no leases. */
    size_t live_generations() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return generations_.size();
    }

    /** Live leases across every generation. */
    size_t live_leases() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t leases = 0;
        for (const auto &generation : generations_)
            leases += generation->leases;
        return leases;
    }

    /** Release every generation, regardless of its leases. */
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (retire_) {
            for (const auto &generation : generations_)
                retire_(generation->id, generation->raw);
        }
        generations_.clear();
        current_ = nullptr;
    }

    /** Forget every generation after a device reset invalidated its backing. */
    void abandon() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &generation : generations_)
            generation->arena->abandon_after_device_failure();
        generations_.clear();
        current_ = nullptr;
    }

private:
    struct Generation {
        uint64_t id{0};
        std::unique_ptr<DeviceArena> arena;
        void *base{nullptr};
        size_t capacity{0};
        void *raw{nullptr};
        size_t leases{0};
        bool current{false};
    };

    static Lease lease_for(const Generation &generation) {
        return Lease{generation.id, generation.base, generation.capacity, generation.raw};
    }

    AllocFn alloc_;
    FreeFn free_;
    void *ctx_{nullptr};
    RetireFn retire_;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Generation>> generations_;
    Generation *current_{nullptr};
    uint64_t next_generation_{1};
};
