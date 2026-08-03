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

#ifndef SRC_COMMON_WORKER_PIPELINE_SLOT_POOL_H_
#define SRC_COMMON_WORKER_PIPELINE_SLOT_POOL_H_

#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>

#include "pto_runtime_c_api.h"

/**
 * Fixed-capacity owner of generation-safe pipeline slot leases.
 *
 * This is capability only: callers decide when admission is allowed. A
 * released lease remains idempotently releasable until the slot is acquired
 * again; after reuse, the old generation can neither access nor release it.
 */
class PipelineSlotPool {
public:
    explicit PipelineSlotPool(uint32_t depth) :
        depth_(depth) {
        if (depth == 0 || depth > PTO_PIPELINE_MAX_DEPTH) {
            throw std::invalid_argument("pipeline slot depth is outside the supported range");
        }
    }

    std::optional<PipelineSlotLease> try_acquire() { return try_acquire(depth_); }

    std::optional<PipelineSlotLease> try_acquire(uint32_t admission_depth) {
        if (admission_depth == 0 || admission_depth > depth_) {
            throw std::invalid_argument("pipeline admission depth is outside the pool range");
        }
        std::lock_guard<std::mutex> lock(mu_);
        for (uint32_t slot = 0; slot < admission_depth; ++slot) {
            SlotState &state = slots_[slot];
            if (state.in_use) continue;
            if (state.generation == std::numeric_limits<uint64_t>::max()) {
                throw std::overflow_error("pipeline slot generation space exhausted");
            }
            ++state.generation;
            state.in_use = true;
            return PipelineSlotLease{slot, 0, state.generation};
        }
        return std::nullopt;
    }

    bool owns(const PipelineSlotLease &lease) const {
        std::lock_guard<std::mutex> lock(mu_);
        return owns_locked(lease);
    }

    bool release(const PipelineSlotLease &lease) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!matches_generation_locked(lease)) return false;
        // Releasing the current generation twice is deliberately idempotent.
        slots_[lease.slot_id].in_use = false;
        return true;
    }

    uint32_t depth() const { return depth_; }

private:
    struct SlotState {
        uint64_t generation{0};
        bool in_use{false};
    };

    bool matches_generation_locked(const PipelineSlotLease &lease) const {
        return lease.reserved == 0 && lease.generation != 0 && lease.slot_id < depth_ &&
               slots_[lease.slot_id].generation == lease.generation;
    }

    bool owns_locked(const PipelineSlotLease &lease) const {
        return matches_generation_locked(lease) && slots_[lease.slot_id].in_use;
    }

    const uint32_t depth_;
    mutable std::mutex mu_;
    std::array<SlotState, PTO_PIPELINE_MAX_DEPTH> slots_{};
};

/**
 * Replay filter for a consumer that executes leases minted elsewhere.
 *
 * `PipelineSlotPool` is the sole mint and the sole authority on who owns a
 * slot. A consumer downstream of it never sees an acquire or a release, so it
 * cannot re-derive ownership; it records the newest generation each slot has
 * presented and refuses anything older. Repeating the current generation stays
 * admissible, because one run may dispatch several times under one lease.
 *
 * This is strictly weaker than an ownership check, and deliberately so. It
 * rejects a *superseded* lease — one whose successor has already presented
 * itself — but a lease that was released while its successor has not yet
 * reached this consumer still passes, because nothing has raised the mark.
 * Closing that window requires gating dispatch on `PipelineSlotPool::owns`
 * before work is handed down, which belongs to whoever admits runs.
 *
 * The filter mints nothing. A consumer that also served unleased work must not
 * invent generations for it, or a later real lease is rejected as stale.
 */
class PipelineSlotGenerationFilter {
public:
    bool is_admissible(const PipelineSlotLease &lease) {
        if (lease.slot_id >= newest_.size()) return false;
        std::lock_guard<std::mutex> lock(mu_);
        return lease.generation >= newest_[lease.slot_id];
    }

    bool admit(const PipelineSlotLease &lease) {
        if (lease.slot_id >= newest_.size()) return false;
        std::lock_guard<std::mutex> lock(mu_);
        uint64_t &newest = newest_[lease.slot_id];
        if (lease.generation < newest) return false;
        newest = lease.generation;
        return true;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mu_);
        newest_.fill(0);
    }

private:
    std::mutex mu_;
    std::array<uint64_t, PTO_PIPELINE_MAX_DEPTH> newest_{};
};

#endif  // SRC_COMMON_WORKER_PIPELINE_SLOT_POOL_H_
