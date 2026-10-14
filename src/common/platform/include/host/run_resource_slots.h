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
 * @file run_resource_slots.h
 * @brief One resource per run-resource set, sized to the grant a context was given.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

/**
 * A runner-owned resource that exists once per run-resource set.
 *
 * Indexing is by set, which is what a run's lease names, and the count is the
 * grant that context was given rather than any compile-time width. Storage is
 * contiguous and sized once before any run can name a set, so element addresses
 * remain stable for the table's lifetime without allocating each slot alone.
 *
 * Sizing happens once, before any run can name a set. Growing on first touch
 * would instead have two runs preparing on two sets race on the container
 * itself — which is the one thing a per-set resource exists to avoid — so an
 * index past the grant is a refusal rather than an extension.
 *
 * Sizing is two-phase and the split is load-bearing. `build` allocates and
 * constructs into a detached container and touches no member, so a failure
 * anywhere in it leaves this object exactly as it was; `commit` is `noexcept`,
 * so a grant made of several of these cannot half-apply. A single call that did
 * both would leave the earlier containers resized and the later ones not, with
 * nothing able to tell which.
 */
template <typename T>
class RunResourceSlots {
public:
    using Storage = std::vector<T>;

    RunResourceSlots() = default;

    /**
     * Construct one default-built resource per granted set, detached.
     *
     * May throw; mutates nothing. Whether a resizing is admissible at all is the
     * owner's judgement: it is the side that knows whether a run holds a set,
     * and a set's index is what a lease names.
     */
    static Storage build(size_t granted) {
        if (granted == 0) throw std::invalid_argument("a context must be granted at least one run-resource set");
        return Storage(granted);
    }

    /** Adopt a built set. Never throws, which is what makes a multi-owner grant atomic. */
    void commit(Storage &&built) noexcept { slots_.swap(built); }

    size_t size() const noexcept { return slots_.size(); }
    bool empty() const noexcept { return slots_.empty(); }
    /** Whether a set index is one this context was granted. */
    bool holds(size_t slot) const noexcept { return slot < slots_.size(); }

    decltype(auto) operator[](size_t slot) { return slots_.at(slot); }
    decltype(auto) operator[](size_t slot) const { return slots_.at(slot); }

    /** The resource for `slot`, or `nullptr` when the grant does not reach it. */
    T *find(size_t slot) noexcept { return slot < slots_.size() ? &slots_[slot] : nullptr; }
    const T *find(size_t slot) const noexcept { return slot < slots_.size() ? &slots_[slot] : nullptr; }

    /** Give every granted set the same value, without changing how many there are. */
    void fill(const T &value) {
        std::fill(slots_.begin(), slots_.end(), value);
    }

    auto begin() noexcept { return slots_.begin(); }
    auto end() noexcept { return slots_.end(); }
    auto begin() const noexcept { return slots_.begin(); }
    auto end() const noexcept { return slots_.end(); }

private:
    Storage slots_;
};
