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
#include <cstring>
#include <vector>

#include "host/raii_scope_guard.h"
#include "runtime_c_api.h"
#include "utils/device_arena.h"

/**
 * The replacement transaction one `setup_static_arena` call performs over the
 * pooled arena regions of a single bank.
 *
 * A region is backed by exactly one allocation, so growing it means allocating
 * a replacement. The transaction stages every replacement it needs before
 * publishing any of them, which is what makes a failed allocation leave the
 * caller's addresses and capacities exactly as they were: an arena's base,
 * region table and committed flag answer from the current backing until
 * publication, and no cached size is written before it.
 *
 * The cost of that ordering is occupancy. A region being replaced holds its
 * old and new backing at the same time, from the first staging until
 * publication frees the superseded block — so a layout that a free-first
 * sequence could have fitted into the remaining device memory can be refused
 * here. The window spans the whole transaction, not one region's replacement,
 * and a superseded block whose free fails stays charged to the allocator until
 * the existing finalize/abandon path retires it.
 *
 * `requested_size == 0` means the region holds nothing. Its release is
 * deferred to publication for the same reason a replacement is: a peer's
 * allocation failure must not have cleared it.
 */

/** Upper bound on the regions one transaction covers; a bank has three. */
inline constexpr size_t kMaxArenaTransactionRegions = 4;

struct ArenaRegionRequest {
    DeviceArena *arena;
    // The capacity the caller publishes for this region. Written only on
    // publication.
    size_t *cached_size;
    size_t requested_size;
    const char *name;
    // Announced immediately before this region's backing is staged, for a
    // caller whose allocation callback needs to know which region is asking —
    // the callback's own arguments carry only a byte count. Null when the
    // caller has no such need, which is every caller that owns one pool.
    void (*announce)(void *ctx, size_t region_index){nullptr};
    void *announce_ctx{nullptr};
};

enum class ArenaRegionAction {
    // The committed backing already covers the request, or nothing is
    // committed and nothing is asked for. No allocation, no free.
    Keep,
    // Needs a replacement backing.
    Replace,
    // Committed, and asked to hold nothing.
    Release,
};

/**
 * What `request` needs, decided from state the transaction has not mutated.
 *
 * Reading this for every region before staging anything is what lets the
 * transaction know its whole plan while every arena is still untouched.
 */
inline ArenaRegionAction arena_region_action(const ArenaRegionRequest &request) {
    const DeviceArena &arena = *request.arena;
    if (request.requested_size == 0) {
        return (arena.is_committed() && *request.cached_size != 0) ? ArenaRegionAction::Release :
                                                                     ArenaRegionAction::Keep;
    }
    if (arena.is_committed() && request.requested_size <= *request.cached_size) return ArenaRegionAction::Keep;
    return ArenaRegionAction::Replace;
}

struct ArenaTransactionResult {
    // Whether the plan took effect. False means every arena still holds the
    // base, region table and cached size it had on entry.
    bool published{false};
    // Which region's staging allocation failed, or -1 when none did.
    int failed_region{-1};
    // How many regions changed base or were released. A cache keyed on a
    // region base stays valid while this is zero.
    size_t changed_regions{0};
};

/**
 * Stage every replacement, publish them together, then free what they
 * superseded.
 *
 * On a staging failure only this call's staged blocks are freed; `requests`
 * and their arenas are left as they were found, including a region whose
 * request was zero. Publication allocates nothing and cannot throw, so no
 * region can be left half-installed. The frees that follow it are reported by
 * neither the arena nor this function — a failed free keeps the block charged
 * to the allocator, and publication does not roll back for it.
 */
inline ArenaTransactionResult run_arena_replacement_transaction(
    ArenaRegionRequest *requests, size_t count, size_t base_align = DeviceArena::kDefaultBaseAlign
) {
    ArenaTransactionResult result;
    // Nothing was staged, so no region failed: `failed_region` stays -1 and the
    // caller's log does not name a region for what is an argument error.
    if (requests == nullptr || count == 0 || count > kMaxArenaTransactionRegions) return result;

    ArenaRegionAction actions[kMaxArenaTransactionRegions] = {};
    for (size_t i = 0; i < count; ++i)
        actions[i] = arena_region_action(requests[i]);

    // Covers a throwing allocation as well as a null return: either way the
    // staged blocks this call created are the only ones to drop.
    auto staged_guard = RAIIScopeGuard([&]() {
        for (size_t i = 0; i < count; ++i)
            requests[i].arena->abort_replacement();
    });

    for (size_t i = 0; i < count; ++i) {
        if (actions[i] != ArenaRegionAction::Replace) continue;
        if (requests[i].announce != nullptr) requests[i].announce(requests[i].announce_ctx, i);
        if (requests[i].arena->stage_replacement(requests[i].requested_size, base_align) != nullptr) continue;
        result.failed_region = static_cast<int>(i);
        return result;
    }
    staged_guard.dismiss();

    void *superseded[kMaxArenaTransactionRegions] = {};
    for (size_t i = 0; i < count; ++i) {
        if (actions[i] == ArenaRegionAction::Replace) {
            superseded[i] = requests[i].arena->publish_replacement();
            *requests[i].cached_size = requests[i].requested_size;
            ++result.changed_regions;
        } else if (actions[i] == ArenaRegionAction::Release) {
            superseded[i] = requests[i].arena->detach_backing();
            *requests[i].cached_size = 0;
            ++result.changed_regions;
        }
    }
    result.published = true;

    for (size_t i = 0; i < count; ++i)
        requests[i].arena->free_superseded(superseded[i]);
    return result;
}

/**
 * The single prebuilt runtime-arena image entry an arena bank can hold.
 *
 * The entry advertises the bank's three region bases alongside the image built
 * for them, so it is only answerable while those bases stand. A publication
 * that moves any of them invalidates it; a transaction that changed nothing
 * leaves it answerable, which is what lets a prepare whose layout the existing
 * regions already cover keep skipping the rebuild.
 */
class PrebuiltRuntimeArenaCache {
public:
    void invalidate() {
        valid_ = false;
        hash_ = 0;
        key_.clear();
        gm_heap_base_ = nullptr;
        sm_base_ = nullptr;
        runtime_arena_base_ = nullptr;
        runtime_off_ = 0;
        image_.clear();
    }

    bool is_valid() const { return valid_; }

    bool lookup(
        uint64_t hash, const void *key_data, size_t key_size, void **gm_heap_base, void **sm_base,
        void **runtime_arena_base, size_t *runtime_off, const void **image_data, size_t *image_size
    ) const {
        if (!valid_ || hash_ != hash || key_.size() != key_size || key_data == nullptr || gm_heap_base == nullptr ||
            sm_base == nullptr || runtime_arena_base == nullptr || runtime_off == nullptr || image_data == nullptr ||
            image_size == nullptr) {
            return false;
        }
        if (std::memcmp(key_.data(), key_data, key_size) != 0) return false;
        *gm_heap_base = gm_heap_base_;
        *sm_base = sm_base_;
        *runtime_arena_base = runtime_arena_base_;
        *runtime_off = runtime_off_;
        *image_data = image_.data();
        *image_size = image_.size();
        return true;
    }

    void store(
        uint64_t hash, const void *key_data, size_t key_size, void *gm_heap_base, void *sm_base,
        void *runtime_arena_base, size_t runtime_off, const void *image_data, size_t image_size
    ) {
        valid_ = false;
        hash_ = hash;
        key_.assign(static_cast<const uint8_t *>(key_data), static_cast<const uint8_t *>(key_data) + key_size);
        gm_heap_base_ = gm_heap_base;
        sm_base_ = sm_base;
        runtime_arena_base_ = runtime_arena_base;
        runtime_off_ = runtime_off;
        image_.assign(static_cast<const uint8_t *>(image_data), static_cast<const uint8_t *>(image_data) + image_size);
        valid_ = true;
    }

private:
    bool valid_{false};
    uint64_t hash_{0};
    std::vector<uint8_t> key_;
    void *gm_heap_base_{nullptr};
    void *sm_base_{nullptr};
    void *runtime_arena_base_{nullptr};
    size_t runtime_off_{0};
    std::vector<uint8_t> image_;
};

struct BankArenaSetupOutcome {
    // What the caller returns. A staging failure is an error rather than a
    // success at the capacity the bank already had.
    int rc{PTO_RUNTIME_ERR_INTERNAL};
    ArenaTransactionResult transaction{};
    // Whether this call ran the entry's invalidation, which it does whenever a
    // base moved on the bank that owns the entry. A no-op when nothing was
    // stored, so this reports the step rather than a retired entry.
    bool cache_invalidated{false};
};

/**
 * Commit one bank's regions to `requests` and settle its prebuilt-image entry.
 *
 * `owns_prebuilt_cache` is whether this bank is the one the entry describes;
 * only that bank's publication can invalidate it. A staging failure returns an
 * error with the entry untouched — no base moved, so what it advertises is
 * still true.
 */
inline BankArenaSetupOutcome run_bank_arena_setup(
    ArenaRegionRequest *requests, size_t count, bool owns_prebuilt_cache, PrebuiltRuntimeArenaCache *cache,
    size_t base_align = DeviceArena::kDefaultBaseAlign
) {
    BankArenaSetupOutcome outcome;
    outcome.transaction = run_arena_replacement_transaction(requests, count, base_align);
    if (!outcome.transaction.published) return outcome;
    if (outcome.transaction.changed_regions != 0 && owns_prebuilt_cache && cache != nullptr) {
        cache->invalidate();
        outcome.cache_invalidated = true;
    }
    outcome.rc = 0;
    return outcome;
}
