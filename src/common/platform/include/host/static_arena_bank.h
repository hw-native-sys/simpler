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

#include "common/unified_log.h"
#include "utils/device_arena.h"
#include "../../../worker/runtime_c_api.h"

/**
 * The capacity rule for one arena bank's three pooled regions, shared by the
 * onboard and simulation runners so both hold the same rule rather than two
 * copies of it.
 */

/**
 * One region of a bank: the arena backing it, the size the bank remembers for
 * it, and the size this call asks for. `cached_size` is updated in place to
 * whatever the region holds when the call returns.
 */
struct StaticArenaRegionRequest {
    DeviceArena *arena;
    size_t *cached_size;
    size_t requested_size;
};

/** The three regions a bank commits together. */
struct StaticArenaBankRequest {
    StaticArenaRegionRequest gm_heap;
    StaticArenaRegionRequest gm_sm;
    StaticArenaRegionRequest runtime_pool;
};

/**
 * `rc` is 0 or PTO_RUNTIME_ERR_INTERNAL. `bases_changed` is true when any
 * region was released or (re)committed, so the pooled base addresses handed
 * out earlier are no longer current and any image keyed on them is stale.
 */
struct StaticArenaBankOutcome {
    int rc{0};
    bool bases_changed{false};
};

/**
 * Commit the bank: grow a region whose request exceeds the remembered size,
 * release one whose request is zero, and leave one whose request already fits
 * untouched — so a steady-state caller mutates nothing.
 *
 * `kernel_mode` selects the context-static capacity rule. A kernel-mode
 * context's config never changes, so each region is committed at most once and
 * is neither grown nor released afterwards: a captured graph replays the base
 * address the region held when it was captured. A request that would re-base
 * or release a committed region is therefore an internal invariant break, not
 * caller-configurable behavior, and is refused with PTO_RUNTIME_ERR_INTERNAL.
 *
 * The refusal is the one failure that leaves the bank exactly as it was. Every
 * other failure rolls the whole bank back — releasing each region and zeroing
 * its cached size, peers from earlier successful calls included, so a caller
 * that retries starts from the post-construction state rather than a partial
 * layout. Rolling a refusal back would free the very base addresses the rule
 * exists to hold still, so the two failures cannot share one exit.
 */
inline StaticArenaBankOutcome commit_static_arena_bank(const StaticArenaBankRequest &request, bool kernel_mode) {
    const StaticArenaRegionRequest regions[] = {request.gm_heap, request.gm_sm, request.runtime_pool};
    StaticArenaBankOutcome outcome;
    bool capacity_refused = false;

    for (const StaticArenaRegionRequest &region : regions) {
        DeviceArena &arena = *region.arena;
        size_t &cached_size = *region.cached_size;
        const size_t requested_size = region.requested_size;

        if (kernel_mode && arena.is_committed() &&
            (requested_size == 0 ? cached_size != 0 : requested_size > cached_size)) {
            LOG_ERROR(
                "setup_static_arena: kernel mode forbids %s a committed region (cached %zu, requested %zu)",
                requested_size == 0 ? "releasing" : "growing", cached_size, requested_size
            );
            capacity_refused = true;
            outcome.rc = PTO_RUNTIME_ERR_INTERNAL;
            break;
        }
        if (requested_size == 0) {
            // hbg's runtime_arena path: caller passed 0 and never reserved a
            // region. Leave the arena uncommitted; acquire_pooled_* will
            // return nullptr.
            if (arena.is_committed() && cached_size != 0) {
                arena.release();
                cached_size = 0;
                outcome.bases_changed = true;
            }
            continue;
        }
        if (arena.is_committed() && requested_size <= cached_size) {
            continue;
        }
        arena.release();
        cached_size = 0;
        outcome.bases_changed = true;
        arena.reserve(requested_size, DeviceArena::kDefaultBaseAlign);
        if (arena.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
            // commit() failure leaves committed_=false, so the rollback below
            // skips this arena's release branch. release() is idempotent on a
            // never-committed arena (zeroes cursor_).
            arena.release();
            outcome.rc = PTO_RUNTIME_ERR_INTERNAL;
            break;
        }
        cached_size = requested_size;
    }

    if (outcome.rc != 0 && !capacity_refused) {
        for (const StaticArenaRegionRequest &region : regions) {
            region.arena->release();
            *region.cached_size = 0;
        }
        outcome.bases_changed = true;
    }
    return outcome;
}
