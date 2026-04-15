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
 * DistScope — scope-depth tracking and scope-owned reference management.
 *
 * A scope is a bracket around a group of submitted tasks.  Each task inside
 * a scope carries one extra "scope reference" (counted in fanout_total).  When
 * scope_end() is called, that reference is released for every task in the scope,
 * allowing tasks that have no downstream consumers to reach CONSUMED.
 *
 * Orch-owned: single-threaded, no locking required.
 *
 * Mirrors L2 scope_begin / scope_end semantics.
 */

#pragma once

#include <functional>
#include <stdexcept>
#include <vector>

#include "dist_types.h"

class DistScope {
public:
    // Open a new scope level.
    void scope_begin();

    // Close innermost scope.
    // Calls release_fn(slot) for every task registered in this scope.
    void scope_end(const std::function<void(DistTaskSlot)> &release_fn);

    // Register a task as belonging to the current innermost scope.
    // Must be called after scope_begin() and before scope_end().
    void register_task(DistTaskSlot slot);

    // Current nesting depth (0 = no open scope).
    int32_t depth() const { return static_cast<int32_t>(stack_.size()); }

    // L2-style 0-based scope index: the innermost open scope, or 0 when
    // none is open. Used by DistRing::alloc to choose a heap ring:
    //   ring_idx = min(current_depth(), DIST_MAX_RING_DEPTH - 1)
    // Matches `PTO2OrchestratorState::current_ring_id` semantics: the
    // first scope opened (depth()==1) maps to ring 0, the next nested
    // scope maps to ring 1, and so on. Returns 0 when no scope is open
    // so tasks submitted outside `Worker::run` still have a deterministic
    // ring assignment.
    int32_t current_depth() const {
        int32_t d = depth();
        return d > 0 ? d - 1 : 0;
    }

private:
    struct ScopeFrame {
        std::vector<DistTaskSlot> tasks;
    };
    std::vector<ScopeFrame> stack_;
};
