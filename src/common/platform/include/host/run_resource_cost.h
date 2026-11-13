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

#include <cstdint>

/**
 * Device bytes one run-resource set committed.
 *
 * A footprint, not a marginal delta: a region the set shares with an
 * already-proven set — the pooled GM heap is the usual one — is reported by
 * every holder of it. `budgeted_bytes()` is the part the workspace ledger's
 * own limit covers, because the ledger allocates the arena regions; the
 * scheduler state and the run-result region are allocated through the device
 * allocator directly, so they are part of the set's footprint but outside that
 * ledger's budget.
 */
struct RunResourceCost {
    /** GM heap, pooled lease capacity or the bank's own committed region. */
    std::uint64_t heap_bytes{0};
    /** The runtime-image region (`runtime_pool`) of the bank the set selected. */
    std::uint64_t image_bytes{0};
    /** The GM shared-memory region of that bank. */
    std::uint64_t shared_memory_bytes{0};
    /** This set's retained scheduler state and ready-pool scratch (a5); zero where the runtime has none. */
    std::uint64_t scheduler_bytes{0};
    /** The run-result region this run's epoch committed, when it has one. */
    std::uint64_t reported_bytes{0};

    /** Bytes the workspace ledger budgets and charges. */
    std::uint64_t budgeted_bytes() const { return heap_bytes + image_bytes + shared_memory_bytes; }

    /** Everything the set committed, budgeted or not. */
    std::uint64_t total_bytes() const { return budgeted_bytes() + scheduler_bytes + reported_bytes; }

    bool any() const { return total_bytes() != 0; }
};
