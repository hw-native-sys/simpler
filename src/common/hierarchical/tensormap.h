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
 * TensorMap — TensorKey → producer task slot mapping.
 *
 * At the hierarchical host level, every tensor is identified by a TensorKey
 * consisting of (ptr, worker).  Host tensors (HeapRing) use worker=-1 because
 * their addresses are globally unique; child_memory tensors use the owning
 * worker's logical id (0..N-1) to disambiguate identical device addresses
 * across different NPUs.
 *
 * Unlike the L2 PTO2TensorMap, this implementation:
 *   - Uses std::unordered_map (no ring buffer entry pool)
 *   - Does not perform overlap detection (each key maps to one producer)
 *   - Cleans up entries actively when a task is CONSUMED
 *
 * Owned exclusively by the Orchestrator (main thread); no locking required.
 */

#pragma once

#include <unordered_map>
#include <vector>

#include "types.h"

class TensorMap {
public:
    // Look up the producer for a tensor key.
    // Returns INVALID_SLOT when not found.
    TaskSlot lookup(TensorKey key) const;

    // Register key → producer mapping.
    // Overwrites any existing entry (re-use of the same buffer by a new producer).
    void insert(TensorKey key, TaskSlot producer);

    // Remove all entries whose key appears in 'keys'.
    // Called when a producer task transitions to CONSUMED.
    void erase_task_outputs(const std::vector<TensorKey> &keys);

    // Number of entries currently tracked.
    int32_t size() const;

private:
    std::unordered_map<TensorKey, TaskSlot, TensorKeyHash> map_;
};
