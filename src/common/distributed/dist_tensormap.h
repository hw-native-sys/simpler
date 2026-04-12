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
 * DistTensorMap — base_ptr → producer task slot mapping.
 *
 * At the distributed host level, every tensor is identified by its base pointer.
 * When a task produces an output, it registers the output's base_ptr here.
 * When a later task lists an input, lookup() finds the producer and creates a
 * fanin dependency edge.
 *
 * Unlike the L2 PTO2TensorMap, this implementation:
 *   - Uses std::unordered_map (no ring buffer entry pool)
 *   - Does not perform overlap detection (each base_ptr maps to one producer)
 *   - Cleans up entries actively when a task is CONSUMED
 *
 * Owned exclusively by the Orchestrator (main thread); no locking required.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "dist_types.h"

class DistTensorMap {
public:
    // Look up the producer for tensor base_ptr.
    // Returns DIST_INVALID_SLOT when not found.
    DistTaskSlot lookup(const DistTensorKey &key) const;

    // Register base_ptr → producer mapping.
    // Overwrites any existing entry (re-use of the same buffer by a new producer).
    void insert(const DistTensorKey &key, DistTaskSlot producer);

    // Remove all entries whose key appears in 'keys'.
    // Called when a producer task transitions to CONSUMED.
    void erase_task_outputs(const std::vector<DistTensorKey> &keys);

    // Number of entries currently tracked.
    int32_t size() const;

private:
    struct DistTensorKeyHash {
        size_t operator()(const DistTensorKey &key) const {
            size_t h1 = std::hash<int32_t>{}(key.worker_index);
            size_t h2 = std::hash<uint64_t>{}(key.base_ptr);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    std::unordered_map<DistTensorKey, DistTaskSlot, DistTensorKeyHash> map_;
};
