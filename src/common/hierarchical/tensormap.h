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
 * TensorMap — base_ptr → producer task slot mapping.
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
#include <unordered_map>
#include <vector>

#include "types.h"

class TensorMap {
public:
    // Look up the producer for tensor base_ptr.
    // Returns INVALID_SLOT when not found.
    TaskSlot lookup(uint64_t base_ptr) const;

    // Register base_ptr → producer mapping.
    // Overwrites any existing entry (re-use of the same buffer by a new producer).
    void insert(uint64_t base_ptr, TaskSlot producer);

    // Remove all entries whose key appears in 'keys'.
    // Called when a producer task transitions to CONSUMED.
    void erase_task_outputs(const std::vector<uint64_t> &keys);

    // Number of entries currently tracked.
    int32_t size() const;

private:
    std::unordered_map<uint64_t, TaskSlot> map_;
};
