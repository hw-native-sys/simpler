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
 * TensorMap — (RunId, TensorKey) → producer task slot mapping.
 *
 * At the hierarchical host level, every tensor is identified by a TensorKey
 * consisting of (ptr, worker_id). Host tensors (HeapRing) use
 * worker_id=-1 because their addresses are globally unique; child_memory
 * tensors use the owning NEXT_LEVEL worker id to disambiguate identical
 * device addresses across different children.
 *
 * Unlike the L2 PTO2TensorMap, this implementation:
 *   - Uses std::unordered_map (no ring buffer entry pool)
 *   - Does not perform overlap detection (each key maps to one producer)
 *   - Cleans up entries actively when a task is CONSUMED
 *
 * Owned exclusively by the Orchestrator (main thread); no locking required.
 */

#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

#include "types.h"

class TensorMap {
public:
    // Look up the producer for a tensor key.
    // Returns INVALID_SLOT when not found.
    TaskSlot lookup(RunId run_id, TensorKey key) const;

    // Register key → producer mapping.
    // Overwrites any existing entry (re-use of the same buffer by a new producer).
    void insert(RunId run_id, TensorKey key, TaskSlot producer);

    // Remove all entries whose key appears in 'keys'.
    // Called when a producer task transitions to CONSUMED.
    void erase_task_outputs(RunId run_id, const std::vector<TensorKey> &keys);

    // Number of entries currently tracked.
    int32_t size() const;

private:
    struct RunTensorKey {
        RunId run_id{INVALID_RUN_ID};
        TensorKey tensor{};

        bool operator==(const RunTensorKey &other) const { return run_id == other.run_id && tensor == other.tensor; }
    };

    struct RunTensorKeyHash {
        size_t operator()(const RunTensorKey &key) const {
            size_t h = std::hash<RunId>{}(key.run_id);
            size_t tensor_hash = TensorKeyHash{}(key.tensor);
            h ^= tensor_hash + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    mutable std::mutex mu_;
    std::unordered_map<RunTensorKey, TaskSlot, RunTensorKeyHash> map_;
};
