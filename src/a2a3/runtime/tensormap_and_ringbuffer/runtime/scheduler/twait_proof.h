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

#include <stdint.h>
#include "tensormap_and_ringbuffer/task_id.h"

namespace twait {

inline constexpr uint32_t max_proof_nodes = 512;
inline constexpr uint32_t proof_words = max_proof_nodes / 64;

struct ProofHandle {
    uint64_t run{0};
    uint32_t index{max_proof_nodes};
};

struct ProofNode {
    uint64_t run{0};
    TaskId task{};
    uint64_t generation{0};
    uint64_t ancestors[proof_words]{};
};

struct ProofStore {
    uint32_t count{0};
    ProofNode nodes[max_proof_nodes]{};

    int32_t find(uint64_t run, TaskId task, uint64_t generation) const {
        for (uint32_t index = 0; index < count; ++index) {
            const auto &node = nodes[index];
            if (node.run == run && node.task == task && node.generation == generation)
                return static_cast<int32_t>(index);
        }
        return -1;
    }

    bool proves(
        ProofHandle handle, uint64_t run, TaskId barrier, uint64_t barrier_generation, TaskId producer,
        uint64_t producer_generation
    ) const {
        if (handle.run != run || handle.index >= count) return false;
        const auto &node = nodes[handle.index];
        if (node.run != run || node.task != barrier || node.generation != barrier_generation) return false;
        const auto ancestor = find(run, producer, producer_generation);
        return ancestor >= 0 && (node.ancestors[ancestor / 64] & (1ull << (ancestor % 64))) != 0;
    }
};

}  // namespace twait
