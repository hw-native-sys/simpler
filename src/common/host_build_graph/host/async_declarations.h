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

#include <unordered_map>
#include <vector>

#include "host_build_graph/async_use_declaration.h"

// Owned by this run's GraphHostState, with its recording_mutex held for writes.
// Graph keys scope body-local task indices; GLOBAL task indices use a separate
// map. The same captured declaration is resolved anew for every Graph instance.
class AsyncHostDeclarations {
public:
    using Uses = std::vector<AsyncUseDeclaration>;
    using Tasks = std::unordered_map<int32_t, Uses>;

    bool add(bool graph, uint64_t graph_key, int32_t task, const AsyncUseDeclaration &use) {
        if (task < 0) return false;
        auto &uses = graph ? graphs_[graph_key][task] : ordinary_[task];
        for (const auto &existing : uses) {
            if (existing == use) return true;
            if (async_use_declarations_conflict(existing, use)) return false;
        }
        uses.push_back(use);
        return true;
    }

    const Uses *find(bool graph, uint64_t graph_key, int32_t task) const {
        const Tasks *tasks = &ordinary_;
        if (graph) {
            const auto found = graphs_.find(graph_key);
            if (found == graphs_.end()) return nullptr;
            tasks = &found->second;
        }
        const auto found = tasks->find(task);
        return found == tasks->end() ? nullptr : &found->second;
    }

    void erase_graph(uint64_t key) { graphs_.erase(key); }

private:
    Tasks ordinary_;
    std::unordered_map<uint64_t, Tasks> graphs_;
};
