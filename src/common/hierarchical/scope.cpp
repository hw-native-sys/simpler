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

#include "scope.h"

void Scope::scope_begin() {
    std::lock_guard<std::mutex> lk(mu_);
    auto &stack = stacks_[std::this_thread::get_id()];
    if (stack.size() >= static_cast<size_t>(MAX_SCOPE_DEPTH)) {
        throw std::runtime_error("Scope: maximum nesting depth exceeded");
    }
    stack.push_back(ScopeFrame{});
}

void Scope::scope_end(const std::function<void(TaskSlot)> &release_fn) {
    ScopeFrame frame;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = stacks_.find(std::this_thread::get_id());
        if (it == stacks_.end() || it->second.empty()) {
            throw std::runtime_error("Scope: scope_end without scope_begin");
        }
        frame = std::move(it->second.back());
        it->second.pop_back();
        if (it->second.empty()) stacks_.erase(it);
    }
    for (TaskSlot slot : frame.tasks) {
        release_fn(slot);
    }
}

void Scope::register_task(TaskSlot slot) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = stacks_.find(std::this_thread::get_id());
    if (it == stacks_.end() || it->second.empty()) return;
    it->second.back().tasks.push_back(slot);
}

int32_t Scope::depth() const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = stacks_.find(std::this_thread::get_id());
    return it == stacks_.end() ? 0 : static_cast<int32_t>(it->second.size());
}

int32_t Scope::current_depth() const {
    int32_t d = depth();
    return d > 0 ? d - 1 : 0;
}
