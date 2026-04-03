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

#include "dist_scope.h"

void DistScope::scope_begin() {
    if (depth() >= DIST_MAX_SCOPE_DEPTH) throw std::runtime_error("DistScope: maximum nesting depth exceeded");
    stack_.push_back(ScopeFrame{});
}

void DistScope::scope_end(const std::function<void(DistTaskSlot)> &release_fn) {
    if (stack_.empty()) throw std::runtime_error("DistScope: scope_end without scope_begin");
    ScopeFrame &frame = stack_.back();
    for (DistTaskSlot slot : frame.tasks)
        release_fn(slot);
    stack_.pop_back();
}

void DistScope::register_task(DistTaskSlot slot) {
    if (stack_.empty()) return;  // no open scope — task has no scope ref
    stack_.back().tasks.push_back(slot);
}
