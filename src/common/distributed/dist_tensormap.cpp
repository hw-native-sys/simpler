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

#include "dist_tensormap.h"

DistTaskSlot DistTensorMap::lookup(uint64_t base_ptr) const {
    auto it = map_.find(base_ptr);
    if (it == map_.end()) return DIST_INVALID_SLOT;
    return it->second;
}

void DistTensorMap::insert(uint64_t base_ptr, DistTaskSlot producer) { map_[base_ptr] = producer; }

void DistTensorMap::erase_task_outputs(const std::vector<uint64_t> &keys) {
    for (uint64_t key : keys)
        map_.erase(key);
}

int32_t DistTensorMap::size() const { return static_cast<int32_t>(map_.size()); }
