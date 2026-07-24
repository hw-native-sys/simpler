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

#include "tensormap.h"

TaskSlot TensorMap::lookup(RunId run_id, TensorKey key) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = map_.find(RunTensorKey{run_id, key});
    if (it == map_.end()) return INVALID_SLOT;
    return it->second;
}

void TensorMap::insert(RunId run_id, TensorKey key, TaskSlot producer) {
    std::lock_guard<std::mutex> lk(mu_);
    map_[RunTensorKey{run_id, key}] = producer;
}

void TensorMap::erase_task_outputs(RunId run_id, const std::vector<TensorKey> &keys) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto &key : keys)
        map_.erase(RunTensorKey{run_id, key});
}

int32_t TensorMap::size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return static_cast<int32_t>(map_.size());
}
