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
#include "host/ccu_execution_registry.h"

#include <limits>

bool CcuOwnerLeaseV1::Retain() {
    if (retained) return true;
    if (owner == nullptr || ops.abi_version != CCU_EXECUTION_ABI_V1 || ops.struct_size != sizeof(ops) ||
        ops.retain == nullptr || ops.release == nullptr)
        return false;
    try {
        if (ops.retain(owner) != 0) return false;
    } catch (...) {
        return false;
    }
    retained = true;
    return true;
}

bool CcuOwnerLeaseV1::Release() {
    if (!retained) return true;
    if (ops.release == nullptr) return false;
    try {
        if (ops.release(owner) != 0) return false;
    } catch (...) {
        return false;
    }
    retained = false;
    return true;
}

CcuOwnerLeaseV1::~CcuOwnerLeaseV1() {
    try {
        if (retained) (void)Release();
    } catch (...) {}
}

CcuExecutionRegistryV1 &CcuExecutionRegistryV1::Instance() {
    // Retained device work and callback owners outlive static destruction.
    // 未确认的设备任务及回调租约不在静态析构阶段释放。
    static auto *registry = new CcuExecutionRegistryV1;
    return *registry;
}

uint64_t CcuExecutionRegistryV1::NextId() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (next_id_ == 0 || next_id_ == std::numeric_limits<uint64_t>::max()) return 0;
    return next_id_++;
}

bool CcuExecutionRegistryV1::AddContext(const std::shared_ptr<CcuContextEntryV1> &context) {
    if (!context || context->id == 0) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    return contexts_.emplace(context->id, context).second;
}

std::shared_ptr<CcuContextEntryV1> CcuExecutionRegistryV1::GetContext(CcuContextId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = contexts_.find(id);
    return it == contexts_.end() ? nullptr : it->second;
}

std::shared_ptr<CcuRequestEntryV1> CcuExecutionRegistryV1::GetRequest(CcuRequestId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = requests_.find(id);
    return it == requests_.end() ? nullptr : it->second;
}

bool CcuExecutionRegistryV1::AddRequest(const std::shared_ptr<CcuRequestEntryV1> &request) {
    if (!request || request->id == 0) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    return requests_.emplace(request->id, request).second;
}

void CcuExecutionRegistryV1::RemoveRequest(CcuRequestId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    requests_.erase(id);
}

void CcuExecutionRegistryV1::RemoveContext(CcuContextId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    contexts_.erase(id);
}
