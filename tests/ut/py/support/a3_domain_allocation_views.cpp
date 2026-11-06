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

#include <cassert>
#include <cstdint>
#include <vector>
#include "comm_hccl.cpp"  // NOLINT(bugprone-suspicious-include): exercise allocation owners with fake ACL

extern "C" aclError aclrtUnmapMem(void *) { return ACL_SUCCESS; }
extern "C" aclError aclrtReleaseMemAddress(void *) { return ACL_SUCCESS; }
extern "C" aclError aclrtFreePhysical(aclrtDrvMemHandle) { return ACL_SUCCESS; }
extern "C" aclError aclrtFree(void *) { return ACL_SUCCESS; }

int main() {
    CommHandle_ handle{};
    auto allocation = std::make_unique<DomainAllocation>();
    allocation->rank = 0;
    allocation->nranks = 2;
    allocation->device_ctx = reinterpret_cast<CommContext *>(0x40000);
    allocation->local_window.base = reinterpret_cast<void *>(0x10000);
    allocation->local_window.size = 4096;
    VmmWindow peer;
    peer.base = reinterpret_cast<void *>(0x20000);
    peer.size = 8192;
    allocation->peer_windows.push_back(std::move(peer));
    auto *owner = allocation.get();
    handle.domain_allocations.emplace(77, std::move(allocation));
    size_t count = 99;
    assert(comm_query_domain_allocation(&handle, 77, nullptr, 0, &count) != 0 && count == 0);
    owner->initialized = true;
    assert(comm_query_domain_allocation(&handle, 77, nullptr, 0, &count) == 0 && count == 3);
    std::vector<CommAllocationSpan> views(count);
    assert(comm_query_domain_allocation(&handle, 77, views.data(), views.size(), &count) == 0);
    assert(views[0].addr == 0x40000 && views[0].bytes == sizeof(CommContext));
    assert(views[1].addr == 0x10000 && views[1].bytes == 4096);
    assert(views[2].addr == 0x20000 && views[2].bytes == 8192);
    CommAllocationSpan sentinel{123, 456};
    assert(comm_query_domain_allocation(&handle, 77, &sentinel, 1, &count) != 0);
    assert(count == 3 && sentinel.addr == 123 && sentinel.bytes == 456);
    owner->release_failed = true;
    assert(comm_query_domain_allocation(&handle, 77, nullptr, 0, &count) != 0 && count == 0);
    owner->release_failed = false;
    owner->local_window.size = 0;
    assert(comm_query_domain_allocation(&handle, 77, nullptr, 0, &count) != 0 && count == 0);
    owner->local_window.size = 4096;
    assert(comm_query_domain_allocation(nullptr, 77, nullptr, 0, &count) != 0 && count == 0);
    assert(comm_query_domain_allocation(&handle, 78, nullptr, 0, &count) != 0 && count == 0);
    handle.domain_allocations.erase(77);
    assert(comm_query_domain_allocation(&handle, 77, nullptr, 0, &count) != 0 && count == 0);
}
