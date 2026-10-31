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
#include <sys/mman.h>

#include <cassert>
#include <cstring>
#include <unordered_map>

#include "comm_workspace_managers.h"
#include "comm_hccl.cpp"

namespace {
std::unordered_map<void *, size_t> mappings;
int physical_handles = 0;
bool fail_context_upload = false;

aclError map_buffer(void **out, size_t bytes) {
    static uintptr_t hint = UINT64_C(0x1000000000);
    void *p = mmap(reinterpret_cast<void *>(hint), bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 1;
    hint += 65536;
    mappings.emplace(p, bytes);
    *out = p;
    return ACL_SUCCESS;
}
aclError unmap_buffer(void *p) {
    auto it = mappings.find(p);
    assert(it != mappings.end());
    const int rc = munmap(p, it->second);
    assert(rc == 0);
    mappings.erase(it);
    return ACL_SUCCESS;
}
}  // namespace

extern "C" aclError aclrtGetDevice(int32_t *id) {
    *id = 0;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMalloc(void **out, size_t bytes, aclrtMemMallocPolicy) { return map_buffer(out, bytes); }
extern "C" aclError aclrtFree(void *p) { return unmap_buffer(p); }
extern "C" aclError aclrtMemcpy(void *out, size_t, const void *in, size_t bytes, aclrtMemcpyKind) {
    if (fail_context_upload && bytes == sizeof(CommContext)) return 1;
    std::memcpy(out, in, bytes);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMemset(void *out, size_t, int32_t value, size_t bytes) {
    std::memset(out, value, bytes);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMallocPhysical(aclrtDrvMemHandle *out, size_t, const aclrtPhysicalMemProp *, uint64_t) {
    *out = reinterpret_cast<aclrtDrvMemHandle>(new int(1));
    ++physical_handles;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtFreePhysical(aclrtDrvMemHandle handle) {
    delete reinterpret_cast<int *>(handle);
    --physical_handles;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtReserveMemAddress(void **out, size_t bytes, size_t, void *, uint64_t) {
    return map_buffer(out, bytes);
}
extern "C" aclError aclrtReleaseMemAddress(void *p) { return unmap_buffer(p); }
extern "C" aclError aclrtMapMem(void *, size_t, size_t, aclrtDrvMemHandle, uint64_t) { return ACL_SUCCESS; }
extern "C" aclError aclrtUnmapMem(void *) { return ACL_SUCCESS; }
extern "C" aclError aclrtMemSetAccess(void *, size_t, aclrtMemAccessDesc *, size_t) { return ACL_SUCCESS; }
extern "C" aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp *, aclrtMemGranularityOptions, size_t *n) {
    *n = 4096;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMemExportToShareableHandle(aclrtDrvMemHandle, aclrtMemHandleType, uint64_t, uint64_t *out) {
    *out = 1;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMemImportFromShareableHandle(uint64_t, int32_t, aclrtDrvMemHandle *) { return 1; }
extern "C" aclError aclrtDeviceEnablePeerAccess(int32_t, uint32_t) { return ACL_SUCCESS; }
extern "C" aclError aclrtDevicePeerAccessStatus(int32_t, int32_t, int32_t *status) {
    *status = 1;
    return ACL_SUCCESS;
}
extern "C" HcclResult HcclCommDestroy(HcclComm) { return HCCL_SUCCESS; }

int main() {
    using namespace workspace_fixture;
    char folder[] = "/tmp/simpler-comm-workspaces-XXXXXX";
    assert(mkdtemp(folder));
    auto *h = new CommHandle_{};
    h->rank = 0;
    h->nranks = 1;
    h->rootinfo_path = std::string(folder) + "/root";
    h->hccl_comm = reinterpret_cast<HcclComm>(1);
    setenv("PTO_ROCE_PHYIDS", "0", 1);
    setenv("PTO_ROCE_LOCAL_IP", "127.0.0.1", 1);
    uint64_t next_id = 1;
    const uint32_t ranks[] = {0};
    auto allocate = [&](uint32_t mask, uint64_t &id, uint64_t &ctx) {
        id = next_id++;
        uint64_t base = 0;
        size_t size = 0;
        return comm_alloc_domain_windows(h, id, ranks, 1, 0, 4096, mask, &ctx, &base, &size);
    };
    auto release = [&](uint64_t id) {
        assert(comm_release_domain_windows(h, id, 1, 0) == 0);
    };
    uint64_t first = 0, first_ctx = 0;
    assert(allocate(0, first, first_ctx) == 0);
    assert(started[0] == 0 && started[1] == 0 && started[2] == 0);
    release(first);
    // Synchronous mappings do not depend on any asynchronous provider or RoCE bootstrap.
    unsetenv("PTO_ROCE_PHYIDS");
    unsetenv("PTO_ROCE_LOCAL_IP");
    for (int unavailable : {0, 1, 2}) {
        fail_engine = throw_engine = unavailable;
        assert(allocate(0, first, first_ctx) == 0);
        assert(started[0] == 0 && started[1] == 0 && started[2] == 0);
        const auto *ctx = reinterpret_cast<CommContext *>(first_ctx);
        for (const auto &workspace : ctx->workspaces)
            assert(workspace.addr == 0 && workspace.size == 0);
        assert(ctx->rankId == 0 && ctx->rankNum == 1 && ctx->winSize >= 4096);
        assert(h->domain_allocations.at(first)->own_handle != nullptr);
        auto *mapped = reinterpret_cast<uint64_t *>(ctx->windowsIn[0]);
        *mapped = 0x12345678;
        assert(*mapped == 0x12345678);
        release(first);
        assert(mappings.empty() && physical_handles == 0);
    }
    fail_engine = throw_engine = -1;
    assert(allocate(UINT32_MAX, first, first_ctx) != 0);
    assert(mappings.empty() && physical_handles == 0);
    setenv("PTO_ROCE_PHYIDS", "0", 1);
    setenv("PTO_ROCE_LOCAL_IP", "127.0.0.1", 1);

    if ((compiled_comm_workspaces() & 2) != 0) {
        uint64_t second = 0, second_ctx = 0;
        assert(allocate(3, first, first_ctx) == 0);
        assert(allocate(3, second, second_ctx) == 0);
        const auto *a = reinterpret_cast<CommContext *>(first_ctx);
        const auto *b = reinterpret_cast<CommContext *>(second_ctx);
        assert(comm_workspace<CommWorkspaceKind::SDMA>(*a).addr == comm_workspace<CommWorkspaceKind::SDMA>(*b).addr);
        assert(comm_workspace<CommWorkspaceKind::URMA>(*a).addr != comm_workspace<CommWorkspaceKind::URMA>(*b).addr);
        assert(comm_workspace<CommWorkspaceKind::SDMA>(*a).addr != comm_workspace<CommWorkspaceKind::URMA>(*a).addr);
        release(first);
        assert(live[1] == 1 && live[0] == 1);
        const auto held_mappings = mappings.size();
        const int held_physical = physical_handles;
        for (int failure : {0, 1, 2}) {
            fail_engine = failure == 0 ? 1 : -1;
            throw_engine = failure == 1 ? 1 : -1;
            fail_context_upload = failure == 2;
            uint64_t rejected = 0, rejected_ctx = 0;
            assert(allocate(3, rejected, rejected_ctx) != 0);
            assert(h->domain_allocations.count(rejected) == 0);
            assert(mappings.size() == held_mappings && physical_handles == held_physical);
            assert(live[1] == 1 && live[0] == 1);
        }
        fail_engine = throw_engine = -1;
        fail_context_upload = false;
        release(second);
    }
    if ((compiled_comm_workspaces() & 4) != 0) {
        uint64_t second = 0, second_ctx = 0;
        assert(allocate(4, first, first_ctx) == 0);
        assert(allocate(1, second, second_ctx) == 0);
        assert(comm_workspace<CommWorkspaceKind::SDMA>(*reinterpret_cast<CommContext *>(first_ctx)).addr == 0);
        assert(comm_workspace<CommWorkspaceKind::RDMA>(*reinterpret_cast<CommContext *>(second_ctx)).addr == 0);
        assert(h->domain_allocations.at(first)->own_handle == nullptr);
        assert(h->domain_allocations.at(second)->own_handle != nullptr);
        release(second);
        assert(live[2] == 1);
        const auto held_mappings = mappings.size();
        const int held_physical = physical_handles;
        for (int failure : {0, 1, 2}) {
            fail_engine = failure == 0 ? 2 : -1;
            throw_engine = failure == 1 ? 2 : -1;
            fail_context_upload = failure == 2;
            uint64_t rejected = 0, rejected_ctx = 0;
            assert(allocate(4, rejected, rejected_ctx) != 0);
            assert(h->domain_allocations.count(rejected) == 0);
            assert(mappings.size() == held_mappings && physical_handles == held_physical);
            assert(live[2] == 1);
        }
        fail_engine = throw_engine = -1;
        fail_context_upload = false;
        release(first);
    }
    const int sdma_starts = started[0];
    assert(allocate(0, first, first_ctx) == 0);
    assert(started[0] == sdma_starts);
    auto *sync_ctx = reinterpret_cast<CommContext *>(first_ctx);
    assert(
        !comm_workspace<CommWorkspaceKind::SDMA>(*sync_ctx).addr &&
        !comm_workspace<CommWorkspaceKind::URMA>(*sync_ctx).addr &&
        !comm_workspace<CommWorkspaceKind::RDMA>(*sync_ctx).addr
    );
    release(first);
    assert(allocate(7, first, first_ctx) != 0);
    assert(allocate(8, first, first_ctx) != 0);
    assert(mappings.empty() && physical_handles == 0);
    // Finalization must also reclaim domains whose caller skipped explicit release.
    assert(allocate(1, first, first_ctx) == 0);
    if ((compiled_comm_workspaces() & 2) != 0) {
        uint64_t held = 0, held_ctx = 0;
        assert(allocate(3, held, held_ctx) == 0);
    }
    if ((compiled_comm_workspaces() & 4) != 0) {
        uint64_t held = 0, held_ctx = 0;
        assert(allocate(4, held, held_ctx) == 0);
    }
    assert(comm_destroy(h) == 0);
    assert(live[0] == 0 && live[1] == 0 && live[2] == 0);
    assert(mappings.empty() && physical_handles == 0);
    h = new CommHandle_{};
    h->rank = 0;
    h->nranks = 1;
    h->rootinfo_path = std::string(folder) + "/second";
    h->hccl_comm = reinterpret_cast<HcclComm>(1);
    throw_engine = 0;
    assert(allocate(1, first, first_ctx) != 0);
    assert(!h->sdma_workspace && mappings.empty() && physical_handles == 0 && live[0] == 0);
    throw_engine = -1;
    fail_engine = 0;
    assert(allocate(1, first, first_ctx) != 0);
    assert(!h->sdma_workspace && mappings.empty() && physical_handles == 0);
    assert(allocate(0, first, first_ctx) == 0);
    assert(!comm_workspace<CommWorkspaceKind::SDMA>(*reinterpret_cast<CommContext *>(first_ctx)).addr);
    release(first);
    fail_engine = -1;
    assert(allocate(1, first, first_ctx) == 0);
    release(first);
    assert(comm_destroy(h) == 0);
    assert(live[0] == 0 && mappings.empty() && physical_handles == 0);
    std::filesystem::remove_all(folder);
}
