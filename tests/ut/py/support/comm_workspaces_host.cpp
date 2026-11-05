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
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "comm_workspace_managers.h"
#include "comm_hccl.cpp"

namespace {
std::unordered_map<void *, size_t> mappings;
int physical_handles = 0;
bool fail_context_upload = false;
enum class PeerFailure { NONE, IMPORT, RESERVE, MAP, ACCESS };
PeerFailure peer_failure = PeerFailure::NONE;
bool peer_reserve_pending = false;
int peer_import_calls = 0;
int injected_peer_failures = 0;
std::unordered_set<aclrtDrvMemHandle> imported_handles;
std::unordered_set<void *> peer_mappings;

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
    imported_handles.erase(handle);
    delete reinterpret_cast<int *>(handle);
    --physical_handles;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtReserveMemAddress(void **out, size_t bytes, size_t, void *, uint64_t) {
    const bool peer = peer_reserve_pending;
    peer_reserve_pending = false;
    if (peer && peer_failure == PeerFailure::RESERVE) {
        ++injected_peer_failures;
        return 1;
    }
    return map_buffer(out, bytes);
}
extern "C" aclError aclrtReleaseMemAddress(void *p) { return unmap_buffer(p); }
extern "C" aclError aclrtMapMem(void *address, size_t, size_t, aclrtDrvMemHandle handle, uint64_t) {
    if (imported_handles.count(handle) != 0) {
        if (peer_failure == PeerFailure::MAP) {
            ++injected_peer_failures;
            return 1;
        }
        peer_mappings.insert(address);
    }
    return ACL_SUCCESS;
}
extern "C" aclError aclrtUnmapMem(void *address) {
    peer_mappings.erase(address);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMemSetAccess(void *address, size_t, aclrtMemAccessDesc *, size_t) {
    if (peer_mappings.count(address) != 0 && peer_failure == PeerFailure::ACCESS) {
        ++injected_peer_failures;
        return 1;
    }
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp *, aclrtMemGranularityOptions, size_t *n) {
    *n = 4096;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMemExportToShareableHandle(aclrtDrvMemHandle, aclrtMemHandleType, uint64_t, uint64_t *out) {
    *out = 1;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMemImportFromShareableHandle(uint64_t, int32_t, aclrtDrvMemHandle *out) {
    ++peer_import_calls;
    if (peer_failure == PeerFailure::IMPORT) {
        ++injected_peer_failures;
        return 1;
    }
    *out = reinterpret_cast<aclrtDrvMemHandle>(new int(2));
    ++physical_handles;
    imported_handles.insert(*out);
    peer_reserve_pending = true;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtDeviceEnablePeerAccess(int32_t, uint32_t) { return ACL_SUCCESS; }
extern "C" aclError aclrtDevicePeerAccessStatus(int32_t, int32_t, int32_t *status) {
    *status = 1;
    return ACL_SUCCESS;
}
extern "C" HcclResult
HcclCreateSubCommConfig(HcclComm *, uint32_t, uint32_t *, uint64_t id, uint32_t, HcclCommConfig *, HcclComm *out) {
    *out = reinterpret_cast<HcclComm>(static_cast<uintptr_t>(id + 2));
    return HCCL_SUCCESS;
}
extern "C" HcclResult HcclCommDestroy(HcclComm) { return HCCL_SUCCESS; }

static void test_peer_import_rollback(CommHandle h, uint64_t &next_id) {
    using namespace workspace_fixture;
    const int old_nranks = h->nranks;
    h->nranks = 2;
    const uint32_t ranks[] = {0, 1};
    auto allocate = [&](uint32_t mask, uint64_t &id, uint64_t &ctx) {
        id = next_id++;
        // Rank 1 is represented by its real announce/barrier file formats;
        // rank 0 runs the production two-rank allocation and cleanup path.
        assert(domain_write_announce(h->rootinfo_path, id, 1, h->run_token, getpid(), 1, 42));
        for (const char *phase : {"p2p_ready", "release"}) {
            std::ofstream marker(barrier_marker_path(h->rootinfo_path, h->run_token, domain_barrier_tag(id, phase), 1));
            marker << "1";
            assert(marker.good());
        }
        uint64_t base = 0;
        size_t size = 0;
        return comm_alloc_domain_windows(h, id, ranks, 2, 0, 4096, mask, &ctx, &base, &size);
    };
    uint64_t held_id = 0, held_ctx = 0;
    assert(allocate(1, held_id, held_ctx) == 0);
    const auto held_mappings = mappings.size();
    const auto held_imports = imported_handles.size();
    const auto held_peers = peer_mappings.size();
    const auto held_domains = h->domain_allocations.size();
    const int held_physical = physical_handles;
    assert(held_imports == 1 && held_peers == 1);
    std::vector<uint32_t> masks{0, 1};
    if ((compiled_comm_workspaces() & 2) != 0) masks.push_back(3);
    for (uint32_t mask : masks) {
        for (auto failure :
             {PeerFailure::IMPORT, PeerFailure::RESERVE, PeerFailure::MAP, PeerFailure::ACCESS, PeerFailure::NONE}) {
            peer_failure = failure;
            fail_context_upload = failure == PeerFailure::NONE;
            const int old_calls = peer_import_calls;
            const int old_failures = injected_peer_failures;
            const std::vector<int> held_live{live[0], live[1], live[2]};
            uint64_t rejected = 0, ctx = 0;
            assert(allocate(mask, rejected, ctx) == COMM_INITIALIZATION_ABORT_REQUIRED);
            assert(peer_import_calls == old_calls + 1);
            assert(injected_peer_failures == old_failures + (failure != PeerFailure::NONE));
            assert(ctx == 0 && h->domain_allocations.count(rejected) == 1);
            assert(comm_abort_domain_windows(h, rejected, 2, 0) == 0);
            assert(h->domain_allocations.size() == held_domains);
            assert(mappings.size() == held_mappings && physical_handles == held_physical);
            assert(imported_handles.size() == held_imports && peer_mappings.size() == held_peers);
            for (int engine = 0; engine < 3; ++engine)
                assert(live[engine] == held_live[engine]);
        }
        peer_failure = PeerFailure::NONE;
        fail_context_upload = false;
        uint64_t id = 0, ctx = 0;
        assert(allocate(mask, id, ctx) == 0);
        const auto *view = reinterpret_cast<const CommContext *>(ctx);
        assert(view->rankNum == 2 && view->windowsIn[0] != 0 && view->windowsIn[1] != 0);
        assert(view->windowsIn[0] != view->windowsIn[1]);
        assert(h->domain_allocations.at(id)->peer_windows.size() == 1);
        assert(comm_release_domain_windows(h, id, 2, 0) == 0);
        assert(mappings.size() == held_mappings && physical_handles == held_physical);
        assert(imported_handles.size() == held_imports && peer_mappings.size() == held_peers);
    }
    assert(comm_release_domain_windows(h, held_id, 2, 0) == 0);
    assert(mappings.empty() && physical_handles == 0 && imported_handles.empty() && peer_mappings.empty());
    h->nranks = old_nranks;
}

int main() {
    using namespace workspace_fixture;
    char folder[] = "/tmp/simpler-comm-workspaces-XXXXXX";
    assert(mkdtemp(folder));
    auto *h = new CommHandle_{};
    h->rank = 0;
    h->nranks = 2;
    h->rootinfo_path = std::string(folder) + "/root";
    h->hccl_comm = reinterpret_cast<HcclComm>(1);
    setenv("PTO_ROCE_PHYIDS", "0", 1);
    setenv("PTO_ROCE_LOCAL_IP", "127.0.0.1", 1);
    uint64_t next_id = 1;
    const uint32_t ranks[] = {0, 1};
    auto allocate = [&](uint32_t mask, uint64_t &id, uint64_t &ctx) {
        id = next_id++;
        const size_t count = (mask & 2) ? 2 : 1;
        if (count == 2) {
            assert(domain_write_announce(h->rootinfo_path, id, 1, h->run_token, getpid(), 1, 42));
            for (const char *phase : {"p2p_ready", "release"}) {
                std::ofstream marker(
                    barrier_marker_path(h->rootinfo_path, h->run_token, domain_barrier_tag(id, phase), 1)
                );
                marker << "1";
                assert(marker.good());
            }
        }
        uint64_t base = 0;
        size_t size = 0;
        return comm_alloc_domain_windows(h, id, ranks, count, 0, 4096, mask, &ctx, &base, &size);
    };
    auto release = [&](uint64_t id) {
        const auto count = h->domain_allocations.at(id)->nranks;
        assert(comm_release_domain_windows(h, id, count, 0) == 0);
    };
    uint64_t first = 0, first_ctx = 0;
    uint64_t base = 0;
    size_t size = 0;
    const uint32_t sparse_ranks[] = {0, 2};
    assert(
        comm_alloc_domain_windows(h, next_id++, sparse_ranks, 2, 0, 4096, 2, &first_ctx, &base, &size) ==
        COMM_DOMAIN_URMA_CONFIG_INVALID
    );
    assert(mappings.empty() && started[1] == 0);
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

    // A single-rank request reaches the provider; its failure is not a domain-policy rejection.
    fail_engine = 1;
    const uint64_t single_rank_id = next_id++;
    assert(
        comm_alloc_domain_windows(h, single_rank_id, ranks, 1, 0, 4096, 2, &first_ctx, &base, &size) ==
        COMM_INITIALIZATION_ABORT_REQUIRED
    );
    assert(started[1] == 1 && live[1] == 1 && h->domain_allocations.count(single_rank_id) == 1);
    assert(comm_abort_domain_windows(h, single_rank_id, 1, 0) == 0);
    assert(live[1] == 0 && h->domain_allocations.empty());
    assert(mappings.empty() && physical_handles == 0);
    fail_engine = -1;

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
        for (int failure : {0, 1, 2, 3, 4}) {
            invalid_addr_engine = failure == 3 ? 1 : -1;
            invalid_size_engine = failure == 4 ? 1 : -1;
            fail_engine = failure == 0 ? 1 : -1;
            throw_engine = failure == 1 ? 1 : -1;
            fail_context_upload = failure == 2;
            uint64_t rejected = 0, rejected_ctx = 0;
            assert(allocate(3, rejected, rejected_ctx) == COMM_INITIALIZATION_ABORT_REQUIRED);
            assert(h->domain_allocations.count(rejected) == 1);
            assert(comm_abort_domain_windows(h, rejected, h->domain_allocations.at(rejected)->nranks, 0) == 0);
            assert(h->domain_allocations.count(rejected) == 0);
            assert(mappings.size() == held_mappings && physical_handles == held_physical);
            assert(live[1] == 1 && live[0] == 1);
        }
        fail_engine = throw_engine = invalid_addr_engine = invalid_size_engine = -1;
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
            assert(allocate(4, rejected, rejected_ctx) == COMM_INITIALIZATION_ABORT_REQUIRED);
            assert(h->domain_allocations.count(rejected) == 1);
            assert(comm_abort_domain_windows(h, rejected, h->domain_allocations.at(rejected)->nranks, 0) == 0);
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
    test_peer_import_rollback(h, next_id);
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
    for (const char *tag : {"destroy", "destroy_departed"}) {
        assert(
            file_marker_handshake::publish_marker(
                file_marker_handshake::marker_path(h->rootinfo_path, h->run_token, tag, 1)
            )
        );
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
    assert(allocate(1, first, first_ctx) == COMM_INITIALIZATION_ABORT_REQUIRED);
    assert(comm_abort_domain_windows(h, first, 1, 0) == 0);
    assert(!h->sdma_workspace && mappings.empty() && physical_handles == 0 && live[0] == 0);
    throw_engine = -1;
    fail_engine = 0;
    assert(allocate(1, first, first_ctx) == COMM_INITIALIZATION_ABORT_REQUIRED);
    assert(comm_abort_domain_windows(h, first, 1, 0) == 0);
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
