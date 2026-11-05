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

#include "comm_hccl.cpp"

namespace {
std::unordered_map<void *, size_t> mappings;
int physical_handles = 0;
bool fail_get_device = false;
int device_calls = 0;
bool fail_physical_free = false;
int physical_free_failures = 0;
struct PhysicalAllocation {};
std::unordered_map<aclrtDrvMemHandle, std::shared_ptr<PhysicalAllocation>> physical_resources;
std::unordered_map<uint64_t, std::weak_ptr<PhysicalAllocation>> exported_resources;
uint64_t last_exported = 0;
bool fabric_supported = true;
int unsupported_attempts = 0;
std::weak_ptr<PhysicalAllocation> unsupported_resource;
bool fail_context_upload = false;
bool reject_explicit_access = false;
enum class PeerFailure { NONE, IMPORT, RESERVE, MAP };
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
    ++device_calls;
    if (fail_get_device && device_calls == (fabric_supported ? 1 : 2)) return 1;
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
    physical_resources.emplace(*out, std::make_shared<PhysicalAllocation>());
    ++physical_handles;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtFreePhysical(aclrtDrvMemHandle handle) {
    imported_handles.erase(handle);
    assert(physical_resources.erase(handle) == 1);
    delete reinterpret_cast<int *>(handle);
    --physical_handles;
    if (fail_physical_free) {
        // Model an error reported after consumption. This only checks that
        // fallback stops on the error, not retention of an unfreed ACL handle.
        fail_physical_free = false;
        ++physical_free_failures;
        return 1;
    }
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
extern "C" aclError aclrtMemSetAccess(void *, size_t, aclrtMemAccessDesc *, size_t) {
    if (reject_explicit_access) return ACL_ERROR_RT_INVALID_DEVICEID;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp *, aclrtMemGranularityOptions, size_t *n) {
    *n = 4096;
    return ACL_SUCCESS;
}
extern "C" aclError
aclrtMemExportToShareableHandle(aclrtDrvMemHandle handle, aclrtMemHandleType, uint64_t, uint64_t *out) {
    if (!fabric_supported) assert(unsupported_resource.expired());
    *out = reinterpret_cast<uint64_t>(handle);
    exported_resources[*out] = physical_resources.at(handle);
    last_exported = *out;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMemImportFromShareableHandle(uint64_t exported, int32_t, aclrtDrvMemHandle *out) {
    ++peer_import_calls;
    if (peer_failure == PeerFailure::IMPORT) {
        ++injected_peer_failures;
        return 1;
    }
    auto resource = exported == 42 ? std::make_shared<PhysicalAllocation>() : exported_resources[exported].lock();
    if (!resource) return 1;
    *out = reinterpret_cast<aclrtDrvMemHandle>(new int(2));
    physical_resources.emplace(*out, std::move(resource));
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
extern "C" aclError
aclrtMemExportToShareableHandleV2(aclrtDrvMemHandle handle, uint64_t, aclrtMemSharedHandleType, void *out) {
    if (!fabric_supported) {
        unsupported_resource = physical_resources.at(handle);
        ++unsupported_attempts;
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }
    uint64_t token = 0;
    const auto rc = aclrtMemExportToShareableHandle(handle, ACL_MEM_HANDLE_TYPE_NONE, 0, &token);
    std::memset(out, 0, sizeof(aclrtMemFabricHandle));
    std::memcpy(out, &token, sizeof(token));
    return rc;
}
extern "C" aclError
aclrtMemImportFromShareableHandleV2(void *in, aclrtMemSharedHandleType, uint64_t, aclrtDrvMemHandle *out) {
    uint64_t token = 0;
    std::memcpy(&token, in, sizeof(token));
    return aclrtMemImportFromShareableHandle(token, 0, out);
}
extern "C" HcclResult HcclCommDestroy(HcclComm) { return HCCL_SUCCESS; }

static void prepare_peer(CommHandle h, uint64_t id, bool release_marker) {
    aclrtMemFabricHandle fabric{};
    const uint64_t token = 42;
    std::memcpy(&fabric, &token, sizeof(token));
    assert(domain_write_announce(h->rootinfo_path, id, 1, h->run_token, getpid(), 1, 4096, fabric));
    IpcAnnounceFile ipc{};
    ipc.pid = getpid();
    ipc.rank = 1;
    ipc.device_id = 1;
    ipc.mapping_size = 4096;
    ipc.shareable_handle = token;
    assert(write_ipc_announce(domain_ipc_announce_path(h->rootinfo_path, id, 1, h->run_token), ipc));
    for (const char *phase : {"fabric_ready", "ipc_p2p_ready", "release"}) {
        if (!release_marker && std::strcmp(phase, "release") == 0) continue;
        std::ofstream marker(barrier_marker_path(h->rootinfo_path, h->run_token, domain_barrier_tag(id, phase), 1));
        marker << "1";
    }
}

int main(int argc, char **argv) {
    assert(argc == 3);
    fabric_supported = std::string(argv[1]) == "fabric";
    reject_explicit_access = std::string(argv[2]) == "map_grants_access";
    const std::string scenario = reject_explicit_access ? "success" : argv[2];
    fail_physical_free = scenario == "fallback_cleanup";
    fail_context_upload = scenario == "late_import";
    fail_get_device = scenario == "early_get_device";
    if (scenario == "peer_import") peer_failure = PeerFailure::IMPORT;
    if (scenario == "peer_reserve") peer_failure = PeerFailure::RESERVE;
    if (scenario == "peer_map") peer_failure = PeerFailure::MAP;
    char folder[] = "/tmp/simpler-a2a3-lifecycle-XXXXXX";
    assert(mkdtemp(folder));
    auto *h = new CommHandle_{};
    h->rank = 0;
    h->nranks = 2;
    h->rootinfo_path = std::string(folder) + "/root";
    h->hccl_comm = reinterpret_cast<HcclComm>(1);
    const uint64_t id = 1;
    prepare_peer(h, id, scenario == "success");
    const uint32_t ranks[] = {0, 1};
    uint64_t ctx = 0, base = 0;
    size_t size = 0;
    const int rc = comm_alloc_domain_windows(h, id, ranks, 2, 0, 4096, 0, &ctx, &base, &size);
    if (scenario == "success") {
        assert(rc == 0 && ctx != 0 && base != 0 && size == 4096);
        const auto saved = mappings.size();
        assert(physical_handles == 2);
        prepare_peer(h, 2, true);
        uint64_t second_ctx = 0, second_base = 0;
        assert(comm_alloc_domain_windows(h, 2, ranks, 2, 0, 4096, 0, &second_ctx, &second_base, &size) == 0);
        assert(physical_handles == 4 && mappings.size() == 2 * saved);
        assert(comm_release_domain_windows(h, id, 2, 0) == 0);
        assert(physical_handles == 2 && mappings.size() == saved);
        assert(reinterpret_cast<CommContext *>(second_ctx)->rankNum == 2);
        assert(mappings.count(reinterpret_cast<void *>(second_base)) == 1);
        assert(h->hccl_comm == reinterpret_cast<HcclComm>(1));
        assert(comm_release_domain_windows(h, 2, 2, 0) == 0);
    } else {
        assert(rc == COMM_INITIALIZATION_ABORT_REQUIRED && ctx == 0);
        assert(h->domain_allocations.count(id) == 1);
        const auto held_mappings = mappings.size();
        const auto held_handles = physical_handles;
        // A failed participant must refuse normal release without waiting for its peers.
        assert(comm_release_domain_windows(h, id, 2, 0) != 0);
        assert(mappings.size() == held_mappings && physical_handles == held_handles);
        aclrtDrvMemHandle late_import = nullptr;
        if (scenario == "early_get_device") {
            assert(mappings.empty() && physical_handles == 0);
            fail_get_device = false;
        } else if (scenario == "fallback_cleanup") {
            assert(physical_free_failures == 1 && last_exported == 0);
        } else {
            assert(physical_handles > 0 && !mappings.empty());
            assert(!exported_resources.at(last_exported).expired());
            if (scenario == "late_import") {
                if (fabric_supported) {
                    aclrtMemFabricHandle handle{};
                    std::memcpy(&handle, &last_exported, sizeof(last_exported));
                    assert(
                        aclrtMemImportFromShareableHandleV2(
                            &handle, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0, &late_import
                        ) == ACL_SUCCESS
                    );
                } else {
                    assert(aclrtMemImportFromShareableHandle(last_exported, 1, &late_import) == ACL_SUCCESS);
                }
            } else {
                assert(injected_peer_failures == 1 && peer_import_calls == 1);
                // Once the peer import succeeds, even an unmapped handle belongs to the failed owner.
                assert(physical_handles == (scenario == "peer_import" ? 1 : 2));
            }
        }
        // No peer release marker exists: abort follows parent join and must not enter a collective.
        assert(comm_abort_domain_windows(h, id, 2, 0) == 0);
        if (late_import != nullptr) {
            assert(physical_handles == 1 && !exported_resources.at(last_exported).expired());
            assert(aclrtFreePhysical(late_import) == ACL_SUCCESS);
            assert(exported_resources.at(last_exported).expired());
        }
    }
    assert(h->domain_allocations.empty() && mappings.empty() && physical_handles == 0);
    assert(physical_resources.empty() && imported_handles.empty() && peer_mappings.empty());
    if (!fabric_supported && scenario != "early_get_device") {
        assert(unsupported_attempts == (scenario == "success" ? 2 : 1));
        assert(unsupported_resource.expired());
        assert(!std::filesystem::exists(domain_fabric_announce_path(h->rootinfo_path, id, 0, h->run_token)));
    }
    if (scenario != "success") {
        fail_context_upload = false;
        peer_failure = PeerFailure::NONE;
        peer_reserve_pending = false;
        prepare_peer(h, 3, true);
        assert(comm_alloc_domain_windows(h, 3, ranks, 2, 0, 4096, 0, &ctx, &base, &size) == 0);
        assert(comm_release_domain_windows(h, 3, 2, 0) == 0);
        assert(h->domain_allocations.empty() && mappings.empty() && physical_handles == 0);
    }
    delete h;
    std::filesystem::remove_all(folder);
    return 0;
}
