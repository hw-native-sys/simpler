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
#include <array>
#include <cstring>
#include <fstream>
#include <future>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
bool fail_host_unlink = false;
int unlink_host_window(const char *path);
}  // namespace

#define shm_unlink unlink_host_window
#include "comm_hccl.cpp"
#undef shm_unlink

namespace {
int unlink_host_window(const char *path) {
    if (fail_host_unlink) {
        errno = EACCES;
        return -1;
    }
    return shm_unlink(path);
}

std::unordered_map<void *, size_t> mappings;
int physical_handles = 0;
std::unordered_set<void *> registered_hosts;
bool fail_host_register = false;
bool fail_host_unregister = false;
bool fail_host_capability = false;
aclrtHostMemMapCapability host_sdma_capability = ACL_RT_HOST_MEM_MAP_SUPPORTED;
int host_capability_calls = 0;
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
int explicit_access_calls = 0;
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

extern "C" aclError aclrtHostRegister(void *cpu, uint64_t, aclrtHostRegisterType, void **device) {
    if (fail_host_register) return ACL_ERROR_FAILURE;
    assert(registered_hosts.insert(cpu).second);
    *device = cpu;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtHostUnregister(void *cpu) {
    if (fail_host_unregister) return ACL_ERROR_FAILURE;
    assert(registered_hosts.erase(cpu) == 1);
    return ACL_SUCCESS;
}

extern "C" aclError aclrtGetDevice(int32_t *id) {
    ++device_calls;
    if (fail_get_device && device_calls == (fabric_supported ? 1 : 2)) return 1;
    *id = 0;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtHostMemMapCapabilities(uint32_t, aclrtHacType engine, aclrtHostMemMapCapability *capability) {
    ++host_capability_calls;
    assert(engine == ACL_RT_HAC_TYPE_SDMA);
    if (fail_host_capability) return ACL_ERROR_FAILURE;
    *capability = host_sdma_capability;
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
    ++explicit_access_calls;
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

static void test_host_registration_lifecycle(const std::string &scenario) {
    const bool fail_init = scenario == "host_register_failure";
    const bool sdma = scenario == "host_borrow_sdma";
    const bool collision = scenario == "host_name_collision";
    fail_host_capability = !sdma;
    auto h = std::make_unique<CommHandle_>();
    h->rank = 0;
    h->nranks = sdma ? 1 : 2;
    char folder[] = "/tmp/simpler-host-lifecycle-XXXXXX";
    if (sdma) {
        assert(mkdtemp(folder));
        h->rootinfo_path = std::string(folder) + "/root";
    }
    const auto size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const auto name = "simpler_host_ut_" + std::to_string(getpid());
    const auto path = "/" + name;
    if (collision) {
        const int fd = shm_open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        assert(fd >= 0 && ftruncate(fd, static_cast<off_t>(size)) == 0 && close(fd) == 0);
    }
    uint64_t ctx = 0, base = 0;
    size_t mapped = 0;
    const uint32_t ranks[] = {0, 1};
    fail_host_register = fail_init;
    const uint32_t mask = sdma ? comm_workspace_bit(CommWorkspaceKind::SDMA) : 0;
    // A2/A3 has no URMA engine, so any non-default engine configuration is rejected.
    const CommWorkspaceConfig urma_config{2, 0};
    assert(
        comm_alloc_host_domain_windows(
            h.get(), 2, ranks, h->nranks, 0, size, 0, name.c_str(), &ctx, &base, &mapped, &urma_config
        ) == COMM_DOMAIN_WORKSPACE_UNSUPPORTED
    );
    const int rc =
        comm_alloc_host_domain_windows(h.get(), 1, ranks, h->nranks, 0, size, mask, name.c_str(), &ctx, &base, &mapped);
    if (fail_init || collision) {
        assert(rc == COMM_INITIALIZATION_ABORT_REQUIRED && ctx == 0 && registered_hosts.empty());
        assert(comm_abort_domain_windows(h.get(), 1, 2, 0) == 0);
    } else {
        assert(rc == 0 && mapped == size && registered_hosts.size() == 1);
        const auto *context = reinterpret_cast<const CommContext *>(ctx);
        assert(context->windowsIn[0] == base && context->windowsIn[1] == 0);
        assert(comm_workspace<CommWorkspaceKind::SDMA>(*context).addr == 0);
        if (!sdma) {
            fail_host_unregister = true;
            assert(comm_destroy(h.get()) == COMM_RECLAMATION_UNCONFIRMED);
            assert(h->domain_allocations.count(1) == 1 && registered_hosts.size() == 1);
            fail_host_unregister = false;
            fail_host_unlink = true;
            assert(comm_release_domain_windows(h.get(), 1, h->nranks, 0) == COMM_RECLAMATION_UNCONFIRMED);
            assert(h->domain_allocations.count(1) == 1 && registered_hosts.empty() && mappings.empty());
            fail_host_unlink = false;
        }
        assert(comm_release_domain_windows(h.get(), 1, h->nranks, 0) == 0);
    }
    assert(h->domain_allocations.empty() && registered_hosts.empty() && mappings.empty());
    assert(host_capability_calls == (sdma ? 1 : 0));
    const int owner_fd = shm_open(path.c_str(), O_RDWR, 0);
    if (collision) {
        assert(owner_fd >= 0 && close(owner_fd) == 0 && shm_unlink(path.c_str()) == 0);
    } else {
        assert(owner_fd == -1 && errno == ENOENT);
    }
    if (sdma) {
        const auto marker =
            barrier_marker_path(h->rootinfo_path, h->run_token, domain_barrier_tag(1, "host_window"), 0);
        assert(std::remove(marker.c_str()) == 0 && rmdir(folder) == 0);
    }
}

static void test_host_sdma_capability_rejection(const std::string &scenario) {
    auto folder = "/tmp/simpler-host-capability-" + std::to_string(getpid()) + "-XXXXXX";
    assert(mkdtemp(folder.data()));
    const auto size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const uint32_t ranks[] = {0, 1};
    std::array<std::unique_ptr<CommHandle_>, 2> handles;
    std::array<std::string, 2> names, markers;
    for (uint32_t rank = 0; rank < 2; ++rank) {
        auto &h = handles[rank];
        h = std::make_unique<CommHandle_>();
        h->rank = static_cast<int>(rank);
        h->nranks = 2;
        h->rootinfo_path = std::string(folder) + "/root";
        names[rank] = "simpler_cap_ut_" + std::to_string(getpid()) + "_" + std::to_string(rank);
        markers[rank] = barrier_marker_path(h->rootinfo_path, h->run_token, domain_barrier_tag(1, "host_window"), rank);
    }
    std::ofstream(markers[1]) << names[1] << '\n';
    for (uint32_t rank = 0; rank < 2; ++rank) {
        host_sdma_capability = rank == 0 ? ACL_RT_HOST_MEM_MAP_NOT_SUPPORTED : ACL_RT_HOST_MEM_MAP_SUPPORTED;
        fail_host_capability = rank == 0 && scenario == "host_sdma_capability_failure";
        fail_get_device = rank == 0 && scenario == "host_sdma_device_failure";
        uint64_t ctx = 0, base = 0;
        size_t mapped = 0;
        assert(
            comm_alloc_host_domain_windows(
                handles[rank].get(), 1, ranks, 2, rank, size, comm_workspace_bit(CommWorkspaceKind::SDMA),
                names[rank].c_str(), &ctx, &base, &mapped
            ) == COMM_INITIALIZATION_ABORT_REQUIRED
        );
        assert(ctx == 0 && base == 0 && mapped == 0 && handles[rank]->domain_allocations.count(1) == 1);
        if (rank == 0) {
            std::string announcement;
            std::ifstream(markers[0]) >> announcement;
            assert(announcement == "!" && registered_hosts.empty());
        }
    }
    assert(host_capability_calls == (scenario == "host_sdma_device_failure" ? 1 : 2));
    assert(registered_hosts.size() == 1);
    for (uint32_t rank = 0; rank < 2; ++rank) {
        assert(comm_abort_domain_windows(handles[rank].get(), 1, 2, rank) == 0);
        assert(handles[rank]->domain_allocations.empty());
        const auto path = "/" + names[rank];
        const int fd = shm_open(path.c_str(), O_RDWR, 0);
        assert(fd == -1 && errno == ENOENT);
        assert(std::remove(markers[rank].c_str()) == 0);
    }
    assert(registered_hosts.empty() && mappings.empty() && rmdir(folder.c_str()) == 0);
}

static void test_host_cleanup_failure_notifies_peer(const std::string &scenario) {
    auto folder = (std::filesystem::temp_directory_path() / "simpler-host-notify-XXXXXX").string();
    assert(mkdtemp(folder.data()));
    const auto root = folder + "/root";
    const uint32_t failed_rank = scenario == "host_notify_abort" ? 1 : 0;
    const int peer_rank = 1 - static_cast<int>(failed_rank);
    auto failed = std::make_unique<CommHandle_>();
    auto peer = std::make_unique<CommHandle_>();
    failed->rank = static_cast<int>(failed_rank);
    peer->rank = peer_rank;
    for (auto *h : {failed.get(), peer.get()}) {
        h->nranks = 2;
        h->rootinfo_path = root;
        h->run_token = 2;
    }
    const uint32_t ranks[] = {0, 1};
    const size_t size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const auto name = "simpler_notify_" + std::to_string(getpid());
    const auto path = "/" + name;
    uint64_t ctx = 0, base = 0;
    size_t mapped = 0;
    assert(
        comm_alloc_host_domain_windows(
            failed.get(), 1, ranks, 2, failed_rank, size, 0, name.c_str(), &ctx, &base, &mapped
        ) == 0
    );
    const auto old_failure = barrier_marker_path(root, 1, "a3_host_cleanup_failed", 0);
    assert(file_marker_handshake::publish_marker(old_failure));
    assert(!host_cleanup_failed(peer.get()));
    auto device_domain = std::make_unique<DomainAllocation>();
    device_domain->rank = peer_rank;
    device_domain->nranks = 2;
    device_domain->initialized = true;
    peer->domain_allocations.emplace(2, std::move(device_domain));
    auto result = std::async(std::launch::async, [&] {
        return comm_destroy(peer.get());
    });
    const auto arrival = barrier_marker_path(root, peer->run_token, "destroy", peer_rank);
    assert(
        file_marker_handshake::wait_for_path(
            arrival, file_marker_handshake::Stage::ARRIVAL_WAIT, peer_rank,
            std::chrono::steady_clock::now() + std::chrono::seconds(2), std::chrono::milliseconds(1)
        )
            .ok()
    );

    fail_host_unregister = true;
    int rc = COMM_RECLAMATION_UNCONFIRMED;
    if (scenario == "host_notify_release") {
        rc = comm_release_domain_windows(failed.get(), 1, 2, failed_rank);
    } else if (scenario == "host_notify_abort") {
        rc = comm_abort_domain_windows(failed.get(), 1, 2, failed_rank);
    } else {
        rc = comm_destroy(failed.get());
    }
    assert(rc == COMM_RECLAMATION_UNCONFIRMED);
    const auto peer_status = result.wait_for(std::chrono::seconds(2));
    assert(failed->domain_allocations.count(1) == 1 && registered_hosts.size() == 1);
    const int fd = shm_open(path.c_str(), O_RDWR, 0);
    assert(fd >= 0 && close(fd) == 0);

    fail_host_unregister = false;
    assert(comm_abort_domain_windows(failed.get(), 1, 2, failed_rank) == 0);
    assert(mappings.empty() && registered_hosts.empty());
    assert(shm_open(path.c_str(), O_RDWR, 0) == -1 && errno == ENOENT);
    assert(peer_status == std::future_status::ready && result.get() == COMM_RECLAMATION_UNCONFIRMED);
    assert(comm_release_domain_windows(peer.get(), 2, 2, peer_rank) == COMM_RECLAMATION_UNCONFIRMED);
    assert(peer->domain_allocations.count(2) == 1);
    assert(
        comm_alloc_domain_windows(peer.get(), 3, ranks, 2, peer_rank, size, 0, &ctx, &base, &mapped) ==
        COMM_RECLAMATION_UNCONFIRMED
    );
    assert(
        comm_alloc_host_domain_windows(
            peer.get(), 3, ranks, 2, peer_rank, size, 0, name.c_str(), &ctx, &base, &mapped
        ) == COMM_RECLAMATION_UNCONFIRMED
    );
    assert(file_marker_handshake::cleanup(root).ok() && rmdir(folder.c_str()) == 0);
}

int main(int argc, char **argv) {
    assert(argc == 3);
    fabric_supported = std::string(argv[1]) == "fabric";
    reject_explicit_access = std::string(argv[2]) == "map_grants_access";
    const std::string scenario = reject_explicit_access ? "success" : argv[2];
    if (scenario == "host_notify_release" || scenario == "host_notify_abort" || scenario == "host_notify_destroy") {
        test_host_cleanup_failure_notifies_peer(scenario);
        return 0;
    }
    if (scenario == "host_sdma_unsupported" || scenario == "host_sdma_capability_failure" ||
        scenario == "host_sdma_device_failure") {
        test_host_sdma_capability_rejection(scenario);
        return 0;
    }
    if (scenario == "host_register_failure" || scenario == "host_unregister_failure" ||
        scenario == "host_borrow_sdma" || scenario == "host_name_collision") {
        test_host_registration_lifecycle(scenario);
        return 0;
    }
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
    assert(explicit_access_calls == 0);
    delete h;
    std::filesystem::remove_all(folder);
    return 0;
}
