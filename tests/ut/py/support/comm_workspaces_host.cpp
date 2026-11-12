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
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "comm_workspace_managers.h"
#include "comm_hccl.cpp"

namespace {
std::unordered_map<void *, size_t> mappings;
int physical_handles = 0;
void *fail_free_once = nullptr;
bool throw_hccl_destroy = false;
bool fail_context_upload = false;
bool fail_host_register = false;
bool unsupported_host_udma = false;
int subcomm_creates = 0;
bool fail_host_pointer = false;
bool fail_host_unregister = false;
std::unordered_map<void *, int> registered_hosts;
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

extern "C" aclError aclrtHostRegisterV2(void *host, uint64_t bytes, uint32_t flags) {
    assert(bytes > 0 && bytes % static_cast<size_t>(sysconf(_SC_PAGESIZE)) == 0);
    assert(flags == ACL_HOST_REG_MAPPED);
    if (fail_host_register) return ACL_ERROR_FAILURE;
    assert(registered_hosts.emplace(host, workspace_fixture::live[1]).second);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtHostGetDevicePointer(void *host, void **device, uint32_t) {
    assert(registered_hosts.count(host) == 1);
    if (fail_host_pointer) return ACL_ERROR_FAILURE;
    *device = host;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtHostUnregister(void *host) {
    assert(registered_hosts.count(host) == 1);
    if (fail_host_unregister) {
        fail_host_unregister = false;
        return ACL_ERROR_FAILURE;
    }
    assert(workspace_fixture::live[1] <= registered_hosts.at(host));
    registered_hosts.erase(host);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtHostMemMapCapabilities(uint32_t, aclrtHacType engine, aclrtHostMemMapCapability *capability) {
    assert(engine == ACL_RT_HAC_TYPE_UDMA);
    *capability = unsupported_host_udma ? ACL_RT_HOST_MEM_MAP_NOT_SUPPORTED : ACL_RT_HOST_MEM_MAP_SUPPORTED;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtGetDevice(int32_t *id) {
    *id = 0;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMalloc(void **out, size_t bytes, aclrtMemMallocPolicy) { return map_buffer(out, bytes); }
extern "C" aclError aclrtFree(void *p) {
    if (p == fail_free_once) {
        fail_free_once = nullptr;
        return ACL_ERROR_FAILURE;
    }
    return unmap_buffer(p);
}
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
    ++subcomm_creates;
    *out = reinterpret_cast<HcclComm>(static_cast<uintptr_t>(id + 2));
    return HCCL_SUCCESS;
}
extern "C" HcclResult HcclCommDestroy(HcclComm) {
    if (throw_hccl_destroy) throw std::runtime_error("injected HCCL destroy failure");
    return HCCL_SUCCESS;
}

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
    assert(mappings.size() == 1 && mappings.count(h->sdma_workspace->GetWorkspaceAddr()) == 1);
    assert(physical_handles == 0 && imported_handles.empty() && peer_mappings.empty());
    h->nranks = old_nranks;
}

static void test_cleanup_failure(const std::string &scenario) {
    using namespace workspace_fixture;
    setenv("PTO_ROCE_PHYIDS", "0", 1);
    setenv("PTO_ROCE_LOCAL_IP", "127.0.0.1", 1);
    char folder[] = "/tmp/simpler-workspace-cleanup-XXXXXX";
    assert(mkdtemp(folder));
    auto *h = new CommHandle_{};
    h->rank = 0;
    h->nranks = 1;
    h->rootinfo_path = std::string(folder) + "/root";
    h->hccl_comm = reinterpret_cast<HcclComm>(1);
    const int engine = scenario == "urma_release" || scenario == "urma_destroy" ? 1 :
                       scenario == "rdma_release"                               ? 2 :
                                                                                  0;
    const uint32_t ranks[] = {0};
    uint64_t ctx = 0, base = 0;
    size_t size = 0;
    assert(comm_alloc_domain_windows(h, 1, ranks, 1, 0, 4096, 1U << engine, &ctx, &base, &size) == 0);
    void *workspace = reinterpret_cast<void *>(reinterpret_cast<CommContext *>(ctx)->workspaces[engine].addr);
    assert(workspace != nullptr && mappings.count(workspace) == 1);
    const bool destroy = scenario == "sdma_destroy" || scenario == "urma_destroy" || scenario == "destroy_exception";
    if (engine == 0) assert(comm_release_domain_windows(h, 1, 1, 0) == 0);
    const bool exception = scenario == "destroy_exception";
    throw_hccl_destroy = exception;
    if (!exception) fail_free_once = workspace;
    const auto held_mappings = mappings.size();
    assert((destroy ? comm_destroy(h) : comm_release_domain_windows(h, 1, 1, 0)) == COMM_RECLAMATION_UNCONFIRMED);
    assert(h->cleanup_failed && mappings.count(workspace) == 1);
    assert(fail_free_once == nullptr);
    if (exception) {
        assert(mappings.size() == held_mappings && live[0] == 1);
    } else {
        assert(live[engine] == 0);
        if (engine == 0) assert(h->sdma_workspace_addr == workspace);
        else {
            const auto &alloc = *h->domain_allocations.at(1);
            assert(alloc.cleanup_failed);
            if (engine == 1) assert(alloc.urma_workspace_addr == workspace);
#ifdef SIMPLER_ENABLE_PTO_RDMA_WORKSPACE
            else assert(alloc.rdma_workspace_addr == workspace);
#endif
        }
    }
    const auto retained_mappings = mappings.size();
    assert(comm_destroy(h) == COMM_RECLAMATION_UNCONFIRMED);
    assert(mappings.size() == retained_mappings);
    std::filesystem::remove_all(folder);
}

static void test_host_domain(const std::string &scenario) {
    using namespace workspace_fixture;
    char folder[] = "/tmp/simpler-host-domain-XXXXXX";
    assert(mkdtemp(folder));
    auto h = std::make_unique<CommHandle_>();
    h->rank = 0;
    h->nranks = 2;
    h->rootinfo_path = std::string(folder) + "/root";
    h->run_token = 2;
    h->hccl_comm = reinterpret_cast<HcclComm>(1);
    const std::string name = "simpler_host_" + std::to_string(getpid());
    const std::string path = "/" + name;
    const uint32_t ranks[] = {0, 1};
    uint64_t ctx = 0, base = 0;
    size_t size = 0;
    const uint32_t mask = scenario == "host_local" ? 0 : 2;
    const bool aiv_preflight_failed = scenario == "host_aiv_preflight_failed";
    const bool peer_prepare_failed = scenario == "host_peer_prepare_failed";
    if (aiv_preflight_failed || peer_prepare_failed) {
        auto peer = std::make_unique<CommHandle_>();
        peer->rank = 1;
        peer->nranks = h->nranks;
        peer->rootinfo_path = h->rootinfo_path;
        peer->run_token = h->run_token;
        peer->hccl_comm = h->hccl_comm;
        const auto aiv_count = device_aiv_count;
        const auto failed_name = name + "_failed";
        if (aiv_preflight_failed) device_aiv_count = 0;
        fail_host_register = peer_prepare_failed;
        const int failed_rc =
            comm_alloc_host_domain_windows(h.get(), 1, ranks, 2, 0, 64, mask, failed_name.c_str(), &ctx, &base, &size);
        assert(
            aiv_preflight_failed ? failed_rc == COMM_DOMAIN_URMA_CONFIG_INVALID :
                                   failed_rc == COMM_INITIALIZATION_ABORT_REQUIRED
        );
        assert(h->domain_allocations.size() == static_cast<size_t>(peer_prepare_failed) && registered_hosts.empty());
        const auto failed = domain_barrier_tag(1, "host_failed");
        assert(std::filesystem::is_regular_file(barrier_marker_path(h->rootinfo_path, h->run_token, failed, 0)));

        device_aiv_count = aiv_count;
        fail_host_register = false;
        assert(
            comm_alloc_host_domain_windows(peer.get(), 1, ranks, 2, 1, 64, mask, name.c_str(), &ctx, &base, &size) ==
            COMM_INITIALIZATION_ABORT_REQUIRED
        );
        assert(subcomm_creates == 0 && ctx == 0 && base == 0 && peer->domain_allocations.count(1) == 1);
        assert(peer->domain_allocations.at(1)->host_registered);
        if (peer_prepare_failed) assert(comm_abort_domain_windows(h.get(), 1, 2, 0) == 0);
        assert(comm_abort_domain_windows(peer.get(), 1, 2, 1) == 0);
        assert(h->domain_allocations.empty() && peer->domain_allocations.empty());
        assert(mappings.empty() && registered_hosts.empty());
        assert(shm_open(("/" + failed_name).c_str(), O_RDWR, 0) < 0 && errno == ENOENT);
        assert(shm_open(path.c_str(), O_RDWR, 0) < 0 && errno == ENOENT);

        const auto ready = domain_barrier_tag(2, "host_ready");
        assert(file_marker_handshake::publish_marker(barrier_marker_path(h->rootinfo_path, h->run_token, ready, 0)));
        assert(
            comm_alloc_host_domain_windows(peer.get(), 2, ranks, 2, 1, 64, mask, name.c_str(), &ctx, &base, &size) == 0
        );
        assert(subcomm_creates == 1 && ctx != 0 && base != 0);
        const auto release = domain_barrier_tag(2, "release");
        assert(file_marker_handshake::publish_marker(barrier_marker_path(h->rootinfo_path, h->run_token, release, 0)));
        assert(comm_release_domain_windows(peer.get(), 2, 2, 1) == 0);
        assert(peer->domain_allocations.empty() && mappings.empty() && registered_hosts.empty());
        std::filesystem::remove_all(folder);
        return;
    }
    uint64_t device_context = 0;
    if (scenario == "host_coexist") {
        assert(domain_write_announce(h->rootinfo_path, 2, 1, h->run_token, getpid(), 1, 42));
        for (const char *phase : {"p2p_ready", "release"}) {
            std::ofstream marker(barrier_marker_path(h->rootinfo_path, h->run_token, domain_barrier_tag(2, phase), 1));
            marker << "1";
        }
        assert(comm_alloc_domain_windows(h.get(), 2, ranks, 2, 0, 4096, 2, &device_context, &base, &size) == 0);
        base = size = 0;
        assert(live[1] == 1);
    }
    fail_host_register = scenario == "host_register";
    fail_host_pointer = scenario == "host_pointer";
    const bool abort = scenario == "host_unregister_abort";
    const bool destroy = scenario == "host_unregister_destroy";
    fail_host_unregister = scenario == "host_unregister" || abort || destroy;
    fail_context_upload = scenario == "host_context";
    fail_engine = scenario == "host_workspace" ? 1 : -1;
    if (scenario == "host_collision") {
        const int existing = shm_open(path.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        assert(existing >= 0);
        assert(ftruncate(existing, 8192) == 0);
        close(existing);
    }
    if (scenario == "host_invalid") {
        for (const char *bad :
             {"", "/already_absolute", "with/slash", "with space", "abcdefghijklmnopqrstuvwxyz0123456789"}) {
            assert(comm_alloc_host_domain_windows(h.get(), 1, ranks, 2, 0, 64, 2, bad, &ctx, &base, &size) != 0);
            assert(h->domain_allocations.empty());
        }
        for (uint32_t unsupported : {1U, 3U, 4U}) {
            assert(
                comm_alloc_host_domain_windows(
                    h.get(), 1, ranks, 2, 0, 64, unsupported, name.c_str(), &ctx, &base, &size
                ) == COMM_DOMAIN_WORKSPACE_UNSUPPORTED
            );
            assert(h->domain_allocations.empty());
        }
        std::filesystem::remove_all(folder);
        return;
    }
    const bool configured = scenario == "host_urma_config";
    unsupported_host_udma = scenario == "host_urma_unsupported";
    const bool ready_publish_failed = scenario == "host_ready_publish_failed";
    if (ready_publish_failed) {
        const auto tag = domain_barrier_tag(1, "host_ready");
        const auto marker = barrier_marker_path(h->rootinfo_path, h->run_token, tag, 0);
        assert(std::filesystem::create_directory(marker + ".tmp"));
    }
    if (mask != 0) {
        const auto tag = domain_barrier_tag(1, "host_ready");
        const auto marker = barrier_marker_path(h->rootinfo_path, h->run_token, tag, 1);
        assert(file_marker_handshake::publish_marker(marker));
    }
    const CommWorkspaceConfig config{2, 512};
    const int rc = comm_alloc_host_domain_windows(
        h.get(), 1, ranks, 2, 0, 64, mask, name.c_str(), &ctx, &base, &size, configured ? &config : nullptr
    );
    if (configured) {
        assert(sized_jetties == 2 && initialized_jetties == 2 && initialized_sq_depth == 512);
        assert(sized_aivs == device_aiv_count && initialized_aivs == device_aiv_count);
    }
    if (scenario == "host_register" || scenario == "host_pointer" || scenario == "host_workspace" ||
        scenario == "host_context" || scenario == "host_collision" || unsupported_host_udma || ready_publish_failed) {
        if (unsupported_host_udma || ready_publish_failed) assert(live[1] == 0);
        if (ready_publish_failed) {
            assert(subcomm_creates == 0);
            const auto tag = domain_barrier_tag(1, "host_failed");
            assert(std::filesystem::exists(barrier_marker_path(h->rootinfo_path, h->run_token, tag, 0)));
        }
        if (unsupported_host_udma) assert(registered_hosts.empty());
        assert(rc == COMM_INITIALIZATION_ABORT_REQUIRED && ctx == 0 && base == 0);
        assert(h->domain_allocations.count(1) == 1);
        assert(comm_abort_domain_windows(h.get(), 1, 2, 0) == 0);
        assert(h->domain_allocations.empty() && mappings.empty() && registered_hosts.empty());
        const int probe = shm_open(path.c_str(), O_RDWR, 0);
        if (scenario == "host_collision") {
            assert(probe >= 0);
            struct stat st{};
            assert(fstat(probe, &st) == 0 && st.st_size == 8192);
            close(probe);
            assert(shm_unlink(path.c_str()) == 0);
        } else {
            assert(probe < 0 && errno == ENOENT);
        }
        assert(live[1] == 0 && physical_handles == 0);
        std::filesystem::remove_all(folder);
        return;
    }
    assert(rc == 0 && ctx != 0 && base != 0 && size >= 64);
    assert(physical_handles == (device_context != 0 ? 2 : 0));
    assert(peer_import_calls == (device_context != 0 ? 1 : 0));
    const auto *context = reinterpret_cast<const CommContext *>(ctx);
    assert(context->windowsIn[0] == base && context->windowsOut[0] == base);
    assert(context->windowsIn[1] == 0 && context->windowsOut[1] == 0);
    assert((comm_workspace<CommWorkspaceKind::URMA>(*context).addr != 0) == (mask != 0));
    const int fd = shm_open(path.c_str(), O_RDWR, 0);
    assert(fd >= 0);
    void *parent = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    assert(parent != MAP_FAILED && parent != reinterpret_cast<void *>(base));
    auto *cpu = static_cast<uint32_t *>(parent);
    auto *device = reinterpret_cast<uint32_t *>(base);
    cpu[7] = 0x12345678;
    assert(device[7] == 0x12345678);
    device[9] = 0x98765432;
    assert(cpu[9] == 0x98765432);
    {
        std::ofstream marker(barrier_marker_path(h->rootinfo_path, h->run_token, domain_barrier_tag(1, "release"), 1));
        marker << "1";
    }
    if (fail_host_unregister) {
        assert(file_marker_handshake::publish_failure(file_marker_handshake::failure_path(h->rootinfo_path, 1)));
        if (destroy) {
            for (const char *tag : {"destroy", "destroy_departed"}) {
                assert(
                    file_marker_handshake::publish_marker(barrier_marker_path(h->rootinfo_path, h->run_token, tag, 1))
                );
            }
        }
        const int cleanup_rc = destroy ? comm_destroy(h.get()) :
                               abort   ? comm_abort_domain_windows(h.get(), 1, 2, 0) :
                                         comm_release_domain_windows(h.get(), 1, 2, 0);
        assert(cleanup_rc == COMM_RECLAMATION_UNCONFIRMED && !fail_host_unregister);
        assert(std::filesystem::is_regular_file(file_marker_handshake::failure_path(h->rootinfo_path, h->run_token)));

        auto peer = std::make_unique<CommHandle_>();
        peer->rank = 1;
        peer->nranks = h->nranks;
        peer->rootinfo_path = h->rootinfo_path;
        peer->run_token = h->run_token;
        peer->hccl_comm = h->hccl_comm;
        assert(comm_destroy(peer.get()) == COMM_RECLAMATION_UNCONFIRMED);
        assert(peer->hccl_comm != nullptr);

        assert(h->cleanup_failed && h->domain_allocations.at(1)->host_registered);
        assert(registered_hosts.size() == 1 && cpu[7] == 0x12345678);
        const int retained = shm_open(path.c_str(), O_RDWR, 0);
        assert(retained >= 0);
        close(retained);
        assert(comm_release_domain_windows(h.get(), 1, 2, 0) == COMM_RECLAMATION_UNCONFIRMED);
        assert(shm_unlink(path.c_str()) == 0);
    } else {
        assert(comm_release_domain_windows(h.get(), 1, 2, 0) == 0);
        assert(registered_hosts.empty());
        if (device_context != 0) {
            assert(h->domain_allocations.size() == 1 && live[1] == 1);
            const auto *device_view = reinterpret_cast<const CommContext *>(device_context);
            assert(device_view->windowsIn[0] != 0 && device_view->windowsIn[1] != 0);
            assert(comm_workspace<CommWorkspaceKind::URMA>(*device_view).addr != 0);
            assert(comm_release_domain_windows(h.get(), 2, 2, 0) == 0);
        }
        assert(h->domain_allocations.empty() && mappings.empty() && physical_handles == 0);
        assert(shm_open(path.c_str(), O_RDWR, 0) < 0 && errno == ENOENT);
        // An exported CPU mapping retains the POSIX backing after native ownership ends.
        assert(cpu[7] == 0x12345678 && cpu[9] == 0x98765432);
    }
    assert(munmap(parent, size) == 0);
    std::filesystem::remove_all(folder);
}

int main(int argc, char **argv) {
    if (argc == 2) {
        if (std::string(argv[1]).find("host_") == 0) {
            test_host_domain(argv[1]);
            return 0;
        }
        test_cleanup_failure(argv[1]);
        return 0;
    }
    assert(argc == 1);
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
    auto allocate = [&](uint32_t mask, uint64_t &id, uint64_t &ctx, const CommWorkspaceConfig *config = nullptr) {
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
        return comm_alloc_domain_windows(h, id, ranks, count, 0, 4096, mask, &ctx, &base, &size, config);
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

    for (const CommWorkspaceConfig config : {CommWorkspaceConfig{0, 0}, {9, 0}, {2, 3}, {2, 1u << 31}}) {
        const int starts = started[1];
        assert(allocate(2, first, first_ctx, &config) == COMM_DOMAIN_URMA_CONFIG_INVALID);
        assert(started[1] == starts && h->domain_allocations.empty());
        assert(mappings.empty() && physical_handles == 0);
    }
    const CommWorkspaceConfig multi{2, 512};
    assert(allocate(0, first, first_ctx, &multi) == COMM_DOMAIN_URMA_CONFIG_INVALID);
    device_aiv_count = 129;
    assert(allocate(2, first, first_ctx, &multi) == COMM_DOMAIN_URMA_CONFIG_INVALID);
    assert(h->domain_allocations.empty());
    device_aiv_count = 2;
    assert(allocate(2, first, first_ctx, &multi) == 0);
    assert(sized_jetties == 2 && initialized_jetties == 2 && initialized_sq_depth == 512);
    assert(sized_aivs == 2 && initialized_aivs == 2);
    auto *multi_info = reinterpret_cast<const pto::comm::urma::UrmaInfo *>(
        comm_workspace<CommWorkspaceKind::URMA>(*reinterpret_cast<CommContext *>(first_ctx)).addr
    );
    assert(multi_info->rowsPerPeer == 4 && multi_info->jettyCount == 8);
    release(first);
    assert(mappings.empty() && physical_handles == 0);

    if ((compiled_comm_workspaces() & 2) != 0) {
        uint64_t second = 0, second_ctx = 0;
        assert(allocate(3, first, first_ctx) == 0);
        assert(allocate(3, second, second_ctx) == 0);
        const auto *a = reinterpret_cast<CommContext *>(first_ctx);
        const auto *b = reinterpret_cast<CommContext *>(second_ctx);
        assert(comm_workspace<CommWorkspaceKind::SDMA>(*a).size == pto::comm::sdma::kSdmaWorkspaceBytes);
        assert(comm_workspace<CommWorkspaceKind::SDMA>(*b).size == pto::comm::sdma::kSdmaWorkspaceBytes);
        assert(comm_workspace<CommWorkspaceKind::SDMA>(*a).addr == comm_workspace<CommWorkspaceKind::SDMA>(*b).addr);
        assert(comm_workspace<CommWorkspaceKind::URMA>(*a).addr != comm_workspace<CommWorkspaceKind::URMA>(*b).addr);
        assert(comm_workspace<CommWorkspaceKind::SDMA>(*a).addr != comm_workspace<CommWorkspaceKind::URMA>(*a).addr);
        release(first);
        assert(live[1] == 1 && live[0] == 1);
        const auto held_mappings = mappings.size();
        const int held_physical = physical_handles;
        zero_size_engine = 1;
        const int starts = started[1];
        uint64_t zero_id = 0, zero_ctx = 0;
        assert(allocate(3, zero_id, zero_ctx) == COMM_INITIALIZATION_ABORT_REQUIRED);
        assert(started[1] == starts);
        assert(comm_abort_domain_windows(h, zero_id, h->domain_allocations.at(zero_id)->nranks, 0) == 0);
        assert(mappings.size() == held_mappings && physical_handles == held_physical);
        zero_size_engine = -1;
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
        zero_size_engine = 2;
        const int starts = started[2];
        uint64_t zero_id = 0, zero_ctx = 0;
        assert(allocate(4, zero_id, zero_ctx) == COMM_INITIALIZATION_ABORT_REQUIRED);
        assert(started[2] == starts);
        assert(comm_abort_domain_windows(h, zero_id, h->domain_allocations.at(zero_id)->nranks, 0) == 0);
        assert(mappings.size() == held_mappings && physical_handles == held_physical);
        zero_size_engine = -1;
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
    assert(mappings.size() == 1 && mappings.count(h->sdma_workspace->GetWorkspaceAddr()) == 1);
    assert(physical_handles == 0);
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
    zero_size_engine = 0;
    const int starts = started[0];
    assert(allocate(1, first, first_ctx) == COMM_INITIALIZATION_ABORT_REQUIRED);
    assert(comm_abort_domain_windows(h, first, 1, 0) == 0);
    assert(started[0] == starts && !h->sdma_workspace && mappings.empty() && physical_handles == 0);
    zero_size_engine = -1;
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
