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

namespace simpler::ccu {
bool query_window_token(uint64_t, uint64_t, uint64_t &) {
    assert(false && "URMA-only fixture must not query CCU tokens");
    return false;
}
}  // namespace simpler::ccu

namespace {
std::unordered_map<HcclComm, std::vector<CommMem>> registrations;
struct SubComm {
    std::vector<uint32_t> base_ranks;
    uint32_t rank;
};
std::unordered_map<HcclComm, SubComm> child_comms;
int fail_registration = 0;
int registration_calls = 0;
bool fail_destroy = false;
int destroy_calls = 0;
std::vector<HcclComm> destroyed;
int unsafe_frees = 0;
int channel_calls = 0;
bool fail_channels = true;
struct ChannelState {
    ChannelEntity entity{};
    SqContext sq{};
    CqContext cq{};
    RegedBufferEntity buffers[2]{};
    RegedBufferEntity remote_buffers[2]{};
    CommMem remote_mems[2]{};
    uint32_t sq_head = 0, sq_tail = 0, cq_head = 0, cq_tail = 0, sq_db = 0, cq_db = 0;
    char sq_buffer[64]{}, cq_buffer[64]{};
};
struct ChannelDeleter {
    void operator()(ChannelState *state) const {
        state->~ChannelState();
        assert(munmap(state, 4096) == 0);
    }
};
std::unordered_map<HcclComm, std::unique_ptr<ChannelState, ChannelDeleter>> channels;
std::unordered_map<void *, size_t> mappings;
int physical_handles = 0;
bool fail_get_device = false;
struct PhysicalAllocation {};
std::unordered_map<aclrtDrvMemHandle, std::shared_ptr<PhysicalAllocation>> physical_resources;
std::unordered_map<uint64_t, std::weak_ptr<PhysicalAllocation>> exported_resources;
uint64_t last_exported = 0;
bool fail_context_upload = false;
enum class PeerFailure { NONE, IMPORT, RESERVE, MAP };
PeerFailure peer_failure = PeerFailure::NONE;
bool peer_reserve_pending = false;
int peer_import_calls = 0;
int injected_peer_failures = 0;
int explicit_access_calls = 0;
std::unordered_set<aclrtDrvMemHandle> imported_handles;
std::unordered_set<void *> peer_mappings;

uint64_t remote_window_address(HcclComm comm, uint32_t base_rank) {
    return 0x300000000000ULL + (reinterpret_cast<uintptr_t>(comm) * COMM_MAX_RANK_NUM + base_rank) * 8192;
}

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
    const auto start = reinterpret_cast<uintptr_t>(p);
    for (const auto &entry : registrations)
        for (const auto &mem : entry.second) {
            const auto address = reinterpret_cast<uintptr_t>(mem.addr);
            if (address >= start && address - start < it->second) ++unsafe_frees;
        }
    const int rc = munmap(p, it->second);
    assert(rc == 0);
    mappings.erase(it);
    return ACL_SUCCESS;
}
}  // namespace

extern "C" aclError aclrtHostRegisterV2(void *, uint64_t, uint32_t) { return ACL_ERROR_FAILURE; }
extern "C" aclError aclrtHostGetDevicePointer(void *, void **, uint32_t) { return ACL_ERROR_FAILURE; }
extern "C" aclError aclrtHostUnregister(void *) { return ACL_ERROR_FAILURE; }
extern "C" aclError aclrtHostMemMapCapabilities(uint32_t, aclrtHacType, aclrtHostMemMapCapability *) {
    return ACL_ERROR_FAILURE;
}
extern "C" aclError aclrtGetDevice(int32_t *id) {
    if (fail_get_device) return 1;
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
    for (const auto &entry : registrations)
        for (const auto &mem : entry.second)
            if (mem.addr == address) ++unsafe_frees;
    peer_mappings.erase(address);
    return ACL_SUCCESS;
}
// A5 window mappings rely on MapMem's owner grant and issue no explicit access
// grant; any such call aborts the fixture.
extern "C" aclError aclrtMemSetAccess(void *, size_t, aclrtMemAccessDesc *, size_t) {
    ++explicit_access_calls;
    assert(explicit_access_calls == 0);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp *, aclrtMemGranularityOptions, size_t *n) {
    *n = 4096;
    return ACL_SUCCESS;
}
extern "C" aclError
aclrtMemExportToShareableHandle(aclrtDrvMemHandle handle, aclrtMemHandleType, uint64_t, uint64_t *out) {
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
extern "C" HcclResult HcclCommDestroy(HcclComm comm) {
    ++destroy_calls;
    destroyed.push_back(comm);
    if (fail_destroy) return HCCL_E_INTERNAL;
    channels.erase(comm);
    registrations.erase(comm);
    child_comms.erase(comm);
    return HCCL_SUCCESS;
}
extern "C" HcclResult HcclCreateSubCommConfig(
    HcclComm *base, uint32_t count, uint32_t *ranks, uint64_t id, uint32_t rank, HcclCommConfig *, HcclComm *out
) {
    assert(*base == reinterpret_cast<HcclComm>(1) && rank < count);
    *out = reinterpret_cast<HcclComm>(id + 100);
    assert(child_comms.emplace(*out, SubComm{{ranks, ranks + count}, rank}).second);
    return HCCL_SUCCESS;
}
extern "C" HcclResult HcclCommMemReg(HcclComm comm, const char *, const CommMem *mem, HcclMemHandle *out) {
    if (++registration_calls == fail_registration) return HCCL_E_INTERNAL;
    registrations[comm].push_back(*mem);
    *out = mem->addr;
    return HCCL_SUCCESS;
}
extern "C" HcclResult
HcclRankGraphGetLinks(HcclComm comm, uint32_t, uint32_t src, uint32_t dst, CommLink **links, uint32_t *count) {
    const auto size = child_comms.at(comm).base_ranks.size();
    assert(src < size && dst < size && src != dst);
    static CommLink link{};
    link.linkAttr.linkProtocol = static_cast<CommProtocol>(4);
    *links = &link;
    *count = 1;
    return HCCL_SUCCESS;
}
extern "C" HcclResult
HcclChannelAcquire(HcclComm comm, CommEngine, const HcclChannelDesc *desc, uint32_t count, ChannelHandle *out) {
    ++channel_calls;
    if (fail_channels) return HCCL_E_INTERNAL;
    assert(count == 1);
    const auto &group = child_comms.at(comm);
    assert(desc->remoteRank != group.rank);
    const auto remote_addr = remote_window_address(comm, group.base_ranks.at(desc->remoteRank));
    static uintptr_t hint = 0x200000000000ULL;
    void *region =
        mmap(reinterpret_cast<void *>(hint), 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(region != MAP_FAILED);
    hint += 65536;
    std::unique_ptr<ChannelState, ChannelDeleter> c(new (region) ChannelState{});
    c->entity.abiHeader.magicWord = 0x0f0f0f0fU;
    c->entity.engine = COMM_ENGINE_AIV;
    c->entity.sqNum = c->entity.cqNum = 1;
    c->entity.localBufferNum = c->entity.remoteBufferNum = 2;
    c->entity.localBufferAddr = c->buffers;
    c->entity.remoteBufferAddr = c->remote_buffers;
    c->entity.sqContextAddr = &c->sq;
    c->entity.cqContextAddr = &c->cq;
    for (size_t i = 0; i < 2; ++i) {
        c->buffers[i].type = REGED_BUFFER_RMA;
        c->buffers[i].bufferInfo.rma.addr = reinterpret_cast<uint64_t>(registrations.at(comm)[i].addr);
        c->buffers[i].bufferInfo.rma.size = registrations.at(comm)[i].size;
        c->remote_buffers[i] = c->buffers[i];
        c->remote_buffers[i].bufferInfo.rma.addr = remote_addr + i * 4096;
        c->remote_mems[i] = registrations.at(comm)[i];
        c->remote_mems[i].addr = reinterpret_cast<void *>(c->remote_buffers[i].bufferInfo.rma.addr);
    }
    auto &sq = c->sq.contextInfo.ubJfs;
    sq.sqVa = reinterpret_cast<uint64_t>(c->sq_buffer);
    sq.headAddr = reinterpret_cast<uint64_t>(&c->sq_head);
    sq.tailAddr = reinterpret_cast<uint64_t>(&c->sq_tail);
    sq.dbVa = reinterpret_cast<uint64_t>(&c->sq_db);
    sq.sqDepth = 256;
    sq.wqeSize = 64;
    auto &cq = c->cq.contextInfo.ubJfc;
    cq.scqVa = reinterpret_cast<uint64_t>(c->cq_buffer);
    cq.headAddr = reinterpret_cast<uint64_t>(&c->cq_head);
    cq.tailAddr = reinterpret_cast<uint64_t>(&c->cq_tail);
    cq.dbVa = reinterpret_cast<uint64_t>(&c->cq_db);
    cq.cqDepth = 256;
    cq.cqeSize = 64;
    *out = reinterpret_cast<ChannelHandle>(&c->entity);
    assert(*out >= 0x100000000000ULL);
    channels[comm] = std::move(c);
    return HCCL_SUCCESS;
}
extern "C" HcclResult
HcclChannelGetRemoteMems(HcclComm comm, ChannelHandle handle, uint32_t *count, CommMem **mems, char ***tags) {
    static char sym[] = "pto_urma_sym";
    static char notify[] = "pto_urma_notify";
    static char *names[] = {sym, notify};
    auto &channel = *channels.at(comm);
    assert(handle == reinterpret_cast<ChannelHandle>(&channel.entity));
    *count = 2;
    *mems = channel.remote_mems;
    *tags = names;
    return HCCL_SUCCESS;
}
extern "C" aclError aclrtGetDeviceInfo(uint32_t, aclrtDevAttr, int64_t *value) {
    *value = 1;
    return ACL_SUCCESS;
}

static void test_overlapping_domains(CommHandle h, const std::vector<uint32_t> &bc, bool fail_release) {
    h->rank = 1;
    h->nranks = 3;
    fail_channels = false;
    const std::vector<uint32_t> ab{0, 1};
    const uint32_t bc_rank = bc.front() == 1 ? 0 : 1;
    const uint32_t urma = comm_workspace_bit(CommWorkspaceKind::URMA);
    auto allocate = [&](uint64_t id, const std::vector<uint32_t> &ranks, uint32_t rank) {
        const uint32_t peer = 1 - rank;
        assert(domain_write_announce(h->rootinfo_path, id, peer, h->run_token, getpid(), ranks[peer], 42));
        for (const char *phase : {"p2p_ready", "release"}) {
            std::ofstream marker(
                barrier_marker_path(h->rootinfo_path, h->run_token, domain_barrier_tag(id, phase), peer)
            );
            marker << "1";
            assert(marker.good());
        }
        uint64_t ctx = 0, base = 0;
        size_t size = 0;
        const int rc =
            comm_alloc_domain_windows(h, id, ranks.data(), ranks.size(), rank, 4096, urma, &ctx, &base, &size);
        if (rc == 0) {
            assert(ctx != 0 && base != 0 && size == 4096);
        } else {
            assert(ctx == 0 && base == 0 && size == 0);
        }
        return rc;
    };
    auto check_domain = [&](uint64_t id, const std::vector<uint32_t> &ranks, uint32_t rank) {
        const auto &alloc = *h->domain_allocations.at(id);
        const auto &group = child_comms.at(alloc.urma_comm);
        assert(group.base_ranks == ranks && group.rank == rank);
        assert(registrations.at(alloc.urma_comm).front().addr == alloc.local_buf);
        const auto &ctx = *alloc.device_ctx;
        const uint32_t peer = 1 - rank;
        assert(ctx.rankId == rank && ctx.rankNum == ranks.size());
        assert(ctx.windowsIn[rank] == reinterpret_cast<uint64_t>(alloc.local_buf));
        assert(ctx.windowsIn[peer] == reinterpret_cast<uint64_t>(alloc.peer_windows.at(0).address));
        const auto *info =
            reinterpret_cast<const pto::comm::urma::UrmaInfo *>(comm_workspace<CommWorkspaceKind::URMA>(ctx).addr);
        const auto *mr = reinterpret_cast<const pto::comm::urma::UrmaMemInfo *>(info->memPtr);
        assert(mr[rank * info->rowsPerPeer].addr == ctx.windowsIn[rank]);
        assert(mr[peer * info->rowsPerPeer].addr == remote_window_address(alloc.urma_comm, ranks[peer]));
    };
    assert(allocate(1, ab, 1) == 0);
    const auto ab_mappings = mappings.size();
    const int ab_channel_calls = channel_calls;
    fail_channels = true;
    assert(allocate(2, bc, bc_rank) == COMM_INITIALIZATION_ABORT_REQUIRED);
    assert(channel_calls == ab_channel_calls + 1 && child_comms.size() == 2 && registrations.size() == 2);
    assert(comm_abort_domain_windows(h, 2, 2, bc_rank) == 0);
    assert(child_comms.size() == 1 && mappings.size() == ab_mappings && unsafe_frees == 0);
    check_domain(1, ab, 1);
    fail_channels = false;
    assert(allocate(3, bc, bc_rank) == 0);
    check_domain(3, bc, bc_rank);
    if (fail_release) {
        fail_destroy = true;
        const auto held = mappings.size();
        assert(comm_release_domain_windows(h, 3, 2, bc_rank) == COMM_RECLAMATION_UNCONFIRMED);
        assert(child_comms.size() == 2 && mappings.size() == held && unsafe_frees == 0);
        assert(h->cleanup_failed && h->domain_allocations.at(3)->cleanup_failed);
        assert(!h->domain_allocations.at(1)->cleanup_failed);
        check_domain(1, ab, 1);
        check_domain(3, bc, bc_rank);
        return;
    }
    assert(comm_release_domain_windows(h, 1, 2, 1) == 0);
    assert(child_comms.size() == 1 && unsafe_frees == 0);
    check_domain(3, bc, bc_rank);
    assert(comm_release_domain_windows(h, 3, 2, bc_rank) == 0);
    assert(child_comms.empty() && registrations.empty() && mappings.empty() && unsafe_frees == 0);
    delete h;
}

int main(int argc, char **argv) {
    assert(argc == 2);
    const std::string scenario = argv[1];
    fail_registration = scenario == "notify_registration" ? 2 : 0;
    fail_destroy = scenario == "destroy_failure";
    fail_channels = scenario == "notify_registration" || scenario == "channel_acquire" || scenario == "destroy_failure";
    fail_context_upload = scenario == "context_upload" || scenario == "late_import";
    fail_get_device = scenario == "early_get_device";
    if (scenario == "peer_import") peer_failure = PeerFailure::IMPORT;
    if (scenario == "peer_reserve") peer_failure = PeerFailure::RESERVE;
    if (scenario == "peer_map") peer_failure = PeerFailure::MAP;
    char folder[] = "/tmp/simpler-urma-lifecycle-XXXXXX";
    assert(mkdtemp(folder));
    auto *h = new CommHandle_{};
    h->rank = 0;
    h->nranks = 2;
    h->rootinfo_path = std::string(folder) + "/root";
    h->hccl_comm = reinterpret_cast<HcclComm>(1);
    if (scenario == "overlap_subset" || scenario == "overlap_reordered" || scenario == "overlap_release_failure") {
        const auto ranks = scenario == "overlap_subset" ? std::vector<uint32_t>{1, 2} : std::vector<uint32_t>{2, 1};
        test_overlapping_domains(h, ranks, scenario == "overlap_release_failure");
        std::filesystem::remove_all(folder);
        return 0;
    }
    if (scenario == "base_destroy") {
        h->nranks = 1;
        fail_destroy = true;
        assert(comm_destroy(h) == COMM_RECLAMATION_UNCONFIRMED);
        assert(destroy_calls == 1 && h->hccl_comm == reinterpret_cast<HcclComm>(1));
        assert(comm_destroy(h) == COMM_RECLAMATION_UNCONFIRMED);
        assert(destroy_calls == 1);
        std::filesystem::remove_all(folder);
        return 0;
    }
    const uint64_t id = 1;
    assert(domain_write_announce(h->rootinfo_path, id, 1, h->run_token, getpid(), 1, 42));
    for (const char *phase : {"p2p_ready", "release"}) {
        if (scenario == "abort_unpublished" && std::strcmp(phase, "release") == 0) continue;
        std::ofstream marker(barrier_marker_path(h->rootinfo_path, h->run_token, domain_barrier_tag(id, phase), 1));
        marker << "1";
    }
    const uint32_t ranks[] = {0, 1};
    uint64_t ctx = 0, base = 0;
    size_t size = 0;
    const int rc = comm_alloc_domain_windows(h, id, ranks, 2, 0, 4096, 2, &ctx, &base, &size);
    if (scenario == "abort_unpublished") {
        assert(rc == 0 && ctx != 0);
        assert(comm_abort_domain_windows(h, id, 2, 0) == 0);
        assert(h->domain_allocations.empty() && registrations.empty() && mappings.empty());
        assert(destroy_calls == 1 && unsafe_frees == 0);
        delete h;
        std::filesystem::remove_all(folder);
        return 0;
    }
    if (scenario == "release_destroy_failure") {
        assert(rc == 0 && ctx != 0);
        fail_destroy = true;
        const auto held = mappings.size();
        assert(comm_release_domain_windows(h, id, 2, 0) == COMM_RECLAMATION_UNCONFIRMED);
        assert(destroy_calls == 1 && h->domain_allocations.count(id) == 1);
        assert(mappings.size() == held && registrations.size() == 1 && unsafe_frees == 0);
        assert(comm_release_domain_windows(h, id, 2, 0) == COMM_RECLAMATION_UNCONFIRMED);
        assert(destroy_calls == 1 && mappings.size() == held);
        std::filesystem::remove_all(folder);
        return 0;
    }
    if (scenario == "finalize_live" || scenario == "finalize_undrained") {
        assert(rc == 0 && ctx != 0);
        if (scenario == "finalize_undrained") {
            channels.begin()->second->sq_head = 1;
            const auto held = mappings.size();
            assert(comm_destroy(h) == COMM_RECLAMATION_UNCONFIRMED);
            assert(destroy_calls == 0 && mappings.size() == held && unsafe_frees == 0);
        } else {
            const auto root = h->rootinfo_path;
            const auto token = h->run_token;
            std::thread peer([&] {
                assert(file_marker_handshake::destroy_barrier(root, 1, 2, token).ok());
                assert(file_marker_handshake::release_after_cleanup(root, 1, 2, token).ok());
            });
            assert(comm_destroy(h) == 0);
            peer.join();
            assert(destroyed.size() == 2 && destroyed[0] == reinterpret_cast<HcclComm>(101));
            assert(destroyed[1] == reinterpret_cast<HcclComm>(1));
            assert(registrations.empty() && mappings.empty() && unsafe_frees == 0);
        }
        std::filesystem::remove_all(folder);
        return 0;
    }
    if (scenario == "success" || scenario == "undrained" || scenario == "cqe_pending") {
        assert(rc == 0 && ctx != 0);
        const auto saved = mappings.size();
        assert(registrations.size() == 1 && child_comms.size() == 1);
        if (scenario == "undrained" || scenario == "cqe_pending") {
            if (scenario == "undrained") channels.begin()->second->sq_head = 1;
            else channels.begin()->second->cq_tail = 1;
            assert(comm_release_domain_windows(h, id, 2, 0) != 0);
            assert(destroy_calls == 0 && mappings.size() == saved && unsafe_frees == 0);
        } else {
            const uint64_t second_id = 2;
            assert(domain_write_announce(h->rootinfo_path, second_id, 1, h->run_token, getpid(), 1, 42));
            for (const char *phase : {"p2p_ready", "release"}) {
                std::ofstream marker(
                    barrier_marker_path(h->rootinfo_path, h->run_token, domain_barrier_tag(second_id, phase), 1)
                );
                marker << "1";
            }
            uint64_t second_ctx = 0;
            assert(comm_alloc_domain_windows(h, second_id, ranks, 2, 0, 4096, 2, &second_ctx, &base, &size) == 0);
            assert(child_comms.size() == 2 && registrations.size() == 2);
            assert(comm_release_domain_windows(h, id, 2, 0) == 0);
            assert(destroy_calls == 1 && mappings.size() == saved && registrations.size() == 1 && unsafe_frees == 0);
            assert(reinterpret_cast<CommContext *>(second_ctx)->rankNum == 2);
            assert(h->hccl_comm == reinterpret_cast<HcclComm>(1));
            const auto prior_registrations = registration_calls;
            uint64_t rejected_ctx = 0;
            assert(comm_alloc_domain_windows(h, id, ranks, 2, 0, 4096, 2, &rejected_ctx, &base, &size) != 0);
            assert(registration_calls == prior_registrations && mappings.size() == saved);
            assert(child_comms.size() == 1 && unsafe_frees == 0);
            assert(comm_release_domain_windows(h, second_id, 2, 0) == 0);
            assert(destroy_calls == 2 && mappings.empty() && registrations.empty() && unsafe_frees == 0);
            delete h;
        }
        std::filesystem::remove_all(folder);
        return 0;
    }
    if (scenario == "early_get_device") {
        assert(rc == COMM_DOMAIN_URMA_CONFIG_INVALID && ctx == 0);
        assert(mappings.empty() && registrations.empty() && physical_handles == 0);
        assert(h->domain_allocations.empty() && destroy_calls == 0 && unsafe_frees == 0);
        delete h;
        std::filesystem::remove_all(folder);
        return 0;
    }
    assert(rc == COMM_INITIALIZATION_ABORT_REQUIRED && ctx == 0);
    assert(h->domain_allocations.count(id) == 1);
    assert(destroy_calls == 0 && unsafe_frees == 0);
    assert(!mappings.empty() && !registrations.empty());
    aclrtDrvMemHandle late_import = nullptr;
    if (scenario == "late_import") {
        assert(last_exported != 0 && !exported_resources.at(last_exported).expired());
        assert(aclrtMemImportFromShareableHandle(last_exported, 1, &late_import) == ACL_SUCCESS);
        assert(physical_resources.at(late_import) == exported_resources.at(last_exported).lock());
    }
    const int abort_rc = comm_abort_domain_windows(h, id, 2, 0);
    assert(abort_rc == (fail_destroy ? COMM_RECLAMATION_UNCONFIRMED : 0));
    if (late_import != nullptr) {
        assert(physical_handles == 1 && imported_handles.count(late_import) == 1);
        assert(!exported_resources.at(last_exported).expired());
        assert(aclrtFreePhysical(late_import) == ACL_SUCCESS);
        assert(exported_resources.at(last_exported).expired());
    }
    if (peer_failure != PeerFailure::NONE) {
        assert(peer_import_calls == 1 && injected_peer_failures == 1);
        assert(imported_handles.empty() && peer_mappings.empty());
    }
    assert(registration_calls == 2);
    assert(channel_calls == (fail_registration == 2 ? 0 : 1));
    assert(unsafe_frees == 0);
    assert(destroy_calls == 1);
    if (fail_destroy) {
        assert(h->domain_allocations.count(id) == 1);
        assert(!mappings.empty() && !registrations.empty());
        const int calls = destroy_calls;
        const auto held = mappings.size();
        assert(comm_release_domain_windows(h, id, 2, 0) != 0);
        assert(destroy_calls == calls && mappings.size() == held && unsafe_frees == 0);
    } else {
        assert(h->domain_allocations.empty());
        assert(mappings.empty() && physical_handles == 0 && registrations.empty() && child_comms.empty());
        const int prior_registrations = registration_calls;
        const int prior_destroy_calls = destroy_calls;
        assert(comm_alloc_domain_windows(h, id, ranks, 2, 0, 4096, 2, &ctx, &base, &size) != 0);
        assert(registration_calls == prior_registrations && destroy_calls == prior_destroy_calls);
        assert(mappings.empty() && physical_handles == 0);
        delete h;
    }
    // Every local and imported window relies on MapMem's owner grant; an
    // explicit access grant fails this fixture.
    assert(explicit_access_calls == 0);
    std::filesystem::remove_all(folder);
}
extern "C" errno_t memcpy_s(void *dest, size_t capacity, const void *src, size_t count) {
    assert(count <= capacity);
    std::memcpy(dest, src, count);
    return 0;
}
extern "C" errno_t memset_s(void *dest, size_t capacity, int value, size_t count) {
    assert(count <= capacity);
    std::memset(dest, value, count);
    return 0;
}
extern "C" HcclResult HcclRankGraphGetLayers(HcclComm, uint32_t **, uint32_t *) { return HCCL_E_INTERNAL; }
extern "C" HcclResult
HcclRankGraphGetEndpointInfo(HcclComm, uint32_t, const EndpointDesc *, EndpointAttr, uint32_t, void *) {
    return HCCL_E_INTERNAL;
}
extern "C" HcclResult HcclChannelConfigCreate(HcclChannelConfig *) { return HCCL_E_INTERNAL; }
extern "C" HcclResult HcclChannelConfigDestroy(HcclChannelConfig) { return HCCL_E_INTERNAL; }
extern "C" HcclResult HcclChannelConfigSetInt(HcclChannelConfig, HcclChannelConfigType, uint32_t) {
    return HCCL_E_INTERNAL;
}
extern "C" HcclResult HcclChannelConfigSetStr(HcclChannelConfig, HcclChannelConfigType, const char *) {
    return HCCL_E_INTERNAL;
}
extern "C" HcclResult HcclChannelAcquireWithConfig(
    HcclComm, CommEngine, const HcclChannelDesc *, uint32_t, HcclChannelConfig, ChannelHandle *
) {
    return HCCL_E_INTERNAL;
}
