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
#include <future>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "comm_hccl.cpp"
#include "../../../cpp/a5/runtime/urma_queue_fixture.h"

// Fail allocations on the calling thread only. The worker keeps its normal allocator.
static thread_local int allocations_before_failure = -1;
extern "C" void *__real__Znwm(size_t);
extern "C" void *__wrap__Znwm(size_t size) {
    if (allocations_before_failure == 0) {
        allocations_before_failure = -1;
        throw std::bad_alloc();
    }
    if (allocations_before_failure > 0) --allocations_before_failure;
    return __real__Znwm(size);
}

namespace simpler::ccu {
// URMA owns no CCU resources; retained host teardown paths still reference these APIs.
bool HcommMissionStreamPool::close() {
    assert(entries_.empty());
    closed_ = true;
    return true;
}
bool HcommDomainChannels::close() {
    assert(communicator == nullptr);
    return true;
}
void create_hcomm_domain_channels(
    HcclComm, uint64_t, const std::vector<uint32_t> &, uint32_t, std::shared_ptr<HcommDomainChannels> &, CommCcuMode
) {
    assert(false && "URMA-only fixture must not create CCU channels");
}
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
    char sq_buffer[64]{}, cq_buffer[3 * 64]{};
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
const void *fail_queue_read = nullptr;
void *fail_queue_write = nullptr;
constexpr uint64_t aiv_alias = uint64_t{1} << 61;
int blocked_control_waits = 0;
int control_launches = 0;
int control_images = 0;
int control_syncs = 0;
int allowed_control_completions = -1;
const auto run_context = reinterpret_cast<aclrtContext>(0x501);
thread_local aclrtContext thread_context = nullptr;
thread_local bool inside_poll = false;
std::atomic<bool> refuse_scan_context{false};
std::atomic<unsigned> rejected_scan_contexts{0};
auto fixture_launch_gate = std::make_shared<UrmaLaunchGate>();
int control_queries = 0;

// Suspend the host copy while the polling thread keeps checking the same Run.
struct ScanProbe {
    std::mutex mutex;
    std::condition_variable changed;
    const void *header{};
    const void *late_cqe{};
    bool hold{};
    bool entered{};
    unsigned copies{};
    unsigned poll_copies{};
    std::thread::id scanner;
    std::atomic<uint32_t> word{0};
    std::atomic<unsigned> pending_reads{0};
} scan_probe;
struct ControlStream {
    std::optional<simpler::urma::QueueCommand> pending;
};
std::unordered_set<ControlStream *> control_streams;

struct FixtureAivMemory {
    static bool read(uint64_t address, uint32_t &word) {
        assert(address & aiv_alias);
        const auto *source = reinterpret_cast<const void *>(address ^ aiv_alias);
        if (source == scan_probe.late_cqe) {
            word = scan_probe.word.load();
            if (word == 0) ++scan_probe.pending_reads;
            return true;
        }
        if (source == fail_queue_read) return false;
        std::memcpy(&word, source, sizeof(word));
        return true;
    }
    static bool write(uint64_t address, uint32_t word) {
        assert(address & aiv_alias);
        auto *destination = reinterpret_cast<void *>(address ^ aiv_alias);
        if (destination == fail_queue_write) return false;
        std::memcpy(destination, &word, sizeof(word));
        return true;
    }
};
enum class PeerFailure { NONE, IMPORT, RESERVE, MAP };
PeerFailure peer_failure = PeerFailure::NONE;
bool peer_reserve_pending = false;
int peer_import_calls = 0;
int peer_enable_calls = 0;
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
extern "C" aclError aclrtGetDevice(int32_t *id) {
    if (fail_get_device) return 1;
    *id = 0;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtGetCurrentContext(aclrtContext *context) {
    *context = thread_context;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtSetCurrentContext(aclrtContext context) {
    if (refuse_scan_context) {
        ++rejected_scan_contexts;
        return ACL_ERROR_FAILURE;
    }
    thread_context = context;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMalloc(void **out, size_t bytes, aclrtMemMallocPolicy) { return map_buffer(out, bytes); }
extern "C" aclError aclrtFree(void *p) { return unmap_buffer(p); }
extern "C" aclError aclrtMemcpy(void *out, size_t, const void *in, size_t bytes, aclrtMemcpyKind kind) {
    assert(!((reinterpret_cast<uint64_t>(out) | reinterpret_cast<uint64_t>(in)) & aiv_alias));
    if (scan_probe.header && kind == ACL_MEMCPY_DEVICE_TO_HOST) {
        std::unique_lock lock(scan_probe.mutex);
        if (inside_poll) {
            ++scan_probe.poll_copies;
            return ACL_ERROR_FAILURE;
        }
        assert(thread_context == run_context);
        if (scan_probe.scanner == std::thread::id{}) scan_probe.scanner = std::this_thread::get_id();
        assert(scan_probe.scanner == std::this_thread::get_id());
        ++scan_probe.copies;
        if (in == scan_probe.header && scan_probe.hold) {
            scan_probe.entered = true;
            scan_probe.changed.notify_all();
            assert(scan_probe.changed.wait_for(lock, std::chrono::seconds(4), [] {
                return !scan_probe.hold;
            }));
        }
    }
    if (fail_context_upload && bytes == sizeof(CommContext)) return 1;
    if (fail_queue_read != nullptr && kind == ACL_MEMCPY_DEVICE_TO_HOST && in == fail_queue_read) return 1;
    if (fail_queue_write != nullptr && kind == ACL_MEMCPY_HOST_TO_DEVICE && out == fail_queue_write) return 1;
    std::memcpy(out, in, bytes);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtBinaryLoadFromFile(const char *path, aclrtBinaryLoadOptions *, aclrtBinHandle *binary) {
    assert(std::filesystem::path(path).filename() == "ccu_event_trigger.o");
    ++control_images;
    *binary = reinterpret_cast<aclrtBinHandle>(1);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtBinaryGetFunction(aclrtBinHandle, const char *name, aclrtFuncHandle *function) {
    assert(std::strcmp(name, "simpler_urma_queue_control") == 0);
    *function = reinterpret_cast<aclrtFuncHandle>(1);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtBinaryUnLoad(aclrtBinHandle) {
    assert(control_images > 0);
    --control_images;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtCreateStream(aclrtStream *stream) {
    auto *state = new ControlStream;
    control_streams.insert(state);
    *stream = reinterpret_cast<aclrtStream>(state);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtDestroyStream(aclrtStream stream) {
    auto *state = reinterpret_cast<ControlStream *>(stream);
    assert(!state->pending && control_streams.erase(state));
    delete state;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle, uint32_t blocks, aclrtStream stream, aclrtLaunchKernelCfg *config, void *args, size_t bytes,
    aclrtPlaceHolderInfo *, size_t
) {
    auto *state = reinterpret_cast<ControlStream *>(stream);
    assert(control_streams.count(state) && !state->pending && blocks == 1);
    assert(config->attrs[0].value.engineType == ACL_RT_ENGINE_TYPE_AIV);
    assert(bytes == sizeof(simpler::urma::QueueCommand));
    state->pending = *static_cast<const simpler::urma::QueueCommand *>(args);
    ++control_launches;
    return ACL_SUCCESS;
}
static void finish_control_command(ControlStream &stream) {
    if (!stream.pending) return;
    const auto &command = *stream.pending;
    *reinterpret_cast<uint32_t *>(command.result_address) =
        static_cast<uint32_t>(simpler::urma::execute_queue_command(FixtureAivMemory{}, command));
    stream.pending.reset();
}
extern "C" aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t) {
    auto *state = reinterpret_cast<ControlStream *>(stream);
    if (!control_streams.count(state)) return ACL_SUCCESS;
    ++control_syncs;
    if (blocked_control_waits > 0) {
        --blocked_control_waits;
        return 1;
    }
    finish_control_command(*state);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtStreamQuery(aclrtStream stream, aclrtStreamStatus *status) {
    auto *state = reinterpret_cast<ControlStream *>(stream);
    assert(control_streams.count(state));
    ++control_queries;
    if (state->pending && allowed_control_completions == 0) {
        *status = ACL_STREAM_STATUS_NOT_READY;
        return ACL_SUCCESS;
    }
    if (state->pending && allowed_control_completions > 0) --allowed_control_completions;
    finish_control_command(*state);
    *status = ACL_STREAM_STATUS_COMPLETE;
    return ACL_SUCCESS;
}

static int acquire_test_run(CommHandle handle, const uint64_t *scalars, int32_t count, UrmaRunLease *use) {
    const int rc = acquire_hbg_urma_run(handle, scalars, count, use);
    if (rc == 0 && use->stopped_scan) {
        fixture_launch_gate->track(use->stopped_scan->thread);
        use->launch_permission = fixture_launch_gate;
    }
    return rc;
}

static bool drained(UrmaRunLease &lease, bool device_usable = true) {
    auto saved = lease.launch_permission;
    if (!device_usable) lease.launch_permission.reset();
    const auto status = lease.check_stopped_queues(UrmaWait::Wait);
    lease.launch_permission = saved;
    return status == UrmaQueueState::READY;
}

static UrmaQueueState poll(UrmaRunLease &lease) {
    inside_poll = true;
    const auto result = lease.check_stopped_queues(UrmaWait::Poll);
    inside_poll = false;
    return result;
}

// Fetch one worker scan submitted in Poll mode before the test edits its queue.
static UrmaQueueState poll_scan(UrmaRunLease &lease) {
    assert(poll(lease) == UrmaQueueState::PENDING);
    return lease.check_stopped_queues(UrmaWait::Wait);
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
extern "C" aclError aclrtDeviceEnablePeerAccess(int32_t, uint32_t) {
    ++peer_enable_calls;
    return ACL_SUCCESS;
}
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
    cq.scqVa = reinterpret_cast<uint64_t>(c->cq_buffer) | aiv_alias;
    cq.headAddr = reinterpret_cast<uint64_t>(&c->cq_head);
    cq.tailAddr = reinterpret_cast<uint64_t>(&c->cq_tail);
    cq.dbVa = reinterpret_cast<uint64_t>(&c->cq_db) | aiv_alias;
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
        assert(domain_write_announce(h->rootinfo_path, id, peer, h->run_token, getpid(), ranks[peer], 0));
        for (const char *phase : {"p2p_ready", "release", "release_consumed"}) {
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
        assert(ctx.windowsIn[peer] == 0 && alloc.peer_windows.empty());
        assert(peer_import_calls == 0 && peer_enable_calls == 0 && last_exported == 0);
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
    if constexpr (hbg_urma) {
        const auto &first = *h->domain_allocations.at(1);
        const auto &second = *h->domain_allocations.at(3);
        const uint64_t both[] = {
            reinterpret_cast<uint64_t>(first.device_ctx), reinterpret_cast<uint64_t>(second.device_ctx),
            reinterpret_cast<uint64_t>(first.device_ctx)
        };
        UrmaRunLease held, next;
        assert(acquire_test_run(h, both + 1, 1, &held) == 0 && held.domains.size() == 1);
        assert(acquire_test_run(h, both, 3, &next) == COMM_DOMAIN_ADMISSION_STOPPED && next.domains.empty());
        assert(acquire_test_run(h, both, 1, &next) == 0 && next.domains.size() == 1);
        next = {};
        held = {};
        assert(acquire_test_run(h, both, 3, &next) == 0 && next.domains.size() == 2);
        assert(next.domains[0].binding.workspace == reinterpret_cast<uint64_t>(first.urma_workspace.addr));
        assert(next.domains[1].binding.workspace == reinterpret_cast<uint64_t>(second.urma_workspace.addr));
        assert(next.domains[0].binding.local_rank == 1 && next.domains[1].binding.local_rank == bc_rank);
        assert(next.domains[0].binding.row_owner != next.domains[1].binding.row_owner && drained(next));
        auto &channel = *channels.at(second.urma_comm);
        ++channel.sq_head;
        assert(!drained(next) && comm_release_domain_windows(h, 3, 2, bc_rank) == COMM_DOMAIN_ADMISSION_STOPPED);
        ++channel.sq_tail;
        assert(drained(next));
        next.retire_queues();
        assert(acquire_test_run(h, both + 1, 1, &held) == 0);
        ++channel.sq_head;
        assert(drained(next) && !drained(held));  // The successor owns these submissions.
        ++channel.sq_tail;
        assert(drained(held));
        held = {};
        assert(comm_release_domain_windows(h, 3, 2, bc_rank) == COMM_DOMAIN_ADMISSION_STOPPED);
    }
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

static void test_background_poll(const std::string &scenario) {
    const bool abandon = scenario == "abandon_poll";
    const bool foreground_only = scenario == "foreground_d2h";
    const bool revoke_during_scan = scenario == "revoke_poll";
    using State = UrmaQueueState;
    using namespace std::chrono_literals;
    simpler::test::UrmaQueueFixture queues;
    queues.info.rankCount = 4;
    queues.info.rowsPerPeer = 2;
    constexpr unsigned last = simpler::test::kRows - 1;
    for (auto &cq : queues.cq) {
        cq.bufAddr |= aiv_alias;
        cq.dbAddr |= aiv_alias;
    }
    queues.sq_head[last] = queues.sq[last].submittedWqeCount = 1;
    CommContext device_ctx{};
    CommHandle_ h{};
    auto allocation = std::make_shared<DomainAllocation>();
    h.domain_allocations.emplace(1, allocation);
    allocation->nranks = queues.info.rankCount;
    allocation->initialized = true;
    allocation->device_ctx = &device_ctx;
    allocation->urma_row_owner = 1;
    allocation->urma_aiv_count = 1;
    allocation->workspace_config.urma_jetties_per_aiv = 2;
    allocation->urma_workspace.addr = &queues.info;
    allocation->urma_workspace.bytes = sizeof(queues);
    allocation->urma_workspace.impl = reinterpret_cast<decltype(allocation->urma_workspace.impl)>(&queues);
    const uint64_t context = reinterpret_cast<uint64_t>(&device_ctx);
    UrmaRunLease use, other;
    if (!foreground_only) {
        thread_context = nullptr;
        assert(acquire_test_run(&h, &context, 1, &use) == COMM_DOMAIN_URMA_CONFIG_INVALID);
        assert(!allocation->urma_run_active && use.domains.empty());
    }
    thread_context = run_context;
    assert(acquire_test_run(&h, &context, 1, &use) == 0);
    if (scenario == "reset_active_poll" || scenario == "reset_idle_poll" || scenario == "reset_unstarted_poll") {
        const bool active = scenario == "reset_active_poll";
        auto scan = use.stopped_scan->thread;
        scan_probe.header = &queues.info;
        if (active) {
            scan_probe.hold = true;
            assert(poll(use) == State::PENDING);
            std::unique_lock lock(scan_probe.mutex);
            assert(scan_probe.changed.wait_for(lock, 4s, [] {
                return scan_probe.entered;
            }));
        } else if (scenario == "reset_idle_poll") {
            queues.sq_head[last] = queues.sq[last].submittedWqeCount = 0;
            assert(use.check_stopped_queues(UrmaWait::Wait) == State::READY);
            assert(scan->worker.joinable());
        } else {
            assert(!scan->worker.joinable());
        }
        fixture_launch_gate->revoke();
        auto resetting = std::async(std::launch::async, [] {
            fixture_launch_gate->stop_scans();
        });
        if (active) {
            {
                std::unique_lock lock(scan->mutex);
                assert(scan->changed.wait_for(lock, 4s, [&] {
                    return scan->stopping;
                }));
            }
            assert(resetting.wait_for(10ms) == std::future_status::timeout);
            std::lock_guard lock(scan_probe.mutex);
            scan_probe.hold = false;
            scan_probe.changed.notify_all();
        }
        resetting.get();
        assert(!scan->worker.joinable() && !control_launches);
        const auto copies_before = scan_probe.copies;
        fixture_launch_gate = std::make_shared<UrmaLaunchGate>();
        assert(use.launch_permission.expired());
        for (int retry = 0; retry < 3; ++retry)
            assert(poll(use) == State::PENDING);
        assert(scan_probe.copies == copies_before && !control_queries && !control_launches);
        return;
    }
    if (foreground_only) {
        queues.sq_head[last] = queues.sq[last].submittedWqeCount = 0;
        scan_probe.header = &queues.info;
        const auto first = poll(use);
        std::fprintf(stderr, "foreground D2H copies in the first poll: %u\n", scan_probe.poll_copies);
        assert(scan_probe.poll_copies == 0);
        assert(first == State::PENDING);
        assert(use.check_stopped_queues(UrmaWait::Wait) == State::READY);
        assert(scan_probe.copies == 1 + 5 * 6);
        return;
    }
    thread_context = reinterpret_cast<aclrtContext>(0x502);
    scan_probe.header = &queues.info;
    scan_probe.late_cqe = queues.entries[last][0].words.data();
    auto poll_until = [&](State expected) {
        const auto deadline = std::chrono::steady_clock::now() + 4s;
        State observed;
        do {
            assert(std::chrono::steady_clock::now() < deadline);
            observed = poll(use);
            if (observed == State::PENDING) std::this_thread::yield();
        } while (observed == State::PENDING);
        assert(observed == expected);
    };
    // First allocation copies the queue claims. When that succeeds, the next allocates the thread's state.
    for (int fail_after : {0, 1}) {
        allocations_before_failure = fail_after;
        assert(poll(use) == State::PENDING && allocations_before_failure == -1);
        assert(!scan_probe.copies && !control_launches && allocation->urma_run_active);
    }
    refuse_scan_context = true;
    assert(poll_scan(use) == State::PENDING);
    assert(rejected_scan_contexts == 1 && !scan_probe.copies && allocation->urma_run_active);
    refuse_scan_context = false;
    auto hold_scan = [&] {
        {
            std::lock_guard lock(scan_probe.mutex);
            scan_probe.hold = true;
            scan_probe.entered = false;
        }
        assert(poll(use) == State::PENDING);
        std::unique_lock lock(scan_probe.mutex);
        assert(scan_probe.changed.wait_for(lock, 4s, [] {
            return scan_probe.entered;
        }));
    };
    auto release_scan = [] {
        std::lock_guard lock(scan_probe.mutex);
        scan_probe.hold = false;
        scan_probe.changed.notify_all();
    };
    if (abandon) allowed_control_completions = 0;
    hold_scan();
    std::future<State> waiter;
    if (abandon) {
        waiter = std::async(std::launch::async, [&] {
            return use.check_stopped_queues(UrmaWait::Wait);
        });
        assert(waiter.wait_for(10ms) == std::future_status::timeout);
    }
    for (int retry = 0; retry < 32; ++retry)
        assert(poll(use) == State::PENDING);
    assert(acquire_test_run(&h, &context, 1, &other) == COMM_DOMAIN_ADMISSION_STOPPED);
    assert(!scan_probe.poll_copies && scan_probe.scanner != std::this_thread::get_id());
    if (revoke_during_scan) fixture_launch_gate->revoke();
    release_scan();
    if (revoke_during_scan) {
        assert(use.check_stopped_queues(UrmaWait::Wait) == State::PENDING);
        assert(!control_launches && !queues.cq_tail[last] && !queues.sq_tail[last]);
        // An already drained row needs no new helper, even after quarantine.
        queues.cq_tail[last] = queues.sq_tail[last] = queues.cq_doorbell[last] = 1;
        poll_until(State::READY);
        assert(!control_launches);
        use.retire_queues();
        return;
    }
    if (abandon) {
        // The result belongs to the poll that launched the helper. Its AIV is still active.
        assert(waiter.get() == State::PENDING);
        assert(allocation->urma_queue_control.has_command() && control_launches == 1 && control_syncs == 0);
        const int prior_queries = control_queries;
        fixture_launch_gate->revoke();
        assert(poll_scan(use) == State::PENDING);
        assert(control_queries == prior_queries + 1 && control_launches == 1);
        hold_scan();
        auto destroying = std::async(std::launch::async, [&] {
            use = {};
        });
        assert(destroying.wait_for(10ms) == std::future_status::timeout);
        assert(acquire_test_run(&h, &context, 1, &other) == COMM_DOMAIN_ADMISSION_STOPPED);
        release_scan();
        destroying.get();
        assert(acquire_test_run(&h, &context, 1, &other) == COMM_DOMAIN_ADMISSION_STOPPED);
        assert(!domain_drained(*allocation) && !allocation->urma_queue_control.close());
        assert(control_images == 1 && control_streams.size() == 1 && !queues.sq_tail[last]);
        return;
    }

    const auto deadline = std::chrono::steady_clock::now() + 4s;
    while (!scan_probe.pending_reads) {
        assert(std::chrono::steady_clock::now() < deadline && poll(use) == State::PENDING);
        std::this_thread::yield();
    }
    scan_probe.word = (1u << 2) | (1u << 16);
    poll_until(State::FAILED);
    assert(!queues.cq_tail[last] && !queues.cq_doorbell[last] && !queues.sq_tail[last]);
    scan_probe.word = 1u << 2;
    poll_until(State::READY);
    assert(queues.cq_tail[last] == 1 && queues.cq_doorbell[last] == 1 && queues.sq_tail[last] == 1);
    assert(scan_probe.copies >= 1 + 5 * 6 && scan_probe.poll_copies == 0 && control_syncs == 0);
    assert(thread_context == reinterpret_cast<aclrtContext>(0x502));
    assert(allocation->urma_run_active);
    use.retire_queues();
    assert(!allocation->urma_run_active);
    use = {};
    assert(allocation->urma_queue_control.close() && mappings.empty() && control_streams.empty());
}

int main(int argc, char **argv) {
    assert(argc == 2);
    const std::string scenario = argv[1];
    thread_context = run_context;
    if (scenario == "wide_poll" || scenario == "abandon_poll" || scenario == "foreground_d2h" ||
        scenario == "revoke_poll" || scenario == "reset_active_poll" || scenario == "reset_idle_poll" ||
        scenario == "reset_unstarted_poll") {
        test_background_poll(scenario);
        return 0;
    }
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
    const bool mapped = peer_failure != PeerFailure::NONE || scenario == "late_import" || scenario == "mapped_success";
    assert(domain_write_announce(h->rootinfo_path, id, 1, h->run_token, getpid(), 1, mapped ? 42 : 0));
    for (const char *phase : {"p2p_ready", "release", "release_consumed"}) {
        if (scenario == "abort_unpublished" && std::strcmp(phase, "release") == 0) continue;
        std::ofstream marker(barrier_marker_path(h->rootinfo_path, h->run_token, domain_barrier_tag(id, phase), 1));
        marker << "1";
    }
    const uint32_t ranks[] = {0, 1};
    uint64_t ctx = 0, base = 0;
    size_t size = 0;
    const uint32_t urma = comm_workspace_bit(CommWorkspaceKind::URMA);
    if (scenario == "mixed_engines") {
        for (auto other : {CommWorkspaceKind::SDMA, CommWorkspaceKind::RDMA, CommWorkspaceKind::CCU}) {
            const auto mask = urma | comm_workspace_bit(other);
            assert(
                comm_alloc_domain_windows(h, id, ranks, 2, 0, 4096, mask, &ctx, &base, &size) ==
                COMM_DOMAIN_WORKSPACE_UNSUPPORTED
            );
        }
        assert(!h->last_allocation_id && h->domain_allocations.empty() && mappings.empty());
        delete h;
        std::filesystem::remove_all(folder);
        return 0;
    }
    const int rc = comm_alloc_domain_windows(h, id, ranks, 2, 0, 4096, mapped ? 0 : urma, &ctx, &base, &size);
    if (mapped) {
        assert(peer_import_calls == 1 && peer_enable_calls == 1);
        if (scenario == "mapped_success") {
            assert(rc == 0 && ctx != 0);
            const auto &windows = reinterpret_cast<CommContext *>(ctx)->windowsIn;
            assert(windows[0] == base && peer_mappings.count(reinterpret_cast<void *>(windows[1])) == 1);
            assert(comm_release_domain_windows(h, id, 2, 0) == 0);
        } else {
            assert(rc == COMM_INITIALIZATION_ABORT_REQUIRED && ctx == 0);
            aclrtDrvMemHandle late_import = nullptr;
            if (scenario == "late_import") {
                assert(last_exported != 0 && !exported_resources.at(last_exported).expired());
                assert(aclrtMemImportFromShareableHandle(last_exported, 1, &late_import) == ACL_SUCCESS);
                assert(physical_resources.at(late_import) == exported_resources.at(last_exported).lock());
            } else {
                assert(injected_peer_failures == 1);
            }
            assert(comm_abort_domain_windows(h, id, 2, 0) == 0);
            if (late_import != nullptr) {
                assert(physical_handles == 1 && !exported_resources.at(last_exported).expired());
                assert(aclrtFreePhysical(late_import) == ACL_SUCCESS);
                assert(exported_resources.at(last_exported).expired());
            }
        }
        assert(imported_handles.empty() && peer_mappings.empty() && mappings.empty() && physical_handles == 0);
        delete h;
        std::filesystem::remove_all(folder);
        return 0;
    }
    assert(peer_import_calls == 0 && peer_enable_calls == 0 && last_exported == 0);
    if (scenario == "abort_unpublished") {
        assert(rc == 0 && ctx != 0);
        assert(comm_abort_domain_windows(h, id, 2, 0) == 0);
        assert(h->domain_allocations.empty() && registrations.empty() && mappings.empty());
        assert(destroy_calls == 1 && unsafe_frees == 0);
        delete h;
        std::filesystem::remove_all(folder);
        return 0;
    }
    if (scenario == "hbg_lease") {
        assert(rc == 0 && ctx);
        UrmaRunLease use, other;
        const uint64_t foreign = ctx + 1;
        assert(acquire_test_run(h, &foreign, 1, &other) == 0 && other.domains.empty());
        assert(acquire_test_run(h, &ctx, 1, &use) == 0 && use.domains.size() == 1 && drained(use));
        const auto &binding = use.domains.front().binding;
        assert(binding.workspace && binding.row_owner && binding.rows_per_peer == 1 && !binding.local_rank);
        assert(binding.rank_count == 2 && binding.jetties_per_core == 1);
        const auto rows = binding.rows_per_peer * binding.rank_count;
        const auto *owners = reinterpret_cast<const UrmaRowOwner *>(binding.row_owner);
        assert(binding.row_owner % alignof(UrmaRowOwner) == 0);
        const auto end = reinterpret_cast<uint64_t>(owners + rows) - binding.workspace;
        const auto bytes = mappings.at(reinterpret_cast<void *>(binding.workspace));
        assert(end <= bytes && bytes - end < alignof(UrmaRowOwner));
        for (uint32_t r = 0; r < rows; ++r)
            assert(owners[r].value == 0);
        assert(acquire_test_run(h, &ctx, 1, &other) == COMM_DOMAIN_ADMISSION_STOPPED);
        assert(comm_release_domain_windows(h, id, 2, 0) == COMM_DOMAIN_ADMISSION_STOPPED);
        auto &channel = *channels.begin()->second;
        channel.sq_head = 3;
        assert(!drained(use) && comm_destroy(h) == COMM_DOMAIN_ADMISSION_STOPPED);
        assert(!h->cleanup_failed && !unsafe_frees && !destroy_calls);
        using namespace pto::comm::urma;
        const auto *info = reinterpret_cast<const UrmaInfo *>(binding.workspace);
        auto &sq = reinterpret_cast<UrmaWQCtx *>(info->sqPtr)[binding.rows_per_peer];
        sq.submittedWqeCount = 2;
        const uint32_t ready = 1u << 2, bad_status = ready | (1u << 16);
        std::memcpy(channel.cq_buffer, &ready, sizeof(ready));
        auto *second_cqe = channel.cq_buffer + 64;
        assert(poll_scan(use) == UrmaQueueState::PENDING);  // Only one CQE arrived.
        std::memcpy(second_cqe, &bad_status, sizeof(bad_status));
        assert(poll_scan(use) == UrmaQueueState::FAILED);
        std::memcpy(second_cqe, &ready, sizeof(ready));
        assert(!domain_drained(*h->domain_allocations.at(id)));  // Domain close only observes the tails.
        const int launches = control_launches;
        assert(!drained(use, false) && control_launches == launches);
        blocked_control_waits = 2;
        assert(!drained(use) && control_launches == launches + 1);
        assert(!drained(use) && control_launches == launches + 1);
        assert(comm_release_domain_windows(h, id, 2, 0) == COMM_DOMAIN_ADMISSION_STOPPED);
        assert(acquire_test_run(h, &ctx, 1, &other) == COMM_DOMAIN_ADMISSION_STOPPED);
        assert(control_images == 1 && control_streams.size() == 1);
        const auto &pending = (*control_streams.begin())->pending;
        assert(pending && mappings.count(reinterpret_cast<void *>(pending->result_address)) == 1);
        assert(channel.cq_tail == 0 && channel.cq_db == 0 && channel.sq_tail == 0);
        fail_queue_read = second_cqe;
        assert(!drained(use));
        assert(channel.cq_tail == 0 && channel.cq_db == 0 && channel.sq_tail == 0);
        fail_queue_read = nullptr;
        fail_queue_write = &channel.cq_db;
        assert(!drained(use));
        assert(channel.cq_tail == 2 && channel.cq_db == 0 && channel.sq_tail == 0);
        // The retained, validated position survives a partial publication.
        fail_queue_read = second_cqe;
        fail_queue_write = &channel.sq_tail;
        assert(!drained(use));
        assert(channel.cq_tail == 2 && channel.cq_db == 2 && channel.sq_tail == 0);
        fail_queue_write = nullptr;
        assert(drained(use));
        assert(channel.cq_tail == 2 && channel.cq_db == 2 && channel.sq_tail == 3);
        // A poller alone delivers the next receipt. Hold the AIV first during its scan, then during the doorbell.
        channel.sq_head = 4;
        sq.submittedWqeCount = 3;
        std::memcpy(channel.cq_buffer + 128, &ready, sizeof(ready));
        const int start_syncs = control_syncs, start_launches = control_launches;
        auto poll_once = [&] {
            return poll_scan(use);
        };
        allowed_control_completions = 0;
        assert(poll_once() == UrmaQueueState::PENDING && poll_once() == UrmaQueueState::PENDING);
        assert(control_launches == start_launches + 1 && channel.cq_tail == 2 && channel.sq_tail == 3);
        allowed_control_completions = 1;
        assert(poll_once() == UrmaQueueState::PENDING);
        assert(poll_once() == UrmaQueueState::PENDING && control_launches == start_launches + 2);
        assert(channel.cq_tail == 3 && channel.cq_db == 2 && channel.sq_tail == 3);
        // Quarantine still permits collecting this already submitted doorbell and publishing SQ tail.
        fixture_launch_gate->revoke();
        allowed_control_completions = 1;
        assert(poll_once() == UrmaQueueState::READY && poll_once() == UrmaQueueState::READY);
        assert(control_launches == start_launches + 2 && control_syncs == start_syncs);
        assert(channel.cq_tail == 3 && channel.cq_db == 3 && channel.sq_tail == 4);

        assert(acquire_test_run(h, &ctx, 1, &other) == COMM_DOMAIN_ADMISSION_STOPPED);
        fail_queue_read = nullptr;
        use = {};
        assert(comm_release_domain_windows(h, id, 2, 0) == 0);
        assert(control_images == 0 && control_streams.empty());
        assert(child_comms.empty() && registrations.empty() && mappings.empty() && unsafe_frees == 0);
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
            assert(domain_write_announce(h->rootinfo_path, second_id, 1, h->run_token, getpid(), 1, 0));
            for (const char *phase : {"p2p_ready", "release", "release_consumed"}) {
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
    const int abort_rc = comm_abort_domain_windows(h, id, 2, 0);
    assert(abort_rc == (fail_destroy ? COMM_RECLAMATION_UNCONFIRMED : 0));
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
