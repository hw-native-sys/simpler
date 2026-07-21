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
/**
 * HCCL backend for the comm_* distributed communication API.
 *
 * Implements the five functions declared in host/comm.h using Ascend
 * HCCL (bundled with CANN) for the bootstrap / barrier / teardown plane
 * and the public ACL VMM shareable-handle API (aclrtMallocPhysical +
 * MemExportToShareableHandle / MemImportFromShareableHandle +
 * EnablePeerAccess) for the per-rank symmetric window pool (Path D).
 *
 * Scope: L3 single-host multi-card only. The VMM shareable-handle
 * exchange is host-local, so cross-host (L4) deployments need a different
 * windows backend -- see .docs/28.l3-comm/ext.01.pr-774-review.md F2 /
 * 05.plan.zh.md for the channel-API direction.
 */

#include "platform_comm/comm.h"
#include "platform_comm/comm_context.h"

#include "common/unified_log.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"
#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
#include "pto/comm/async/sdma/sdma_workspace_manager.hpp"
#endif

// Thin wrappers around the HCCL public APIs we use. Kept as a translation
// layer in case we need to swap (e.g., InitConfig variant) later.
static inline HcclResult hccl_get_root_info(HcclRootInfo *ri) { return HcclGetRootInfo(ri); }
static inline HcclResult hccl_comm_init_root_info(uint32_t n, const HcclRootInfo *ri, uint32_t r, HcclComm *c) {
    return HcclCommInitRootInfo(n, ri, r, c);
}
static inline HcclResult hccl_barrier(HcclComm c, aclrtStream s) { return HcclBarrier(c, s); }
static inline HcclResult hccl_comm_destroy(HcclComm c) { return HcclCommDestroy(c); }

// ============================================================================
// Internal state
// ============================================================================

// Per-domain dynamic allocation.  One of these per orch.allocate_domain call.
// Tracks the local VMM window (mapped VA + its physical handle + the
// granularity-aligned byte size) and every peer VMM import (mapped VA +
// imported physical handle), all torn down in comm_release_domain_windows,
// plus the device CommContext we materialise for the subset.  Only the
// EnablePeerAccess routes are left to aclrtResetDevice at finalize: they are
// process-global and per device-pair, so releasing them per allocation would
// tear down a route a concurrent allocation still uses.
struct DomainAllocation {
    int rank = 0;                            // this rank's index within the subset (domain_rank)
    int nranks = 0;                          // subset size
    void *local_buf = nullptr;               // VMM-mapped device VA
    uint64_t alloc_size = 0;                 // granularity-aligned byte size
    aclrtDrvMemHandle own_handle = nullptr;  // physical handle backing local_buf
    // Per-peer imports: (mapped VA, imported physical handle).  allocate_domain
    // can cycle repeatedly within one comm handle before any device reset, so
    // these are released explicitly at domain teardown rather than left to reset.
    std::vector<std::pair<void *, aclrtDrvMemHandle>> peer_windows;
    CommContext *device_ctx = nullptr;  // aclrtMalloc'd CommContext mirror
};

struct CommHandle_ {
    int rank;
    int nranks;
    std::string rootinfo_path;
    uint64_t run_token = 0;

    // Caller-owned: supplied to comm_init, never created or destroyed here.
    aclrtStream stream = nullptr;
    HcclComm hccl_comm = nullptr;

    CommContext host_ctx{};
    CommContext *device_ctx = nullptr;
    bool owns_device_ctx = false;
    std::vector<CommContext *> derived_contexts;
    std::unordered_map<uint64_t, std::unique_ptr<DomainAllocation>> domain_allocations;
#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
    std::unique_ptr<pto::comm::sdma::SdmaWorkspaceManager> sdma_workspace;
#endif
};

// ============================================================================
// Helpers
// ============================================================================

namespace {

static constexpr uint64_t ROOTINFO_MAGIC = 0x50544f5f4843434cULL;  // "PTO_HCCL"

struct RootInfoFileHeader {
    uint64_t magic = ROOTINFO_MAGIC;
    uint64_t run_token = 0;
    uint32_t payload_size = HCCL_ROOT_INFO_BYTES;
    uint32_t reserved = 0;
};

static std::string handshake_dir(const std::string &rootinfo_path) {
    auto last_slash = rootinfo_path.rfind('/');
    if (last_slash == std::string::npos) return ".";
    return rootinfo_path.substr(0, last_slash);
}

static std::string handshake_prefix(const std::string &rootinfo_path) {
    auto last_slash = rootinfo_path.rfind('/');
    return last_slash == std::string::npos ? rootinfo_path : rootinfo_path.substr(last_slash + 1);
}

static std::string run_token_hex(uint64_t run_token) {
    std::ostringstream oss;
    oss << std::hex << run_token;
    return oss.str();
}

static uint64_t make_run_token(int rank) {
    // steady_clock is monotonic and unaffected by NTP or wall-clock jumps;
    // we only need within-host uniqueness for the handshake file naming.
    auto now = std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now())
                   .time_since_epoch()
                   .count();
    uint64_t token = static_cast<uint64_t>(now);
    token ^= static_cast<uint64_t>(getpid()) << 16;
    token ^= static_cast<uint64_t>(rank & 0xFFFF);
    return token;
}

static std::string
barrier_marker_path(const std::string &rootinfo_path, uint64_t run_token, const std::string &tag, int rank) {
    return handshake_dir(rootinfo_path) + "/barrier_" + handshake_prefix(rootinfo_path) + "_" + tag + "_" +
           run_token_hex(run_token) + "_" + std::to_string(rank) + ".ready";
}

static void cleanup_handshake_files(const std::string &rootinfo_path) {
    std::error_code ec;
    std::filesystem::remove(rootinfo_path, ec);

    const std::string prefix = "barrier_" + handshake_prefix(rootinfo_path) + "_";
    const std::string dir = handshake_dir(rootinfo_path);
    for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) != 0) continue;
        if (name.size() < 6 || name.substr(name.size() - 6) != ".ready") continue;
        std::filesystem::remove(entry.path(), ec);
        ec.clear();
    }
}

static bool
wait_for_rootinfo(const std::string &path, HcclRootInfo *root_info, uint64_t *run_token, int timeout_sec = 120) {
    constexpr int kLogEverySec = 5;
    for (int i = 0; i < timeout_sec * 10; ++i) {
        std::ifstream f(path, std::ios::binary);
        if (f.good()) {
            RootInfoFileHeader header{};
            f.read(reinterpret_cast<char *>(&header), sizeof(header));
            if (!f.good()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            if (header.magic != ROOTINFO_MAGIC || header.payload_size != HCCL_ROOT_INFO_BYTES) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            f.read(root_info->internal, HCCL_ROOT_INFO_BYTES);
            if (!f.good()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            *run_token = header.run_token;
            return true;
        }
        if (i > 0 && i % (kLogEverySec * 10) == 0) {
            LOG_INFO_V0("[comm] wait_for_rootinfo: still waiting (%ds elapsed) path=%s", i / 10, path.c_str());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

static bool file_barrier(
    const std::string &rootinfo_path, int rank, int nranks, const std::string &tag, uint64_t run_token,
    int timeout_sec = 120
) {
    std::string my_marker = barrier_marker_path(rootinfo_path, run_token, tag, rank);
    { std::ofstream(my_marker) << "1"; }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    for (int r = 0; r < nranks; ++r) {
        std::string marker = barrier_marker_path(rootinfo_path, run_token, tag, r);
        while (true) {
            std::ifstream f(marker);
            if (f.good()) break;
            if (std::chrono::steady_clock::now() >= deadline) {
                LOG_ERROR(
                    "[comm rank %d] file_barrier('%s') timed out after %ds waiting for rank %d", rank, tag.c_str(),
                    timeout_sec, r
                );
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    return true;
}

// Release one VMM window — either an own-rank allocation or a peer import; the
// teardown sequence is identical.  `va` must be the mapped address returned by
// aclrtMapMem and `handle` the backing physical handle (from aclrtMallocPhysical
// for an own window, or aclrtMemImportFromShareableHandle for a peer import);
// either may be nullptr before its respective step completed.  Base-comm peer
// imports (alloc_windows_via_ipc) are not tracked and remain reclaimed by
// aclrtResetDevice; per-domain peer imports are tracked and freed via this call.
static void release_own_vmm_window(void *va, aclrtDrvMemHandle handle) {
    if (va != nullptr) {
        aclrtUnmapMem(va);
        aclrtReleaseMemAddress(va);
    }
    if (handle != nullptr) {
        aclrtFreePhysical(handle);
    }
}

// Release every per-peer VMM import recorded on a domain allocation and clear
// the list, so a re-release is a no-op.  Own window and device_ctx are freed
// separately by the caller.
static void release_domain_peer_windows(DomainAllocation &alloc) {
    for (auto &pw : alloc.peer_windows) {
        release_own_vmm_window(pw.first, pw.second);
    }
    alloc.peer_windows.clear();
}

}  // namespace

// ============================================================================
// API implementation
// ============================================================================

extern "C" CommHandle comm_init(int rank, int nranks, void *stream, const char *rootinfo_path) try {
    if (stream == nullptr) {
        LOG_ERROR("[comm rank %d] comm_init: caller-supplied stream is null", rank);
        return nullptr;
    }
    if (rootinfo_path == nullptr || *rootinfo_path == '\0') {
        LOG_ERROR("[comm rank %d] comm_init: rootinfo_path is null or empty", rank);
        return nullptr;
    }
    if (nranks <= 0 || rank < 0 || rank >= nranks) {
        LOG_ERROR("[comm rank %d] comm_init: invalid rank/nranks (rank=%d, nranks=%d)", rank, rank, nranks);
        return nullptr;
    }
    if (static_cast<uint32_t>(nranks) > COMM_MAX_RANK_NUM) {
        LOG_ERROR("[comm rank %d] comm_init: nranks=%d exceeds COMM_MAX_RANK_NUM=%u", rank, nranks, COMM_MAX_RANK_NUM);
        return nullptr;
    }

    auto *h = new (std::nothrow) CommHandle_{};
    if (!h) return nullptr;

    h->rank = rank;
    h->nranks = nranks;
    h->rootinfo_path = rootinfo_path;
    h->stream = static_cast<aclrtStream>(stream);

    // NOTE: aclInit / aclrtSetDevice / stream creation are intentionally NOT
    // performed here — the caller (DeviceRunner::ensure_acl_ready + a stream
    // it owns) is responsible for them.  This keeps ACL lifecycle ownership
    // in one place (DeviceRunner) and matches HCCL's API shape, which already
    // takes a caller-supplied stream.

    // RootInfo exchange
    HcclRootInfo rootInfo{};
    if (rank == 0) {
        cleanup_handshake_files(h->rootinfo_path);
        h->run_token = make_run_token(rank);
        HcclResult hret = hccl_get_root_info(&rootInfo);
        if (hret != HCCL_SUCCESS) {
            LOG_ERROR("[comm rank 0] HcclGetRootInfo failed: %d", (int)hret);
            delete h;
            return nullptr;
        }
        RootInfoFileHeader header{};
        header.run_token = h->run_token;
        std::string tmp_path = h->rootinfo_path + ".tmp." + std::to_string(getpid());
        std::ofstream fout(tmp_path, std::ios::binary | std::ios::trunc);
        fout.write(reinterpret_cast<const char *>(&header), sizeof(header));
        fout.write(rootInfo.internal, HCCL_ROOT_INFO_BYTES);
        fout.close();
        if (!fout.good() || std::rename(tmp_path.c_str(), h->rootinfo_path.c_str()) != 0) {
            std::remove(tmp_path.c_str());
            delete h;
            return nullptr;
        }
    } else {
        if (!wait_for_rootinfo(h->rootinfo_path, &rootInfo, &h->run_token)) {
            LOG_ERROR("[comm rank %d] Timeout waiting for rootinfo", rank);
            delete h;
            return nullptr;
        }
    }

    if (!file_barrier(h->rootinfo_path, h->rank, h->nranks, "rootinfo_ready", h->run_token)) {
        delete h;
        return nullptr;
    }

    // Init communicator
    HcclResult hret =
        hccl_comm_init_root_info(static_cast<uint32_t>(nranks), &rootInfo, static_cast<uint32_t>(rank), &h->hccl_comm);
    if (hret != HCCL_SUCCESS) {
        LOG_ERROR("[comm rank %d] HcclCommInitRootInfo failed: %d", rank, (int)hret);
        delete h;
        return nullptr;
    }

    return h;
} catch (const std::exception &e) {
    LOG_ERROR("[comm rank %d] comm_init: exception: %s", rank, e.what());
    return nullptr;
} catch (...) {
    LOG_ERROR("[comm rank %d] comm_init: unknown exception", rank);
    return nullptr;
}

namespace {

// Path D: build the per-rank symmetric pool ourselves via the public ACL
// VMM shareable-handle API (aclrtMallocPhysical + ReserveMemAddress +
// MapMem + MemExportToShareableHandle / MemImportFromShareableHandle) and
// open cross-card P2P via aclrtDeviceEnablePeerAccess. aclrtIpcMem* is
// gated on the HCCS exbus capability and returns 507899 on PCIe-attached
// boxes (910B2C reports support_shmem_map_exbus=0), so the VMM path is
// the one that works across both HCCS and PCIe topologies. Cross-device
// import (alloc on devN, import onto devM) is supported. See #1037.

// Default per-rank symmetric pool size when comm_alloc_windows is called
// with win_size == 0. Picked to match the HCCL_BUFFSIZE default of the
// pre-Path-D backend so existing callers see no behavioural change.
constexpr uint64_t kDefaultIpcWinSize = 200ULL * 1024 * 1024;
constexpr uint64_t kIpcAnnounceMagic = 0x49504344334d4549ULL;  // "IPCD3MEI"

struct IpcAnnounceFile {
    uint64_t magic;
    int32_t pid;
    uint32_t rank;
    int32_t device_id;          // ACL logic device id this rank is bound to.
    uint64_t shareable_handle;  // aclrtMemExportToShareableHandle output
};

// Announce file path shares the `barrier_<prefix>_..._<rank>.ready` shape so
// cleanup_handshake_files picks it up alongside the file_barrier markers.
// Without this convention these files would accumulate across re-runs.
static std::string ipc_announce_path(const std::string &rootinfo, int rank, uint64_t run_token) {
    return handshake_dir(rootinfo) + "/barrier_" + handshake_prefix(rootinfo) + "_ipc_announce_" +
           run_token_hex(run_token) + "_" + std::to_string(rank) + ".ready";
}

static bool ipc_write_announce(
    const std::string &rootinfo, int rank, uint64_t run_token, int32_t pid, int32_t device_id, uint64_t shareable_handle
) {
    IpcAnnounceFile a{};
    a.magic = kIpcAnnounceMagic;
    a.pid = pid;
    a.rank = static_cast<uint32_t>(rank);
    a.device_id = device_id;
    a.shareable_handle = shareable_handle;
    std::string p = ipc_announce_path(rootinfo, rank, run_token);
    std::string tmp = p + ".tmp." + std::to_string(getpid());
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char *>(&a), sizeof(a));
        if (!f.good()) {
            std::remove(tmp.c_str());
            return false;
        }
    }
    if (std::rename(tmp.c_str(), p.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

static bool ipc_read_announce(
    const std::string &rootinfo, int peer, uint64_t run_token, IpcAnnounceFile *out, int timeout_sec = 60
) {
    std::string p = ipc_announce_path(rootinfo, peer, run_token);
    for (int i = 0; i < timeout_sec * 10; ++i) {
        std::ifstream f(p, std::ios::binary);
        if (f.good()) {
            IpcAnnounceFile a{};
            f.read(reinterpret_cast<char *>(&a), sizeof(a));
            if (f.good() && a.magic == kIpcAnnounceMagic && a.rank == static_cast<uint32_t>(peer)) {
                *out = a;
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

// Fills h->host_ctx with rankId/rankNum/winSize/windowsIn[] via VMM
// shareable-handle exchange.  `win_size` is the per-rank pool byte size
// requested by the caller (kDefaultIpcWinSize when 0); it is rounded up to
// the device allocation granularity before mapping, and winSize is stored
// as that aligned value so kernel window math matches the mapped range.
//
// On failure or normal exit, the device-side resources allocated here
// (the own VMM physical handle + mapped VA, peer VMM imports, and any P2P
// routes enabled) are NOT explicitly released. DeviceRunner::finalize
// calls aclrtResetDevice at Worker teardown, which reclaims VMM handles
// and mappings alongside other per-device state. simpler's current usage
// is one comm_init/destroy per Worker lifetime, so the absence of explicit
// cleanup does not accumulate across runs. If a future caller starts
// cycling comm contexts within a single Worker, explicit VMM teardown
// (aclrtUnmapMem + aclrtReleaseMemAddress + aclrtFreePhysical) will need
// to land here.
static int alloc_windows_via_ipc(CommHandle h, uint64_t win_size) {
    const int rank = h->rank;
    const int nranks = h->nranks;
    const std::string &rootinfo = h->rootinfo_path;
    const uint64_t run_token = h->run_token;

    // Discover our own device id. Rank != device in general (e.g. simpler's
    // chip_process spawns rank N on whatever device the resource pool gives
    // it). We need real device ids before any cross-rank ACL setup --
    // EnablePeerAccess takes a peer DEVICE id, not a peer rank.
    int32_t myDevice = -1;
    if (aclrtGetDevice(&myDevice) != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] ipc: aclrtGetDevice failed", rank);
        return -1;
    }

    // VMM own-window allocation: allocate a physical handle, reserve a VA
    // range, map the handle into it, and grant our device read/write access.
    aclrtPhysicalMemProp prop{};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_NORMAL;
    prop.location.id = static_cast<uint32_t>(myDevice);
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    size_t granularity = 0;
    aclError aret = aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] ipc: GetAllocationGranularity -> %d", rank, static_cast<int>(aret));
        return -1;
    }
    const uint64_t aligned_size =
        granularity == 0 ? win_size : ((win_size + granularity - 1) / granularity) * granularity;

    aclrtDrvMemHandle handle = nullptr;
    aret = aclrtMallocPhysical(&handle, aligned_size, &prop, 0);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] ipc: MallocPhysical -> %d", rank, static_cast<int>(aret));
        return -1;
    }
    void *localBuf = nullptr;
    aret = aclrtReserveMemAddress(&localBuf, aligned_size, 0, nullptr, 0);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] ipc: ReserveMemAddress -> %d", rank, static_cast<int>(aret));
        aclrtFreePhysical(handle);
        return -1;
    }
    aret = aclrtMapMem(localBuf, aligned_size, 0, handle, 0);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] ipc: MapMem -> %d", rank, static_cast<int>(aret));
        aclrtReleaseMemAddress(localBuf);
        aclrtFreePhysical(handle);
        return -1;
    }
    aclrtMemAccessDesc accessDesc{};
    accessDesc.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
    accessDesc.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = static_cast<uint32_t>(myDevice);
    aret = aclrtMemSetAccess(localBuf, aligned_size, &accessDesc, 1);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] ipc: MemSetAccess -> %d", rank, static_cast<int>(aret));
        release_own_vmm_window(localBuf, handle);
        return -1;
    }
    // DISABLE_PID_VALIDATION drops the pid-whitelist requirement: any peer
    // process may import the handle directly, with no separate authorization
    // step or auth barrier.
    uint64_t shareableHandle = 0;
    aret = aclrtMemExportToShareableHandle(
        handle, ACL_MEM_HANDLE_TYPE_NONE, ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION, &shareableHandle
    );
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] ipc: ExportToShareableHandle -> %d", rank, static_cast<int>(aret));
        release_own_vmm_window(localBuf, handle);
        return -1;
    }

    // Announce (pid, device, shareable handle) and read every peer's announcement.
    const int32_t myPid = static_cast<int32_t>(getpid());
    if (!ipc_write_announce(rootinfo, rank, run_token, myPid, myDevice, shareableHandle)) {
        LOG_ERROR("[comm rank %d] ipc: write_announce failed", rank);
        release_own_vmm_window(localBuf, handle);
        return -1;
    }
    std::vector<IpcAnnounceFile> peers(nranks);
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) {
            peers[p].magic = kIpcAnnounceMagic;
            peers[p].pid = myPid;
            peers[p].rank = static_cast<uint32_t>(rank);
            peers[p].device_id = myDevice;
            peers[p].shareable_handle = shareableHandle;
            continue;
        }
        if (!ipc_read_announce(rootinfo, p, run_token, &peers[p])) {
            LOG_ERROR("[comm rank %d] ipc: read_announce(peer=%d) timed out", rank, p);
            release_own_vmm_window(localBuf, handle);
            return -1;
        }
    }

    // Now we know every peer's device id. Enable cross-card P2P, then run a
    // best-effort confirmation poll. aclrtDeviceEnablePeerAccess is the
    // operative call: it resolves the peer's physical id via the HCCL adapter
    // and opens the HCCS route. Its success is what matters.
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) continue;
        aclError r = aclrtDeviceEnablePeerAccess(peers[p].device_id, 0);
        if (r != ACL_SUCCESS) {
            // CANN 9.x has no dedicated "already enabled" code, so a non-success
            // here may be a benign re-enable. The poll below is confirmation only.
            LOG_WARN(
                "[comm rank %d] ipc: EnablePeerAccess(peer_dev=%d) -> %d", rank, peers[p].device_id, static_cast<int>(r)
            );
        }
    }
    // Confirmation poll. aclrtDevicePeerAccessStatus and EnablePeerAccess
    // interpret the peer device-id differently under ASCEND_VISIBLE_DEVICES
    // remapping: in the fork-per-chip model each process aclrtSetDevice's only
    // its own device, so the status query cannot resolve the peer's logical id
    // to a physical one and reports status=0 indefinitely even when P2P is up
    // (enable succeeded + full-HCCS topology). So confirm quickly where the API
    // is reliable (ARM/openEuler, non-remapped) and fall through with a warning
    // otherwise; the file_barrier below still synchronizes every rank's enable,
    // and a genuinely dead link surfaces as a kernel-side TWAIT hang. See #1018.
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) continue;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (true) {
            int32_t status = 0;
            aclError r = aclrtDevicePeerAccessStatus(myDevice, peers[p].device_id, &status);
            if (r != ACL_SUCCESS) {
                LOG_ERROR(
                    "[comm rank %d] ipc: PeerAccessStatus(local_dev=%d peer_dev=%d) -> %d", rank, myDevice,
                    peers[p].device_id, static_cast<int>(r)
                );
                release_own_vmm_window(localBuf, handle);
                return -1;
            }
            if (status == 1) break;
            if (std::chrono::steady_clock::now() >= deadline) {
                LOG_WARN(
                    "[comm rank %d] ipc: P2P status unconfirmed peer=%d peer_dev=%d status=%d "
                    "(proceeding after best-effort enable attempt, see device-remap note)",
                    rank, p, peers[p].device_id, status
                );
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Barrier so every rank has finished its outbound P2P enable+wait. With
    // DISABLE_PID_VALIDATION the import can proceed once peers have published
    // their shareable handles (read above) and P2P is up.
    if (!file_barrier(rootinfo, rank, nranks, "ipc_p2p_ready", run_token)) {
        release_own_vmm_window(localBuf, handle);
        return -1;
    }

    // windowsOut[] is intentionally left zero: no kernel path reads it
    // (verified by grep across simpler + pto-isa). The field is kept in
    // CommContext only to preserve byte-equivalence with pto-isa's parallel
    // HcclDeviceContext declaration; removing it is gated on the F4
    // private-ization decision (see .docs/28.l3-comm/ext.01.pr-774-review.md).
    // host_ctx was value-initialized at handle construction (CommContext{}),
    // and the idempotency guard in comm_alloc_windows prevents a second entry;
    // no re-zero needed before populating it here.
    h->host_ctx.rankId = static_cast<uint32_t>(rank);
    h->host_ctx.rankNum = static_cast<uint32_t>(nranks);
    h->host_ctx.winSize = aligned_size;
    h->host_ctx.windowsIn[rank] = reinterpret_cast<uint64_t>(localBuf);

    // Import each peer's shareable handle onto our device. The symmetric pool
    // uses one win_size across ranks and all ranks share the chip-type
    // granularity, so every peer's allocation size equals our aligned_size.
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) continue;
        aclrtDrvMemHandle peerHandle = nullptr;
        aret = aclrtMemImportFromShareableHandle(peers[p].shareable_handle, myDevice, &peerHandle);
        if (aret != ACL_SUCCESS) {
            LOG_ERROR("[comm rank %d] ipc: ImportFromShareableHandle(peer=%d) -> %d", rank, p, static_cast<int>(aret));
            release_own_vmm_window(localBuf, handle);
            return -1;
        }
        void *peerVa = nullptr;
        aret = aclrtReserveMemAddress(&peerVa, aligned_size, 0, nullptr, 0);
        if (aret != ACL_SUCCESS) {
            LOG_ERROR("[comm rank %d] ipc: peer ReserveMemAddress(peer=%d) -> %d", rank, p, static_cast<int>(aret));
            aclrtFreePhysical(peerHandle);
            release_own_vmm_window(localBuf, handle);
            return -1;
        }
        aret = aclrtMapMem(peerVa, aligned_size, 0, peerHandle, 0);
        if (aret != ACL_SUCCESS) {
            LOG_ERROR("[comm rank %d] ipc: peer MapMem(peer=%d) -> %d", rank, p, static_cast<int>(aret));
            aclrtReleaseMemAddress(peerVa);
            aclrtFreePhysical(peerHandle);
            release_own_vmm_window(localBuf, handle);
            return -1;
        }
        aret = aclrtMemSetAccess(peerVa, aligned_size, &accessDesc, 1);
        if (aret != ACL_SUCCESS) {
            LOG_ERROR("[comm rank %d] ipc: peer MemSetAccess(peer=%d) -> %d", rank, p, static_cast<int>(aret));
            aclrtUnmapMem(peerVa);
            aclrtReleaseMemAddress(peerVa);
            aclrtFreePhysical(peerHandle);
            release_own_vmm_window(localBuf, handle);
            return -1;
        }
        h->host_ctx.windowsIn[p] = reinterpret_cast<uint64_t>(peerVa);
    }

    return 0;
}

// ============================================================================
// Per-domain dynamic allocation (for orch.allocate_domain).
//
// Same Path-D VMM dance as alloc_windows_via_ipc, but on a fresh per-allocation
// local buffer.  Every barrier filename and announce filename is scoped by
// allocation_id so concurrent allocations from different orch.allocate_domain
// calls do not collide.  Participation is by subset (domain_rank within
// rank_count), so non-members of the subset are not involved.
// ============================================================================

// Announce file path scoped by allocation_id so two concurrent allocations
// from different orch calls do not collide.  Same dir + cleanup-friendly
// prefix as the base-comm IPC announce.
static std::string
domain_announce_path(const std::string &rootinfo, uint64_t allocation_id, uint32_t domain_rank, uint64_t run_token) {
    return handshake_dir(rootinfo) + "/barrier_" + handshake_prefix(rootinfo) + "_alloc_" +
           std::to_string(allocation_id) + "_ipc_announce_" + run_token_hex(run_token) + "_" +
           std::to_string(domain_rank) + ".ready";
}

static bool domain_write_announce(
    const std::string &rootinfo, uint64_t allocation_id, uint32_t domain_rank, uint64_t run_token, int32_t pid,
    int32_t device_id, uint64_t shareable_handle
) {
    IpcAnnounceFile a{};
    a.magic = kIpcAnnounceMagic;
    a.pid = pid;
    a.rank = domain_rank;
    a.device_id = device_id;
    a.shareable_handle = shareable_handle;
    std::string p = domain_announce_path(rootinfo, allocation_id, domain_rank, run_token);
    std::string tmp = p + ".tmp." + std::to_string(getpid());
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char *>(&a), sizeof(a));
        if (!f.good()) {
            std::remove(tmp.c_str());
            return false;
        }
    }
    if (std::rename(tmp.c_str(), p.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

static bool domain_read_announce(
    const std::string &rootinfo, uint64_t allocation_id, uint32_t peer_domain_rank, uint64_t run_token,
    IpcAnnounceFile *out, int timeout_sec = 60
) {
    std::string p = domain_announce_path(rootinfo, allocation_id, peer_domain_rank, run_token);
    for (int i = 0; i < timeout_sec * 10; ++i) {
        std::ifstream f(p, std::ios::binary);
        if (f.good()) {
            IpcAnnounceFile a{};
            f.read(reinterpret_cast<char *>(&a), sizeof(a));
            if (f.good() && a.magic == kIpcAnnounceMagic && a.rank == peer_domain_rank) {
                *out = a;
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

// Tag helper for allocation-scoped file barriers.  Tag is fed straight into
// `file_barrier`, which already namespaces the marker filename by
// rootinfo prefix + run_token + rank, so adding allocation_id to `tag` is
// enough to keep concurrent allocations from sharing a marker file.
static std::string domain_barrier_tag(uint64_t allocation_id, const char *phase) {
    return std::string("alloc_") + std::to_string(allocation_id) + "_" + phase;
}

// Idempotently provision the process-global PTO-ISA async-SDMA scratch
// workspace on the comm handle and mirror its address into host_ctx.  Both
// the base-window path and the dynamic per-domain path call this; only the
// first call allocates.  CANN 9.0+ feature: on 8.5 the aclnn dlsym fails by
// design, so we leave workSpace == 0 and SDMA demos self-skip.  No-op when
// the build-time PTO-ISA dependency is absent.
static void ensure_sdma_workspace(CommHandle h) {
#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
    if (h->sdma_workspace) return;
    h->sdma_workspace = std::make_unique<pto::comm::sdma::SdmaWorkspaceManager>();
    if (h->sdma_workspace->Init()) {
        h->host_ctx.workSpace = reinterpret_cast<uint64_t>(h->sdma_workspace->GetWorkspaceAddr());
        h->host_ctx.workSpaceSize = 16 * 1024;
    } else {
        h->sdma_workspace.reset();
    }
#else
    (void)h;
#endif
}

// Performs the per-allocation Path-D dance for one subset rank.  rank_ids
// must list participating BASE-COMM rank ids in domain rank order; this
// rank's domain_rank must match its base rank for the same invariant
// alloc_windows_via_ipc relies on (rank_ids[domain_rank] == h->rank).
//
// Failure paths tear down the own VMM window if it was mapped, plus every peer
// import already recorded on `out` (release_domain_peer_windows).  On success
// the peer imports live on `out->peer_windows` for comm_release_domain_windows.
static int domain_alloc_via_ipc(
    CommHandle h, uint64_t allocation_id, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank,
    uint64_t win_size, DomainAllocation *out
) {
    const std::string &rootinfo = h->rootinfo_path;
    const uint64_t run_token = h->run_token;
    const int subset_n = static_cast<int>(rank_count);
    const int my_dr = static_cast<int>(domain_rank);

    int32_t myDevice = -1;
    if (aclrtGetDevice(&myDevice) != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: aclrtGetDevice failed", h->rank);
        return -1;
    }

    // VMM own-window allocation; see alloc_windows_via_ipc for step-by-step
    // rationale. aligned_size is also stored in the DomainAllocation so the
    // caller can zero the full mapped range.
    aclrtPhysicalMemProp prop{};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_NORMAL;
    prop.location.id = static_cast<uint32_t>(myDevice);
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    size_t granularity = 0;
    aclError aret = aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: GetAllocationGranularity -> %d", h->rank, static_cast<int>(aret));
        return -1;
    }
    const uint64_t aligned_size =
        granularity == 0 ? win_size : ((win_size + granularity - 1) / granularity) * granularity;

    aclrtDrvMemHandle handle = nullptr;
    aret = aclrtMallocPhysical(&handle, aligned_size, &prop, 0);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: MallocPhysical -> %d", h->rank, static_cast<int>(aret));
        return -1;
    }
    void *localBuf = nullptr;
    aret = aclrtReserveMemAddress(&localBuf, aligned_size, 0, nullptr, 0);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: ReserveMemAddress -> %d", h->rank, static_cast<int>(aret));
        aclrtFreePhysical(handle);
        return -1;
    }
    aret = aclrtMapMem(localBuf, aligned_size, 0, handle, 0);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: MapMem -> %d", h->rank, static_cast<int>(aret));
        aclrtReleaseMemAddress(localBuf);
        aclrtFreePhysical(handle);
        return -1;
    }
    aclrtMemAccessDesc accessDesc{};
    accessDesc.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
    accessDesc.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = static_cast<uint32_t>(myDevice);
    aret = aclrtMemSetAccess(localBuf, aligned_size, &accessDesc, 1);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: MemSetAccess -> %d", h->rank, static_cast<int>(aret));
        release_own_vmm_window(localBuf, handle);
        return -1;
    }
    uint64_t shareableHandle = 0;
    aret = aclrtMemExportToShareableHandle(
        handle, ACL_MEM_HANDLE_TYPE_NONE, ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION, &shareableHandle
    );
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: ExportToShareableHandle -> %d", h->rank, static_cast<int>(aret));
        release_own_vmm_window(localBuf, handle);
        return -1;
    }

    const int32_t myPid = static_cast<int32_t>(getpid());
    if (!domain_write_announce(rootinfo, allocation_id, domain_rank, run_token, myPid, myDevice, shareableHandle)) {
        LOG_ERROR("[comm rank %d] alloc_domain: write_announce failed", h->rank);
        release_own_vmm_window(localBuf, handle);
        return -1;
    }
    std::vector<IpcAnnounceFile> peers(subset_n);
    for (int p = 0; p < subset_n; ++p) {
        if (p == my_dr) {
            peers[p].magic = kIpcAnnounceMagic;
            peers[p].pid = myPid;
            peers[p].rank = domain_rank;
            peers[p].device_id = myDevice;
            peers[p].shareable_handle = shareableHandle;
            continue;
        }
        if (!domain_read_announce(rootinfo, allocation_id, static_cast<uint32_t>(p), run_token, &peers[p])) {
            LOG_ERROR("[comm rank %d] alloc_domain: read_announce(peer_dr=%d) timed out", h->rank, p);
            release_own_vmm_window(localBuf, handle);
            return -1;
        }
    }

    // Enable cross-card P2P for every domain peer, then a best-effort
    // confirmation poll. The orch-only allocate_domain model has no base
    // comm_alloc_windows to own the P2P route, so each allocation must
    // (idempotently) ensure it. aclrtDeviceEnablePeerAccess is process-global
    // and per device-pair; once any allocation opens a pair, later ones simply
    // observe it. The enable is the operative call (resolves the peer physical
    // id via the HCCL adapter and opens the HCCS route).
    for (int p = 0; p < subset_n; ++p) {
        if (p == my_dr) continue;
        aclError r = aclrtDeviceEnablePeerAccess(peers[p].device_id, 0);
        if (r != ACL_SUCCESS) {
            LOG_WARN(
                "[comm rank %d] alloc_domain: EnablePeerAccess(peer_dev=%d) -> %d", h->rank, peers[p].device_id,
                static_cast<int>(r)
            );
        }
    }
    // See the device-remap note in alloc_windows_via_ipc: under
    // ASCEND_VISIBLE_DEVICES remapping aclrtDevicePeerAccessStatus cannot
    // resolve a peer that this fork'd single-device process never set, so it
    // reports status=0 even when P2P is up. Confirm quickly where reliable,
    // else fall through with a warning; the file_barrier below synchronizes
    // every rank's enable and a dead link surfaces as a kernel-side hang.
    for (int p = 0; p < subset_n; ++p) {
        if (p == my_dr) continue;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (true) {
            int32_t status = 0;
            aclError r = aclrtDevicePeerAccessStatus(myDevice, peers[p].device_id, &status);
            if (r != ACL_SUCCESS) {
                LOG_ERROR(
                    "[comm rank %d] alloc_domain: PeerAccessStatus(local_dev=%d peer_dev=%d) -> %d", h->rank, myDevice,
                    peers[p].device_id, static_cast<int>(r)
                );
                release_own_vmm_window(localBuf, handle);
                return -1;
            }
            if (status == 1) break;
            if (std::chrono::steady_clock::now() >= deadline) {
                LOG_WARN(
                    "[comm rank %d] alloc_domain: P2P status unconfirmed peer_dr=%d peer_dev=%d status=%d "
                    "(proceeding after best-effort enable attempt, see device-remap note)",
                    h->rank, p, peers[p].device_id, status
                );
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // With DISABLE_PID_VALIDATION the import can proceed once peers have
    // published their shareable handles (read above) and P2P is up.
    if (!file_barrier(rootinfo, my_dr, subset_n, domain_barrier_tag(allocation_id, "p2p_ready"), run_token)) {
        release_own_vmm_window(localBuf, handle);
        return -1;
    }

    out->rank = my_dr;
    out->nranks = subset_n;
    out->local_buf = localBuf;
    out->alloc_size = aligned_size;
    out->own_handle = handle;
    // Build a host-side CommContext for the subset and upload it as device_ctx.
    // PTO-ISA async SDMA ops (SdmaTget) read the scratch workspace off
    // CommContext::workSpace.  The dynamic-domain path does not go through
    // comm_alloc_windows, so provision the workspace here; without it a
    // freshly zero-initialized per-domain ctx would leave workSpace == 0 and
    // those kernels early-return on the workSpace guard.
    ensure_sdma_workspace(h);

    CommContext ctx{};
    ctx.rankId = domain_rank;
    ctx.rankNum = static_cast<uint32_t>(subset_n);
    ctx.winSize = aligned_size;
    ctx.workSpace = h->host_ctx.workSpace;
    ctx.workSpaceSize = h->host_ctx.workSpaceSize;
    ctx.windowsIn[my_dr] = reinterpret_cast<uint64_t>(localBuf);
    // Import each peer's shareable handle onto our device; see the symmetry
    // note in alloc_windows_via_ipc (one win_size, shared chip granularity).
    for (int p = 0; p < subset_n; ++p) {
        if (p == my_dr) continue;
        aclrtDrvMemHandle peerHandle = nullptr;
        aret = aclrtMemImportFromShareableHandle(peers[p].shareable_handle, myDevice, &peerHandle);
        if (aret != ACL_SUCCESS) {
            LOG_ERROR(
                "[comm rank %d] alloc_domain: ImportFromShareableHandle(peer_dr=%d) -> %d", h->rank, p,
                static_cast<int>(aret)
            );
            release_domain_peer_windows(*out);
            release_own_vmm_window(localBuf, handle);
            return -1;
        }
        void *peerVa = nullptr;
        aret = aclrtReserveMemAddress(&peerVa, aligned_size, 0, nullptr, 0);
        if (aret != ACL_SUCCESS) {
            LOG_ERROR(
                "[comm rank %d] alloc_domain: peer ReserveMemAddress(peer_dr=%d) -> %d", h->rank, p,
                static_cast<int>(aret)
            );
            aclrtFreePhysical(peerHandle);
            release_domain_peer_windows(*out);
            release_own_vmm_window(localBuf, handle);
            return -1;
        }
        aret = aclrtMapMem(peerVa, aligned_size, 0, peerHandle, 0);
        if (aret != ACL_SUCCESS) {
            LOG_ERROR("[comm rank %d] alloc_domain: peer MapMem(peer_dr=%d) -> %d", h->rank, p, static_cast<int>(aret));
            aclrtReleaseMemAddress(peerVa);
            aclrtFreePhysical(peerHandle);
            release_domain_peer_windows(*out);
            release_own_vmm_window(localBuf, handle);
            return -1;
        }
        aret = aclrtMemSetAccess(peerVa, aligned_size, &accessDesc, 1);
        if (aret != ACL_SUCCESS) {
            LOG_ERROR(
                "[comm rank %d] alloc_domain: peer MemSetAccess(peer_dr=%d) -> %d", h->rank, p, static_cast<int>(aret)
            );
            aclrtUnmapMem(peerVa);
            aclrtReleaseMemAddress(peerVa);
            aclrtFreePhysical(peerHandle);
            release_domain_peer_windows(*out);
            release_own_vmm_window(localBuf, handle);
            return -1;
        }
        out->peer_windows.emplace_back(peerVa, peerHandle);
        ctx.windowsIn[p] = reinterpret_cast<uint64_t>(peerVa);
    }

    void *newDevMem = nullptr;
    aret = aclrtMalloc(&newDevMem, sizeof(CommContext), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: ctx aclrtMalloc -> %d", h->rank, static_cast<int>(aret));
        release_own_vmm_window(localBuf, handle);
        return -1;
    }
    aret = aclrtMemcpy(newDevMem, sizeof(CommContext), &ctx, sizeof(CommContext), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: ctx Memcpy H2D -> %d", h->rank, static_cast<int>(aret));
        aclrtFree(newDevMem);
        release_own_vmm_window(localBuf, handle);
        return -1;
    }
    out->device_ctx = reinterpret_cast<CommContext *>(newDevMem);
    return 0;
}

}  // namespace

extern "C" int comm_alloc_windows(CommHandle h, size_t win_size, uint64_t *device_ctx_out) try {
    if (!h || !device_ctx_out) return -1;

    // Idempotency guard: comm_alloc_windows is not re-entrant. The localBuf
    // allocated by alloc_windows_via_ipc is owned by the handle's windowsIn[]
    // entries and is only reclaimed at aclrtResetDevice; calling this twice
    // would leak a full per-rank pool. device_ctx is set on first success.
    if (h->device_ctx != nullptr) {
        LOG_ERROR("[comm rank %d] comm_alloc_windows: already allocated on this handle", h->rank);
        return -1;
    }

    // Path D: DIY symmetric pool on stable ACL VMM shareable handles +
    // EnablePeerAccess. Replaced the prior HcclAllocComResourceByTiling
    // reverse-parse path (broken on CANN 9.0 due to HcclOpResParam ABI
    // drift; see project history). One backend, works on 8.5 and 9.0
    // unchanged.
    const uint64_t effective_win_size = win_size != 0 ? static_cast<uint64_t>(win_size) : kDefaultIpcWinSize;
    if (alloc_windows_via_ipc(h, effective_win_size) != 0) return -1;

    // Optional PTO-ISA async SDMA workspace pre-allocation (overlays the comm
    // backend's output; comm-side flow does not care about workSpace).
    ensure_sdma_workspace(h);

    void *newDevMem = nullptr;
    aclError aRet = aclrtMalloc(&newDevMem, sizeof(CommContext), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS) return -1;
    aRet = aclrtMemcpy(newDevMem, sizeof(CommContext), &h->host_ctx, sizeof(CommContext), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aRet != ACL_SUCCESS) {
        aclrtFree(newDevMem);
        return -1;
    }
    h->device_ctx = reinterpret_cast<CommContext *>(newDevMem);
    h->owns_device_ctx = true;
    *device_ctx_out = reinterpret_cast<uint64_t>(h->device_ctx);
    return 0;
} catch (const std::exception &e) {
    LOG_ERROR("[comm] comm_alloc_windows: exception: %s", e.what());
    return -1;
} catch (...) {
    LOG_ERROR("[comm] comm_alloc_windows: unknown exception");
    return -1;
}

extern "C" int comm_get_local_window_base(CommHandle h, uint64_t *base_out) {
    if (!h || !base_out) return -1;
    *base_out = h->host_ctx.windowsIn[h->rank];
    return 0;
}

extern "C" int comm_get_window_size(CommHandle h, size_t *size_out) {
    if (!h || !size_out) return -1;
    *size_out = static_cast<size_t>(h->host_ctx.winSize);
    return 0;
}

extern "C" int comm_derive_context(
    CommHandle h, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank, size_t window_offset,
    size_t window_size, uint64_t *device_ctx_out
) try {
    if (!h || !rank_ids || !device_ctx_out) return -1;
    if (h->host_ctx.rankNum == 0) {
        LOG_ERROR("[comm rank %d] comm_derive_context: base windows are not allocated", h->rank);
        return -1;
    }
    if (rank_count == 0 || rank_count > COMM_MAX_RANK_NUM || domain_rank >= rank_count) {
        LOG_ERROR(
            "[comm rank %d] comm_derive_context: invalid rank_count=%zu domain_rank=%u", h->rank, rank_count,
            domain_rank
        );
        return -1;
    }
    if (window_offset + window_size > static_cast<size_t>(h->host_ctx.winSize)) {
        LOG_ERROR(
            "[comm rank %d] comm_derive_context: window range [%zu, %zu) exceeds base window size %llu", h->rank,
            window_offset, window_offset + window_size, static_cast<unsigned long long>(h->host_ctx.winSize)
        );
        return -1;
    }

    CommContext ctx{};
    ctx.workSpace = h->host_ctx.workSpace;
    ctx.workSpaceSize = h->host_ctx.workSpaceSize;
    ctx.rankId = domain_rank;
    ctx.rankNum = static_cast<uint32_t>(rank_count);
    ctx.winSize = window_size;
    for (size_t i = 0; i < rank_count; ++i) {
        uint32_t base_rank = rank_ids[i];
        if (base_rank >= static_cast<uint32_t>(h->nranks)) {
            LOG_ERROR(
                "[comm rank %d] comm_derive_context: rank_ids[%zu]=%u out of range [0, %d)", h->rank, i, base_rank,
                h->nranks
            );
            return -1;
        }
        ctx.windowsIn[i] = h->host_ctx.windowsIn[base_rank] + window_offset;
        ctx.windowsOut[i] = h->host_ctx.windowsOut[base_rank] + window_offset;
    }

    void *newDevMem = nullptr;
    aclError aRet = aclrtMalloc(&newDevMem, sizeof(CommContext), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] comm_derive_context: aclrtMalloc failed: %d", h->rank, static_cast<int>(aRet));
        return -1;
    }
    aRet = aclrtMemcpy(newDevMem, sizeof(CommContext), &ctx, sizeof(CommContext), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aRet != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] comm_derive_context: aclrtMemcpy H2D failed: %d", h->rank, static_cast<int>(aRet));
        aclrtFree(newDevMem);
        return -1;
    }

    auto *derived = reinterpret_cast<CommContext *>(newDevMem);
    h->derived_contexts.push_back(derived);
    *device_ctx_out = reinterpret_cast<uint64_t>(derived);
    return 0;
} catch (const std::exception &e) {
    LOG_ERROR("[comm] comm_derive_context: exception: %s", e.what());
    return -1;
} catch (...) {
    LOG_ERROR("[comm] comm_derive_context: unknown exception");
    return -1;
}

extern "C" int comm_barrier(CommHandle h) {
    if (!h) return -1;
    // HcclBarrier is synchronous — it blocks until all ranks arrive.
    // Do NOT call aclrtSynchronizeStream after it: HcclBarrier internally
    // switches the thread's ACL context, which invalidates the caller-owned
    // stream for context-checked ACL calls (error 507018).
    HcclResult hret = hccl_barrier(h->hccl_comm, h->stream);
    if (hret != HCCL_SUCCESS) {
        LOG_ERROR("[comm rank %d] HcclBarrier failed: %d", h->rank, static_cast<int>(hret));
        return static_cast<int>(hret);
    }
    return 0;
}

extern "C" int comm_alloc_domain_windows(
    CommHandle h, uint64_t allocation_id, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank,
    size_t window_size, uint64_t *device_ctx_out, uint64_t *local_window_base_out
) try {
    if (!h || !rank_ids || !device_ctx_out || !local_window_base_out) return -1;
    if (rank_count == 0 || rank_count > COMM_MAX_RANK_NUM || domain_rank >= rank_count || window_size == 0) {
        LOG_ERROR(
            "[comm rank %d] alloc_domain: bad args (rank_count=%zu domain_rank=%u window_size=%zu)", h->rank,
            rank_count, domain_rank, window_size
        );
        return -1;
    }
    if (h->domain_allocations.count(allocation_id) > 0) {
        LOG_ERROR(
            "[comm rank %d] alloc_domain: allocation_id=%llu already live", h->rank,
            static_cast<unsigned long long>(allocation_id)
        );
        return -1;
    }
    if (rank_ids[domain_rank] != static_cast<uint32_t>(h->rank)) {
        LOG_ERROR(
            "[comm rank %d] alloc_domain: rank_ids[%u]=%u does not match base rank", h->rank, domain_rank,
            rank_ids[domain_rank]
        );
        return -1;
    }
    // The base communicator only needs comm_init to have run (rootinfo_path
    // + run_token are set, used to scope barrier filenames).  We do NOT
    // require comm_alloc_windows on the base in the orch-only model — the
    // dynamic alloc path does its own per-allocation aclrtMalloc + IPC dance.
    if (h->rootinfo_path.empty() || h->hccl_comm == nullptr) {
        LOG_ERROR("[comm rank %d] alloc_domain: base communicator not initialised", h->rank);
        return -1;
    }

    auto alloc = std::make_unique<DomainAllocation>();
    int rc = domain_alloc_via_ipc(h, allocation_id, rank_ids, rank_count, domain_rank, window_size, alloc.get());
    if (rc != 0) return rc;

    // Zero the freshly-allocated local pool so kernels do not observe stale
    // bytes (parity with the sim backend's memset). The full granularity-aligned
    // mapped range is zeroed to match ctx.winSize.
    aclError aret = aclrtMemset(alloc->local_buf, alloc->alloc_size, 0, alloc->alloc_size);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: aclrtMemset -> %d", h->rank, static_cast<int>(aret));
        aclrtFree(alloc->device_ctx);
        release_own_vmm_window(alloc->local_buf, alloc->own_handle);
        return -1;
    }

    *device_ctx_out = reinterpret_cast<uint64_t>(alloc->device_ctx);
    *local_window_base_out = reinterpret_cast<uint64_t>(alloc->local_buf);
    h->domain_allocations.emplace(allocation_id, std::move(alloc));
    return 0;
} catch (const std::exception &e) {
    LOG_ERROR("[comm] alloc_domain: exception: %s", e.what());
    return -1;
} catch (...) {
    LOG_ERROR("[comm] alloc_domain: unknown exception");
    return -1;
}

extern "C" int
comm_release_domain_windows(CommHandle h, uint64_t allocation_id, size_t rank_count, uint32_t domain_rank) try {
    if (!h) return -1;
    auto it = h->domain_allocations.find(allocation_id);
    if (it == h->domain_allocations.end()) {
        LOG_ERROR(
            "[comm rank %d] release_domain: allocation_id=%llu not found", h->rank,
            static_cast<unsigned long long>(allocation_id)
        );
        return -1;
    }
    auto &alloc = it->second;
    if (static_cast<size_t>(alloc->nranks) != rank_count || static_cast<uint32_t>(alloc->rank) != domain_rank) {
        LOG_ERROR(
            "[comm rank %d] release_domain: caller (rank_count=%zu, domain_rank=%u) "
            "disagrees with alloc-time (nranks=%d, rank=%d)",
            h->rank, rank_count, domain_rank, alloc->nranks, alloc->rank
        );
        return -1;
    }
    int rc = 0;
    // Best-effort subset barrier so peers don't free local memory under each
    // other.  If a peer crashed mid-allocation, the timeout returns false and
    // we proceed with local teardown anyway — same shape as comm_destroy.
    if (!file_barrier(
            h->rootinfo_path, static_cast<int>(domain_rank), static_cast<int>(rank_count),
            domain_barrier_tag(allocation_id, "release"), h->run_token
        )) {
        LOG_WARN("[comm rank %d] release_domain: barrier timed out; releasing local state anyway", h->rank);
        rc = -1;
    }

    if (alloc->device_ctx) {
        aclError aret = aclrtFree(alloc->device_ctx);
        if (aret != ACL_SUCCESS && rc == 0) rc = -1;
    }
    // local_buf and every peer import are VMM-mapped VAs, not aclrtMalloc
    // pointers: unmap + release the VA reservation, then free the physical
    // handle.
    release_domain_peer_windows(*alloc);
    if (alloc->local_buf) {
        release_own_vmm_window(alloc->local_buf, alloc->own_handle);
        alloc->local_buf = nullptr;
        alloc->own_handle = nullptr;
    }
    h->domain_allocations.erase(it);
    return rc;
} catch (const std::exception &e) {
    LOG_ERROR("[comm] release_domain: exception: %s", e.what());
    return -1;
} catch (...) {
    LOG_ERROR("[comm] release_domain: unknown exception");
    return -1;
}

extern "C" int comm_destroy(CommHandle h) try {
    if (!h) return -1;

    // Final barrier is best-effort: if a peer already crashed we still need to
    // release the local resources we own, so timeout just logs and proceeds.
    int rc = 0;
    if (!file_barrier(h->rootinfo_path, h->rank, h->nranks, "destroy", h->run_token)) {
        LOG_WARN("[comm rank %d] comm_destroy: final barrier timed out; releasing local state anyway", h->rank);
        rc = -1;
    }

    if (h->owns_device_ctx && h->device_ctx) {
        aclrtFree(h->device_ctx);
    }
    for (CommContext *ctx : h->derived_contexts) {
        if (ctx != nullptr) {
            aclrtFree(ctx);
        }
    }
    h->derived_contexts.clear();
    // Reclaim any still-live domain allocations as a safety net.  Caller
    // should release them explicitly via comm_release_domain_windows; this
    // path runs only when an exception or shutdown bypassed that.
    for (auto &kv : h->domain_allocations) {
        auto &alloc = kv.second;
        if (alloc->device_ctx) aclrtFree(alloc->device_ctx);
        release_domain_peer_windows(*alloc);
        if (alloc->local_buf) release_own_vmm_window(alloc->local_buf, alloc->own_handle);
    }
    h->domain_allocations.clear();
    if (h->hccl_comm) {
        HcclResult hret = hccl_comm_destroy(h->hccl_comm);
        if (hret != HCCL_SUCCESS) {
            LOG_ERROR("[comm rank %d] HcclCommDestroy failed: %d", h->rank, static_cast<int>(hret));
            if (rc == 0) rc = -1;
        }
    }

    // NOTE: we do NOT destroy h->stream — it is caller-owned.
    // We also do NOT call aclrtResetDevice / aclFinalize here.  Device/ACL
    // lifecycle belongs to DeviceRunner, whose finalize() releases all
    // device memory before resetting the device and running aclFinalize.

    // Only rank 0 sweeps the on-disk handshake markers, and only if the
    // final barrier succeeded.  Deleting them after a timeout would strand
    // any peer that hasn't observed our marker yet, and leak that peer
    // into the next run with no rootinfo to discover.  Letting cleanup
    // ride on the next rank-0 init is the safer recovery path.
    if (h->rank == 0 && rc == 0) {
        cleanup_handshake_files(h->rootinfo_path);
    }

    delete h;
    return rc;
} catch (const std::exception &e) {
    LOG_ERROR("[comm] comm_destroy: exception: %s", e.what());
    if (h) delete h;
    return -1;
} catch (...) {
    LOG_ERROR("[comm] comm_destroy: unknown exception");
    if (h) delete h;
    return -1;
}
