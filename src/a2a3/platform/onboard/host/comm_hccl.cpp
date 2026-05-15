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
 * and the public ACL IPC primitives (aclrtIpcMem* + EnablePeerAccess)
 * for the per-rank symmetric window pool (Path D).
 *
 * Scope: L3 single-host multi-card only. aclrtIpcMem* is host-local, so
 * cross-host (L4) deployments need a different windows backend -- see
 * .docs/28.l3-comm/ext.01.pr-774-review.md F2 / 05.plan.zh.md for the
 * channel-API direction.
 */

#include "platform_comm/comm.h"
#include "platform_comm/comm_context.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
#include <memory>
#endif
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>
#include <unistd.h>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"
#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
#include "pto/npu/comm/async/sdma/sdma_workspace_manager.hpp"
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
                fprintf(
                    stderr, "[comm rank %d] file_barrier('%s') timed out after %ds waiting for rank %d\n", rank,
                    tag.c_str(), timeout_sec, r
                );
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    return true;
}

}  // namespace

// ============================================================================
// API implementation
// ============================================================================

extern "C" CommHandle comm_init(int rank, int nranks, void *stream, const char *rootinfo_path) try {
    if (stream == nullptr) {
        fprintf(stderr, "[comm rank %d] comm_init: caller-supplied stream is null\n", rank);
        return nullptr;
    }
    if (rootinfo_path == nullptr || *rootinfo_path == '\0') {
        fprintf(stderr, "[comm rank %d] comm_init: rootinfo_path is null or empty\n", rank);
        return nullptr;
    }
    if (nranks <= 0 || rank < 0 || rank >= nranks) {
        fprintf(stderr, "[comm rank %d] comm_init: invalid rank/nranks (rank=%d, nranks=%d)\n", rank, rank, nranks);
        return nullptr;
    }
    if (static_cast<uint32_t>(nranks) > COMM_MAX_RANK_NUM) {
        fprintf(
            stderr, "[comm rank %d] comm_init: nranks=%d exceeds COMM_MAX_RANK_NUM=%u\n", rank, nranks,
            COMM_MAX_RANK_NUM
        );
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
            fprintf(stderr, "[comm rank 0] HcclGetRootInfo failed: %d\n", (int)hret);
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
            fprintf(stderr, "[comm rank %d] Timeout waiting for rootinfo\n", rank);
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
        fprintf(stderr, "[comm rank %d] HcclCommInitRootInfo failed: %d\n", rank, (int)hret);
        delete h;
        return nullptr;
    }

    return h;
} catch (const std::exception &e) {
    fprintf(stderr, "[comm rank %d] comm_init: exception: %s\n", rank, e.what());
    return nullptr;
} catch (...) {
    fprintf(stderr, "[comm rank %d] comm_init: unknown exception\n", rank);
    return nullptr;
}

namespace {

// Path D: build the per-rank symmetric pool ourselves via the public ACL
// IPC primitives (aclrtMalloc + aclrtIpcMemGetExportKey + SetImportPid +
// ImportByKey). This mirrors HCCL's own internal cross-rank IPC pattern
// (refs/hcomm adapter_rts.cc::hrtIpc* + p2p_mgmt.cc::EnableP2P) so we
// depend only on stable ACL surface, no HCCL-private struct ABI.
// Spike-validated in hw-native-sys/comm-spike; see project memory.

// Default per-rank symmetric pool size when comm_alloc_windows is called
// with win_size == 0. Picked to match the HCCL_BUFFSIZE default of the
// pre-Path-D backend so existing callers see no behavioural change.
constexpr uint64_t kDefaultIpcWinSize = 200ULL * 1024 * 1024;
constexpr size_t kIpcNameLen = 65;
constexpr uint64_t kIpcAnnounceMagic = 0x49504344334d4549ULL;  // "IPCD3MEI"

struct IpcAnnounceFile {
    uint64_t magic;
    int32_t pid;
    uint32_t rank;
    int32_t device_id;  // ACL logic device id this rank is bound to.
    char name[kIpcNameLen];
    char pad[3];  // keep struct size a multiple of 8
};

// Announce file path shares the `barrier_<prefix>_..._<rank>.ready` shape so
// cleanup_handshake_files picks it up alongside the file_barrier markers.
// Without this convention these files would accumulate across re-runs.
static std::string ipc_announce_path(const std::string &rootinfo, int rank, uint64_t run_token) {
    return handshake_dir(rootinfo) + "/barrier_" + handshake_prefix(rootinfo) + "_ipc_announce_" +
           run_token_hex(run_token) + "_" + std::to_string(rank) + ".ready";
}

static bool ipc_write_announce(
    const std::string &rootinfo, int rank, uint64_t run_token, int32_t pid, int32_t device_id, const char *name
) {
    IpcAnnounceFile a{};
    a.magic = kIpcAnnounceMagic;
    a.pid = pid;
    a.rank = static_cast<uint32_t>(rank);
    a.device_id = device_id;
    memcpy(a.name, name, kIpcNameLen);
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

// Fills h->host_ctx with rankId/rankNum/winSize/windowsIn[] via DIY IPC.
// `win_size` is the per-rank pool byte size requested by the caller
// (kDefaultIpcWinSize when 0).
//
// On failure or normal exit, the device-side resources allocated here
// (localBuf via aclrtMalloc, the IPC export key, peer imports, and any
// P2P routes enabled) are NOT explicitly released. DeviceRunner::finalize
// calls aclrtResetDevice at Worker teardown, which reclaims all of the
// above. simpler's current usage is one comm_init/destroy per Worker
// lifetime, so the absence of explicit cleanup does not accumulate
// across runs. If a future caller starts cycling comm contexts within a
// single Worker, explicit teardown will need to land here.
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
        fprintf(stderr, "[comm rank %d] ipc: aclrtGetDevice failed\n", rank);
        return -1;
    }

    // Allocate local buffer + export its IPC name.
    void *localBuf = nullptr;
    aclError aret = aclrtMalloc(&localBuf, win_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aret != ACL_SUCCESS) {
        fprintf(stderr, "[comm rank %d] ipc: aclrtMalloc -> %d\n", rank, static_cast<int>(aret));
        return -1;
    }
    char myName[kIpcNameLen]{};
    aret = aclrtIpcMemGetExportKey(localBuf, win_size, myName, kIpcNameLen, 0);
    if (aret != ACL_SUCCESS) {
        fprintf(stderr, "[comm rank %d] ipc: GetExportKey -> %d\n", rank, static_cast<int>(aret));
        aclrtFree(localBuf);
        return -1;
    }

    // Announce (pid, device, name) and read every peer's announcement.
    const int32_t myPid = static_cast<int32_t>(getpid());
    if (!ipc_write_announce(rootinfo, rank, run_token, myPid, myDevice, myName)) {
        fprintf(stderr, "[comm rank %d] ipc: write_announce failed\n", rank);
        aclrtFree(localBuf);
        return -1;
    }
    std::vector<IpcAnnounceFile> peers(nranks);
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) {
            peers[p].magic = kIpcAnnounceMagic;
            peers[p].pid = myPid;
            peers[p].rank = static_cast<uint32_t>(rank);
            peers[p].device_id = myDevice;
            memcpy(peers[p].name, myName, kIpcNameLen);
            continue;
        }
        if (!ipc_read_announce(rootinfo, p, run_token, &peers[p])) {
            fprintf(stderr, "[comm rank %d] ipc: read_announce(peer=%d) timed out\n", rank, p);
            aclrtFree(localBuf);
            return -1;
        }
    }

    // Now we know every peer's device id. Enable cross-card P2P and poll
    // until ENABLED. Mirrors hcomm/.../p2p_mgmt.cc::EnableP2P + WaitP2PEnabled.
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) continue;
        aclError r = aclrtDeviceEnablePeerAccess(peers[p].device_id, 0);
        if (r != ACL_SUCCESS) {
            // Non-fatal: may already be enabled from a prior init in this process.
            fprintf(
                stderr, "[comm rank %d] ipc: EnablePeerAccess(peer_dev=%d) -> %d (continuing)\n", rank,
                peers[p].device_id, static_cast<int>(r)
            );
        }
    }
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) continue;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (true) {
            int32_t status = 0;
            aclError r = aclrtDevicePeerAccessStatus(myDevice, peers[p].device_id, &status);
            if (r != ACL_SUCCESS) {
                fprintf(
                    stderr, "[comm rank %d] ipc: PeerAccessStatus(local_dev=%d peer_dev=%d) -> %d\n", rank, myDevice,
                    peers[p].device_id, static_cast<int>(r)
                );
                aclrtFree(localBuf);
                return -1;
            }
            if (status == 1) break;
            if (std::chrono::steady_clock::now() >= deadline) {
                fprintf(
                    stderr, "[comm rank %d] ipc: P2P enable timeout peer=%d peer_dev=%d status=%d\n", rank, p,
                    peers[p].device_id, status
                );
                aclrtFree(localBuf);
                return -1;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Barrier so every rank has finished its outbound P2P enable+wait.
    if (!file_barrier(rootinfo, rank, nranks, "ipc_p2p_ready", run_token)) {
        aclrtFree(localBuf);
        return -1;
    }

    // Authorize every peer's pid against MY name (batched).
    std::vector<int32_t> peerPids;
    peerPids.reserve(nranks - 1);
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) continue;
        peerPids.push_back(peers[p].pid);
    }
    aret = aclrtIpcMemSetImportPid(myName, peerPids.data(), peerPids.size());
    if (aret != ACL_SUCCESS) {
        fprintf(stderr, "[comm rank %d] ipc: SetImportPid -> %d\n", rank, static_cast<int>(aret));
        aclrtFree(localBuf);
        return -1;
    }
    if (!file_barrier(rootinfo, rank, nranks, "ipc_auth_done", run_token)) {
        aclrtFree(localBuf);
        return -1;
    }

    // Import every peer's buffer into our local VA space.
    // windowsOut[] is intentionally left zero by the memset below: no kernel
    // path reads it (verified by grep across simpler + pto-isa). The field
    // is kept in CommContext only to preserve byte-equivalence with pto-isa's
    // parallel HcclDeviceContext declaration; removing it is gated on the
    // F4 private-ization decision (see .docs/28.l3-comm/ext.01.pr-774-review.md).
    memset(&h->host_ctx, 0, sizeof(h->host_ctx));
    h->host_ctx.rankId = static_cast<uint32_t>(rank);
    h->host_ctx.rankNum = static_cast<uint32_t>(nranks);
    h->host_ctx.winSize = win_size;
    h->host_ctx.windowsIn[rank] = reinterpret_cast<uint64_t>(localBuf);

    for (int p = 0; p < nranks; ++p) {
        if (p == rank) continue;
        void *peerVa = nullptr;
        aret = aclrtIpcMemImportByKey(&peerVa, peers[p].name, 0);
        if (aret != ACL_SUCCESS) {
            fprintf(
                stderr, "[comm rank %d] ipc: ImportByKey(peer=%d pid=%d) -> %d\n", rank, p, peers[p].pid,
                static_cast<int>(aret)
            );
            aclrtFree(localBuf);
            return -1;
        }
        h->host_ctx.windowsIn[p] = reinterpret_cast<uint64_t>(peerVa);
    }

    return 0;
}

}  // namespace

extern "C" int comm_alloc_windows(CommHandle h, size_t win_size, uint64_t *device_ctx_out) try {
    if (!h || !device_ctx_out) return -1;

    // Path D: DIY symmetric pool on stable ACL IPC + EnablePeerAccess.
    // Replaced the prior HcclAllocComResourceByTiling reverse-parse path
    // (broken on CANN 9.0 due to HcclOpResParam ABI drift; see project
    // history). One backend, works on 8.5 and 9.0 unchanged.
    const uint64_t effective_win_size = win_size != 0 ? static_cast<uint64_t>(win_size) : kDefaultIpcWinSize;
    if (alloc_windows_via_ipc(h, effective_win_size) != 0) return -1;

    // Optional PTO-ISA async SDMA workspace pre-allocation. This is a
    // CANN 9.0+ feature at runtime: aclnnShmemSdmaStarsQuery does not
    // exist in libopapi.so on 8.5, so SdmaWorkspaceManager::Init() fails
    // at dlsym and we fall back to workSpace == 0 (demos that need this
    // field check it and skip). The macro itself only gates the build-
    // time dependency on PTO_ISA_ROOT. Located here so it overlays the
    // comm backend's output -- comm-side flow does not care about
    // workSpace.
#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
    h->sdma_workspace = std::make_unique<pto::comm::sdma::SdmaWorkspaceManager>();
    if (h->sdma_workspace->Init()) {
        h->host_ctx.workSpace = reinterpret_cast<uint64_t>(h->sdma_workspace->GetWorkspaceAddr());
        h->host_ctx.workSpaceSize = 16 * 1024;
    } else {
        // On CANN 8.5 the dlsym for the aclnn ops fails by design; demote
        // to "no SDMA workspace" rather than failing the whole comm init.
        // Demos that require workSpace != 0 will skip themselves.
        h->sdma_workspace.reset();
    }
#endif

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
    fprintf(stderr, "[comm] comm_alloc_windows: exception: %s\n", e.what());
    return -1;
} catch (...) {
    fprintf(stderr, "[comm] comm_alloc_windows: unknown exception\n");
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

extern "C" int comm_barrier(CommHandle h) {
    if (!h) return -1;
    // HcclBarrier is synchronous — it blocks until all ranks arrive.
    // Do NOT call aclrtSynchronizeStream after it: HcclBarrier internally
    // switches the thread's ACL context, which invalidates the caller-owned
    // stream for context-checked ACL calls (error 507018).
    HcclResult hret = hccl_barrier(h->hccl_comm, h->stream);
    if (hret != HCCL_SUCCESS) {
        fprintf(stderr, "[comm rank %d] HcclBarrier failed: %d\n", h->rank, static_cast<int>(hret));
        return static_cast<int>(hret);
    }
    return 0;
}

extern "C" int comm_destroy(CommHandle h) try {
    if (!h) return -1;

    // Final barrier is best-effort: if a peer already crashed we still need to
    // release the local resources we own, so timeout just logs and proceeds.
    int rc = 0;
    if (!file_barrier(h->rootinfo_path, h->rank, h->nranks, "destroy", h->run_token)) {
        fprintf(
            stderr, "[comm rank %d] comm_destroy: final barrier timed out; releasing local state anyway\n", h->rank
        );
        rc = -1;
    }

    if (h->owns_device_ctx && h->device_ctx) {
        aclrtFree(h->device_ctx);
    }
    if (h->hccl_comm) {
        HcclResult hret = hccl_comm_destroy(h->hccl_comm);
        if (hret != HCCL_SUCCESS) {
            fprintf(stderr, "[comm rank %d] HcclCommDestroy failed: %d\n", h->rank, static_cast<int>(hret));
            if (rc == 0) rc = -1;
        }
    }

    // NOTE: we do NOT destroy h->stream — it is caller-owned.
    // We also do NOT call aclrtResetDevice / aclFinalize here.  Device/ACL
    // lifecycle belongs to DeviceRunner, whose finalize() releases all
    // device memory before resetting the device and running aclFinalize.

    if (h->rank == 0) {
        cleanup_handshake_files(h->rootinfo_path);
    }

    delete h;
    return rc;
} catch (const std::exception &e) {
    fprintf(stderr, "[comm] comm_destroy: exception: %s\n", e.what());
    if (h) delete h;
    return -1;
} catch (...) {
    fprintf(stderr, "[comm] comm_destroy: unknown exception\n");
    if (h) delete h;
    return -1;
}
