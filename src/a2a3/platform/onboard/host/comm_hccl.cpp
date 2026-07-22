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
 * and ACL VMM for the per-rank symmetric window pool. A3 uses Fabric V2
 * handles; platforms without Fabric support
 * use the legacy VMM IPC handle.
 * Scope: L3 single-host multi-card only. Cross-host (L4) deployments also need a
 * cross-host launcher and control plane.
 */

#include "platform_comm/comm.h"
#include "platform_comm/comm_context.h"
#include "pto_runtime_c_api.h"

#include "common/unified_log.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
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

struct VmmWindow {
    void *base = nullptr;
    aclrtDrvMemHandle handle = nullptr;
    uint64_t size = 0;

    VmmWindow() = default;
    ~VmmWindow();
    VmmWindow(const VmmWindow &) = delete;
    VmmWindow &operator=(const VmmWindow &) = delete;
    VmmWindow(VmmWindow &&other) noexcept;
    VmmWindow &operator=(VmmWindow &&other) noexcept;
};

// Per-domain dynamic allocation. One of these exists per
// orch.allocate_domain call and owns every VMM mapping for that domain.
struct DomainAllocation {
    ~DomainAllocation();

    int rank = 0;
    int nranks = 0;
    VmmWindow local_window;
    std::vector<VmmWindow> peer_windows;
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
    VmmWindow base_local_window;
    std::vector<VmmWindow> base_peer_windows;
    std::vector<CommContext *> derived_contexts;
    std::unordered_map<uint64_t, std::unique_ptr<DomainAllocation>> domain_allocations;
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

static void record_first_error(aclError status, aclError *first_error) {
    if (*first_error == ACL_SUCCESS && status != ACL_SUCCESS) {
        *first_error = status;
    }
}

static aclError release_vmm_window(VmmWindow *window) {
    aclError first_error = ACL_SUCCESS;
    if (window->base != nullptr) {
        record_first_error(aclrtUnmapMem(window->base), &first_error);
        record_first_error(aclrtReleaseMemAddress(window->base), &first_error);
        window->base = nullptr;
    }
    if (window->handle != nullptr) {
        record_first_error(aclrtFreePhysical(window->handle), &first_error);
        window->handle = nullptr;
    }
    window->size = 0;
    return first_error;
}

static aclError release_vmm_windows(std::vector<VmmWindow> *windows) {
    aclError first_error = ACL_SUCCESS;
    for (auto &window : *windows) {
        record_first_error(release_vmm_window(&window), &first_error);
    }
    windows->clear();
    return first_error;
}

static aclError reserve_and_map_vmm_window(
    int32_t device_id, uint64_t size, aclrtDrvMemHandle handle, uint64_t page_type, VmmWindow *window
) {
    void *base = nullptr;
    aclError status = aclrtReserveMemAddress(&base, size, 0, nullptr, page_type);
    if (status != ACL_SUCCESS) return status;

    status = aclrtMapMem(base, size, 0, handle, 0);
    if (status != ACL_SUCCESS) {
        aclrtReleaseMemAddress(base);
        return status;
    }

    aclrtMemAccessDesc access_desc{};
    access_desc.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
    access_desc.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = static_cast<uint32_t>(device_id);
    status = aclrtMemSetAccess(base, size, &access_desc, 1);
    if (status != ACL_SUCCESS) {
        aclrtUnmapMem(base);
        aclrtReleaseMemAddress(base);
        return status;
    }

    window->base = base;
    window->handle = handle;
    window->size = size;
    return ACL_SUCCESS;
}

static aclError create_local_vmm_window(
    int32_t device_id, uint64_t requested_size, decltype(aclrtPhysicalMemProp{}.memAttr) mem_attr, uint64_t page_type,
    VmmWindow *window
) {
    aclrtPhysicalMemProp prop{};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = static_cast<uint32_t>(device_id);
    prop.memAttr = mem_attr;

    size_t granularity = 0;
    aclError status = aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity);
    if (status != ACL_SUCCESS) return status;
    if (granularity == 0 || requested_size > std::numeric_limits<uint64_t>::max() - (granularity - 1)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint64_t aligned_size = ((requested_size + granularity - 1) / granularity) * granularity;

    aclrtDrvMemHandle handle = nullptr;
    status = aclrtMallocPhysical(&handle, aligned_size, &prop, 0);
    if (status != ACL_SUCCESS) return status;

    status = reserve_and_map_vmm_window(device_id, aligned_size, handle, page_type, window);
    if (status != ACL_SUCCESS) aclrtFreePhysical(handle);
    return status;
}

static aclError create_local_fabric_window(int32_t device_id, uint64_t requested_size, VmmWindow *window) {
    return create_local_vmm_window(device_id, requested_size, ACL_HBM_MEM_HUGE, HUGE_PAGE_TYPE, window);
}

static aclError create_local_ipc_window(int32_t device_id, uint64_t requested_size, VmmWindow *window) {
    return create_local_vmm_window(device_id, requested_size, ACL_HBM_MEM_NORMAL, 0, window);
}

static aclError export_fabric_window(const VmmWindow &window, aclrtMemFabricHandle *fabric_handle) {
    return aclrtMemExportToShareableHandleV2(
        window.handle, ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC,
        fabric_handle
    );
}

static aclError
import_fabric_window(int32_t device_id, const aclrtMemFabricHandle &fabric_handle, uint64_t size, VmmWindow *window) {
    aclrtMemFabricHandle mutable_handle = fabric_handle;
    aclrtDrvMemHandle imported_handle = nullptr;
    aclError status = aclrtMemImportFromShareableHandleV2(
        &mutable_handle, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT, &imported_handle
    );
    if (status != ACL_SUCCESS) return status;

    status = reserve_and_map_vmm_window(device_id, size, imported_handle, HUGE_PAGE_TYPE, window);
    if (status != ACL_SUCCESS) aclrtFreePhysical(imported_handle);
    return status;
}

static aclError export_ipc_window(const VmmWindow &window, uint64_t *shareable_handle) {
    return aclrtMemExportToShareableHandle(
        window.handle, ACL_MEM_HANDLE_TYPE_NONE, ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION, shareable_handle
    );
}

static aclError import_ipc_window(int32_t device_id, uint64_t shareable_handle, uint64_t size, VmmWindow *window) {
    aclrtDrvMemHandle imported_handle = nullptr;
    aclError status = aclrtMemImportFromShareableHandle(shareable_handle, device_id, &imported_handle);
    if (status != ACL_SUCCESS) return status;

    status = reserve_and_map_vmm_window(device_id, size, imported_handle, 0, window);
    if (status != ACL_SUCCESS) aclrtFreePhysical(imported_handle);
    return status;
}

static aclError release_domain_windows(DomainAllocation *alloc) {
    aclError first_error = release_vmm_windows(&alloc->peer_windows);
    record_first_error(release_vmm_window(&alloc->local_window), &first_error);
    return first_error;
}

static aclError release_base_windows(CommHandle h) {
    aclError first_error = release_vmm_windows(&h->base_peer_windows);
    record_first_error(release_vmm_window(&h->base_local_window), &first_error);
    const uint64_t workspace = h->host_ctx.workSpace;
    const uint64_t workspace_size = h->host_ctx.workSpaceSize;
    h->host_ctx = CommContext{};
    h->host_ctx.workSpace = workspace;
    h->host_ctx.workSpaceSize = workspace_size;
    return first_error;
}

}  // namespace

VmmWindow::~VmmWindow() { release_vmm_window(this); }

VmmWindow::VmmWindow(VmmWindow &&other) noexcept :
    base(std::exchange(other.base, nullptr)),
    handle(std::exchange(other.handle, nullptr)),
    size(std::exchange(other.size, 0)) {}

VmmWindow &VmmWindow::operator=(VmmWindow &&other) noexcept {
    if (this != &other) {
        release_vmm_window(this);
        base = std::exchange(other.base, nullptr);
        handle = std::exchange(other.handle, nullptr);
        size = std::exchange(other.size, 0);
    }
    return *this;
}

DomainAllocation::~DomainAllocation() {
    if (device_ctx != nullptr) {
        aclrtFree(device_ctx);
        device_ctx = nullptr;
    }
}

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

// Build the per-rank symmetric pool with ACL VMM and Fabric V2 handles.

// Default per-rank symmetric pool size when comm_alloc_windows is called
// with win_size == 0. Picked to match the HCCL_BUFFSIZE default of the
// pre-Path-D backend so existing callers see no behavioural change.
constexpr uint64_t kDefaultWinSize = 200ULL * 1024 * 1024;
constexpr uint64_t kFabricAnnounceMagic = 0x464142334c334d45ULL;  // "FAB3L3ME"
constexpr uint32_t kFabricAnnounceVersion = 1;

enum class FabricAttempt { kSuccess, kUnsupported, kError };

struct FabricAnnounceFile {
    uint64_t magic = kFabricAnnounceMagic;
    uint32_t version = kFabricAnnounceVersion;
    uint32_t rank = 0;
    int32_t pid = 0;
    int32_t device_id = -1;
    uint32_t handle_type = static_cast<uint32_t>(ACL_MEM_SHARE_HANDLE_TYPE_FABRIC);
    uint32_t handle_size = sizeof(aclrtMemFabricHandle);
    uint64_t mapping_size = 0;
    aclrtMemFabricHandle fabric_handle{};
};

constexpr uint64_t kIpcAnnounceMagic = 0x49504344334d4549ULL;  // "IPCD3MEI"
constexpr uint32_t kIpcAnnounceVersion = 1;

struct IpcAnnounceFile {
    uint64_t magic = kIpcAnnounceMagic;
    uint32_t version = kIpcAnnounceVersion;
    uint32_t rank = 0;
    int32_t pid = 0;
    int32_t device_id = -1;
    uint64_t mapping_size = 0;
    uint64_t shareable_handle = 0;
};

static bool write_ipc_announce(const std::string &path, const IpcAnnounceFile &announce) {
    const std::string tmp = path + ".tmp." + std::to_string(getpid());
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char *>(&announce), sizeof(announce));
        if (!f.good()) {
            std::remove(tmp.c_str());
            return false;
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

static bool
read_ipc_announce(const std::string &path, uint32_t expected_rank, IpcAnnounceFile *out, int timeout_sec = 60) {
    for (int i = 0; i < timeout_sec * 10; ++i) {
        std::ifstream f(path, std::ios::binary);
        if (f.good()) {
            IpcAnnounceFile announce{};
            f.read(reinterpret_cast<char *>(&announce), sizeof(announce));
            if (f.good() && announce.magic == kIpcAnnounceMagic && announce.version == kIpcAnnounceVersion &&
                announce.rank == expected_rank && announce.mapping_size != 0) {
                *out = announce;
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

static bool
enable_ipc_peer_access(int log_rank, int local_rank, int32_t device_id, const std::vector<IpcAnnounceFile> &peers) {
    for (size_t p = 0; p < peers.size(); ++p) {
        if (static_cast<int>(p) == local_rank) continue;
        const aclError status = aclrtDeviceEnablePeerAccess(peers[p].device_id, 0);
        if (status != ACL_SUCCESS) {
            LOG_WARN(
                "[comm rank %d] ipc: EnablePeerAccess(peer_dev=%d) -> %d", log_rank, peers[p].device_id,
                static_cast<int>(status)
            );
        }
    }

    for (size_t p = 0; p < peers.size(); ++p) {
        if (static_cast<int>(p) == local_rank) continue;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (true) {
            int32_t enabled = 0;
            const aclError status = aclrtDevicePeerAccessStatus(device_id, peers[p].device_id, &enabled);
            if (status != ACL_SUCCESS) {
                LOG_ERROR(
                    "[comm rank %d] ipc: PeerAccessStatus(local_dev=%d peer_dev=%d) -> %d", log_rank, device_id,
                    peers[p].device_id, static_cast<int>(status)
                );
                return false;
            }
            if (enabled == 1) break;
            if (std::chrono::steady_clock::now() >= deadline) {
                LOG_WARN(
                    "[comm rank %d] ipc: P2P status unconfirmed peer=%zu peer_dev=%d", log_rank, p, peers[p].device_id
                );
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    return true;
}

// Announce file path shares the `barrier_<prefix>_..._<rank>.ready` shape so
// cleanup_handshake_files picks it up alongside the file_barrier markers.
// Without this convention these files would accumulate across re-runs.
static std::string fabric_announce_path(const std::string &rootinfo, int rank, uint64_t run_token) {
    return handshake_dir(rootinfo) + "/barrier_" + handshake_prefix(rootinfo) + "_fabric_announce_" +
           run_token_hex(run_token) + "_" + std::to_string(rank) + ".ready";
}

static bool fabric_write_announce(
    const std::string &rootinfo, int rank, uint64_t run_token, int32_t pid, int32_t device_id, uint64_t mapping_size,
    const aclrtMemFabricHandle &fabric_handle
) {
    FabricAnnounceFile a{};
    a.pid = pid;
    a.rank = static_cast<uint32_t>(rank);
    a.device_id = device_id;
    a.mapping_size = mapping_size;
    a.fabric_handle = fabric_handle;
    std::string p = fabric_announce_path(rootinfo, rank, run_token);
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

static bool fabric_read_announce(
    const std::string &rootinfo, int peer, uint64_t run_token, FabricAnnounceFile *out, int timeout_sec = 60
) {
    std::string p = fabric_announce_path(rootinfo, peer, run_token);
    for (int i = 0; i < timeout_sec * 10; ++i) {
        std::ifstream f(p, std::ios::binary);
        if (f.good()) {
            FabricAnnounceFile a{};
            f.read(reinterpret_cast<char *>(&a), sizeof(a));
            if (f.good() && a.magic == kFabricAnnounceMagic && a.version == kFabricAnnounceVersion &&
                a.rank == static_cast<uint32_t>(peer) &&
                a.handle_type == static_cast<uint32_t>(ACL_MEM_SHARE_HANDLE_TYPE_FABRIC) &&
                a.handle_size == sizeof(aclrtMemFabricHandle) && a.mapping_size != 0) {
                *out = a;
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

static std::string ipc_announce_path(const std::string &rootinfo, int rank, uint64_t run_token) {
    return handshake_dir(rootinfo) + "/barrier_" + handshake_prefix(rootinfo) + "_ipc_announce_" +
           run_token_hex(run_token) + "_" + std::to_string(rank) + ".ready";
}

static int alloc_windows_via_ipc(CommHandle h, uint64_t win_size) {
    const int rank = h->rank;
    const int nranks = h->nranks;

    int32_t device_id = -1;
    if (aclrtGetDevice(&device_id) != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] ipc: aclrtGetDevice failed", rank);
        return -1;
    }

    VmmWindow local_window;
    aclError status = create_local_ipc_window(device_id, win_size, &local_window);
    if (status != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] ipc: create local window -> %d", rank, static_cast<int>(status));
        return -1;
    }

    uint64_t shareable_handle = 0;
    status = export_ipc_window(local_window, &shareable_handle);
    if (status != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] ipc: ExportToShareableHandle -> %d", rank, static_cast<int>(status));
        return -1;
    }

    IpcAnnounceFile local_announce{};
    local_announce.pid = static_cast<int32_t>(getpid());
    local_announce.rank = static_cast<uint32_t>(rank);
    local_announce.device_id = device_id;
    local_announce.mapping_size = local_window.size;
    local_announce.shareable_handle = shareable_handle;
    if (!write_ipc_announce(ipc_announce_path(h->rootinfo_path, rank, h->run_token), local_announce)) {
        LOG_ERROR("[comm rank %d] ipc: write_announce failed", rank);
        return -1;
    }

    std::vector<IpcAnnounceFile> peers(nranks);
    peers[rank] = local_announce;
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) continue;
        if (!read_ipc_announce(
                ipc_announce_path(h->rootinfo_path, p, h->run_token), static_cast<uint32_t>(p), &peers[p]
            )) {
            LOG_ERROR("[comm rank %d] ipc: read_announce(peer=%d) timed out or invalid", rank, p);
            return -1;
        }
        if (peers[p].mapping_size != local_window.size) {
            LOG_ERROR(
                "[comm rank %d] ipc: peer=%d mapping_size=%llu differs from local=%llu", rank, p,
                static_cast<unsigned long long>(peers[p].mapping_size),
                static_cast<unsigned long long>(local_window.size)
            );
            return -1;
        }
    }

    if (!enable_ipc_peer_access(rank, rank, device_id, peers)) return -1;
    if (!file_barrier(h->rootinfo_path, rank, nranks, "ipc_p2p_ready", h->run_token)) return -1;

    CommContext ctx{};
    ctx.rankId = static_cast<uint32_t>(rank);
    ctx.rankNum = static_cast<uint32_t>(nranks);
    ctx.winSize = local_window.size;
    ctx.windowsIn[rank] = reinterpret_cast<uint64_t>(local_window.base);

    std::vector<VmmWindow> peer_windows;
    peer_windows.reserve(static_cast<size_t>(nranks - 1));
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) continue;
        VmmWindow peer_window;
        status = import_ipc_window(device_id, peers[p].shareable_handle, peers[p].mapping_size, &peer_window);
        if (status != ACL_SUCCESS) {
            LOG_ERROR("[comm rank %d] ipc: import peer=%d -> %d", rank, p, static_cast<int>(status));
            return -1;
        }
        ctx.windowsIn[p] = reinterpret_cast<uint64_t>(peer_window.base);
        peer_windows.push_back(std::move(peer_window));
    }

    h->host_ctx = ctx;
    h->base_local_window = std::move(local_window);
    h->base_peer_windows = std::move(peer_windows);
    return 0;
}

// Fills h->host_ctx with rankId/rankNum/winSize/windowsIn[] via Fabric V2
// handle exchange. All local and imported mappings are owned by CommHandle.
static FabricAttempt alloc_windows_via_fabric(CommHandle h, uint64_t win_size) {
    const int rank = h->rank;
    const int nranks = h->nranks;
    const std::string &rootinfo = h->rootinfo_path;
    const uint64_t run_token = h->run_token;

    // Rank and logical device id can differ when workers use a device pool.
    int32_t myDevice = -1;
    if (aclrtGetDevice(&myDevice) != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] fabric: aclrtGetDevice failed", rank);
        return FabricAttempt::kError;
    }

    VmmWindow local_window;
    std::vector<VmmWindow> peer_windows;
    aclError aret = create_local_fabric_window(myDevice, win_size, &local_window);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] fabric: create local window -> %d", rank, static_cast<int>(aret));
        return FabricAttempt::kError;
    }

    aclrtMemFabricHandle fabric_handle{};
    aret = export_fabric_window(local_window, &fabric_handle);
    if (aret != ACL_SUCCESS) {
        if (aret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
            release_vmm_window(&local_window);
            return FabricAttempt::kUnsupported;
        }
        LOG_ERROR("[comm rank %d] fabric: ExportToShareableHandleV2 -> %d", rank, static_cast<int>(aret));
        release_vmm_window(&local_window);
        return FabricAttempt::kError;
    }

    // Publish the full Fabric handle and mapped size before importing peers.
    const int32_t myPid = static_cast<int32_t>(getpid());
    if (!fabric_write_announce(rootinfo, rank, run_token, myPid, myDevice, local_window.size, fabric_handle)) {
        LOG_ERROR("[comm rank %d] fabric: write_announce failed", rank);
        release_vmm_window(&local_window);
        return FabricAttempt::kError;
    }
    std::vector<FabricAnnounceFile> peers(nranks);
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) {
            peers[p].pid = myPid;
            peers[p].rank = static_cast<uint32_t>(rank);
            peers[p].device_id = myDevice;
            peers[p].mapping_size = local_window.size;
            peers[p].fabric_handle = fabric_handle;
            continue;
        }
        if (!fabric_read_announce(rootinfo, p, run_token, &peers[p])) {
            LOG_ERROR("[comm rank %d] fabric: read_announce(peer=%d) timed out or invalid", rank, p);
            release_vmm_window(&local_window);
            return FabricAttempt::kError;
        }
        if (peers[p].mapping_size != local_window.size) {
            LOG_ERROR(
                "[comm rank %d] fabric: peer=%d mapping_size=%llu differs from local=%llu", rank, p,
                static_cast<unsigned long long>(peers[p].mapping_size),
                static_cast<unsigned long long>(local_window.size)
            );
            release_vmm_window(&local_window);
            return FabricAttempt::kError;
        }
    }

    if (!file_barrier(rootinfo, rank, nranks, "fabric_handles_ready", run_token)) {
        release_vmm_window(&local_window);
        return FabricAttempt::kError;
    }

    // windowsOut[] is intentionally left zero: no kernel path reads it
    // (verified by grep across simpler + pto-isa). The field is kept in
    // CommContext only to preserve byte-equivalence with pto-isa's parallel
    // HcclDeviceContext declaration; removing it is gated on the F4
    // private-ization decision (see .docs/28.l3-comm/ext.01.pr-774-review.md).
    // host_ctx was value-initialized at handle construction (CommContext{}),
    // and the idempotency guard in comm_alloc_windows prevents a second entry;
    // no re-zero needed before populating it here.
    CommContext ctx{};
    ctx.rankId = static_cast<uint32_t>(rank);
    ctx.rankNum = static_cast<uint32_t>(nranks);
    ctx.winSize = local_window.size;
    ctx.windowsIn[rank] = reinterpret_cast<uint64_t>(local_window.base);

    // Import each peer's shareable handle onto our device. The symmetric pool
    // uses one win_size across ranks and rejects mismatched mapped sizes.
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) continue;
        VmmWindow peer_window;
        aret = import_fabric_window(myDevice, peers[p].fabric_handle, peers[p].mapping_size, &peer_window);
        if (aret != ACL_SUCCESS) {
            LOG_ERROR("[comm rank %d] fabric: import peer=%d -> %d", rank, p, static_cast<int>(aret));
            release_vmm_windows(&peer_windows);
            release_vmm_window(&local_window);
            return FabricAttempt::kError;
        }
        ctx.windowsIn[p] = reinterpret_cast<uint64_t>(peer_window.base);
        peer_windows.push_back(std::move(peer_window));
    }

    h->host_ctx = ctx;
    h->base_local_window = std::move(local_window);
    h->base_peer_windows = std::move(peer_windows);
    return FabricAttempt::kSuccess;
}

// ============================================================================
// Per-domain dynamic allocation (for orch.allocate_domain).
//
// Same Fabric V2 exchange as alloc_windows_via_fabric, but on a fresh per-allocation
// local buffer.  Every barrier filename and announce filename is scoped by
// allocation_id so concurrent allocations from different orch.allocate_domain
// calls do not collide.  Participation is by subset (domain_rank within
// rank_count), so non-members of the subset are not involved.
// ============================================================================

// Announce file path scoped by allocation_id so two concurrent allocations
// from different orch calls do not collide.  Same dir + cleanup-friendly
// prefix as the base-comm Fabric announce.
static std::string domain_fabric_announce_path(
    const std::string &rootinfo, uint64_t allocation_id, uint32_t domain_rank, uint64_t run_token
) {
    return handshake_dir(rootinfo) + "/barrier_" + handshake_prefix(rootinfo) + "_alloc_" +
           std::to_string(allocation_id) + "_fabric_announce_" + run_token_hex(run_token) + "_" +
           std::to_string(domain_rank) + ".ready";
}

static bool domain_write_announce(
    const std::string &rootinfo, uint64_t allocation_id, uint32_t domain_rank, uint64_t run_token, int32_t pid,
    int32_t device_id, uint64_t mapping_size, const aclrtMemFabricHandle &fabric_handle
) {
    FabricAnnounceFile a{};
    a.pid = pid;
    a.rank = domain_rank;
    a.device_id = device_id;
    a.mapping_size = mapping_size;
    a.fabric_handle = fabric_handle;
    std::string p = domain_fabric_announce_path(rootinfo, allocation_id, domain_rank, run_token);
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
    FabricAnnounceFile *out, int timeout_sec = 60
) {
    std::string p = domain_fabric_announce_path(rootinfo, allocation_id, peer_domain_rank, run_token);
    for (int i = 0; i < timeout_sec * 10; ++i) {
        std::ifstream f(p, std::ios::binary);
        if (f.good()) {
            FabricAnnounceFile a{};
            f.read(reinterpret_cast<char *>(&a), sizeof(a));
            if (f.good() && a.magic == kFabricAnnounceMagic && a.version == kFabricAnnounceVersion &&
                a.rank == peer_domain_rank &&
                a.handle_type == static_cast<uint32_t>(ACL_MEM_SHARE_HANDLE_TYPE_FABRIC) &&
                a.handle_size == sizeof(aclrtMemFabricHandle) && a.mapping_size != 0) {
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

static std::string domain_ipc_announce_path(
    const std::string &rootinfo, uint64_t allocation_id, uint32_t domain_rank, uint64_t run_token
) {
    return handshake_dir(rootinfo) + "/barrier_" + handshake_prefix(rootinfo) + "_alloc_" +
           std::to_string(allocation_id) + "_ipc_announce_" + run_token_hex(run_token) + "_" +
           std::to_string(domain_rank) + ".ready";
}

static int domain_alloc_via_ipc(
    CommHandle h, uint64_t allocation_id, const uint32_t *, size_t rank_count, uint32_t domain_rank, uint64_t win_size,
    DomainAllocation *out
) {
    const int subset_n = static_cast<int>(rank_count);
    const int local_rank = static_cast<int>(domain_rank);

    int32_t device_id = -1;
    if (aclrtGetDevice(&device_id) != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain ipc: aclrtGetDevice failed", h->rank);
        return -1;
    }

    VmmWindow local_window;
    aclError status = create_local_ipc_window(device_id, win_size, &local_window);
    if (status != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain ipc: create local window -> %d", h->rank, static_cast<int>(status));
        return -1;
    }

    uint64_t shareable_handle = 0;
    status = export_ipc_window(local_window, &shareable_handle);
    if (status != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain ipc: ExportToShareableHandle -> %d", h->rank, static_cast<int>(status));
        return -1;
    }

    IpcAnnounceFile local_announce{};
    local_announce.pid = static_cast<int32_t>(getpid());
    local_announce.rank = domain_rank;
    local_announce.device_id = device_id;
    local_announce.mapping_size = local_window.size;
    local_announce.shareable_handle = shareable_handle;
    if (!write_ipc_announce(
            domain_ipc_announce_path(h->rootinfo_path, allocation_id, domain_rank, h->run_token), local_announce
        )) {
        LOG_ERROR("[comm rank %d] alloc_domain ipc: write_announce failed", h->rank);
        return -1;
    }

    std::vector<IpcAnnounceFile> peers(static_cast<size_t>(subset_n));
    peers[static_cast<size_t>(local_rank)] = local_announce;
    for (int p = 0; p < subset_n; ++p) {
        if (p == local_rank) continue;
        if (!read_ipc_announce(
                domain_ipc_announce_path(h->rootinfo_path, allocation_id, static_cast<uint32_t>(p), h->run_token),
                static_cast<uint32_t>(p), &peers[static_cast<size_t>(p)]
            )) {
            LOG_ERROR("[comm rank %d] alloc_domain ipc: read_announce(peer_dr=%d) timed out or invalid", h->rank, p);
            return -1;
        }
        if (peers[static_cast<size_t>(p)].mapping_size != local_window.size) {
            LOG_ERROR(
                "[comm rank %d] alloc_domain ipc: peer_dr=%d mapping_size=%llu differs from local=%llu", h->rank, p,
                static_cast<unsigned long long>(peers[static_cast<size_t>(p)].mapping_size),
                static_cast<unsigned long long>(local_window.size)
            );
            return -1;
        }
    }

    if (!enable_ipc_peer_access(h->rank, local_rank, device_id, peers)) return -1;
    if (!file_barrier(
            h->rootinfo_path, local_rank, subset_n, domain_barrier_tag(allocation_id, "ipc_p2p_ready"), h->run_token
        )) {
        return -1;
    }

    CommContext ctx{};
    ctx.rankId = domain_rank;
    ctx.rankNum = static_cast<uint32_t>(subset_n);
    ctx.winSize = local_window.size;
    ctx.windowsIn[local_rank] = reinterpret_cast<uint64_t>(local_window.base);

    std::vector<VmmWindow> peer_windows;
    peer_windows.reserve(static_cast<size_t>(subset_n - 1));
    for (int p = 0; p < subset_n; ++p) {
        if (p == local_rank) continue;
        VmmWindow peer_window;
        const IpcAnnounceFile &peer = peers[static_cast<size_t>(p)];
        status = import_ipc_window(device_id, peer.shareable_handle, peer.mapping_size, &peer_window);
        if (status != ACL_SUCCESS) {
            LOG_ERROR("[comm rank %d] alloc_domain ipc: import peer_dr=%d -> %d", h->rank, p, static_cast<int>(status));
            return -1;
        }
        ctx.windowsIn[p] = reinterpret_cast<uint64_t>(peer_window.base);
        peer_windows.push_back(std::move(peer_window));
    }

    void *new_dev_mem = nullptr;
    status = aclrtMalloc(&new_dev_mem, sizeof(CommContext), ACL_MEM_MALLOC_HUGE_FIRST);
    if (status != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain ipc: ctx aclrtMalloc -> %d", h->rank, static_cast<int>(status));
        return -1;
    }
    status = aclrtMemcpy(new_dev_mem, sizeof(CommContext), &ctx, sizeof(CommContext), ACL_MEMCPY_HOST_TO_DEVICE);
    if (status != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain ipc: ctx Memcpy H2D -> %d", h->rank, static_cast<int>(status));
        aclrtFree(new_dev_mem);
        return -1;
    }

    out->rank = local_rank;
    out->nranks = subset_n;
    out->local_window = std::move(local_window);
    out->peer_windows = std::move(peer_windows);
    out->device_ctx = reinterpret_cast<CommContext *>(new_dev_mem);
    return 0;
}

// Host wrapper owning one SDMA-enabled Worker's provisioned resources. The
// opaque handle returned to the runner IS this manager; dma_workspace_release()
// destroys it. There is no per-device generation gate: an SDMA-enabled Worker
// owns its provider for its whole life and releases it at finalize.
#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
using SdmaManager = pto::comm::sdma::SdmaWorkspaceManager;
#endif

extern "C" uint32_t dma_workspace_supported_mask(void) {
#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
    return uint32_t{1} << DMA_WORKSPACE_SDMA;
#else
    return 0;
#endif
}

extern "C" int dma_workspace_provision(uint32_t required_mask, uint64_t *addr_out, int count, void **handle_out) {
    if (!addr_out || !handle_out || count < 0) return -1;
    *handle_out = nullptr;
    for (int i = 0; i < count; ++i)
        addr_out[i] = 0;
    const uint32_t supported = dma_workspace_supported_mask();
    if ((required_mask & ~supported) != 0) return -1;
#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
    if ((required_mask & (uint32_t{1} << DMA_WORKSPACE_SDMA)) == 0) return 0;
    if (count <= DMA_WORKSPACE_SDMA) return -1;
    try {
        auto manager = std::make_unique<SdmaManager>();
        bool init_ok = false;
        try {
            // Init creates the 48 STARS streams + 16KB workspace. It may fail
            // after creating only a subset; destructing that partial manager on
            // an error-state card can itself stall, so leak it on failure rather
            // than risk the stall.
            init_ok = manager->Init();
        } catch (...) {
            LOG_ERROR("SdmaWorkspaceManager::Init threw; abandoning its partial resources");
        }
        if (!init_ok) {
            (void)manager.release();
            return -1;
        }
        const uint64_t addr = reinterpret_cast<uint64_t>(manager->GetWorkspaceAddr());
        if (addr == 0) {
            (void)manager.release();
            LOG_ERROR("dma_workspace_provision: manager returned a null workspace address");
            return -1;
        }
        addr_out[DMA_WORKSPACE_SDMA] = addr;
        *handle_out = manager.release();
        return 0;
    } catch (...) {
        LOG_ERROR("dma_workspace_provision: exception while provisioning SDMA");
        return -1;
    }
#else
    return required_mask == 0 ? 0 : -1;
#endif
}

extern "C" void dma_workspace_release(void *handle) {
#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
    if (!handle) return;
    try {
        std::unique_ptr<SdmaManager> manager(static_cast<SdmaManager *>(handle));
        manager.reset();
    } catch (...) {
        LOG_ERROR("dma_workspace_release: exception while releasing SDMA resources");
    }
#else
    (void)handle;
#endif
}

// Performs the per-allocation Fabric V2 exchange for one subset rank. rank_ids
// must list participating BASE-COMM rank ids in domain rank order; this
// rank's domain_rank must match its base rank for the same invariant
// alloc_windows_via_fabric relies on (rank_ids[domain_rank] == h->rank).
//
// Failure paths tear down the local VMM window plus every peer import already
// recorded on `out`. On success
// the peer imports live on `out->peer_windows` for comm_release_domain_windows.
static FabricAttempt domain_alloc_via_fabric(
    CommHandle h, uint64_t allocation_id, const uint32_t *, size_t rank_count, uint32_t domain_rank, uint64_t win_size,
    DomainAllocation *out
) {
    const std::string &rootinfo = h->rootinfo_path;
    const uint64_t run_token = h->run_token;
    const int subset_n = static_cast<int>(rank_count);
    const int my_dr = static_cast<int>(domain_rank);

    int32_t myDevice = -1;
    if (aclrtGetDevice(&myDevice) != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: aclrtGetDevice failed", h->rank);
        return FabricAttempt::kError;
    }

    aclError aret = create_local_fabric_window(myDevice, win_size, &out->local_window);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: create local Fabric window -> %d", h->rank, static_cast<int>(aret));
        return FabricAttempt::kError;
    }

    aclrtMemFabricHandle fabric_handle{};
    aret = export_fabric_window(out->local_window, &fabric_handle);
    if (aret != ACL_SUCCESS) {
        if (aret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
            release_domain_windows(out);
            return FabricAttempt::kUnsupported;
        }
        LOG_ERROR("[comm rank %d] alloc_domain: ExportToShareableHandleV2 -> %d", h->rank, static_cast<int>(aret));
        release_domain_windows(out);
        return FabricAttempt::kError;
    }

    const int32_t myPid = static_cast<int32_t>(getpid());
    if (!domain_write_announce(
            rootinfo, allocation_id, domain_rank, run_token, myPid, myDevice, out->local_window.size, fabric_handle
        )) {
        LOG_ERROR("[comm rank %d] alloc_domain: write_announce failed", h->rank);
        release_domain_windows(out);
        return FabricAttempt::kError;
    }
    std::vector<FabricAnnounceFile> peers(subset_n);
    for (int p = 0; p < subset_n; ++p) {
        if (p == my_dr) {
            peers[p].pid = myPid;
            peers[p].rank = domain_rank;
            peers[p].device_id = myDevice;
            peers[p].mapping_size = out->local_window.size;
            peers[p].fabric_handle = fabric_handle;
            continue;
        }
        if (!domain_read_announce(rootinfo, allocation_id, static_cast<uint32_t>(p), run_token, &peers[p])) {
            LOG_ERROR("[comm rank %d] alloc_domain: read_announce(peer_dr=%d) timed out or invalid", h->rank, p);
            release_domain_windows(out);
            return FabricAttempt::kError;
        }
        if (peers[p].mapping_size != out->local_window.size) {
            LOG_ERROR(
                "[comm rank %d] alloc_domain: peer_dr=%d mapping_size=%llu differs from local=%llu", h->rank, p,
                static_cast<unsigned long long>(peers[p].mapping_size),
                static_cast<unsigned long long>(out->local_window.size)
            );
            release_domain_windows(out);
            return FabricAttempt::kError;
        }
    }

    if (!file_barrier(rootinfo, my_dr, subset_n, domain_barrier_tag(allocation_id, "fabric_ready"), run_token)) {
        release_domain_windows(out);
        return FabricAttempt::kError;
    }

    out->rank = my_dr;
    out->nranks = subset_n;
    // Build a host-side CommContext for the subset and upload it as device_ctx.
    // Async-engine workspaces are runtime-owned and injected through each
    // kernel's GlobalContext, independent of this communication domain.
    CommContext ctx{};
    ctx.rankId = domain_rank;
    ctx.rankNum = static_cast<uint32_t>(subset_n);
    ctx.winSize = out->local_window.size;
    ctx.windowsIn[my_dr] = reinterpret_cast<uint64_t>(out->local_window.base);
    for (int p = 0; p < subset_n; ++p) {
        if (p == my_dr) continue;
        VmmWindow peer_window;
        aret = import_fabric_window(myDevice, peers[p].fabric_handle, peers[p].mapping_size, &peer_window);
        if (aret != ACL_SUCCESS) {
            LOG_ERROR(
                "[comm rank %d] alloc_domain: import Fabric peer_dr=%d -> %d", h->rank, p, static_cast<int>(aret)
            );
            release_domain_windows(out);
            return FabricAttempt::kError;
        }
        ctx.windowsIn[p] = reinterpret_cast<uint64_t>(peer_window.base);
        out->peer_windows.push_back(std::move(peer_window));
    }

    void *newDevMem = nullptr;
    aret = aclrtMalloc(&newDevMem, sizeof(CommContext), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: ctx aclrtMalloc -> %d", h->rank, static_cast<int>(aret));
        release_domain_windows(out);
        return FabricAttempt::kError;
    }
    aret = aclrtMemcpy(newDevMem, sizeof(CommContext), &ctx, sizeof(CommContext), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: ctx Memcpy H2D -> %d", h->rank, static_cast<int>(aret));
        aclrtFree(newDevMem);
        release_domain_windows(out);
        return FabricAttempt::kError;
    }
    out->device_ctx = reinterpret_cast<CommContext *>(newDevMem);
    return FabricAttempt::kSuccess;
}

}  // namespace

extern "C" int comm_alloc_windows(CommHandle h, size_t win_size, uint64_t *device_ctx_out) try {
    if (!h || !device_ctx_out) return -1;

    // A comm handle owns one base window set and one device context.
    if (h->device_ctx != nullptr) {
        LOG_ERROR("[comm rank %d] comm_alloc_windows: already allocated on this handle", h->rank);
        return -1;
    }

    const uint64_t effective_win_size = win_size != 0 ? static_cast<uint64_t>(win_size) : kDefaultWinSize;
    const FabricAttempt fabric_result = alloc_windows_via_fabric(h, effective_win_size);
    if (fabric_result == FabricAttempt::kError) return -1;
    if (fabric_result == FabricAttempt::kUnsupported) {
        LOG_INFO_V0("[comm rank %d] Fabric V2 unsupported; using VMM IPC windows", h->rank);
        if (alloc_windows_via_ipc(h, effective_win_size) != 0) return -1;
    }

    void *newDevMem = nullptr;
    aclError aRet = aclrtMalloc(&newDevMem, sizeof(CommContext), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] comm_alloc_windows: ctx aclrtMalloc -> %d", h->rank, static_cast<int>(aRet));
        release_base_windows(h);
        return -1;
    }
    aRet = aclrtMemcpy(newDevMem, sizeof(CommContext), &h->host_ctx, sizeof(CommContext), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aRet != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] comm_alloc_windows: ctx Memcpy H2D -> %d", h->rank, static_cast<int>(aRet));
        aclrtFree(newDevMem);
        release_base_windows(h);
        return -1;
    }
    h->device_ctx = reinterpret_cast<CommContext *>(newDevMem);
    h->owns_device_ctx = true;
    *device_ctx_out = reinterpret_cast<uint64_t>(h->device_ctx);
    return 0;
} catch (const std::exception &e) {
    LOG_ERROR("[comm] comm_alloc_windows: exception: %s", e.what());
    if (h) release_base_windows(h);
    return -1;
} catch (...) {
    LOG_ERROR("[comm] comm_alloc_windows: unknown exception");
    if (h) release_base_windows(h);
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
    // dynamic alloc path creates its own per-allocation VMM window.
    if (h->rootinfo_path.empty() || h->hccl_comm == nullptr) {
        LOG_ERROR("[comm rank %d] alloc_domain: base communicator not initialised", h->rank);
        return -1;
    }

    auto alloc = std::make_unique<DomainAllocation>();
    const FabricAttempt fabric_result =
        domain_alloc_via_fabric(h, allocation_id, rank_ids, rank_count, domain_rank, window_size, alloc.get());
    if (fabric_result == FabricAttempt::kError) return -1;
    if (fabric_result == FabricAttempt::kUnsupported) {
        LOG_INFO_V0("[comm rank %d] Fabric V2 unsupported; using VMM IPC domain windows", h->rank);
        const int rc =
            domain_alloc_via_ipc(h, allocation_id, rank_ids, rank_count, domain_rank, window_size, alloc.get());
        if (rc != 0) return rc;
    }

    // Zero the freshly-allocated local pool so kernels do not observe stale
    // bytes (parity with the sim backend's memset). The full granularity-aligned
    // mapped range is zeroed to match ctx.winSize.
    aclError aret = aclrtMemset(alloc->local_window.base, alloc->local_window.size, 0, alloc->local_window.size);
    if (aret != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] alloc_domain: aclrtMemset -> %d", h->rank, static_cast<int>(aret));
        aclrtFree(alloc->device_ctx);
        alloc->device_ctx = nullptr;
        release_domain_windows(alloc.get());
        return -1;
    }

    *device_ctx_out = reinterpret_cast<uint64_t>(alloc->device_ctx);
    *local_window_base_out = reinterpret_cast<uint64_t>(alloc->local_window.base);
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
        alloc->device_ctx = nullptr;
    }
    aclError release_status = release_domain_windows(alloc.get());
    if (release_status != ACL_SUCCESS) {
        LOG_ERROR(
            "[comm rank %d] release_domain: release VMM windows -> %d", h->rank, static_cast<int>(release_status)
        );
        if (rc == 0) rc = -1;
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
        if (alloc->device_ctx) {
            aclrtFree(alloc->device_ctx);
            alloc->device_ctx = nullptr;
        }
        if (release_domain_windows(alloc.get()) != ACL_SUCCESS && rc == 0) rc = -1;
    }
    h->domain_allocations.clear();
    if (release_base_windows(h) != ACL_SUCCESS && rc == 0) rc = -1;
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
