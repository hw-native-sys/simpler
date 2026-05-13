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
 * HCCL (bundled with CANN).  Handles both MESH and RING topologies
 * when extracting per-rank RDMA window addresses.
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

using CommTopo = uint32_t;

// Internal HCCL helpers are exported by libhcomm on CANN 9.x.  The public
// HCCL APIs below intentionally use the standard, non-V2 entry points to match
// the working pto-isa initialization sequence.
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle);
extern "C" HcclResult HcomGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t isSetDevice);

static inline HcclResult hccl_get_root_info(HcclRootInfo *ri) { return HcclGetRootInfo(ri); }
static inline HcclResult hccl_comm_init_root_info(uint32_t n, const HcclRootInfo *ri, uint32_t r, HcclComm *c) {
    return HcclCommInitRootInfo(n, ri, r, c);
}
static inline HcclResult hccl_create_sub_comm_config(
    HcclComm *base, uint32_t n, uint32_t *rank_ids, uint64_t sub_comm_id, uint32_t sub_comm_rank_id,
    HcclCommConfig *config, HcclComm *sub_comm
) {
    return HcclCreateSubCommConfig(base, n, rank_ids, sub_comm_id, sub_comm_rank_id, config, sub_comm);
}
static inline HcclResult hccl_get_comm_name(HcclComm c, char *name) { return HcclGetCommName(c, name); }
static inline HcclResult hccl_barrier(HcclComm c, aclrtStream s) { return HcclBarrier(c, s); }
static inline HcclResult hccl_comm_destroy(HcclComm c) { return HcclCommDestroy(c); }
static inline HcclResult hccl_alloc_com_resource(HcclComm c, void *s, void *t, void **ctx) {
    return HcclAllocComResourceByTiling(c, s, t, ctx);
}
static inline HcclResult hccl_get_comm_handle_by_group(const char *g, HcclComm *c) {
    return HcomGetCommHandleByGroup(g, c);
}
static inline HcclResult hccl_get_l0_topo_type_ex(const char *g, CommTopo *t, uint32_t f) {
    return HcomGetL0TopoTypeEx(g, t, f);
}

static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
static constexpr uint32_t COMM_TOPO_MESH = 0b1u;

// ============================================================================
// HCCL tiling structures (required by HcclAllocComResourceByTiling)
// ============================================================================

namespace {

struct Mc2CommConfigV2 {
    struct Mc2InitTilingInner {
        uint32_t version;
        uint32_t mc2HcommCnt;
        uint32_t offset[8];
        uint8_t debugMode;
        uint8_t preparePosition;
        uint16_t queueNum;
        uint16_t commBlockNum;
        uint8_t devType;
        char reserved[17];
    } init;

    struct Mc2HcommCfg {
        uint8_t skipLocalRankCopy;
        uint8_t skipBufferWindowCopy;
        uint8_t stepSize;
        char reserved[13];
        char groupName[128];
        char algConfig[128];
        uint32_t opType;
        uint32_t reduceType;
    } inner;
};
static_assert(sizeof(Mc2CommConfigV2::Mc2InitTilingInner) == 64, "Mc2InitTilingInner size drift");
static_assert(sizeof(Mc2CommConfigV2::Mc2HcommCfg) == 280, "Mc2HcommCfg size drift");
static_assert(offsetof(Mc2CommConfigV2, inner) == 64, "Mc2CommConfigV2 layout drift");

// HCCL compat structs for RING topology parsing
struct HcclSignalInfo {
    uint64_t resId;
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
    uint32_t flag;
};

struct HcclStreamInfo {
    int32_t streamIds;
    uint32_t sqIds;
    uint32_t cqIds;
    uint32_t logicCqids;
};

struct ListCommon {
    uint64_t nextHost;
    uint64_t preHost;
    uint64_t nextDevice;
    uint64_t preDevice;
};

static constexpr uint32_t COMPAT_LOCAL_NOTIFY_MAX_NUM = 64;
static constexpr uint32_t COMPAT_LOCAL_STREAM_MAX_NUM = 19;
static constexpr uint32_t COMPAT_AICPU_OP_NOTIFY_MAX_NUM = 2;

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    HcclSignalInfo localSignals[COMPAT_LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[COMPAT_LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[COMPAT_AICPU_OP_NOTIFY_MAX_NUM];
    ListCommon nextTagRes;
};

struct AlgoTopoInfo {
    uint32_t userRank;
    uint32_t userRankSize;
    int32_t deviceLogicId;
    bool isSingleMeshAggregation;
    uint32_t deviceNumPerAggregation;
    uint32_t superPodNum;
    uint32_t devicePhyId;
    uint32_t topoType;
    uint32_t deviceType;
    uint32_t serverNum;
    uint32_t meshAggregationRankSize;
    uint32_t multiModuleDiffDeviceNumMode;
    uint32_t multiSuperPodDiffServerNumMode;
    uint32_t realUserRank;
    bool isDiffDeviceModule;
    bool isDiffDeviceType;
    uint32_t gcdDeviceNumPerAggregation;
    uint32_t moduleNum;
    uint32_t isUsedRdmaRankPairNum;
    uint64_t isUsedRdmaRankPair;
    uint32_t pairLinkCounterNum;
    uint64_t pairLinkCounter;
    uint32_t nicNum;
    uint64_t nicList;
    uint64_t complanRankLength;
    uint64_t complanRank;
    uint64_t bridgeRankNum;
    uint64_t bridgeRank;
    uint64_t serverAndsuperPodRankLength;
    uint64_t serverAndsuperPodRank;
};

struct HcclOpConfig {
    uint8_t deterministic;
    uint8_t retryEnable;
    uint8_t highPerfEnable;
    uint8_t padding[5];
    uint8_t linkTimeOut[8];
    uint64_t notifyWaitTime;
    uint32_t retryHoldTime;
    uint32_t retryIntervalTime;
    bool interXLinkDisable;
    uint32_t floatOverflowMode;
    uint32_t multiQpThreshold;
};

struct RemoteResPtr {
    uint64_t nextHostPtr;
    uint64_t nextDevicePtr;
};

struct HcclMC2WorkSpace {
    uint64_t workspace;
    uint64_t workspaceSize;
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
    ListCommon nextTagRes;
};

struct HcclOpResParamHead {
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
};

struct HcclOpResParam {
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;
    uint32_t rWinOffset;
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;
    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;
    uint32_t remoteResNum;
    RemoteResPtr remoteRes[1];
};

// Layout contract with CANN 9.x libhcomm.  These asserts convert a silent
// "field offset shifted -> we read garbage from aclrtMemcpy" failure mode
// into a compile error.  If CANN upgrades change any of these, re-verify
// the struct against the new libhcomm source before bumping the numbers.
static_assert(sizeof(HcclRankRelationResV2) == 64, "HcclRankRelationResV2 size drift");
static_assert(offsetof(HcclRankRelationResV2, windowsIn) == 8, "HcclRankRelationResV2 layout drift");
static_assert(sizeof(LocalResInfoV2) == 2472, "LocalResInfoV2 size drift");
static_assert(sizeof(HcclOpResParam) == 3000, "HcclOpResParam size drift");
static_assert(offsetof(HcclOpResParam, localUsrRankId) == 16, "HcclOpResParam layout drift");
static_assert(offsetof(HcclOpResParam, rankSize) == 20, "HcclOpResParam layout drift");
static_assert(offsetof(HcclOpResParam, winSize) == 24, "HcclOpResParam layout drift");
static_assert(offsetof(HcclOpResParam, localWindowsIn) == 32, "HcclOpResParam layout drift");
static_assert(offsetof(HcclOpResParam, remoteRes) == 2984, "HcclOpResParam layout drift");

static constexpr uint32_t HCCL_MESH_RANK_NUM = 32;
struct HcclMeshOpParamHead {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
};

static_assert(sizeof(HcclMeshOpParamHead) == 32, "HcclMeshOpParamHead size drift");

// Magic numbers required by HcclAllocComResourceByTiling.  These are CANN
// internal enum values with no public header; names + comments record intent.
// Changing any of them changes the semantics of the MC2 resource request.
static constexpr uint32_t kMc2CommBlockNum = 48U;      // Hardware comm block count (A2/A3 topology)
static constexpr const char *kMc2AlgConfig = "BatchWrite=level0:fullmesh";
static constexpr uint32_t kMc2TilingVersion = 100U;    // MC2 init tiling V2 path
static constexpr uint32_t kMc2DevType = 4U;            // 910_93 / A2-A3 MC2 device type
static constexpr uint32_t kMc2OpTypeBatchWrite = HCCL_CMD_BATCH_WRITE;

}  // anonymous namespace

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
    std::vector<CommContext *> derived_contexts;
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

extern "C" CommHandle comm_create_subcomm(
    CommHandle base, uint64_t sub_comm_id, const uint32_t *rank_ids, size_t rank_count, uint32_t sub_comm_rank_id,
    void *stream
) try {
    if (base == nullptr) {
        fprintf(stderr, "[comm] comm_create_subcomm: base is null\n");
        return nullptr;
    }
    if (base->hccl_comm == nullptr) {
        fprintf(stderr, "[comm rank %d] comm_create_subcomm: base HCCL communicator is null\n", base->rank);
        return nullptr;
    }
    if (stream == nullptr) {
        fprintf(stderr, "[comm rank %d] comm_create_subcomm: caller-supplied stream is null\n", base->rank);
        return nullptr;
    }
    if (rank_ids == nullptr) {
        fprintf(stderr, "[comm rank %d] comm_create_subcomm: rank_ids is null\n", base->rank);
        return nullptr;
    }
    if (rank_count == 0 || rank_count > COMM_MAX_RANK_NUM) {
        fprintf(
            stderr, "[comm rank %d] comm_create_subcomm: invalid rank_count=%zu (max=%u)\n", base->rank, rank_count,
            COMM_MAX_RANK_NUM
        );
        return nullptr;
    }
    if (sub_comm_rank_id >= rank_count) {
        fprintf(
            stderr, "[comm rank %d] comm_create_subcomm: invalid sub_comm_rank_id=%u rank_count=%zu\n", base->rank,
            sub_comm_rank_id, rank_count
        );
        return nullptr;
    }

    for (size_t i = 0; i < rank_count; ++i) {
        if (rank_ids[i] >= static_cast<uint32_t>(base->nranks)) {
            fprintf(
                stderr, "[comm rank %d] comm_create_subcomm: rank_ids[%zu]=%u out of range [0, %d)\n", base->rank, i,
                rank_ids[i], base->nranks
            );
            return nullptr;
        }
        for (size_t j = 0; j < i; ++j) {
            if (rank_ids[j] == rank_ids[i]) {
                fprintf(stderr, "[comm rank %d] comm_create_subcomm: duplicate rank id %u\n", base->rank, rank_ids[i]);
                return nullptr;
            }
        }
    }
    if (rank_ids[sub_comm_rank_id] != static_cast<uint32_t>(base->rank)) {
        fprintf(
            stderr, "[comm rank %d] comm_create_subcomm: rank_ids[%u]=%u does not match base rank\n", base->rank,
            sub_comm_rank_id, rank_ids[sub_comm_rank_id]
        );
        return nullptr;
    }

    std::vector<uint32_t> sub_rank_ids(rank_ids, rank_ids + rank_count);
    HcclCommConfig config{};
    HcclCommConfigInit(&config);

    HcclComm sub_comm = nullptr;
    HcclComm base_comm = base->hccl_comm;
    HcclResult hret = hccl_create_sub_comm_config(
        &base_comm, static_cast<uint32_t>(sub_rank_ids.size()), sub_rank_ids.data(), sub_comm_id, sub_comm_rank_id,
        &config, &sub_comm
    );
    if (hret != HCCL_SUCCESS || sub_comm == nullptr) {
        fprintf(
            stderr, "[comm rank %d] HcclCreateSubCommConfig failed for sub_comm_id=%llu: %d\n", base->rank,
            static_cast<unsigned long long>(sub_comm_id), static_cast<int>(hret)
        );
        return nullptr;
    }

    auto *h = new (std::nothrow) CommHandle_{};
    if (!h) {
        HcclResult destroy_ret = hccl_comm_destroy(sub_comm);
        if (destroy_ret != HCCL_SUCCESS) {
            fprintf(
                stderr, "[comm rank %d] HcclCommDestroy failed after subcomm handle allocation failure: %d\n",
                base->rank, static_cast<int>(destroy_ret)
            );
        }
        return nullptr;
    }

    h->rank = static_cast<int>(sub_comm_rank_id);
    h->nranks = static_cast<int>(rank_count);
    h->rootinfo_path = base->rootinfo_path + ".subcomm." + std::to_string(sub_comm_id);
    h->run_token = base->run_token;
    h->stream = static_cast<aclrtStream>(stream);
    h->hccl_comm = sub_comm;
    return h;
} catch (const std::exception &e) {
    fprintf(stderr, "[comm] comm_create_subcomm: exception: %s\n", e.what());
    return nullptr;
} catch (...) {
    fprintf(stderr, "[comm] comm_create_subcomm: unknown exception\n");
    return nullptr;
}

extern "C" int comm_alloc_windows(CommHandle h, size_t /*win_size*/, uint64_t *device_ctx_out) try {
    if (!h || !device_ctx_out) return -1;

    aclError aRet;
    char group[128] = {};
    HcclResult hret = hccl_get_comm_name(h->hccl_comm, group);
    if (hret != HCCL_SUCCESS) {
        fprintf(stderr, "[comm rank %d] HcclGetCommName failed: %d\n", h->rank, static_cast<int>(hret));
        return -1;
    }

    CommTopo topoType = 0;
    hret = hccl_get_l0_topo_type_ex(group, &topoType, COMM_IS_NOT_SET_DEVICE);
    if (hret != HCCL_SUCCESS) {
        fprintf(stderr, "[comm rank %d] HcomGetL0TopoType failed: %d\n", h->rank, static_cast<int>(hret));
        return -1;
    }

    HcclComm commHandle = nullptr;
    hret = hccl_get_comm_handle_by_group(group, &commHandle);
    if (hret != HCCL_SUCCESS) {
        fprintf(stderr, "[comm rank %d] HcomGetCommHandleByGroup failed: %d\n", h->rank, static_cast<int>(hret));
        return -1;
    }

    // File barrier so all ranks have completed HcclCommInitRootInfo
    if (!file_barrier(h->rootinfo_path, h->rank, h->nranks, "hccl_init", h->run_token)) {
        return -1;
    }
    hret = hccl_barrier(h->hccl_comm, h->stream);
    if (hret != HCCL_SUCCESS) {
        fprintf(stderr, "[comm rank %d] HcclBarrier before HCCL alloc failed: %d\n", h->rank, static_cast<int>(hret));
        return -1;
    }
    aRet = aclrtSynchronizeStream(h->stream);
    if (aRet != ACL_SUCCESS) {
        fprintf(stderr, "[comm rank %d] aclrtSynchronizeStream before HCCL alloc failed: %d\n", h->rank, (int)aRet);
        return -1;
    }

    // Tiling configuration for HcclAllocComResourceByTiling.  This is the
    // offset-list V2 form used by HCCL's MC2 resource allocator; it avoids
    // linking the optiling builder symbols into the host runtime.
    Mc2CommConfigV2 tiling{};
    memset(&tiling, 0, sizeof(tiling));
    tiling.init.version = kMc2TilingVersion;
    tiling.init.mc2HcommCnt = 1U;
    tiling.init.commBlockNum = kMc2CommBlockNum;
    tiling.init.devType = kMc2DevType;
    tiling.init.offset[0] = static_cast<uint32_t>(offsetof(Mc2CommConfigV2, inner));
    tiling.inner.opType = kMc2OpTypeBatchWrite;
    strncpy(tiling.inner.groupName, group, sizeof(tiling.inner.groupName) - 1);
    strncpy(tiling.inner.algConfig, kMc2AlgConfig, sizeof(tiling.inner.algConfig) - 1);

    void *ctxPtr = nullptr;
    hret = hccl_alloc_com_resource(commHandle, h->stream, &tiling, &ctxPtr);
    if (hret != HCCL_SUCCESS || ctxPtr == nullptr) {
        fprintf(
            stderr, "[comm rank %d] HcclAllocComResourceByTiling failed for group=%s: %d ctx=%p\n", h->rank, group,
            static_cast<int>(hret), ctxPtr
        );
        return -1;
    }
    aRet = aclrtSynchronizeStream(h->stream);
    if (aRet != ACL_SUCCESS) {
        fprintf(stderr, "[comm rank %d] aclrtSynchronizeStream after HCCL alloc failed: %d\n", h->rank, (int)aRet);
        return -1;
    }

    // Extract CommContext (topology-dependent)
    if (topoType == COMM_TOPO_MESH) {
        h->device_ctx = reinterpret_cast<CommContext *>(ctxPtr);
        memset(&h->host_ctx, 0, sizeof(h->host_ctx));
        HcclMeshOpParamHead head{};
        aRet = aclrtMemcpy(&head, sizeof(head), ctxPtr, sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) return -1;
        if (head.rankNum == 0 || head.rankNum > HCCL_MESH_RANK_NUM || head.rankNum > COMM_MAX_RANK_NUM) {
            fprintf(
                stderr, "[comm rank %d] MESH rankNum=%u out of range [1, %u]\n", h->rank, head.rankNum,
                HCCL_MESH_RANK_NUM
            );
            return -1;
        }
        h->host_ctx.workSpace = head.workSpace;
        h->host_ctx.workSpaceSize = head.workSpaceSize;
        h->host_ctx.rankId = head.rankId;
        h->host_ctx.rankNum = head.rankNum;
        h->host_ctx.winSize = head.winSize;

        const size_t windowsInOff = sizeof(HcclMeshOpParamHead);
        const size_t windowsOutOff = windowsInOff + HCCL_MESH_RANK_NUM * sizeof(uint64_t);
        aRet = aclrtMemcpy(
            h->host_ctx.windowsIn, head.rankNum * sizeof(uint64_t),
            reinterpret_cast<uint8_t *>(ctxPtr) + windowsInOff, head.rankNum * sizeof(uint64_t),
            ACL_MEMCPY_DEVICE_TO_HOST
        );
        if (aRet != ACL_SUCCESS) return -1;
        aRet = aclrtMemcpy(
            h->host_ctx.windowsOut, head.rankNum * sizeof(uint64_t),
            reinterpret_cast<uint8_t *>(ctxPtr) + windowsOutOff, head.rankNum * sizeof(uint64_t),
            ACL_MEMCPY_DEVICE_TO_HOST
        );
        if (aRet != ACL_SUCCESS) return -1;
    } else {
        // RING topology: parse HcclOpResParam structure on device
        auto *rawCtx = reinterpret_cast<uint8_t *>(ctxPtr);

        HcclOpResParamHead head{};
        const size_t headOff = offsetof(HcclOpResParam, localUsrRankId);
        aRet = aclrtMemcpy(&head, sizeof(head), rawCtx + headOff, sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) return -1;

        // rankSize comes from device memory; cap against our static windowsIn
        // buffer (COMM_MAX_RANK_NUM) before using it to index or size.
        if (head.rankSize == 0 || head.rankSize > COMM_MAX_RANK_NUM) {
            fprintf(
                stderr, "[comm rank %d] HcclOpResParam.rankSize=%u out of range [1, %u]\n", h->rank, head.rankSize,
                COMM_MAX_RANK_NUM
            );
            return -1;
        }

        const size_t remoteResOff = offsetof(HcclOpResParam, remoteRes);
        const size_t remoteResBytes = head.rankSize * sizeof(RemoteResPtr);
        std::vector<RemoteResPtr> remoteResArr(head.rankSize);
        aRet = aclrtMemcpy(
            remoteResArr.data(), remoteResBytes, rawCtx + remoteResOff, remoteResBytes, ACL_MEMCPY_DEVICE_TO_HOST
        );
        if (aRet != ACL_SUCCESS) return -1;

        memset(&h->host_ctx, 0, sizeof(h->host_ctx));

        uint64_t wsFields[2] = {0, 0};
        aRet = aclrtMemcpy(wsFields, sizeof(wsFields), rawCtx, sizeof(wsFields), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) return -1;
        h->host_ctx.workSpace = wsFields[0];
        h->host_ctx.workSpaceSize = wsFields[1];
        h->host_ctx.rankId = head.localUsrRankId;
        h->host_ctx.rankNum = head.rankSize;
        h->host_ctx.winSize = head.winSize;

        for (uint32_t i = 0; i < head.rankSize; ++i) {
            if (i == head.localUsrRankId) {
                h->host_ctx.windowsIn[i] = head.localWindowsIn;
                h->host_ctx.windowsOut[i] = head.localWindowsOut;
                continue;
            }
            uint64_t devPtr = remoteResArr[i].nextDevicePtr;
            if (devPtr == 0) return -1;

            HcclRankRelationResV2 remoteInfo{};
            aRet = aclrtMemcpy(
                &remoteInfo, sizeof(remoteInfo), reinterpret_cast<void *>(devPtr), sizeof(remoteInfo),
                ACL_MEMCPY_DEVICE_TO_HOST
            );
            if (aRet != ACL_SUCCESS) return -1;
            h->host_ctx.windowsIn[i] = remoteInfo.windowsIn;
            h->host_ctx.windowsOut[i] = remoteInfo.windowsOut;
        }
    }

    bool need_device_ctx_copy = topoType != COMM_TOPO_MESH;
#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
    h->sdma_workspace = std::make_unique<pto::comm::sdma::SdmaWorkspaceManager>();
    if (!h->sdma_workspace->Init()) {
        fprintf(stderr, "[comm rank %d] SDMA workspace initialization failed\n", h->rank);
        h->sdma_workspace.reset();
        return -1;
    }
    h->host_ctx.workSpace = reinterpret_cast<uint64_t>(h->sdma_workspace->GetWorkspaceAddr());
    h->host_ctx.workSpaceSize = 16 * 1024;
    need_device_ctx_copy = true;
#endif

    if (need_device_ctx_copy) {
        void *newDevMem = nullptr;
        aRet = aclrtMalloc(&newDevMem, sizeof(CommContext), ACL_MEM_MALLOC_HUGE_FIRST);
        if (aRet != ACL_SUCCESS) return -1;

        aRet =
            aclrtMemcpy(newDevMem, sizeof(CommContext), &h->host_ctx, sizeof(CommContext), ACL_MEMCPY_HOST_TO_DEVICE);
        if (aRet != ACL_SUCCESS) {
            aclrtFree(newDevMem);
            return -1;
        }
        h->device_ctx = reinterpret_cast<CommContext *>(newDevMem);
        h->owns_device_ctx = true;
    }

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

extern "C" int comm_derive_context(
    CommHandle h, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank, size_t window_offset,
    size_t window_size, uint64_t *device_ctx_out
) try {
    if (!h || !rank_ids || !device_ctx_out) return -1;
    if (h->device_ctx == nullptr || h->host_ctx.rankNum == 0) {
        fprintf(stderr, "[comm rank %d] comm_derive_context: base windows are not allocated\n", h->rank);
        return -1;
    }
    if (rank_count == 0 || rank_count > COMM_MAX_RANK_NUM || domain_rank >= rank_count) {
        fprintf(
            stderr, "[comm rank %d] comm_derive_context: invalid rank_count=%zu domain_rank=%u\n", h->rank,
            rank_count, domain_rank
        );
        return -1;
    }
    if (window_offset + window_size > static_cast<size_t>(h->host_ctx.winSize)) {
        fprintf(
            stderr,
            "[comm rank %d] comm_derive_context: window range [%zu, %zu) exceeds base window size %llu\n", h->rank,
            window_offset, window_offset + window_size, static_cast<unsigned long long>(h->host_ctx.winSize)
        );
        return -1;
    }

    CommContext derived{};
    derived.workSpace = h->host_ctx.workSpace;
    derived.workSpaceSize = h->host_ctx.workSpaceSize;
    derived.rankId = domain_rank;
    derived.rankNum = static_cast<uint32_t>(rank_count);
    derived.winSize = window_size;
    for (size_t i = 0; i < rank_count; ++i) {
        uint32_t base_rank = rank_ids[i];
        if (base_rank >= h->host_ctx.rankNum) {
            fprintf(
                stderr, "[comm rank %d] comm_derive_context: rank_ids[%zu]=%u out of range [0, %u)\n", h->rank, i,
                base_rank, h->host_ctx.rankNum
            );
            return -1;
        }
        derived.windowsIn[i] = h->host_ctx.windowsIn[base_rank] + window_offset;
        derived.windowsOut[i] = h->host_ctx.windowsOut[base_rank] + window_offset;
    }
    void *newDevMem = nullptr;
    aclError aRet = aclrtMalloc(&newDevMem, sizeof(CommContext), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS || newDevMem == nullptr) {
        fprintf(stderr, "[comm rank %d] comm_derive_context: aclrtMalloc failed: %d\n", h->rank, static_cast<int>(aRet));
        return -1;
    }

    aRet = aclrtMemcpy(newDevMem, sizeof(CommContext), &derived, sizeof(CommContext), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aRet != ACL_SUCCESS) {
        fprintf(
            stderr, "[comm rank %d] comm_derive_context: aclrtMemcpy H2D failed: %d\n", h->rank, static_cast<int>(aRet)
        );
        aclrtFree(newDevMem);
        return -1;
    }

    auto *ctx = reinterpret_cast<CommContext *>(newDevMem);
    h->derived_contexts.push_back(ctx);
    *device_ctx_out = reinterpret_cast<uint64_t>(ctx);
    return 0;
} catch (const std::exception &e) {
    fprintf(stderr, "[comm] comm_derive_context: exception: %s\n", e.what());
    return -1;
} catch (...) {
    fprintf(stderr, "[comm] comm_derive_context: unknown exception\n");
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
    for (auto *ctx : h->derived_contexts) {
        if (ctx) {
            aclrtFree(ctx);
        }
    }
    h->derived_contexts.clear();
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
