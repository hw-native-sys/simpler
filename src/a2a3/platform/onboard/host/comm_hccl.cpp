/**
 * HCCL backend for the comm_* distributed communication API.
 *
 * Implements the five functions declared in host/comm.h using Ascend
 * HCCL (bundled with CANN).  Handles both MESH and RING topologies
 * when extracting per-rank RDMA window addresses.
 */

#include "host/comm.h"
#include "common/comm_context.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>
#include <unistd.h>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"

using CommTopo = uint32_t;

// Internal HCCL helpers are exported by libhcomm on CANN 9.x.  The public
// HCCL APIs below intentionally use the standard, non-V2 entry points to match
// the working pto-isa initialization sequence.
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream,
                                                   void* mc2Tiling, void** commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(const char* group, HcclComm* commHandle);
extern "C" HcclResult HcomGetL0TopoTypeEx(const char* group, CommTopo* topoType,
                                          uint32_t isSetDevice);

static inline HcclResult hccl_get_root_info(HcclRootInfo* ri)
    { return HcclGetRootInfo(ri); }
static inline HcclResult hccl_comm_init_root_info(uint32_t n, const HcclRootInfo* ri, uint32_t r, HcclComm* c)
    { return HcclCommInitRootInfo(n, ri, r, c); }
static inline HcclResult hccl_get_comm_name(HcclComm c, char* name)
    { return HcclGetCommName(c, name); }
static inline HcclResult hccl_barrier(HcclComm c, aclrtStream s)
    { return HcclBarrier(c, s); }
static inline HcclResult hccl_comm_destroy(HcclComm c)
    { return HcclCommDestroy(c); }
static inline HcclResult hccl_alloc_com_resource(HcclComm c, void* s, void* t, void** ctx)
    { return HcclAllocComResourceByTiling(c, s, t, ctx); }
static inline HcclResult hccl_get_comm_handle_by_group(const char* g, HcclComm* c)
    { return HcomGetCommHandleByGroup(g, c); }
static inline HcclResult hccl_get_l0_topo_type_ex(const char* g, CommTopo* t, uint32_t f)
    { return HcomGetL0TopoTypeEx(g, t, f); }

static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
static constexpr uint32_t COMM_TOPO_MESH = 0b1u;

using rtStream_t = void*;
static constexpr int32_t RT_STREAM_PRIORITY_DEFAULT = 0;
extern "C" int32_t rtSetDevice(int32_t device);
extern "C" int32_t rtStreamCreate(rtStream_t* stream, int32_t priority);
extern "C" int32_t rtStreamDestroy(rtStream_t stream);

// ============================================================================
// HCCL tiling structures (required by HcclAllocComResourceByTiling)
// ============================================================================

namespace {

static constexpr uint32_t MAX_CC_TILING_NUM = 8U;
static constexpr uint32_t GROUP_NAME_SIZE = 128U;
static constexpr uint32_t ALG_CONFIG_SIZE = 128U;

struct Mc2InitTilingInner {
    uint32_t version;
    uint32_t mc2HcommCnt;
    uint32_t offset[MAX_CC_TILING_NUM];
    uint8_t debugMode;
    uint8_t preparePosition;
    uint16_t queueNum;
    uint16_t commBlockNum;
    uint8_t devType;
    char reserved[17];
};

struct Mc2cCTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[9];
    uint8_t commEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[GROUP_NAME_SIZE];
    char algConfig[ALG_CONFIG_SIZE];
    uint32_t opType;
    uint32_t reduceType;
};

struct Mc2CommConfigV2 {
    Mc2InitTilingInner init;
    Mc2cCTilingInner inner;
};

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

}  // anonymous namespace

// ============================================================================
// Internal state
// ============================================================================

struct CommHandle_ {
    int rank;
    int nranks;
    std::string rootinfo_path;
    uint64_t run_token = 0;

    rtStream_t stream = nullptr;
    HcclComm hccl_comm = nullptr;

    CommDeviceContext host_ctx{};
    CommDeviceContext* device_ctx = nullptr;
    bool owns_device_ctx = false;
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
    auto now = std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now())
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

static bool wait_for_rootinfo(
    const std::string &path, HcclRootInfo *root_info, uint64_t *run_token, int timeout_sec = 120) {
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

static void file_barrier(
    const std::string &rootinfo_path, int rank, int nranks, const std::string &tag, uint64_t run_token) {
    std::string my_marker = barrier_marker_path(rootinfo_path, run_token, tag, rank);
    { std::ofstream(my_marker) << "1"; }

    for (int r = 0; r < nranks; ++r) {
        std::string marker = barrier_marker_path(rootinfo_path, run_token, tag, r);
        while (true) {
            std::ifstream f(marker);
            if (f.good()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

}  // namespace

// ============================================================================
// API implementation
// ============================================================================

extern "C" CommHandle comm_init(int rank, int nranks, int device_id, const char* rootinfo_path) {
    auto* h = new (std::nothrow) CommHandle_{};
    if (!h) return nullptr;

    h->rank = rank;
    h->nranks = nranks;
    h->rootinfo_path = rootinfo_path;

    // ACL init
    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        fprintf(stderr, "[comm rank %d] aclInit failed: %d\n", rank, (int)aRet);
        delete h;
        return nullptr;
    }

    if (rank == 0) {
        int32_t rtRet = rtSetDevice(device_id);
        if (rtRet != 0) {
            fprintf(stderr, "[comm rank %d] rtSetDevice(%d) failed: %d\n",
                    rank, device_id, rtRet);
            delete h;
            return nullptr;
        }
    }

    // HCCL requires an ACL runtime context bound to the physical device.
    // This cannot be inferred from rank because distributed runs may map
    // ranks to arbitrary device lists (for example devices=[2,4,5,7]).
    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        fprintf(stderr, "[comm rank %d] aclrtSetDevice(%d) failed: %d\n",
                rank, device_id, (int)aRet);
        delete h;
        return nullptr;
    }

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
        fout.write(reinterpret_cast<const char*>(&header), sizeof(header));
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

    file_barrier(h->rootinfo_path, h->rank, h->nranks, "rootinfo_ready", h->run_token);

    // Create stream for HCCL operations
    rtStreamCreate(&h->stream, RT_STREAM_PRIORITY_DEFAULT);

    // Init communicator
    HcclResult hret = hccl_comm_init_root_info(
        static_cast<uint32_t>(nranks), &rootInfo, static_cast<uint32_t>(rank), &h->hccl_comm);
    if (hret != HCCL_SUCCESS) {
        fprintf(stderr, "[comm rank %d] HcclCommInitRootInfo failed: %d\n", rank, (int)hret);
        if (h->stream) rtStreamDestroy(h->stream);
        delete h;
        return nullptr;
    }

    return h;
}

extern "C" int comm_alloc_windows(CommHandle h, size_t /*win_size*/, uint64_t* device_ctx_out) {
    if (!h || !device_ctx_out) return -1;

    char group[128] = {};
    HcclResult hret = hccl_get_comm_name(h->hccl_comm, group);
    if (hret != HCCL_SUCCESS) return -1;

    CommTopo topoType = 0;
    hret = hccl_get_l0_topo_type_ex(group, &topoType, COMM_IS_NOT_SET_DEVICE);
    if (hret != HCCL_SUCCESS) return -1;

    HcclComm commHandle = nullptr;
    hret = hccl_get_comm_handle_by_group(group, &commHandle);
    if (hret != HCCL_SUCCESS) return -1;

    // File barrier so all ranks have completed HcclCommInitRootInfo
    file_barrier(h->rootinfo_path, h->rank, h->nranks, "hccl_init", h->run_token);

    // Tiling configuration for HcclAllocComResourceByTiling
    Mc2CommConfigV2 tiling{};
    memset(&tiling, 0, sizeof(tiling));
    tiling.init.version = 100U;
    tiling.init.mc2HcommCnt = 1U;
    tiling.init.commBlockNum = 48U;
    tiling.init.devType = 4U;
    tiling.init.offset[0] = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(&tiling.inner) - reinterpret_cast<uint64_t>(&tiling.init));
    tiling.inner.opType = 18U;
    tiling.inner.commEngine = 3U;
    tiling.inner.version = 1U;
    strncpy(tiling.inner.groupName, group, GROUP_NAME_SIZE - 1);
    strncpy(tiling.inner.algConfig, "BatchWrite=level0:fullmesh", ALG_CONFIG_SIZE - 1);

    void* ctxPtr = nullptr;
    hret = hccl_alloc_com_resource(commHandle, h->stream, &tiling, &ctxPtr);
    if (hret != HCCL_SUCCESS || ctxPtr == nullptr) return -1;

    // Extract CommDeviceContext (topology-dependent)
    aclError aRet;
    if (topoType == COMM_TOPO_MESH) {
        h->device_ctx = reinterpret_cast<CommDeviceContext*>(ctxPtr);
        aRet = aclrtMemcpy(&h->host_ctx, sizeof(h->host_ctx),
                           h->device_ctx, sizeof(h->host_ctx), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) return -1;
    } else {
        // RING topology: parse HcclOpResParam structure on device
        auto* rawCtx = reinterpret_cast<uint8_t*>(ctxPtr);

        HcclOpResParamHead head{};
        const size_t headOff = offsetof(HcclOpResParam, localUsrRankId);
        aRet = aclrtMemcpy(&head, sizeof(head), rawCtx + headOff, sizeof(head),
                           ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) return -1;

        const size_t remoteResOff = offsetof(HcclOpResParam, remoteRes);
        const size_t remoteResBytes = head.rankSize * sizeof(RemoteResPtr);
        std::vector<RemoteResPtr> remoteResArr(head.rankSize);
        aRet = aclrtMemcpy(remoteResArr.data(), remoteResBytes,
                           rawCtx + remoteResOff, remoteResBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) return -1;

        memset(&h->host_ctx, 0, sizeof(h->host_ctx));

        uint64_t wsFields[2] = {0, 0};
        aclrtMemcpy(wsFields, sizeof(wsFields), rawCtx, sizeof(wsFields), ACL_MEMCPY_DEVICE_TO_HOST);
        h->host_ctx.workSpace = wsFields[0];
        h->host_ctx.workSpaceSize = wsFields[1];
        h->host_ctx.rankId = head.localUsrRankId;
        h->host_ctx.rankNum = head.rankSize;
        h->host_ctx.winSize = head.winSize;

        for (uint32_t i = 0; i < head.rankSize; ++i) {
            if (i == head.localUsrRankId) {
                h->host_ctx.windowsIn[i] = head.localWindowsIn;
                continue;
            }
            uint64_t devPtr = remoteResArr[i].nextDevicePtr;
            if (devPtr == 0) return -1;

            HcclRankRelationResV2 remoteInfo{};
            aRet = aclrtMemcpy(&remoteInfo, sizeof(remoteInfo),
                               reinterpret_cast<void*>(devPtr), sizeof(remoteInfo),
                               ACL_MEMCPY_DEVICE_TO_HOST);
            if (aRet != ACL_SUCCESS) return -1;
            h->host_ctx.windowsIn[i] = remoteInfo.windowsIn;
        }

        void* newDevMem = nullptr;
        aRet = aclrtMalloc(&newDevMem, sizeof(CommDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST);
        if (aRet != ACL_SUCCESS) return -1;

        aRet = aclrtMemcpy(newDevMem, sizeof(CommDeviceContext),
                           &h->host_ctx, sizeof(CommDeviceContext), ACL_MEMCPY_HOST_TO_DEVICE);
        if (aRet != ACL_SUCCESS) {
            aclrtFree(newDevMem);
            return -1;
        }
        h->device_ctx = reinterpret_cast<CommDeviceContext*>(newDevMem);
        h->owns_device_ctx = true;
    }

    *device_ctx_out = reinterpret_cast<uint64_t>(h->device_ctx);
    return 0;
}

extern "C" int comm_get_local_window_base(CommHandle h, uint64_t* base_out) {
    if (!h || !base_out) return -1;
    *base_out = h->host_ctx.windowsIn[h->rank];
    return 0;
}

extern "C" int comm_get_window_size(CommHandle h, size_t* size_out) {
    if (!h || !size_out) return -1;
    *size_out = static_cast<size_t>(h->host_ctx.winSize);
    return 0;
}

extern "C" int comm_barrier(CommHandle h) {
    if (!h) return -1;
    hccl_barrier(h->hccl_comm, (aclrtStream)h->stream);
    aclrtSynchronizeStream((aclrtStream)h->stream);
    return 0;
}

extern "C" int comm_destroy(CommHandle h) {
    if (!h) return -1;

    file_barrier(h->rootinfo_path, h->rank, h->nranks, "destroy", h->run_token);

    if (h->owns_device_ctx && h->device_ctx) {
        aclrtFree(h->device_ctx);
    }
    if (h->stream) rtStreamDestroy(h->stream);
    if (h->hccl_comm) hccl_comm_destroy(h->hccl_comm);

    // NOTE: Do NOT call aclrtResetDevice / aclFinalize here.
    // Device lifecycle is owned by DeviceRunner (static singleton) whose
    // destructor frees all tracked device memory before resetting the device.
    // Resetting early would invalidate pointers still held by MemoryAllocator.

    if (h->rank == 0) {
        cleanup_handshake_files(h->rootinfo_path);
    }

    delete h;
    return 0;
}
