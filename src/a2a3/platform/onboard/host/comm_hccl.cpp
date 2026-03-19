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
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"

using CommTopo = uint32_t;

// Internal HCCL APIs (not in public headers).
// CANN 9.x renames all HCCL symbols with a V2 suffix and ships libhccl_v2.so.
// The public header still declares weak non-V2 symbols, so we declare V2 variants
// separately and dispatch via inline wrappers.
#ifdef HCCL_USE_V2_API
extern "C" HcclResult HcclGetRootInfoV2(HcclRootInfo* rootInfo);
extern "C" HcclResult HcclCommInitRootInfoV2(uint32_t nRanks, const HcclRootInfo* rootInfo,
                                              uint32_t rank, HcclComm* comm);
extern "C" HcclResult HcclGetCommNameV2(HcclComm comm, char* commName);
extern "C" HcclResult HcclBarrierV2(HcclComm comm, aclrtStream stream);
extern "C" HcclResult HcclCommDestroyV2(HcclComm comm);
extern "C" HcclResult HcclAllocComResourceByTilingV2(HcclComm comm, void* stream,
                                                      void* mc2Tiling, void** commContext);
extern "C" HcclResult HcomGetCommHandleByGroupV2(const char* group, HcclComm* commHandle);
extern "C" HcclResult HcomGetL0TopoTypeExV2(const char* group, CommTopo* topoType,
                                             uint32_t isSetDevice);

static inline HcclResult hccl_get_root_info(HcclRootInfo* ri)
    { return HcclGetRootInfoV2(ri); }
static inline HcclResult hccl_comm_init_root_info(uint32_t n, const HcclRootInfo* ri, uint32_t r, HcclComm* c)
    { return HcclCommInitRootInfoV2(n, ri, r, c); }
static inline HcclResult hccl_get_comm_name(HcclComm c, char* name)
    { return HcclGetCommNameV2(c, name); }
static inline HcclResult hccl_barrier(HcclComm c, aclrtStream s)
    { return HcclBarrierV2(c, s); }
static inline HcclResult hccl_comm_destroy(HcclComm c)
    { return HcclCommDestroyV2(c); }
static inline HcclResult hccl_alloc_com_resource(HcclComm c, void* s, void* t, void** ctx)
    { return HcclAllocComResourceByTilingV2(c, s, t, ctx); }
static inline HcclResult hccl_get_comm_handle_by_group(const char* g, HcclComm* c)
    { return HcomGetCommHandleByGroupV2(g, c); }
static inline HcclResult hccl_get_l0_topo_type_ex(const char* g, CommTopo* t, uint32_t f)
    { return HcomGetL0TopoTypeExV2(g, t, f); }
#else
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
#endif

static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
static constexpr uint32_t COMM_TOPO_MESH = 0b1u;

using rtStream_t = void*;
static constexpr int32_t RT_STREAM_PRIORITY_DEFAULT = 0;
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

    rtStream_t stream = nullptr;
    HcclComm hccl_comm = nullptr;

    CommDeviceContext host_ctx{};
    CommDeviceContext* device_ctx = nullptr;
    bool owns_device_ctx = false;
};

// ============================================================================
// Helpers
// ============================================================================

static bool wait_for_file(const std::string& path, int timeout_sec = 120) {
    for (int i = 0; i < timeout_sec * 10; ++i) {
        std::ifstream f(path, std::ios::binary);
        if (f.good()) {
            auto sz = f.seekg(0, std::ios::end).tellg();
            if (sz >= static_cast<std::streamoff>(HCCL_ROOT_INFO_BYTES)) return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

static void file_barrier(const std::string& dir, int rank, int nranks, const std::string& tag) {
    std::string my_marker = dir + "/barrier_" + tag + "_" + std::to_string(rank) + ".ready";
    { std::ofstream(my_marker) << "1"; }

    for (int r = 0; r < nranks; ++r) {
        std::string marker = dir + "/barrier_" + tag + "_" + std::to_string(r) + ".ready";
        while (true) {
            std::ifstream f(marker);
            if (f.good()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

// ============================================================================
// API implementation
// ============================================================================

extern "C" CommHandle comm_init(int rank, int nranks, const char* rootinfo_path) {
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

    // NOTE: Do NOT call aclrtSetDevice here — the caller (distributed_worker)
    // already set the correct physical device via set_device(device_id).
    // Calling aclrtSetDevice(rank) would override the context when
    // rank != device_id (e.g. devices=[2,4,5,7]).

    // RootInfo exchange
    HcclRootInfo rootInfo{};
    if (rank == 0) {
        HcclResult hret = hccl_get_root_info(&rootInfo);
        if (hret != HCCL_SUCCESS) {
            fprintf(stderr, "[comm rank 0] HcclGetRootInfo failed: %d\n", (int)hret);
            delete h;
            return nullptr;
        }
        std::ofstream fout(rootinfo_path, std::ios::binary);
        fout.write(rootInfo.internal, HCCL_ROOT_INFO_BYTES);
        fout.close();
    } else {
        if (!wait_for_file(rootinfo_path)) {
            fprintf(stderr, "[comm rank %d] Timeout waiting for rootinfo\n", rank);
            delete h;
            return nullptr;
        }
        std::ifstream fin(rootinfo_path, std::ios::binary);
        fin.read(rootInfo.internal, HCCL_ROOT_INFO_BYTES);
    }

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
    std::string barrier_dir = h->rootinfo_path;
    auto last_slash = barrier_dir.rfind('/');
    if (last_slash != std::string::npos)
        barrier_dir = barrier_dir.substr(0, last_slash);
    file_barrier(barrier_dir, h->rank, h->nranks, "hccl_init");

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

extern "C" int comm_barrier(CommHandle h) {
    if (!h) return -1;
    hccl_barrier(h->hccl_comm, (aclrtStream)h->stream);
    aclrtSynchronizeStream((aclrtStream)h->stream);
    return 0;
}

extern "C" int comm_destroy(CommHandle h) {
    if (!h) return -1;

    if (h->owns_device_ctx && h->device_ctx) {
        aclrtFree(h->device_ctx);
    }
    if (h->stream) rtStreamDestroy(h->stream);
    if (h->hccl_comm) hccl_comm_destroy(h->hccl_comm);

    // NOTE: Do NOT call aclrtResetDevice / aclFinalize here.
    // Device lifecycle is owned by DeviceRunner (static singleton) whose
    // destructor frees all tracked device memory before resetting the device.
    // Resetting early would invalidate pointers still held by MemoryAllocator.

    delete h;
    return 0;
}
