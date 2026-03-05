/**
 * HCCL helper shared library — same link as pto-comm-isa (ascendcl, hcomm, runtime).
 * Python loads this .so and calls the C API; no direct libacl/libhccl in Python.
 *
 * Build: from examples/scripts/hccl_helper with CANN env set (source set_env.sh):
 *   mkdir build && cd build && cmake .. && make
 * Output: libhccl_helper.so
 */

#include <cstdint>
#include <cstring>
#include <cerrno>
#include <cstddef>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"

#define HCCL_HELPER_ROOT_INFO_BYTES 1024

using CommTopo = uint32_t;
static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
static constexpr uint32_t COMM_TOPO_MESH = 0b1u;
static constexpr uint32_t GROUP_NAME_SIZE = 128U;
static constexpr uint32_t ALG_CONFIG_SIZE = 128U;
static constexpr uint32_t MAX_CC_TILING_NUM = 8U;

extern "C" int rtSetDevice(int32_t device);
extern "C" int rtStreamCreate(void** stream, int32_t priority);
extern "C" int rtStreamDestroy(void* stream);
extern "C" int HcclAllocComResourceByTiling(void* comm, void* stream, void* mc2Tiling, void** commContext);
extern "C" int HcomGetCommHandleByGroup(const char* group, void** commHandle);
extern "C" int HcomGetL0TopoTypeEx(const char* group, CommTopo* topoType, uint32_t isSetDevice);

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

static constexpr uint32_t HCCL_MAX_RANK_NUM = 64;
struct HcclDeviceContext {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[HCCL_MAX_RANK_NUM];
    uint64_t windowsOut[HCCL_MAX_RANK_NUM];
};

static constexpr int RT_STREAM_PRIORITY_DEFAULT = 0;

// ---------------------------------------------------------------------------
// HcclOpResParam compat structs — binary-compatible copies of HCCL internal
// types (from pto-comm-isa common.hpp). Used only on host side to compute
// windowsIn[...] for RING topology.
// ---------------------------------------------------------------------------

namespace hccl_compat {

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

struct HDCommunicateParams {
    uint64_t hostAddr;
    uint64_t deviceAddr;
    uint64_t readCacheAddr;
    uint32_t devMemSize;
    uint32_t buffLen;
    uint32_t flag;
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

// Full struct layout for offsetof(remoteRes) computation.
// Array size of remoteRes does not affect the offset calculation.
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

} // namespace hccl_compat

// ---------------------------------------------------------------------------
// C API for Python ctypes
// ---------------------------------------------------------------------------

extern "C" {

unsigned int hccl_helper_root_info_bytes(void) {
    return HCCL_HELPER_ROOT_INFO_BYTES;
}

// Rank 0: set device, get root info into out_buf. Returns 0 on success.
int hccl_helper_get_root_info(int device_id, void* out_buf, unsigned buf_size) {
    if (out_buf == nullptr || buf_size < HCCL_HELPER_ROOT_INFO_BYTES)
        return -EINVAL;
    aclError e = aclrtSetDevice(device_id);
    if (e != ACL_SUCCESS)
        return -static_cast<int>(e);
    // HcclGetRootInfo expects HcclRootInfo* (opaque struct, typically 1024 bytes)
    auto* root = reinterpret_cast<HcclRootInfo*>(out_buf);
    int ret = HcclGetRootInfo(root);
    return (ret == HCCL_SUCCESS) ? 0 : -ret;
}

// All ranks: init comm. Fills out_comm, out_ctx_ptr, out_win_base, out_stream. Returns 0 on success.
// root_info from rank 0 (hccl_helper_get_root_info). No MPI — Python uses Barrier.
int hccl_helper_init_comm(
    int rank_id,
    int n_ranks,
    int n_devices,
    int first_device_id,
    const void* root_info,
    unsigned root_info_len,
    void** out_comm,
    void** out_ctx_ptr,
    uint64_t* out_win_base,
    void** out_stream
) {
    if (out_comm == nullptr || out_ctx_ptr == nullptr || out_win_base == nullptr || out_stream == nullptr ||
        root_info == nullptr || root_info_len < HCCL_HELPER_ROOT_INFO_BYTES)
        return -EINVAL;

    int device_id = rank_id % n_devices + first_device_id;

    int rtRet = rtSetDevice(device_id);
    if (rtRet != 0)
        return -rtRet;

    void* stream = nullptr;
    rtRet = rtStreamCreate(&stream, RT_STREAM_PRIORITY_DEFAULT);
    if (rtRet != 0 || stream == nullptr)
        return rtRet != 0 ? -rtRet : -1;

    HcclComm comm = nullptr;
    auto* root = reinterpret_cast<const HcclRootInfo*>(root_info);
    int hret = HcclCommInitRootInfo(
        static_cast<uint32_t>(n_ranks),
        root,
        static_cast<uint32_t>(rank_id),
        &comm);
    if (hret != HCCL_SUCCESS || comm == nullptr) {
        rtStreamDestroy(stream);
        return hret != HCCL_SUCCESS ? -hret : -1;
    }

    char group[128] = {};
    hret = HcclGetCommName(comm, group);
    if (hret != HCCL_SUCCESS) {
        HcclCommDestroy(comm);
        rtStreamDestroy(stream);
        return -hret;
    }

    CommTopo topo = 0;
    hret = HcomGetL0TopoTypeEx(group, &topo, COMM_IS_NOT_SET_DEVICE);
    if (hret != HCCL_SUCCESS) {
        HcclCommDestroy(comm);
        rtStreamDestroy(stream);
        return -hret;
    }

    void* commHandle = nullptr;
    hret = HcomGetCommHandleByGroup(group, &commHandle);
    if (hret != HCCL_SUCCESS) {
        HcclCommDestroy(comm);
        rtStreamDestroy(stream);
        return -hret;
    }

    Mc2CommConfigV2 tiling{};
    memset(&tiling, 0, sizeof(tiling));
    tiling.init.version = 100U;
    tiling.init.mc2HcommCnt = 1U;
    tiling.init.commBlockNum = 48U;
    tiling.init.devType = 4U;
    tiling.init.offset[0] = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&tiling.inner) - reinterpret_cast<uintptr_t>(&tiling.init));
    tiling.inner.opType = 18U;
    tiling.inner.commEngine = 3U;
    tiling.inner.version = 1U;
    strncpy(tiling.inner.groupName, group, GROUP_NAME_SIZE - 1);
    strncpy(tiling.inner.algConfig, "BatchWrite=level0:fullmesh", ALG_CONFIG_SIZE - 1);

    void* ctxPtr = nullptr;
    hret = HcclAllocComResourceByTiling(commHandle, stream, &tiling, &ctxPtr);
    if (hret != HCCL_SUCCESS || ctxPtr == nullptr) {
        HcclCommDestroy(comm);
        rtStreamDestroy(stream);
        return hret != HCCL_SUCCESS ? -hret : -1;
    }

    // Build host-side HcclDeviceContext for both MESH and RING topo.
    HcclDeviceContext hostCtx{};
    void* deviceCtxPtr = nullptr;

    if (topo == COMM_TOPO_MESH) {
        // MESH: ctxPtr is HcclCombinOpParamA5 whose first fields match HcclDeviceContext.
        aclError aRet = aclrtMemcpy(&hostCtx, sizeof(hostCtx), ctxPtr, sizeof(hostCtx), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            HcclCommDestroy(comm);
            rtStreamDestroy(stream);
            return -static_cast<int>(aRet);
        }
        deviceCtxPtr = ctxPtr;
    } else {
        // RING: ctxPtr is HcclOpResParam. Extract remote windows and build our own HcclDeviceContext.
        using namespace hccl_compat;
        auto* rawCtx = reinterpret_cast<uint8_t*>(ctxPtr);

        // 1. Read HcclOpResParam head (from localUsrRankId through localWindowsExp).
        HcclOpResParamHead head{};
        const size_t headOff = offsetof(HcclOpResParam, localUsrRankId);
        aclError aRet = aclrtMemcpy(&head, sizeof(head), rawCtx + headOff, sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            HcclCommDestroy(comm);
            rtStreamDestroy(stream);
            return -static_cast<int>(aRet);
        }

        if (head.rankSize == 0 || head.rankSize > HCCL_MAX_RANK_NUM) {
            HcclCommDestroy(comm);
            rtStreamDestroy(stream);
            return -EINVAL;
        }

        // 2. Read remoteRes[0..rankSize-1] (array of device-pointer pairs).
        const size_t remoteResOff = offsetof(HcclOpResParam, remoteRes);
        const size_t remoteResBytes = head.rankSize * sizeof(RemoteResPtr);
        std::vector<RemoteResPtr> remoteResArr(head.rankSize);

        aRet = aclrtMemcpy(remoteResArr.data(), remoteResBytes, rawCtx + remoteResOff, remoteResBytes,
                           ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            HcclCommDestroy(comm);
            rtStreamDestroy(stream);
            return -static_cast<int>(aRet);
        }

        // 3. Build hostCtx with correct per-rank RDMA window addresses.
        std::memset(&hostCtx, 0, sizeof(hostCtx));

        // Read mc2WorkSpace (first 16 bytes of HcclOpResParam).
        uint64_t wsFields[2] = {0, 0};
        aRet = aclrtMemcpy(wsFields, sizeof(wsFields), rawCtx, sizeof(wsFields), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet == ACL_SUCCESS) {
            hostCtx.workSpace = wsFields[0];
            hostCtx.workSpaceSize = wsFields[1];
        }

        hostCtx.rankId = head.localUsrRankId;
        hostCtx.rankNum = head.rankSize;
        hostCtx.winSize = head.winSize;

        for (uint32_t i = 0; i < head.rankSize; ++i) {
            if (i == head.localUsrRankId) {
                hostCtx.windowsIn[i] = head.localWindowsIn;
                continue;
            }

            uint64_t devPtr = remoteResArr[i].nextDevicePtr;
            if (devPtr == 0) {
                HcclCommDestroy(comm);
                rtStreamDestroy(stream);
                return -EINVAL;
            }

            HcclRankRelationResV2 remoteInfo{};
            aRet = aclrtMemcpy(&remoteInfo, sizeof(remoteInfo), reinterpret_cast<void*>(devPtr), sizeof(remoteInfo),
                               ACL_MEMCPY_DEVICE_TO_HOST);
            if (aRet != ACL_SUCCESS) {
                HcclCommDestroy(comm);
                rtStreamDestroy(stream);
                return -static_cast<int>(aRet);
            }

            hostCtx.windowsIn[i] = remoteInfo.windowsIn;
        }

        // 4. Allocate new device memory and copy our correctly-built HcclDeviceContext.
        void* newDevMem = nullptr;
        aRet = aclrtMalloc(&newDevMem, sizeof(HcclDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST);
        if (aRet != ACL_SUCCESS || newDevMem == nullptr) {
            HcclCommDestroy(comm);
            rtStreamDestroy(stream);
            return -static_cast<int>(aRet);
        }

        aRet = aclrtMemcpy(newDevMem, sizeof(HcclDeviceContext), &hostCtx, sizeof(HcclDeviceContext),
                           ACL_MEMCPY_HOST_TO_DEVICE);
        if (aRet != ACL_SUCCESS) {
            aclrtFree(newDevMem);
            HcclCommDestroy(comm);
            rtStreamDestroy(stream);
            return -static_cast<int>(aRet);
        }

        deviceCtxPtr = newDevMem;
    }

    *out_comm = comm;
    *out_ctx_ptr = deviceCtxPtr;
    // 使用 HcclDeviceContext 自带的 rankId 作为本地窗口索引，避免 Python 侧 rank_id 与 HCCL 内部 localUsrRankId 不一致
    *out_win_base = hostCtx.windowsIn[hostCtx.rankId];
    *out_stream = stream;
    return 0;
}

// Barrier + stream sync.
int hccl_helper_barrier(void* comm, void* stream) {
    if (comm == nullptr || stream == nullptr)
        return -EINVAL;
    int hret = HcclBarrier(comm, stream);
    if (hret != HCCL_SUCCESS)
        return -hret;
    aclError e = aclrtSynchronizeStream(stream);
    return (e == ACL_SUCCESS) ? 0 : -static_cast<int>(e);
}

} // extern "C"
