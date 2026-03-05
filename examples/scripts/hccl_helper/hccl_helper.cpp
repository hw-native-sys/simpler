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

    // MESH: ctxPtr is HcclDeviceContext; read windowsIn[rank_id]
    HcclDeviceContext hostCtx;
    aclError aRet = aclrtMemcpy(&hostCtx, sizeof(hostCtx), ctxPtr, sizeof(hostCtx), ACL_MEMCPY_DEVICE_TO_HOST);
    if (aRet != ACL_SUCCESS) {
        HcclCommDestroy(comm);
        rtStreamDestroy(stream);
        return -static_cast<int>(aRet);
    }

    *out_comm = comm;
    *out_ctx_ptr = ctxPtr;
    *out_win_base = hostCtx.windowsIn[rank_id];
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
