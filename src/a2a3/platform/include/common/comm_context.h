/**
 * CommDeviceContext — device-side distributed communication context.
 *
 * This struct is the ABI contract between host (comm_hccl.cpp / comm_sim.cpp)
 * and device kernels.  PTO communication instructions (TREDUCE, TGET, TPUT)
 * access remote data through the GVA addresses in windowsIn[]/windowsOut[]
 * via MTE2 DMA.
 *
 * On HCCL MESH topology the struct layout matches what HCCL returns directly.
 * On RING topology the host builds it by extracting remote RDMA addresses
 * from HcclOpResParam's remoteRes array.
 * On simulation the host fills it with malloc'd pointers.
 */

#pragma once

#include <cstdint>

static constexpr uint32_t COMM_MAX_RANK_NUM = 64;

struct CommDeviceContext {
    uint64_t workSpace;
    uint64_t workSpaceSize;

    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[COMM_MAX_RANK_NUM];
    uint64_t windowsOut[COMM_MAX_RANK_NUM];
};
