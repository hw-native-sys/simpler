/**
 * HcclDeviceContext - device-side context for HCCL collective ops.
 * Extracted from pto-comm-isa for use in simpler-PTO cpt_and_comm.
 */

#pragma once

#include <cstdint>

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
