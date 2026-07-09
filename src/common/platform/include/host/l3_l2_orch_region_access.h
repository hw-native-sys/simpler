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

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_L3_L2_ORCH_REGION_ACCESS_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_L3_L2_ORCH_REGION_ACCESS_H_

#include <stddef.h>
#include <stdint.h>

enum class L3L2RegionAccessProfile : uint32_t {
    INVALID = 0,
    ONBOARD_ACL_IPC = 1,
    SIM_POSIX_SHM = 2,
};

struct L3HostRegionMappingHandle {
    uint64_t id{0};
    L3L2RegionAccessProfile profile{L3L2RegionAccessProfile::INVALID};
    uint64_t mapping_bytes{0};
};

struct L3L2RegionCreateRequest {
    uint64_t magic_version;
    uint64_t request_bytes;
    uint64_t payload_bytes;
    uint64_t counter_bytes;
    int32_t l3_host_pid;
};

struct L3L2RegionCreateReply {
    uint64_t desc[6];
    uint32_t access_profile;
    uint32_t reserved;
    int32_t device_id;
    uint8_t export_key[65];
    uint8_t backing_shm[32];
    uint64_t mapping_bytes;
};

static constexpr size_t L3L2_REGION_CREATE_REQUEST_BYTES = sizeof(L3L2RegionCreateRequest);
static constexpr size_t L3L2_REGION_CREATE_REPLY_BYTES = sizeof(L3L2RegionCreateReply);

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_L3_L2_ORCH_REGION_ACCESS_H_
