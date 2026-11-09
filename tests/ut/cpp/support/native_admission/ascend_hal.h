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

#pragma once
#include <cstddef>
#include <cstdint>

// Only the HAL leaves reached by the real context-control helpers are replaced.
constexpr int MODULE_TYPE_AICORE = 0;
constexpr int MODULE_TYPE_AICPU = 1;
constexpr int MODULE_TYPE_SYSTEM = 2;
constexpr int INFO_TYPE_OCCUPY = 0;
constexpr int INFO_TYPE_PHY_DIE_ID = 1;
constexpr int ADDR_MAP_TYPE_REG_AIC_CTRL = 2;
constexpr int ADDR_MAP_TYPE_REG_AIC_PMU_CTRL = 3;
constexpr int PROCESS_CP1 = 1;
constexpr int RES_AICORE = 0;
struct AddrMapInPara {
    unsigned int addr_type;
    unsigned int devid;
};
struct AddrMapOutPara {
    unsigned long long ptr;
    unsigned long long len;
};
struct res_map_info {
    int target_proc_type;
    int res_type;
    unsigned int res_id;
    unsigned int flag;
    unsigned int rsv[1];
};
extern "C" {
int halGetDeviceInfo(uint32_t device, int32_t module, int32_t info, int64_t *out);
int halGetDeviceInfoByBuff(uint32_t device, int32_t module, int32_t info, void *out, int32_t *bytes);
int halMemCtl(int type, void *input, size_t bytes, void *output, size_t *output_bytes);
int halResMap(unsigned int device, res_map_info *info, uint64_t *address, unsigned int *bytes);
}
