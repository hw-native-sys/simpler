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
 * @file host_regs.cpp
 * @brief Host-side AICore register address retrieval implementation
 */

#include "host/host_regs.h"
#include "host/memory_allocator.h"
#include "common/unified_log.h"
#include "common/platform_config.h"
#include "runtime/rt.h"
#include "ascend_hal.h"  // CANN HAL API definitions (MODULE_TYPE_AICORE, INFO_TYPE_OCCUPY, etc.)
#include <dlfcn.h>
#include <iostream>

static int kind_to_addr_type(AicoreRegKind kind) {
    switch (kind) {
    case AicoreRegKind::Ctrl:
        return ADDR_MAP_TYPE_REG_AIC_CTRL;
    case AicoreRegKind::Pmu:
        return ADDR_MAP_TYPE_REG_AIC_PMU_CTRL;
    }
    return ADDR_MAP_TYPE_REG_AIC_CTRL;
}

static const char *kind_to_name(AicoreRegKind kind) {
    switch (kind) {
    case AicoreRegKind::Ctrl:
        return "AIC_CTRL";
    case AicoreRegKind::Pmu:
        return "AIC_PMU_CTRL";
    }
    return "UNKNOWN";
}

/**
 * Query valid AICore cores via HAL API
 */
static bool get_pg_mask(uint64_t &valid, int64_t device_id) {
    uint64_t aicore_bitmap[PLATFORM_AICORE_MAP_BUFF_LEN] = {0};
    int32_t size_n = static_cast<int32_t>(sizeof(uint64_t)) * PLATFORM_AICORE_MAP_BUFF_LEN;

    auto halFuncDevInfo = (int (*)(uint64_t deviceId, int32_t moduleType, int32_t infoType, void *buf, int32_t *size))
        dlsym(nullptr, "halGetDeviceInfoByBuff");

    if (halFuncDevInfo == nullptr) {
        LOG_WARN("halGetDeviceInfoByBuff not found, assuming all cores valid");
        return false;
    }

    auto ret = halFuncDevInfo(
        static_cast<uint32_t>(device_id), MODULE_TYPE_AICORE, INFO_TYPE_OCCUPY,
        reinterpret_cast<void *>(&aicore_bitmap[0]), &size_n
    );

    if (ret != 0) {
        LOG_ERROR("halGetDeviceInfoByBuff failed with rc=%d", ret);
        return false;
    }

    valid = aicore_bitmap[0];
    return true;
}

/**
 * Retrieve AICore register base addresses via HAL API for one addr_type.
 */
static int
get_aicore_reg_info(std::vector<int64_t> &aic, std::vector<int64_t> &aiv, const int &addr_type, int64_t device_id) {
    uint64_t valid = 0;
    if (!get_pg_mask(valid, device_id)) {
        // If can't get mask, assume all cores valid
        valid = 0xFFFFFFFF;
        LOG_WARN("Using default valid mask 0xFFFFFFFF");
    }

    uint64_t core_stride = 8 * 1024 * 1024;  // 8M
    uint64_t sub_core_stride = 0x100000ULL;

    auto is_valid = [&valid](int id) {
        return (valid & (1ULL << id)) != 0;
    };

    auto halFunc = (int (*)(int type, void *paramValue, size_t paramValueSize, void *outValue, size_t *outSizeRet))
        dlsym(nullptr, "halMemCtl");

    if (halFunc == nullptr) {
        LOG_ERROR("halMemCtl not found in symbol table");
        return -1;
    }

    struct AddrMapInPara in_map_para;
    struct AddrMapOutPara out_map_para;
    in_map_para.devid = device_id;
    in_map_para.addr_type = addr_type;

    auto ret = halFunc(
        0, reinterpret_cast<void *>(&in_map_para), sizeof(struct AddrMapInPara),
        reinterpret_cast<void *>(&out_map_para), nullptr
    );

    if (ret != 0) {
        LOG_ERROR("halMemCtl failed with rc=%d", ret);
        return ret;
    }

    LOG_INFO_V0("Register base: ptr=0x%llx, len=0x%llx", out_map_para.ptr, out_map_para.len);

    // Iterate over all cores and subcores
    for (uint32_t i = 0; i < DAV_2201::PLATFORM_MAX_PHYSICAL_CORES; i++) {
        for (uint32_t j = 0; j < PLATFORM_SUB_CORES_PER_AICORE; j++) {
            uint64_t vaddr = 0UL;
            if (is_valid(i)) {
                vaddr = out_map_para.ptr + (i * core_stride + j * sub_core_stride);
            }
            if (j == 0) {
                aic.push_back(vaddr);
            } else {
                aiv.push_back(vaddr);
            }
        }
    }

    return 0;
}

/**
 * Get one flat AIC-then-AIV address array for the requested register kind.
 * Returns a negative code on HAL failure; does NOT generate placeholder
 * addresses (callers must treat failure as fatal for that kind).
 */
static int get_aicore_regs(std::vector<int64_t> &regs, uint64_t device_id, AicoreRegKind kind) {
    std::vector<int64_t> aic;
    std::vector<int64_t> aiv;

    int rc = get_aicore_reg_info(aic, aiv, kind_to_addr_type(kind), device_id);
    if (rc != 0) {
        LOG_ERROR("get_aicore_regs(%s): halMemCtl failed: %d", kind_to_name(kind), rc);
        return rc;
    }

    // AIC cores first, then AIV cores
    regs.insert(regs.end(), aic.begin(), aic.end());
    regs.insert(regs.end(), aiv.begin(), aiv.end());

    LOG_INFO_V0(
        "get_aicore_regs(%s): Retrieved %zu AIC and %zu AIV register addresses", kind_to_name(kind), aic.size(),
        aiv.size()
    );
    return 0;
}

int init_aicore_register_addresses(
    uint64_t *runtime_regs_ptr, uint64_t device_id, MemoryAllocator &allocator, AicoreRegKind kind
) {
    if (runtime_regs_ptr == nullptr) {
        LOG_ERROR("init_aicore_register_addresses(%s): Invalid parameters", kind_to_name(kind));
        return -1;
    }

    LOG_INFO_V0("Retrieving and allocating AICore %s register addresses...", kind_to_name(kind));

    std::vector<int64_t> host_regs;
    int rc = get_aicore_regs(host_regs, device_id, kind);
    if (rc != 0) {
        return rc;
    }
    if (host_regs.empty()) {
        LOG_ERROR("init_aicore_register_addresses(%s): Empty address array", kind_to_name(kind));
        return -1;
    }

    size_t regs_size = host_regs.size() * sizeof(int64_t);
    void *reg_ptr = allocator.alloc(regs_size);
    if (reg_ptr == nullptr) {
        LOG_ERROR("Failed to allocate device memory for %s register addresses", kind_to_name(kind));
        return -1;
    }

    int ret = rtMemcpy(reg_ptr, regs_size, host_regs.data(), regs_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        LOG_ERROR("Failed to copy %s register addresses to device (rc=%d)", kind_to_name(kind), ret);
        allocator.free(reg_ptr);
        return -1;
    }

    *runtime_regs_ptr = reinterpret_cast<uint64_t>(reg_ptr);

    LOG_INFO_V0(
        "Successfully initialized %s register addresses: %zu addresses at device 0x%llx", kind_to_name(kind),
        host_regs.size(), *runtime_regs_ptr
    );

    return 0;
}
