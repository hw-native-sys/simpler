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

/**
 * Query valid AICore cores via HAL API
 */
static bool get_pg_mask(uint64_t& valid, int64_t device_id) {
    uint64_t aicore_bitmap[PLATFORM_AICORE_BITMAP_LEN] = {0};
    int32_t size_n = static_cast<int32_t>(sizeof(uint64_t)) * PLATFORM_AICORE_BITMAP_LEN;

    auto halFuncDevInfo =
        (int (*)(uint64_t deviceId, int32_t moduleType, int32_t infoType, void* buf, int32_t* size))dlsym(
            nullptr, "halGetDeviceInfoByBuff");

    if (halFuncDevInfo == nullptr) {
        LOG_WARN("halGetDeviceInfoByBuff not found, assuming all cores valid");
        return false;
    }

    auto ret = halFuncDevInfo(static_cast<uint32_t>(device_id),
        MODULE_TYPE_AICORE,
        INFO_TYPE_OCCUPY,
        reinterpret_cast<void*>(&aicore_bitmap[0]),
        &size_n);

    if (ret != 0) {
        LOG_ERROR("halGetDeviceInfoByBuff failed with rc=%d", ret);
        return false;
    }

    valid = aicore_bitmap[0];
    return true;
}

/**
 * Retrieve AICore register base addresses via HAL API
 */
static int get_aicore_reg_info(std::vector<int64_t>& aic, std::vector<int64_t>& aiv,
                           const int& addr_type, int64_t device_id) {
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

    auto halFunc =
        (int (*)(int type, void* paramValue, size_t paramValueSize, void* outValue, size_t* outSizeRet))dlsym(
            nullptr, "halMemCtl");

    if (halFunc == nullptr) {
        LOG_ERROR("halMemCtl not found in symbol table");
        return -1;
    }

    struct AddrMapInPara in_map_para;
    struct AddrMapOutPara out_map_para;
    in_map_para.devid = device_id;
    in_map_para.addr_type = addr_type;

    auto ret = halFunc(0,
        reinterpret_cast<void*>(&in_map_para),
        sizeof(struct AddrMapInPara),
        reinterpret_cast<void*>(&out_map_para),
        nullptr);

    if (ret != 0) {
        LOG_ERROR("halMemCtl failed with rc=%d", ret);
        return ret;
    }

    LOG_INFO("Register base: ptr=0x%llx, len=0x%llx", out_map_para.ptr, out_map_para.len);

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

void get_aicore_regs(std::vector<int64_t>& regs, uint64_t device_id) {
    std::vector<int64_t> aiv;
    std::vector<int64_t> aic;

    int rt = get_aicore_reg_info(aic, aiv, ADDR_MAP_TYPE_REG_AIC_CTRL, device_id);

    if (rt != 0) {
        LOG_ERROR("get_aicore_reg_info failed, using placeholder addresses");
        // Fallback: generate placeholder addresses
        for (int i = 0; i < 25; i++) {
            aic.push_back(0xDEADBEEF00000000ULL + (i * 0x800000));  // 8M stride
            aiv.push_back(0xDEADBEEF00000000ULL + (i * 0x800000) + 0x100000);
            aiv.push_back(0xDEADBEEF00000000ULL + (i * 0x800000) + 0x200000);
        }
    }

    // AIC cores first, then AIV cores
    regs.insert(regs.end(), aic.begin(), aic.end());
    regs.insert(regs.end(), aiv.begin(), aiv.end());

    LOG_INFO("get_aicore_regs: Retrieved %zu AIC and %zu AIV register addresses",
             aic.size(), aiv.size());
}

int init_aicore_register_addresses(
    uint64_t* runtime_regs_ptr,
    uint64_t device_id,
    MemoryAllocator& allocator) {

    if (runtime_regs_ptr == nullptr) {
        LOG_ERROR("init_aicore_register_addresses: Invalid parameters");
        return -1;
    }

    LOG_INFO("Retrieving and allocating AICore register addresses...");

    // Step 1: Get register addresses from HAL
    std::vector<int64_t> host_regs;
    get_aicore_regs(host_regs, device_id);

    if (host_regs.empty()) {
        LOG_ERROR("Failed to get AICore register addresses");
        return -1;
    }

    // Step 2: Allocate device memory for register address array
    size_t regs_size = host_regs.size() * sizeof(int64_t);
    void* reg_ptr = allocator.alloc(regs_size);
    if (reg_ptr == nullptr) {
        LOG_ERROR("Failed to allocate device memory for register addresses");
        return -1;
    }

    // Step 3: Copy register addresses to device memory
    int ret = rtMemcpy(reg_ptr, regs_size, host_regs.data(), regs_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        LOG_ERROR("Failed to copy register addresses to device (rc=%d)", ret);
        return -1;
    }

    // Step 4: Store device pointer in output regs field
    *runtime_regs_ptr = reinterpret_cast<uint64_t>(reg_ptr);

    LOG_INFO("Successfully initialized register addresses: %zu addresses at device 0x%llx",
             host_regs.size(), *runtime_regs_ptr);

    return 0;
}
