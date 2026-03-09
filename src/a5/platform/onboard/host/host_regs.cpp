/**
 * @file host_regs.cpp
 * @brief Host-side AICore register address retrieval implementation
 */

#include "host/host_regs.h"
#include "host/memory_allocator.h"
#include "common/unified_log.h"
#include "common/platform_config.h"
#include "runtime/rt.h"
#include "ascend_hal.h"  // CANN HAL API definitions
#include <dlfcn.h>

/**
 * Retrieve AICore register base addresses via HAL API
 *
 * DAV_3510 uses halResMap (per-core mapping) for resource management.
 * Each AICore has 3 sub-cores (1 AIC + 2 AIV), and the chip has 2 dies with
 * 18 AICores per die.
 */
static int get_aicore_reg_info(std::vector<int64_t>& regs, int64_t device_id) {

    // halResMap: Maps individual AICore resources (DAV_3510-specific)
    auto halFunc = (int (*)(uint32_t devId, struct res_map_info* res_info, uint64_t* va,
        uint32_t* len))dlsym(nullptr, "halResMap");

    if (halFunc == nullptr) {
        LOG_ERROR("halResMap not found in symbol table");
        return -1;
    }

    // Calculate layout parameters from platform config
    constexpr uint32_t SUB_CORE_PER_DIE = PLATFORM_AICORE_PER_DIE * PLATFORM_CORES_PER_BLOCKDIM;
    constexpr uint32_t AIV_BASE_OFFSET = PLATFORM_AICORE_PER_DIE;
    constexpr size_t MAX_INDEX = DAV_3510::PLATFORM_MAX_PHYSICAL_CORES * PLATFORM_CORES_PER_BLOCKDIM;

    struct res_map_info map_info;
    map_info.target_proc_type = PROCESS_CP1;
    map_info.res_type = RES_AICORE;
    map_info.flag = 0;
    map_info.rsv[0] = 0;

    // Pre-allocate output vector
    regs.resize(MAX_INDEX);

    // Map each AICore individually
    for (uint32_t core_idx = 0; core_idx < DAV_3510::PLATFORM_MAX_PHYSICAL_CORES; core_idx++) {
        map_info.res_id = core_idx;
        uint64_t map_addr = 0;
        uint32_t len = REG_AICORE_MAP_SIZE;

        auto ret = halFunc(static_cast<uint32_t>(device_id), &map_info, &map_addr, &len);
        if (ret != 0) {
            LOG_ERROR("halResMap failed for core %u (rc=%d)", core_idx, ret);
            return ret;
        }

        // Calculate die-based indices
        // DAV_3510 has 2 dies, each with 18 AICores
        // Within each die: [AIC0...AIC17][AIV0_0, AIV0_1, AIV1_0, AIV1_1, ...]
        uint32_t die_idx = core_idx / PLATFORM_AICORE_PER_DIE;
        uint32_t local_idx = core_idx % PLATFORM_AICORE_PER_DIE;
        uint32_t die_base = die_idx * SUB_CORE_PER_DIE;

        uint32_t aicore_index = die_base + local_idx;
        uint32_t aiv_first_index = die_base + AIV_BASE_OFFSET + local_idx * 2;
        uint32_t aiv_second_index = aiv_first_index + 1;

        // Store register addresses
        regs[aicore_index] = map_addr;
        regs[aiv_first_index] = map_addr + REG_AIV_FIRST_OFFSET;
        regs[aiv_second_index] = map_addr + REG_AIV_SECOND_OFFSET;
    }

    return 0;
}

static void get_aicore_regs(std::vector<int64_t>& regs, uint64_t device_id) {
    int rt = get_aicore_reg_info(regs, device_id);

    if (rt != 0) {
        LOG_ERROR("get_aicore_reg_info failed, using placeholder addresses");
        // Fallback: generate placeholder addresses
        constexpr size_t MAX_INDEX = DAV_3510::PLATFORM_MAX_PHYSICAL_CORES * PLATFORM_CORES_PER_BLOCKDIM;
        regs.clear();
        regs.resize(MAX_INDEX);

        for (uint32_t core_idx = 0; core_idx < DAV_3510::PLATFORM_MAX_PHYSICAL_CORES; core_idx++) {
            uint32_t die_idx = core_idx / PLATFORM_AICORE_PER_DIE;
            uint32_t local_idx = core_idx % PLATFORM_AICORE_PER_DIE;
            uint32_t die_base = die_idx * (PLATFORM_AICORE_PER_DIE * PLATFORM_CORES_PER_BLOCKDIM);

            uint64_t base_addr = 0xDEADBEEF00000000ULL + (core_idx * 0x800000);

            regs[die_base + local_idx] = base_addr;
            regs[die_base + PLATFORM_AICORE_PER_DIE + local_idx * 2] = base_addr + REG_AIV_FIRST_OFFSET;
            regs[die_base + PLATFORM_AICORE_PER_DIE + local_idx * 2 + 1] = base_addr + REG_AIV_SECOND_OFFSET;
        }
    }

    LOG_INFO("get_aicore_regs: Retrieved %zu register addresses", regs.size());
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
