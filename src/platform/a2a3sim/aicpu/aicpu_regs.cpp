/**
 * @file aicpu_regs.cpp
 * @brief Platform-specific AICPU register initialization for a2a3sim
 */

#include "aicpu/platform_regs.h"
#include "inner_platform_config.h"

void platform_init_aicore_regs(uint64_t reg_addr) {
    // a2a3sim simulation uses the same initialization as a2a3
    write_reg(reg_addr, RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);

    // Initialize task dispatch register to idle state
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, 0);
}

void platform_deinit_aicore_regs(uint64_t reg_addr) {
    // Send exit signal to AICore
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);

    // Close fast path control
    write_reg(reg_addr, RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
}

uint32_t platform_get_physical_cores_count() {
    return DAV_2201::PLATFORM_MAX_PHYSICAL_CORES;
}
