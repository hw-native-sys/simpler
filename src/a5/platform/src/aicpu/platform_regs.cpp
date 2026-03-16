/**
 * @file platform_regs.cpp
 * @brief AICPU register interface - shared implementation
 *
 * Contains platform-agnostic functions shared across all platforms.
 * Platform-specific read_reg/write_reg are in:
 *   sim/aicpu/inner_platform_regs.cpp    -- sparse_reg_ptr mapping for simulation
 *   onboard/aicpu/inner_platform_regs.cpp -- direct MMIO offset for hardware
 */

#include <cstdint>
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

static uint64_t g_platform_regs = 0;

void set_platform_regs(uint64_t regs) {
    g_platform_regs = regs;
}

uint64_t get_platform_regs() {
    return g_platform_regs;
}

void platform_init_aicore_regs(uint64_t reg_addr) {
    // Initialize task dispatch register to idle state
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
}

void platform_deinit_aicore_regs(uint64_t reg_addr) {
    // Send exit signal to AICore
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);

    // Wait for AICore to acknowledge exit by writing AICORE_EXITED_VALUE to COND
    while (read_reg(reg_addr, RegId::COND) != AICORE_EXITED_VALUE) {
    }

    // Initialize task dispatch register to idle state
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
}

uint32_t platform_get_physical_cores_count() {
    return DAV_3510::PLATFORM_MAX_PHYSICAL_CORES * PLATFORM_CORES_PER_BLOCKDIM;
}
