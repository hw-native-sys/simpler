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
#include "common/unified_log.h"
#include "spin_hint.h"

constexpr int32_t AICORE_EXIT_ACK_SPIN_LIMIT = 100000;

static uint64_t g_platform_regs = 0;

void set_platform_regs(uint64_t regs) { g_platform_regs = regs; }

uint64_t get_platform_regs() { return g_platform_regs; }

void platform_init_aicore_regs(uint64_t reg_addr) {
    // Initialize task dispatch register to idle state
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
}

int32_t platform_deinit_aicore_regs(uint64_t reg_addr) {
    // Send exit signal to AICore
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);

    // Wait for AICore to acknowledge exit by writing AICORE_EXITED_VALUE to COND
    for (int32_t spin = 0; spin < AICORE_EXIT_ACK_SPIN_LIMIT; spin++) {
        if (read_reg(reg_addr, RegId::COND) == AICORE_EXITED_VALUE) {
            // Initialize task dispatch register to idle state
            write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
            return 0;
        }
        SPIN_WAIT_HINT();
    }

    LOG_ERROR("Timed out waiting for AICore exit ack at reg_addr=0x%lx", static_cast<unsigned long>(reg_addr));
    return -1;
}

uint32_t platform_get_physical_cores_count() {
    return DAV_3510::PLATFORM_MAX_PHYSICAL_CORES * PLATFORM_CORES_PER_BLOCKDIM;
}
