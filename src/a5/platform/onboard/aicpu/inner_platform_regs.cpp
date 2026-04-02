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
 * @file inner_platform_regs.cpp
 * @brief AICPU register read/write for real hardware (a5)
 *
 * halResMap maps each AICore as 3MB of contiguous MMIO. Hardware offsets
 * (e.g. DATA_MAIN_BASE=0xD0, COND=0x5108) are applied directly to the
 * virtual address with no remapping.
 */

#include <cstdint>
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

uint64_t read_reg(uint64_t reg_base_addr, RegId reg) {
    uint32_t offset = reg_offset(reg);
    volatile uint32_t *ptr = reinterpret_cast<volatile uint32_t *>(reg_base_addr + offset);

    __sync_synchronize();
    uint64_t value = static_cast<uint64_t>(*ptr);
    __sync_synchronize();

    return value;
}

void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value) {
    uint32_t offset = reg_offset(reg);
    volatile uint32_t *ptr = reinterpret_cast<volatile uint32_t *>(reg_base_addr + offset);

    __sync_synchronize();
    *ptr = static_cast<uint32_t>(value);
    __sync_synchronize();
}
