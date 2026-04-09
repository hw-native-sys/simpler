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
 * @brief AICPU register read/write for simulation (a5sim)
 *
 * Simulated registers are two compact 4KB pages per core (8KB total).
 * sparse_reg_ptr() remaps hardware offsets to this layout:
 *   offset < 0x5000  -> page 0: reg_base + offset
 *   offset >= 0x5000 -> page 1: reg_base + 0x1000 + (offset - 0x5000)
 */

#include <cstdint>
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

uint64_t read_reg(uint64_t reg_base_addr, RegId reg) {  // NOLINT(bugprone-easily-swappable-parameters)
    uint32_t offset = reg_offset(reg);
    volatile uint8_t *reg_base = reinterpret_cast<volatile uint8_t *>(reg_base_addr);
    volatile uint32_t *ptr = reinterpret_cast<volatile uint32_t *>(sparse_reg_ptr(reg_base, offset));

    __sync_synchronize();
    uint64_t value = static_cast<uint64_t>(*ptr);  // NOLINT(modernize-use-auto)
    __sync_synchronize();

    return value;
}

uint64_t poll_reg(uint64_t reg_base_addr, RegId reg) {  // NOLINT(bugprone-easily-swappable-parameters)
    uint32_t offset = reg_offset(reg);
    volatile uint8_t *reg_base = reinterpret_cast<volatile uint8_t *>(reg_base_addr);
    volatile uint32_t *ptr = reinterpret_cast<volatile uint32_t *>(sparse_reg_ptr(reg_base, offset));
    return static_cast<uint64_t>(*ptr);
}

void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value) {  // NOLINT(bugprone-easily-swappable-parameters)
    uint32_t offset = reg_offset(reg);
    volatile uint8_t *reg_base = reinterpret_cast<volatile uint8_t *>(reg_base_addr);
    volatile uint32_t *ptr = reinterpret_cast<volatile uint32_t *>(sparse_reg_ptr(reg_base, offset));

    __sync_synchronize();
    *ptr = static_cast<uint32_t>(value);
    __sync_synchronize();
}
