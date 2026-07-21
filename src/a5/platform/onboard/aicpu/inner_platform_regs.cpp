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
 *
 * Ordering: read_reg / write_reg emit only the volatile MMIO load/store.
 * ARM64 Device-nGnRnE memory orders accesses within the same region; cross
 * Device <-> Normal-cacheable ordering is the caller's responsibility
 * (wmb() before a publishing register write, rmb() after observing a
 * register hand-off bit).
 */

#include <cstdint>
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

volatile uint32_t *get_reg_ptr(uint64_t reg_base_addr, RegId reg) {
    return reinterpret_cast<volatile uint32_t *>(reg_base_addr + reg_offset(reg));
}

// Plain Device-nGnRnE MMIO load/store (atomics are not valid on Device
// memory); cross Device<->Normal ordering is the caller's rmb()/wmb(). See
// platform_regs.h.
uint32_t reg_load_acquire(const volatile uint32_t *p) { return *p; }

void reg_store_release(volatile uint32_t *p, uint32_t v) { *p = v; }

uint64_t read_reg(uint64_t reg_base_addr, RegId reg) { return static_cast<uint64_t>(*get_reg_ptr(reg_base_addr, reg)); }

void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value) {
    *get_reg_ptr(reg_base_addr, reg) = static_cast<uint32_t>(value);
}

/**
 * @brief Deinit ACK-wait budget on real hardware: 1 s.
 *
 * On silicon a non-response this long means the op was STARS-killed or the
 * core is wedged; aclrtResetDevice will clean up. The tight budget preserves
 * hardware hang detection. See the declaration in platform_regs.h for the
 * full rationale.
 *
 * @return Timeout in profiling system-counter ticks.
 */
uint64_t inner_get_deinit_timeout_ticks() { return PLATFORM_PROF_SYS_CNT_FREQ; }
