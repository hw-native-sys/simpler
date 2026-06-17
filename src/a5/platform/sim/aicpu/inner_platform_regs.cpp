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
 * Simulated registers are three compact pages per core (16KB total):
 *   0x0000-0x0FFF -> page 0 (AICore SPR low: CTRL, DATA_MAIN_BASE)
 *   0x2400-0x43FF -> page 2 (PMU MMIO)
 *   0x5000-0x5FFF -> page 1 (AICore SPR high: COND)
 * sparse_reg_ptr() performs the offset remapping.
 */

#include <cstdint>
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

volatile uint32_t *get_reg_ptr(uint64_t reg_base_addr, RegId reg) {
    volatile uint8_t *reg_base = reinterpret_cast<volatile uint8_t *>(reg_base_addr);
    return reinterpret_cast<volatile uint32_t *>(sparse_reg_ptr(reg_base, reg_offset(reg)));
}

// Atomic acquire/release: simulated registers are plain host memory shared
// across the AICPU and AICore host threads, so the access itself must carry
// happens-before (and be visible to TSAN). See platform_regs.h.
uint32_t reg_load_acquire(const volatile uint32_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }

void reg_store_release(volatile uint32_t *p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }

uint64_t read_reg(uint64_t reg_base_addr, RegId reg) {
    return static_cast<uint64_t>(reg_load_acquire(get_reg_ptr(reg_base_addr, reg)));
}

void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value) {
    reg_store_release(get_reg_ptr(reg_base_addr, reg), static_cast<uint32_t>(value));
}

/**
 * @brief Deinit ACK-wait budget on sim: 10 s.
 *
 * On sim "AICore" is a host CPU thread, so a missing exit ACK usually just
 * means the OS scheduler hasn't given that thread a slice on a CPU-starved CI
 * runner — not a wedged op. The wide budget tolerates that jitter. See the
 * declaration in platform_regs.h for the full rationale.
 *
 * @return Timeout in profiling system-counter ticks.
 */
uint64_t inner_get_deinit_timeout_ticks() { return 10 * PLATFORM_PROF_SYS_CNT_FREQ; }
