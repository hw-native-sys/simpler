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
 * @file inner_kernel.h
 * @brief Platform-specific AICore definitions for real hardware (a2a3)
 *
 * This header provides platform-specific macro definitions for AICore kernels
 * running on real Ascend hardware with CANN compiler support.
 */

// NOLINT(build/header_guard) -- PLATFORM_* include guards are the project convention here

#ifndef PLATFORM_A2A3_AICORE_INNER_KERNEL_H_
#define PLATFORM_A2A3_AICORE_INNER_KERNEL_H_

#include <cstdint>

#include "common/platform_config.h"

// AICore function attribute for CANN compiler
#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

// dcci (Data Cache Clean and Invalidate) is provided by CANN headers
// No need to define it here - it's a hardware instruction

// SPIN_WAIT_HINT - no-op on real hardware (AICore has dedicated polling support)
#define SPIN_WAIT_HINT() ((void)0)

// OUT_OF_ORDER_STORE_BARRIER - no-op on real hardware (dcci handles cache coherency)
#define OUT_OF_ORDER_STORE_BARRIER() ((void)0)

// OUT_OF_ORDER_LOAD_BARRIER - no-op on real hardware (dcci handles cache coherency)
#define OUT_OF_ORDER_LOAD_BARRIER() ((void)0)

// OUT_OF_ORDER_FULL_BARRIER - no-op on real hardware (dcci handles full cache coherency)
#define OUT_OF_ORDER_FULL_BARRIER() ((void)0)

/**
 * Read an AICore register via SPR access
 *
 * @param reg  Register identifier
 * @return Register value (zero-extended to uint64_t)
 */
__aicore__ inline uint64_t read_reg(RegId reg) {
    switch (reg) {
    case RegId::DATA_MAIN_BASE: {
        uint32_t val;
        __asm__ volatile("MOV %0, DATA_MAIN_BASE\n" : "=l"(val));
        return static_cast<uint64_t>(val);
    }
    case RegId::COND:
    case RegId::FAST_PATH_ENABLE:
        return 0;
    }
}

/**
 * Write to an AICore register
 *
 * @param reg    Register identifier
 * @param value  Value to write
 */
__aicore__ inline void write_reg(RegId reg, uint64_t value) {
    switch (reg) {
    case RegId::COND:
        set_cond(static_cast<uint32_t>(value));
        break;
    case RegId::DATA_MAIN_BASE:
    case RegId::FAST_PATH_ENABLE:
        break;
    }
}

/**
 * Get the physical core ID from hardware
 *
 * @return Physical core ID (masked to 12 bits)
 */
__aicore__ inline uint32_t get_physical_core_id() { return static_cast<uint32_t>(get_coreid()) & AICORE_COREID_MASK; }

// CPU_SIM_SET_EXECUTION_CONTEXT - no-op on real hardware (context provided by CANN runtime)
#define CPU_SIM_SET_EXECUTION_CONTEXT(block_idx, subblock_id, subblock_dim) ((void)0)

// CPU_SIM_SET_TASK_COOKIE - no-op on real hardware
#define CPU_SIM_SET_TASK_COOKIE(cookie) ((void)0)

// platform_get_cpu_sim_task_cookie - no-op on real hardware
__aicore__ inline uint64_t platform_get_cpu_sim_task_cookie(uint32_t core_id, uint32_t reg_task_id) {
    (void)core_id;
    (void)reg_task_id;
    return 0;
}

// =============================================================================
// System Counter
// =============================================================================

/**
 * Get AICore system counter
 *
 * @return Hardware counter value (ticks)
 */
__aicore__ __attribute__((always_inline)) inline uint64_t get_sys_cnt_aicore() { return get_sys_cnt(); }

#endif  // PLATFORM_A2A3_AICORE_INNER_KERNEL_H_
