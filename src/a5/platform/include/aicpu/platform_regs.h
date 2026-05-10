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
 * @file platform_regs.h
 * @brief Platform-level register access interface for AICPU
 *
 * Provides unified interface for:
 * 1. Platform register base address management (set/get_platform_regs)
 * 2. Register read/write operations (read_reg/write_reg)
 *
 * The platform layer calls set_platform_regs() before aicpu_execute(),
 * and runtime code calls get_platform_regs() and read_reg/write_reg()
 * for register communication with AICore.
 *
 * Implementation split:
 *   src/aicpu/platform_regs.cpp            -- shared: set/get_platform_regs, init/deinit, core count
 *   sim/aicpu/inner_platform_regs.cpp      -- read_reg/write_reg via sparse_reg_ptr (simulation)
 *   onboard/aicpu/inner_platform_regs.cpp  -- read_reg/write_reg via direct MMIO offset (hardware)
 */

#ifndef PLATFORM_AICPU_PLATFORM_REGS_H_
#define PLATFORM_AICPU_PLATFORM_REGS_H_

#include <cstdint>
#include <cstddef>
#include "common/platform_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Set the platform register base address array.
 * Called by the platform layer before aicpu_execute().
 *
 * @param regs  Pointer (as uint64_t) to per-core register base address array
 */
void set_platform_regs(uint64_t regs);

/**
 * Get the platform register base address array.
 * Called by runtime AICPU executor code that needs register access.
 *
 * On DAV_3510 the per-core halResMap(RES_AICORE) mapping covers both the
 * AICore SPR page (DATA_MAIN_BASE, COND, CTRL) and the PMU MMIO page, so
 * PMU helpers also read counters through this same register-base array.
 *
 * @return Pointer (as uint64_t) to per-core register base address array
 */
uint64_t get_platform_regs();

#ifdef __cplusplus
}
#endif

/**
 * Read a register value from an AICore's register block
 *
 * @param reg_base_addr  Base address of the AICore's register block
 * @param reg            Register identifier (C++ enum class)
 * @return Register value (zero-extended to uint64_t)
 */
uint64_t read_reg(uint64_t reg_base_addr, RegId reg);

/**
 * Write a value to an AICore's register
 *
 * @param reg_base_addr  Base address of the AICore's register block
 * @param reg            Register identifier (C++ enum class)
 * @param value          Value to write (truncated to register width)
 */
void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value);

/**
 * Initialize AICore registers after core discovery
 *
 * This function performs platform-agnostic register initialization that works
 * for both a5 and a5sim, including enabling fast path control and clearing
 * dispatch registers.
 *
 * @param reg_addr  Register base address of the AICore
 */
void platform_init_aicore_regs(uint64_t reg_addr);

/**
 * Deinitialize AICore registers before termination
 *
 * This function sends exit signal and closes fast path control.
 *
 * @param reg_addr  Register base address of the AICore
 * @return 0 if the core acknowledged exit, non-zero on timeout
 */
int32_t platform_deinit_aicore_regs(uint64_t reg_addr);

/**
 * Get physical core count for current platform
 *
 * This function returns the maximum valid physical_core_id value (exclusive upper bound).
 * Used for validating physical_core_id from AICore handshake before using as array index.
 *
 * @return Physical core count (exclusive upper bound)
 */
uint32_t platform_get_physical_cores_count();

/**
 * Invalidate data cache for a memory range.
 *
 * On ARM64 AICPU, DMA writes from the host (rtMemcpy) go directly to HBM
 * without invalidating the AICPU's data cache.  When rtMalloc returns the
 * same device address across rounds, the AICPU may read stale cached data
 * instead of the fresh values written by the host.
 *
 * On real hardware (onboard): performs DC CIVAC per cache line + DSB/ISB.
 * On simulation (sim): no-op.
 *
 * @param addr  Start address of the memory range
 * @param size  Size of the memory range in bytes
 */
void cache_invalidate_range(const void *addr, size_t size);

/**
 * Clean data cache for a memory range back to global memory.
 *
 * On real hardware (onboard): performs DC CVAC per cache line + DSB/ISB.
 * On simulation (sim): no-op.
 *
 * @param addr  Start address of the memory range
 * @param size  Size of the memory range in bytes
 */
void cache_flush_range(const void *addr, size_t size);

#endif  // PLATFORM_AICPU_PLATFORM_REGS_H_
