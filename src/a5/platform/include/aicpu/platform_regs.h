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

#include <cstddef>
#include <cstdint>
#include "aicpu/cache_maintenance.h"
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
 * Resolve a register identifier to its volatile MMIO pointer.
 *
 * Onboard: simple base + offset. Sim: uses sparse_reg_ptr() for the
 * compact 16KB sparse layout. Callers cache the result for hot-path
 * register access (see scheduler completion polling).
 *
 * @param reg_base_addr  Base address of the AICore's register block
 * @param reg            Register identifier
 * @return Volatile pointer to the 32-bit register
 */
volatile uint32_t *get_reg_ptr(uint64_t reg_base_addr, RegId reg);

/**
 * Register-cell synchronizing accessors for the AICPU<->AICore handshake gate
 * (COND / DATA_MAIN_BASE). Implemented per-variant:
 *   sim/aicpu/inner_platform_regs.cpp     -- atomic acquire/release. The
 *       simulated registers are plain host memory shared across host threads,
 *       so the access itself must carry inter-thread happens-before (and be
 *       visible to ThreadSanitizer) against the AICore side.
 *   onboard/aicpu/inner_platform_regs.cpp -- plain Device-nGnRnE MMIO load/store
 *       (atomics are not valid on Device memory); ordering is the caller's
 *       explicit rmb()/wmb(), so the hardware path is the legacy behavior.
 */
uint32_t reg_load_acquire(const volatile uint32_t *p);
void reg_store_release(volatile uint32_t *p, uint32_t v);

/**
 * Read a register value from an AICore's register block
 *
 * No memory barrier is emitted. Callers that read a hand-off bit
 * written by AICore and then read AICore-published cacheable data
 * must insert an explicit rmb() between the two loads (ARM64 allows
 * Device-nGnRnE -> Normal-cacheable load reorder).
 *
 * @param reg_base_addr  Base address of the AICore's register block
 * @param reg            Register identifier (C++ enum class)
 * @return Register value (zero-extended to uint64_t)
 */
uint64_t read_reg(uint64_t reg_base_addr, RegId reg);

/**
 * Write a value to an AICore's register
 *
 * No memory barrier is emitted. Callers publishing cacheable data
 * that AICore will read after observing this register write must
 * insert an explicit wmb() before the call.
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

// Signal one core to exit, without waiting for its acknowledgement. The store is
// posted -- the window is Device-nGnRE, see docs/hardware/mmio-performance.md --
// so a caller signalling several cores owes one wmb() before it starts polling.
void platform_signal_aicore_exit(uint64_t reg_addr);

// One absolute timeout deadline shared by a group of exiting cores.
uint64_t platform_aicore_exit_deadline();

// Quiesce a core whose COND the caller has already observed as EXITED: dispatch
// back to idle, with that posted store read back so it is complete. The readback
// targets DATA_MAIN_BASE, the register just written: nR orders accesses within
// one peripheral, and DATA_MAIN_BASE (0xD0) and COND (0x5108) do not share a
// 4 KB granule, so a COND load would not order against this store. Issues no
// fence of its own -- a caller closing several windows owes one rmb() after the
// last call and before it publishes anything those closes must precede.
void platform_close_aicore_window(uint64_t reg_addr);

// Retires one exclusively-claimed set of cores: signal every member, collect
// every ACK against one shared deadline, then close every acknowledged window.
// Callers claim their targets first, so concurrent callers never name the same
// core and the set need not be the whole chip. An unacknowledged core is not
// closed and stays the host recovery path's responsibility.
//
// `released`, when non-null, receives one flag per target and lets the caller
// name the cores it failed to retire; this layer takes no logging dependency.
// Every path that returns fills those entries first, rejection included, so the
// caller may read them without initializing the buffer; a `count` above
// PLATFORM_MAX_CORES is rejected and only that many are filled.
// Returns 0 when every target was closed, -1 on timeout or invalid targets.
int32_t
platform_retire_aicore_group(const uint64_t *reg_addrs, size_t count, uint64_t deadline, bool *released = nullptr);

/**
 * Retire one AICore: signal it, wait for its acknowledgement, close its
 * window. Equivalent to platform_retire_aicore_group over a single core;
 * callers retiring several should use the group directly so they share one
 * deadline and one sweep.
 *
 * @param reg_addr  Register base address of the AICore
 * @return 0 if the core acknowledged exit, non-zero on timeout
 */
int32_t platform_deinit_aicore_regs(uint64_t reg_addr);

/**
 * Variant-specific AICore deinit wait timeout, in ticks of get_sys_cnt_aicpu.
 *
 * Implemented per-variant in:
 *   sim/aicpu/inner_platform_regs.cpp    -- larger budget (OS scheduling)
 *   onboard/aicpu/inner_platform_regs.cpp -- 1 s (hardware hang detection)
 *
 * Rationale: on hardware, AICore is independent silicon and 1 s of
 * non-response means the op got STARS-killed or the core is wedged. In
 * sim, "AICore" is a host CPU thread; "no response in 1 s" can just mean
 * the OS scheduler hasn't given it a slice on a CPU-starved CI runner.
 * Keeping the hardware budget at 1 s preserves fast hang detection;
 * widening the sim budget tolerates scheduler jitter without false
 * positives.
 */
uint64_t inner_get_deinit_timeout_ticks();

/**
 * Get physical core count for current platform
 *
 * This function returns the maximum valid physical_core_id value (exclusive upper bound).
 * Used for validating physical_core_id from AICore handshake before using as array index.
 *
 * @return Physical core count (exclusive upper bound)
 */
uint32_t platform_get_physical_cores_count();

#endif  // PLATFORM_AICPU_PLATFORM_REGS_H_
