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
 * Implementation: src/platform/shared/aicpu/platform_regs.cpp (shared across all platforms)
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
 * @return Pointer (as uint64_t) to per-core register base address array
 */
uint64_t get_platform_regs();

/**
 * Set the per-core PMU MMIO register base address array.
 * On hardware this is distinct from set_platform_regs (different HAL addr_type).
 * On sim this points at host-allocated memory that backs the same offsets.
 *
 * @param pmu_regs  Pointer (as uint64_t) to per-core PMU register base address array, 0 if PMU unsupported
 */
void set_platform_pmu_reg_addrs(uint64_t pmu_regs);

/**
 * Get the per-core PMU MMIO register base address array.
 *
 * @return Pointer (as uint64_t) to per-core PMU register base address array, 0 if unset
 */
uint64_t get_platform_pmu_reg_addrs();

/**
 * Set the ACL device ordinal for the current run. Pushed by the platform layer
 * (kernel.cpp) before aicpu_execute() from KernelArgs.device_id; the executor
 * reads it to make the staged orchestration SO filename unique per device so
 * paired dies sharing the preinstall filesystem never collide.
 */
void set_orch_device_id(int device_id);

/** Get the ACL device ordinal set for the current run (0 if unset). */
int get_orch_device_id();

#ifdef __cplusplus
}
#endif

/**
 * Resolve a register identifier to its volatile MMIO pointer.
 *
 * Callers cache the result for hot-path register access (see scheduler
 * completion polling).
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
inline void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value) {
    reg_store_release(
        reinterpret_cast<volatile uint32_t *>(reg_base_addr + reg_offset(reg)), static_cast<uint32_t>(value)
    );
}

/**
 * Initialize AICore registers after core discovery
 *
 * This function performs platform-agnostic register initialization that works
 * for both a2a3 and a2a3sim, including enabling fast path control and clearing
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
