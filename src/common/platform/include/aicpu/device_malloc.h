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
 * @file device_malloc.h
 * @brief Device Memory Allocation Interface for AICPU
 *
 * Provides device-side memory allocation functions that work on both
 * real hardware (using HAL memory API for HBM allocation) and
 * simulation (using standard malloc/free).
 *
 * Platform Support (same shape for both arches):
 * - onboard: Real hardware with HAL memory API (halMemAlloc/halMemFree)
 * - sim: Host-based simulation using malloc/free
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEVICE_MALLOC_H_
#define SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEVICE_MALLOC_H_

#include <cstddef>

/**
 * Allocate device memory (HBM on real hardware, heap on simulation).
 *
 * On onboard: Allocates HBM memory via halMemAlloc. The returned pointer is
 * a device virtual address accessible by AIV/AIC cores. This is NOT the
 * same address space as AICPU-local malloc().
 *
 * On sim: Allocates host heap memory via malloc(). In simulation, all
 * address spaces are shared, so this is equivalent to regular malloc.
 *
 * @param size  Number of bytes to allocate
 * @return Pointer to allocated memory, or nullptr on failure
 */
void *aicpu_device_malloc(size_t size);

/**
 * Return the cached→uncached VA alias offset for device HBM (A5 double page
 * table), or 0 if unavailable.
 *
 * On A5, HBM is mapped twice: a cacheable view and an uncacheable view
 * separated by this fixed offset. The distributed engine's cross-core shared
 * segment (DistGlobal) must be accessed through the uncacheable alias so plain
 * loads/stores + device atomics stay coherent across AICores. A host-allocated
 * rtMalloc(RT_MEMORY_HBM) reserve is cacheable; adding this offset yields its
 * uncacheable alias. Returns 0 on sim or when the driver lacks the double page
 * table (caller falls back to the cacheable address).
 *
 * @return Uncacheable-alias VA offset in bytes, or 0 if unavailable.
 */
unsigned long long aicpu_device_nocache_offset();

/**
 * Free device memory previously allocated by aicpu_device_malloc().
 *
 * Safe to call with nullptr (no-op).
 *
 * @param ptr  Pointer to free (may be nullptr)
 */
void aicpu_device_free(void *ptr);

/**
 * DIAG: probe whether this device can hand out an AICore-uncacheable alias of a
 * given device VA. Fills every field; all halMemCtl calls are best-effort and
 * failures are reported (not fatal). Used to decide whether the distributed
 * engine's DistGlobal can be made uncacheable (route A) on this specific A5.
 *
 * @param va  A device VA (e.g. the DistGlobal reserve base) to query aliases for
 * @param out Filled with query results (see field comments)
 */
struct AicpuUncacheProbe {
    int hal_ctl_resolved;             // 1 if halMemCtl was dlsym-resolved
    int dpt_rc;                       // rc from CTRL_TYPE_GET_DOUBLE_PGTABLE_OFFSET
    unsigned long long dpt_offset;    // returned global nocache offset (0 == none)
    int feature_rc;                   // rc from CTRL_TYPE_SUPPORT_FEATURE
    unsigned long long feature_bits;  // returned feature bitmask
    int dcache_rc;                    // rc from CTRL_TYPE_GET_DCACHE_ADDR for `va`
    unsigned long long dcache_addr;   // returned dcache/alias addr for `va`
};
void aicpu_device_probe_uncacheable(void *va, struct AicpuUncacheProbe *out);

#endif  // SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEVICE_MALLOC_H_
