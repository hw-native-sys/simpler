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
 * @file host_regs.h
 * @brief AICore register address retrieval via CANN HAL APIs
 *
 * Provides register base addresses for AICPU/AICore to perform MMIO access
 * (task dispatch via CTRL, counter reads via PMU).
 */

#ifndef PLATFORM_HOST_HOST_REGS_H_
#define PLATFORM_HOST_HOST_REGS_H_

#include <cstdint>
#include <vector>

// Forward declaration
class MemoryAllocator;

/**
 * AICore bitmap buffer length for DAV_2201
 * Used for querying valid AICore cores via halGetDeviceInfoByBuff
 */
constexpr uint8_t PLATFORM_AICORE_MAP_BUFF_LEN = 2;

/**
 * Which MMIO register page to query from the HAL.
 *
 * Each kind maps to a distinct HAL ADDR_MAP_TYPE constant; all other logic
 * (per-core stride, AIC/AIV layout, device copy) is identical.
 */
enum class AicoreRegKind {
    Ctrl,  // Task dispatch MMIO (DATA_MAIN_BASE / COND / CTRL SPR frame)
    Pmu,   // PMU counter MMIO (per-core CNT/CTRL/IDX pages)
};

/**
 * Initialize per-core AICore register addresses for runtime.
 *
 * Retrieves addresses from the HAL for the requested register kind, allocates
 * device memory, copies the address array to device, and stores the device
 * pointer via *runtime_regs_ptr.
 *
 * Failure returns a negative code; no placeholder addresses are generated.
 * Callers must treat a failed return as fatal for that register kind (e.g. a
 * failed Ctrl query means runtime cannot dispatch; a failed Pmu query means
 * PMU must be disabled).
 *
 * @param runtime_regs_ptr  Pointer to the KernelArgs slot (e.g. &args.regs or
 *                          &args.pmu_reg_addrs) that will receive the device
 *                          pointer to the address array.
 * @param device_id         Device ID
 * @param allocator         Memory allocator for device memory
 * @param kind              Which register page to query (Ctrl or Pmu)
 * @return 0 on success, negative on failure
 */
int init_aicore_register_addresses(
    uint64_t *runtime_regs_ptr, uint64_t device_id, MemoryAllocator &allocator, AicoreRegKind kind
);

#endif  // PLATFORM_HOST_HOST_REGS_H_
