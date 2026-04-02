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
 * Provides register base addresses for AICPU to perform MMIO-based
 * task dispatch to AICore cores.
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
 * Initialize AICore register addresses for runtime
 *
 * Retrieves register addresses from HAL, allocates device memory,
 * copies addresses to device, and stores the device pointer in runtime.
 *
 * @param runtime_regs_ptr Pointer to the regs field (e.g., KernelArgs.regs)
 * @param device_id Device ID
 * @param allocator Memory allocator for device memory
 * @return 0 on success, negative on failure
 */
int init_aicore_register_addresses(uint64_t *runtime_regs_ptr, uint64_t device_id, MemoryAllocator &allocator);

#endif  // PLATFORM_HOST_HOST_REGS_H_
