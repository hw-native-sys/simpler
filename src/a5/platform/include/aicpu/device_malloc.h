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
 * Platform Support:
 * - a5: Real hardware with HAL memory API (halMemAlloc/halMemFree)
 * - a5sim: Host-based simulation using malloc/free
 */

#ifndef PLATFORM_DEVICE_MALLOC_H_
#define PLATFORM_DEVICE_MALLOC_H_

#include <cstddef>

/**
 * Allocate device memory (HBM on real hardware, heap on simulation).
 *
 * On a5: Allocates HBM memory via halMemAlloc. The returned pointer is a
 * device virtual address accessible by AIV/AIC cores. This is NOT the same
 * address space as AICPU-local malloc().
 *
 * On a5sim: Allocates host heap memory via malloc(). In simulation, all
 * address spaces are shared, so this is equivalent to regular malloc.
 *
 * @param size  Number of bytes to allocate
 * @return Pointer to allocated memory, or nullptr on failure
 */
void *aicpu_device_malloc(size_t size);

/**
 * Free device memory previously allocated by aicpu_device_malloc().
 *
 * Safe to call with nullptr (no-op).
 *
 * @param ptr  Pointer to free (may be nullptr)
 */
void aicpu_device_free(void *ptr);

#endif  // PLATFORM_DEVICE_MALLOC_H_
