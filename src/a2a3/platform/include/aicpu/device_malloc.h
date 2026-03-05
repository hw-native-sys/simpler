/**
 * @file device_malloc.h
 * @brief Device Memory Allocation Interface for AICPU
 *
 * Provides device-side memory allocation functions that work on both
 * real hardware (using HAL memory API for HBM allocation) and
 * simulation (using standard malloc/free).
 *
 * Platform Support:
 * - a2a3: Real hardware with HAL memory API (halMemAlloc/halMemFree)
 * - a2a3sim: Host-based simulation using malloc/free
 */

#ifndef PLATFORM_DEVICE_MALLOC_H_
#define PLATFORM_DEVICE_MALLOC_H_

#include <cstddef>

/**
 * Allocate device memory (HBM on real hardware, heap on simulation).
 *
 * On a2a3: Allocates HBM memory via halMemAlloc. The returned pointer is a
 * device virtual address accessible by AIV/AIC cores. This is NOT the same
 * address space as AICPU-local malloc().
 *
 * On a2a3sim: Allocates host heap memory via malloc(). In simulation, all
 * address spaces are shared, so this is equivalent to regular malloc.
 *
 * @param size  Number of bytes to allocate
 * @return Pointer to allocated memory, or nullptr on failure
 */
void* aicpu_device_malloc(size_t size);

/**
 * Free device memory previously allocated by aicpu_device_malloc().
 *
 * Safe to call with nullptr (no-op).
 *
 * @param ptr  Pointer to free (may be nullptr)
 */
void aicpu_device_free(void* ptr);

#endif  // PLATFORM_DEVICE_MALLOC_H_
