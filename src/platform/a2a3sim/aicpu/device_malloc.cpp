/**
 * @file device_malloc.cpp
 * @brief Device Memory Allocation for Simulation (a2a3sim)
 *
 * In simulation, all address spaces are shared between AICPU and AICore,
 * so standard malloc/free is sufficient for device memory allocation.
 */

#include "aicpu/device_malloc.h"

#include <cstdlib>

void* aicpu_device_malloc(size_t size) {
    return malloc(size);
}

void aicpu_device_free(void* ptr) {
    free(ptr);
}
