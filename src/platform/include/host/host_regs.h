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
 * Get AICore register base addresses for all cores
 *
 * @param regs Output vector (AIC cores followed by AIV cores)
 * @param device_id Device ID
 */
void get_aicore_regs(std::vector<int64_t>& regs, uint64_t device_id);

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
int init_aicore_register_addresses(
    uint64_t* runtime_regs_ptr,
    uint64_t device_id,
    MemoryAllocator& allocator);

#endif  // PLATFORM_HOST_HOST_REGS_H_
