/**
 * @file function_cache.h
 * @brief Function Cache Structures for Kernel Binary Management
 *
 * Defines data structures for caching compiled kernel binaries and managing
 * their addresses in memory (device GM memory for a2a3, host memory for a2a3sim).
 *
 * Platform Support:
 * - a2a3: Real hardware with device GM memory
 * - a2a3sim: Host-based simulation with host memory
 *
 * These structures follow the production system design from:
 * - src/interface/cache/core_func_data.h
 * - src/interface/cache/function_cache.h
 *
 * Memory Layout:
 * ┌────────────────────────────────────────────────┐
 * │ CoreFunctionBinCache                            │
 * │ ┌────────────────────────────────────────────┐ │
 * │ │ data_size                                  │ │
 * │ ├────────────────────────────────────────────┤ │
 * │ │ offset[0]                                  │ │
 * │ │ offset[1]                                  │ │
 * │ │ ...                                        │ │
 * │ ├────────────────────────────────────────────┤ │
 * │ │ CoreFunctionBin[0]                         │ │
 * │ │   size                                     │ │
 * │ │   data[...binary...]                       │ │
 * │ ├────────────────────────────────────────────┤ │
 * │ │ CoreFunctionBin[1]                         │ │
 * │ │   size                                     │ │
 * │ │   data[...binary...]                       │ │
 * │ └────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────┘
 */

#ifndef PLATFORM_FUNCTION_CACHE_H_
#define PLATFORM_FUNCTION_CACHE_H_

#include <cstdint>

/**
 * Single kernel binary container
 *
 * Contains the size and binary data for one compiled kernel.
 * The data field is a flexible array member that extends beyond
 * the struct boundary.
 */
#pragma pack(1)
struct CoreFunctionBin {
    uint64_t size;    // Size of binary data in bytes
    uint8_t data[0];  // Flexible array member for kernel binary
};
#pragma pack()

/**
 * Binary cache structure for all kernels
 *
 * This structure packs multiple kernel binaries into a single contiguous
 * memory block for efficient memory allocation and copying.
 *
 * Memory Layout:
 * [data_size][num_kernels][offset0][offset1]...[offsetN][CoreFunctionBin0][CoreFunctionBin1]...
 *
 * Each offset points to the start of a CoreFunctionBin structure relative
 * to the beginning of the cache.
 *
 * Platform Behavior:
 * - a2a3: Binaries are copied to device GM memory
 * - a2a3sim: Binaries are kept in host memory or registered as function pointers
 */
struct CoreFunctionBinCache {
    uint64_t data_size;    // Total size of all data (excluding this header)
    uint64_t num_kernels;  // Number of kernels in this cache

    /**
     * Get offset array pointer
     * @return Pointer to array of offsets
     */
    uint64_t* get_offsets() {
        return reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(this) + sizeof(CoreFunctionBinCache));
    }

    /**
     * Get pointer to binary data region
     * @return Pointer to start of binary data
     */
    uint8_t* get_binary_data() {
        return reinterpret_cast<uint8_t*>(get_offsets()) + num_kernels * sizeof(uint64_t);
    }

    /**
     * Get CoreFunctionBin by index
     * @param index  Kernel index
     * @return Pointer to CoreFunctionBin structure, nullptr if invalid index
     */
    CoreFunctionBin* get_kernel(uint64_t index) {
        if (index >= num_kernels) {
            return nullptr;
        }
        uint64_t offset = get_offsets()[index];
        return reinterpret_cast<CoreFunctionBin*>(get_binary_data() + offset);
    }

    /**
     * Calculate total cache size including header
     * @return Total size in bytes
     */
    uint64_t get_total_size() const {
        return sizeof(CoreFunctionBinCache) + num_kernels * sizeof(uint64_t) + data_size;
    }
};

#endif  // PLATFORM_FUNCTION_CACHE_H_
