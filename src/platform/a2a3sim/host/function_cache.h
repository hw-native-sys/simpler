/**
 * Function Cache Structures (Simulation)
 *
 * Defines data structures for caching compiled kernel binaries and managing
 * their addresses. This is a copy from a2a3 platform for API compatibility.
 *
 * For simulation, these structures are simplified since kernels are
 * registered as function pointers directly.
 */

#ifndef RUNTIME_FUNCTION_CACHE_H
#define RUNTIME_FUNCTION_CACHE_H

#include <cstdint>

/**
 * Single kernel binary container
 *
 * Contains the size and binary data for one compiled kernel.
 * In simulation mode, this is mainly used for API compatibility.
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
 * For simulation, we don't actually cache binary data since kernels
 * are registered as function pointers. This structure is kept for
 * API compatibility.
 */
struct CoreFunctionBinCache {
    uint64_t data_size;    // Total size of all data
    uint64_t num_kernels;  // Number of kernels

    uint64_t* get_offsets() {
        return reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(this) + sizeof(CoreFunctionBinCache));
    }

    uint8_t* get_binary_data() {
        return reinterpret_cast<uint8_t*>(get_offsets()) + num_kernels * sizeof(uint64_t);
    }

    CoreFunctionBin* get_kernel(uint64_t index) {
        if (index >= num_kernels) {
            return nullptr;
        }
        uint64_t offset = get_offsets()[index];
        return reinterpret_cast<CoreFunctionBin*>(get_binary_data() + offset);
    }

    uint64_t get_total_size() const {
        return sizeof(CoreFunctionBinCache) + num_kernels * sizeof(uint64_t) + data_size;
    }
};

#endif  // RUNTIME_FUNCTION_CACHE_H
