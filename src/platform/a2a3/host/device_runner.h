/**
 * Device Runner - Ascend Device Execution Utilities
 *
 * This module provides utilities for launching and managing AICPU and AICore
 * kernels on Ascend devices using CANN runtime APIs.
 *
 * Key Components:
 * - DeviceArgs: AICPU device argument structure
 * - KernelArgsHelper: Helper for managing kernel arguments with device memory
 * - AicpuSoInfo: AICPU shared object (.so) file management
 * - DeviceRunner: Singleton for kernel launching and execution
 */

#ifndef RUNTIME_DEVICERUNNER_H
#define RUNTIME_DEVICERUNNER_H

#include <runtime/rt.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common/kernel_args.h"
#include "common/platform_config.h"
#include "host/function_cache.h"
#include "host/memory_allocator.h"
#include "runtime.h"

/**
 * DeviceArgs structure for AICPU device arguments
 *
 * This structure contains pointers to device memory for the AICPU shared
 * object. The layout is hardcoded in libaicpu_extend_kernels.so, which expects
 * specific offsets for aicpu_so_bin and aicpu_so_len fields.
 */
struct DeviceArgs {
    uint64_t unused[12] = {0};
    uint64_t aicpu_so_bin{0};
    uint64_t aicpu_so_len{0};
};

/**
 * Helper class for managing KernelArgs with device memory
 *
 * This class wraps KernelArgs and provides host-side initialization methods
 * for allocating device memory and copying data to the device. It separates
 * the concerns of device memory management (host-only) from the structure
 * layout (shared with kernels).
 *
 * The helper provides implicit conversion to KernelArgs* for seamless use
 * with runtime APIs.
 */
struct KernelArgsHelper {
    KernelArgs args;
    MemoryAllocator* allocator_{nullptr};

    /**
     * Initialize device arguments by allocating device memory and copying data
     *
     * @param host_device_args  Host-side device arguments to copy
     * @param allocator       Memory allocator to use
     * @return 0 on success, error code on failure
     */
    int init_device_args(const DeviceArgs& host_device_args, MemoryAllocator& allocator);

    /**
     * Free device memory allocated for device arguments
     *
     * @return 0 on success, error code on failure
     */
    int finalize_device_args();

    /**
     * Initialize runtime arguments by allocating device memory and copying data
     *
     * @param host_runtime  Host-side runtime to copy to device
     * @param allocator  Memory allocator to use
     * @return 0 on success, error code on failure
     */
    int init_runtime_args(const Runtime& host_runtime, MemoryAllocator& allocator);

    /**
     * Free device memory allocated for runtime arguments
     *
     * @return 0 on success, error code on failure
     */
    int finalize_runtime_args();

    /**
     * Implicit conversion operators for seamless use with runtime APIs
     *
     * These operators allow KernelArgsHelper to be used wherever KernelArgs*
     * is expected, enabling transparent device memory management while
     * maintaining API compatibility.
     */
    operator KernelArgs*() { return &args; }
    KernelArgs* operator&() { return &args; }
};

/**
 * AICPU shared object information and management
 *
 * This class manages loading and device memory allocation for AICPU
 * shared object (.so) files.
 */
struct AicpuSoInfo {
    uint64_t aicpu_so_bin{0};
    uint64_t aicpu_so_len{0};
    MemoryAllocator* allocator_{nullptr};

    /**
     * Load shared object binary data and copy to device memory
     *
     * @param aicpu_so_binary  Binary data of the AICPU shared object
     * @param allocator      Memory allocator to use
     * @return 0 on success, error code on failure
     */
    int init(const std::vector<uint8_t>& aicpu_so_binary, MemoryAllocator& allocator);

    /**
     * Free device memory allocated for shared object
     *
     * @return 0 on success, error code on failure
     */
    int finalize();
};

/**
 * Device runner singleton for kernel execution
 *
 * This class provides a unified interface for launching AICPU and AICore
 * kernels on Ascend devices. It handles:
 * - Device initialization and resource management
 * - Tensor memory allocation and data transfer
 * - AICPU kernel launching with dynamic arguments
 * - AICore kernel registration and launching
 * - Coordinated execution of both kernel types
 * - Runtime execution workflow
 */
class DeviceRunner {
public:
    /**
     * Get singleton instance
     *
     * @return Reference to the singleton DeviceRunner instance
     */
    static DeviceRunner& get();

    /**
     * Allocate device tensor memory
     *
     * @param bytes  Size of tensor in bytes
     * @return Device pointer on success, nullptr on failure
     */
    void* allocate_tensor(size_t bytes);

    /**
     * Free device tensor memory
     *
     * @param dev_ptr  Device pointer to free
     */
    void free_tensor(void* dev_ptr);

    /**
     * Copy data from host to device
     *
     * @param dev_ptr   Device pointer
     * @param host_ptr  Host pointer
     * @param bytes    Number of bytes to copy
     * @return 0 on success, error code on failure
     */
    int copy_to_device(void* dev_ptr, const void* host_ptr, size_t bytes);

    /**
     * Copy data from device to host
     *
     * @param host_ptr  Host pointer
     * @param dev_ptr   Device pointer
     * @param bytes    Number of bytes to copy
     * @return 0 on success, error code on failure
     */
    int copy_from_device(void* host_ptr, const void* dev_ptr, size_t bytes);

    /**
     * Execute a runtime
     *
     * This method:
     * 1. Initializes device if not already done (lazy initialization)
     * 2. Initializes worker handshake buffers in the runtime based on block_dim
     * 3. Transfers runtime to device memory
     * 4. Launches AICPU init kernel
     * 5. Launches AICPU main kernel
     * 6. Launches AICore kernel
     * 7. Synchronizes streams
     * 8. Cleans up runtime memory
     *
     * @param runtime             Runtime to execute (will be modified to
     * initialize workers)
     * @param block_dim            Number of blocks (1 block = 1 AIC + 2 AIV)
     * @param device_id            Device ID (0-15)
     * @param aicpu_so_binary       Binary data of AICPU shared object
     * @param aicore_kernel_binary  Binary data of AICore kernel
     * @param launch_aicpu_num      Number of AICPU instances (default: 1)
     * @return 0 on success, error code on failure
     */
    int run(Runtime& runtime,
        int block_dim,
        int device_id,
        const std::vector<uint8_t>& aicpu_so_binary,
        const std::vector<uint8_t>& aicore_kernel_binary,
        int launch_aicpu_num = 1);

    /**
     * Print handshake results from device
     *
     * Copies handshake buffers from device and prints their status.
     * Must be called after run() and before finalize().
     */
    void print_handshake_results();

    /**
     * Cleanup all resources
     *
     * Frees all device memory, destroys streams, and resets state.
     *
     * @return 0 on success, error code on failure
     */
    int finalize();

    /**
     * Launch an AICPU kernel
     *
     * Internal method used by run(). Can be called directly for custom
     * workflows.
     *
     * @param stream      AICPU stream
     * @param k_args       Kernel arguments
     * @param kernel_name  Name of the kernel to launch
     * @param aicpu_num    Number of AICPU instances to launch
     * @return 0 on success, error code on failure
     */
    int launch_aicpu_kernel(rtStream_t stream, KernelArgs* k_args, const char* kernel_name, int aicpu_num);

    /**
     * Launch an AICore kernel
     *
     * Internal method used by run(). Can be called directly for custom
     * workflows.
     *
     * @param stream  AICore stream
     * @param runtime   Pointer to device runtime
     * @return 0 on success, error code on failure
     */
    int launch_aicore_kernel(rtStream_t stream, Runtime* runtime);

    /**
     * Upload a kernel binary to device memory
     *
     * IMPORTANT: ensure_device_set() must be called before this function.
     * Kernels are immediately copied to device memory.
     *
     * Receives pre-extracted .text section binary data,
     * allocates device GM memory, copies the binary to device,
     * and returns the device GM address. The caller is responsible
     * for storing this address (typically in Runtime::func_id_to_addr_[]).
     *
     * If the kernel is already uploaded (same func_id), returns the
     * cached address without re-uploading.
     *
     * @param func_id   Function identifier (0, 1, 2, ...) for caching
     * @param bin_data  Kernel .text section binary data
     * @param bin_size  Size of binary data in bytes
     * @return Device GM address of kernel on success, 0 on error
     */
    uint64_t upload_kernel_binary(int func_id, const uint8_t* bin_data, size_t bin_size);

    /**
     * Ensure device is set and streams are created (minimal initialization)
     *
     * This is called by set_device() C API to enable memory allocation
     * before init_runtime(). Only performs:
     * - rtSetDevice(device_id)
     * - Create AICPU and AICore streams
     *
     * @param device_id  Device ID (0-15)
     * @return 0 on success, error code on failure
     */
    int ensure_device_set(int device_id);

private:
    DeviceRunner() = default;
    ~DeviceRunner();

    // Internal state
    int device_id_{-1};
    int block_dim_{0};
    int cores_per_blockdim_{PLATFORM_CORES_PER_BLOCKDIM};
    int worker_count_{0};  // Stored for print_handshake_results in destructor
    std::vector<uint8_t> aicore_kernel_binary_;

    // Memory management
    MemoryAllocator mem_alloc_;

    // Device resources
    rtStream_t stream_aicpu_{nullptr};
    rtStream_t stream_aicore_{nullptr};
    AicpuSoInfo so_info_;
    KernelArgsHelper kernel_args_;
    DeviceArgs device_args_;

    // Kernel binary management
    bool binaries_loaded_{false};            // true after AICPU SO loaded
    std::map<int, uint64_t> func_id_to_addr_;  // func_id -> function_bin_addr (device GM)

    /**
     * Ensure device is initialized (lazy initialization)
     *
     * Checks if device is already initialized. If not, performs:
     * - rtSetDevice(device_id)
     * - Create AICPU and AICore streams
     * - Load AICPU SO to device memory
     * - Initialize device args
     *
     * @param device_id            Device ID (0-15)
     * @param aicpu_so_binary       Binary data of AICPU shared object
     * @param aicore_kernel_binary  Binary data of AICore kernel
     * @return 0 on success, error code on failure
     */
    int ensure_device_initialized(int device_id,
                                const std::vector<uint8_t>& aicpu_so_binary,
                                const std::vector<uint8_t>& aicore_kernel_binary);

    /**
     * Load AICPU SO and initialize device args
     *
     * Called by run() after ensure_device_set(). Performs:
     * - Load AICPU SO to device memory
     * - Initialize device args
     *
     * @param aicpu_so_binary       Binary data of AICPU shared object
     * @param aicore_kernel_binary  Binary data of AICore kernel
     * @return 0 on success, error code on failure
     */
    int ensure_binaries_loaded(const std::vector<uint8_t>& aicpu_so_binary, const std::vector<uint8_t>& aicore_kernel_binary);
};

#endif  // RUNTIME_DEVICERUNNER_H
