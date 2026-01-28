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

#include "function_cache.h"
#include "kernel_args.h"
#include "memoryallocator.h"
#include "runtime.h"

/**
 * DeviceArgs structure for AICPU device arguments
 *
 * This structure contains pointers to device memory for the AICPU shared
 * object. The layout is hardcoded in libaicpu_extend_kernels.so, which expects
 * specific offsets for aicpuSoBin and aicpuSoLen fields.
 */
struct DeviceArgs {
    uint64_t unused[12] = {0};
    uint64_t aicpuSoBin{0};
    uint64_t aicpuSoLen{0};
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
     * @param hostDeviceArgs  Host-side device arguments to copy
     * @param allocator       Memory allocator to use
     * @return 0 on success, error code on failure
     */
    int InitDeviceArgs(const DeviceArgs& hostDeviceArgs, MemoryAllocator& allocator);

    /**
     * Free device memory allocated for device arguments
     *
     * @return 0 on success, error code on failure
     */
    int FinalizeDeviceArgs();

    /**
     * Initialize runtime arguments by allocating device memory and copying data
     *
     * @param hostRuntime  Host-side runtime to copy to device
     * @param allocator  Memory allocator to use
     * @return 0 on success, error code on failure
     */
    int InitRuntimeArgs(const Runtime& hostRuntime, MemoryAllocator& allocator);

    /**
     * Free device memory allocated for runtime arguments
     *
     * @return 0 on success, error code on failure
     */
    int FinalizeRuntimeArgs();

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
    uint64_t aicpuSoBin{0};
    uint64_t aicpuSoLen{0};
    MemoryAllocator* allocator_{nullptr};

    /**
     * Load shared object binary data and copy to device memory
     *
     * @param aicpuSoBinary  Binary data of the AICPU shared object
     * @param allocator      Memory allocator to use
     * @return 0 on success, error code on failure
     */
    int Init(const std::vector<uint8_t>& aicpuSoBinary, MemoryAllocator& allocator);

    /**
     * Free device memory allocated for shared object
     *
     * @return 0 on success, error code on failure
     */
    int Finalize();
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
    static DeviceRunner& Get();

    /**
     * Allocate device tensor memory
     *
     * @param bytes  Size of tensor in bytes
     * @return Device pointer on success, nullptr on failure
     */
    void* AllocateTensor(size_t bytes);

    /**
     * Free device tensor memory
     *
     * @param devPtr  Device pointer to free
     */
    void FreeTensor(void* devPtr);

    /**
     * Copy data from host to device
     *
     * @param devPtr   Device pointer
     * @param hostPtr  Host pointer
     * @param bytes    Number of bytes to copy
     * @return 0 on success, error code on failure
     */
    int CopyToDevice(void* devPtr, const void* hostPtr, size_t bytes);

    /**
     * Copy data from device to host
     *
     * @param hostPtr  Host pointer
     * @param devPtr   Device pointer
     * @param bytes    Number of bytes to copy
     * @return 0 on success, error code on failure
     */
    int CopyFromDevice(void* hostPtr, const void* devPtr, size_t bytes);

    /**
     * Execute a runtime
     *
     * This method:
     * 1. Initializes device if not already done (lazy initialization)
     * 2. Initializes worker handshake buffers in the runtime based on blockDim
     * 3. Transfers runtime to device memory
     * 4. Launches AICPU init kernel
     * 5. Launches AICPU main kernel
     * 6. Launches AICore kernel
     * 7. Synchronizes streams
     * 8. Cleans up runtime memory
     *
     * @param runtime             Runtime to execute (will be modified to
     * initialize workers)
     * @param blockDim            Number of blocks (1 block = 1 AIC + 2 AIV)
     * @param deviceId            Device ID (0-15)
     * @param aicpuSoBinary       Binary data of AICPU shared object
     * @param aicoreKernelBinary  Binary data of AICore kernel
     * @param launchAicpuNum      Number of AICPU instances (default: 1)
     * @return 0 on success, error code on failure
     */
    int Run(Runtime& runtime,
        int blockDim,
        int deviceId,
        const std::vector<uint8_t>& aicpuSoBinary,
        const std::vector<uint8_t>& aicoreKernelBinary,
        int launchAicpuNum = 1);

    /**
     * Print handshake results from device
     *
     * Copies handshake buffers from device and prints their status.
     * Must be called after Run() with the same runtime.
     *
     * @param runtime  The runtime whose handshake results should be printed
     */
    void PrintHandshakeResults(Runtime& runtime);

    /**
     * Cleanup all resources
     *
     * Frees all device memory, destroys streams, and resets state.
     *
     * @return 0 on success, error code on failure
     */
    int Finalize();

    /**
     * Launch an AICPU kernel
     *
     * Internal method used by Run(). Can be called directly for custom
     * workflows.
     *
     * @param stream      AICPU stream
     * @param kArgs       Kernel arguments
     * @param kernelName  Name of the kernel to launch
     * @param aicpuNum    Number of AICPU instances to launch
     * @return 0 on success, error code on failure
     */
    int LaunchAiCpuKernel(rtStream_t stream, KernelArgs* kArgs, const char* kernelName, int aicpuNum);

    /**
     * Launch an AICore kernel
     *
     * Internal method used by Run(). Can be called directly for custom
     * workflows.
     *
     * @param stream  AICore stream
     * @param runtime   Pointer to device runtime
     * @return 0 on success, error code on failure
     */
    int LaunchAicoreKernel(rtStream_t stream, Runtime* runtime);

    /**
     * Register a kernel binary for a func_id
     *
     * Receives pre-extracted .text section binary data from Python,
     * allocates device GM memory, copies the binary to device,
     * and stores the GM address in funcIdToAddr_.
     *
     * @param funcId   Function identifier (0, 1, 2, ...)
     * @param binData  Kernel .text section binary data
     * @param binSize  Size of binary data in bytes
     * @return 0 on success, -1 on error
     */
    int RegisterKernel(int funcId, const uint8_t* binData, size_t binSize);

    /**
     * Get functionBinAddr for a given func_id
     *
     * Returns the device GM address where the kernel binary resides.
     * This address can be cast to a function pointer and called.
     *
     * @param funcId  Function identifier
     * @return Device GM address of kernel, or 0 if not found
     */
    uint64_t GetFunctionBinAddr(int funcId);

    /**
     * Ensure device is set and streams are created (minimal initialization)
     *
     * This is called by set_device() C API to enable memory allocation
     * before InitRuntime(). Only performs:
     * - rtSetDevice(deviceId)
     * - Create AICPU and AICore streams
     *
     * @param deviceId  Device ID (0-15)
     * @return 0 on success, error code on failure
     */
    int EnsureDeviceSet(int deviceId);

private:
    DeviceRunner() = default;
    ~DeviceRunner();

    // Internal state
    int deviceId_{-1};
    int blockDim_{0};
    int coresPerBlockdim_{3};
    std::vector<uint8_t> aicoreKernelBinary_;

    // Memory management
    MemoryAllocator memAlloc_;

    // Device resources
    rtStream_t streamAicpu_{nullptr};
    rtStream_t streamAicore_{nullptr};
    AicpuSoInfo soInfo_;
    KernelArgsHelper kernelArgs_;
    DeviceArgs deviceArgs_;

    // Kernel binary management
    // Host-side storage for pending kernels (before device init)
    struct PendingKernel {
        std::vector<uint8_t> data;
    };
    std::map<int, PendingKernel> pendingKernels_;  // func_id -> host binary
    bool kernelsCommitted_{false};                 // true after kernels copied to device
    bool binariesLoaded_{false};                   // true after AICPU SO loaded
    std::map<int, uint64_t> funcIdToAddr_;         // func_id -> functionBinAddr (device GM)

    /**
     * Ensure device is initialized (lazy initialization)
     *
     * Checks if device is already initialized. If not, performs:
     * - rtSetDevice(deviceId)
     * - Create AICPU and AICore streams
     * - Load AICPU SO to device memory
     * - Initialize device args
     *
     * @param deviceId            Device ID (0-15)
     * @param aicpuSoBinary       Binary data of AICPU shared object
     * @param aicoreKernelBinary  Binary data of AICore kernel
     * @return 0 on success, error code on failure
     */
    int EnsureDeviceInitialized(
        int deviceId, const std::vector<uint8_t>& aicpuSoBinary, const std::vector<uint8_t>& aicoreKernelBinary);

    /**
     * Load AICPU SO and initialize device args
     *
     * Called by Run() after EnsureDeviceSet(). Performs:
     * - Load AICPU SO to device memory
     * - Initialize device args
     *
     * @param aicpuSoBinary       Binary data of AICPU shared object
     * @param aicoreKernelBinary  Binary data of AICore kernel
     * @return 0 on success, error code on failure
     */
    int EnsureBinariesLoaded(const std::vector<uint8_t>& aicpuSoBinary, const std::vector<uint8_t>& aicoreKernelBinary);

    /**
     * Copy pending kernels from host to device memory
     *
     * Called automatically by Run() if kernels haven't been committed yet.
     *
     * @return 0 on success, -1 on error
     */
    int CommitKernels();
};

#endif  // RUNTIME_DEVICERUNNER_H
