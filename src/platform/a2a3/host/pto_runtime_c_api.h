/**
 * PTO Runtime C API
 *
 * Pure C interface for Python ctypes bindings. Wraps C++ classes (Runtime,
 * DeviceRunner) as opaque pointers and provides C functions to manipulate them.
 *
 * Key design:
 * - All functions use C linkage (extern "C")
 * - Opaque pointers hide C++ implementation details
 * - Error codes: 0 = success, negative = error
 * - Memory management: User allocates Runtime with malloc(GetRuntimeSize())
 */

#ifndef PTO_RUNTIME_C_API_H
#define PTO_RUNTIME_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque pointer types for C interface.
 * These hide the C++ class implementations.
 */
typedef void* RuntimeHandle;

/**
 * Orchestration function signature.
 *
 * @param runtime   RuntimeHandle (opaque pointer to Runtime)
 * @param args      Unified argument array:
 *                  - args[0..input_count-1]: device pointers (cast to uint64_t)
 *                  - args[input_count..]: additional arguments
 * @param arg_count Total number of arguments
 * @return 0 on success, negative on error
 */
typedef int (*OrchestrationFunc)(void* runtime, uint64_t* args, int arg_count);

/* ===========================================================================
 */
/* Runtime API */
/* ===========================================================================
 */

/**
 * Get the size of Runtime structure for memory allocation.
 *
 * User should allocate: Runtime* r = (Runtime*)malloc(GetRuntimeSize());
 *
 * @return Size of Runtime structure in bytes
 */
size_t GetRuntimeSize(void);

/**
 * Initialize a runtime with dynamic orchestration.
 *
 * Uses placement new to construct Runtime in user-allocated memory.
 * Calls orchestration function to build the task graph.
 * The orchestration function is responsible for device memory management.
 *
 * @param runtime         User-allocated memory of size GetRuntimeSize()
 * @param orch_func       Orchestration function to build task graph
 * @param func_args       Arguments for orchestration (host pointers, sizes, etc.)
 * @param func_args_count Number of arguments
 * @return 0 on success, -1 on failure
 */
int InitRuntime(RuntimeHandle runtime,
                OrchestrationFunc orch_func,
                uint64_t* func_args,
                int func_args_count);

/* =========================================================================== */
/* Device Memory API (for use by orchestration functions) */
/* =========================================================================== */

/**
 * Allocate device memory.
 *
 * @param size  Size in bytes to allocate
 * @return Device pointer on success, NULL on failure
 */
void* DeviceMalloc(size_t size);

/**
 * Free device memory.
 *
 * @param devPtr  Device pointer to free
 */
void DeviceFree(void* devPtr);

/**
 * Copy data from host to device.
 *
 * @param devPtr   Device destination pointer
 * @param hostPtr  Host source pointer
 * @param size     Size in bytes to copy
 * @return 0 on success, error code on failure
 */
int CopyToDevice(void* devPtr, const void* hostPtr, size_t size);

/**
 * Copy data from device to host.
 *
 * @param hostPtr  Host destination pointer
 * @param devPtr   Device source pointer
 * @param size     Size in bytes to copy
 * @return 0 on success, error code on failure
 */
int CopyFromDevice(void* hostPtr, const void* devPtr, size_t size);

/**
 * Execute a runtime on the device.
 *
 * Initializes DeviceRunner singleton (if first call), registers kernel
 * addresses, copies runtime to device, launches kernels, synchronizes,
 * and copies runtime back from device.
 *
 * @param runtime         Initialized runtime handle
 * @param aicpu_thread_num Number of AICPU scheduler threads
 * @param block_dim        Number of blocks (1 block = 1 AIC + 2 AIV)
 * @param device_id        Device ID (0-15)
 * @param aicpu_binary     AICPU shared object binary data
 * @param aicpu_size       Size of AICPU binary in bytes
 * @param aicore_binary    AICore kernel binary data
 * @param aicore_size      Size of AICore binary in bytes
 * @return 0 on success, error code on failure
 */
int launch_runtime(RuntimeHandle runtime,
    int aicpu_thread_num,
    int block_dim,
    int device_id,
    const uint8_t* aicpu_binary,
    size_t aicpu_size,
    const uint8_t* aicore_binary,
    size_t aicore_size);

/**
 * Finalize and cleanup a runtime instance.
 *
 * Validates results, frees device tensors, calls Runtime destructor.
 * After this call, user can free(runtime).
 *
 * @param runtime  Runtime handle to finalize
 * @return 0 on success, -1 on failure
 */
int FinalizeRuntime(RuntimeHandle runtime);

/**
 * Set device and create streams for memory operations.
 *
 * Must be called before InitRuntime() to enable device tensor allocation.
 * Only performs minimal initialization:
 * - rtSetDevice(device_id)
 * - Create AICPU and AICore streams
 *
 * Binary loading happens later in launch_runtime().
 *
 * @param device_id  Device ID (0-15)
 * @return 0 on success, error code on failure
 */
int set_device(int device_id);

/**
 * Register a kernel binary for a func_id.
 *
 * Receives pre-extracted .text section binary data from Python,
 * allocates device GM memory, copies the binary to device,
 * and stores the GM address for later use by launch_runtime().
 *
 * @param func_id   Function identifier (0, 1, 2, ...)
 * @param bin_data  Kernel .text section binary data
 * @param bin_size  Size of binary data in bytes
 * @return 0 on success, error code on failure
 */
int RegisterKernel(int func_id, const uint8_t* bin_data, size_t bin_size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PTO_RUNTIME_C_API_H */
