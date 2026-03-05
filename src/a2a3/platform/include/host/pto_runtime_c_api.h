/**
 * PTO Runtime C API
 *
 * Pure C interface for Python ctypes bindings. Wraps C++ classes (Runtime,
 * DeviceRunner) as opaque pointers and provides C functions to manipulate them.
 *
 * This interface is shared across all platforms (a2a3, a2a3sim, etc.) to ensure
 * compatibility and consistent behavior.
 *
 * Key design:
 * - All functions use C linkage (extern "C")
 * - Opaque pointers hide C++ implementation details
 * - Error codes: 0 = success, negative = error
 * - Memory management: User allocates Runtime with malloc(get_runtime_size())
 */

#ifndef PTO_RUNTIME_C_API_H
#define PTO_RUNTIME_C_API_H

#include <stddef.h>
#include <stdint.h>

#include "common/compile_strategy.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===========================================================================
 * Compile Strategy API
 *
 * get_incore_compiler() and get_orchestration_compiler() are declared in
 * host/runtime_compile_info.h and linked into this library. They return
 * ToolchainType values indicating which compiler to use.
 * get_platform() is declared in host/platform_compile_info.h.
 * ===========================================================================
 */

/**
 * Argument type for init_runtime.
 * Describes how each argument should be handled for device orchestration.
 */
typedef enum {
    ARG_SCALAR = 0,      // Scalar value, passed directly
    ARG_INPUT_PTR = 1,   // Input pointer: device_malloc + copy_to_device
    ARG_OUTPUT_PTR = 2,  // Output pointer: device_malloc + record for copy-back
    ARG_INOUT_PTR = 3,   // Input/output: copy_to_device + copy-back
} ArgType;

/**
 * Opaque pointer types for C interface.
 * These hide the C++ class implementations.
 */
typedef void* RuntimeHandle;

/* ===========================================================================
 * Runtime API
 * ===========================================================================
 */

/**
 * Get the size of Runtime structure for memory allocation.
 *
 * User should allocate: Runtime* r = (Runtime*)malloc(get_runtime_size());
 *
 * @return Size of Runtime structure in bytes
 */
size_t get_runtime_size(void);

/**
 * Initialize a runtime with dynamic orchestration and kernel binaries.
 *
 * Uses placement new to construct Runtime in user-allocated memory.
 * Registers kernel binaries to device memory, storing addresses directly
 * in Runtime's func_id_to_addr_[] array.
 * Then loads the orchestration shared library from binary data, resolves the
 * specified function, and calls it to build the task graph.
 *
 * IMPORTANT: set_device() MUST be called before this function if kernel_count > 0.
 *
 * @param runtime           User-allocated memory of size get_runtime_size()
 * @param orch_so_binary    Orchestration shared library binary data
 * @param orch_so_size      Size of orchestration SO binary in bytes
 * @param orch_func_name    Name of the orchestration function to call
 * @param func_args         Arguments for orchestration (host pointers, sizes, etc.)
 * @param func_args_count   Number of arguments
 * @param arg_types         Array describing each argument's type (ArgType enum)
 * @param arg_sizes         Array of sizes for pointer arguments (0 for scalars)
 * @param kernel_func_ids   Array of kernel function IDs (can be NULL if kernel_count == 0)
 * @param kernel_binaries   Array of pointers to kernel binary data
 * @param kernel_sizes      Array of kernel binary sizes in bytes
 * @param kernel_count      Number of kernels to register
 * @return 0 on success, -1 on failure
 */
int init_runtime(RuntimeHandle runtime,
                const uint8_t* orch_so_binary,
                size_t orch_so_size,
                const char* orch_func_name,
                uint64_t* func_args,
                int func_args_count,
                int* arg_types,
                uint64_t* arg_sizes,
                const int* kernel_func_ids,
                const uint8_t* const* kernel_binaries,
                const size_t* kernel_sizes,
                int kernel_count);

/* ===========================================================================
 * Device Memory API (for use by orchestration functions)
 * ===========================================================================
 */

/**
 * Allocate device memory.
 *
 * @param size  Size in bytes to allocate
 * @return Device pointer on success, NULL on failure
 */
void* device_malloc(size_t size);

/**
 * Free device memory.
 *
 * @param dev_ptr  Device pointer to free
 */
void device_free(void* dev_ptr);

/**
 * Copy data from host to device.
 *
 * @param dev_ptr   Device destination pointer
 * @param host_ptr  Host source pointer
 * @param size     Size in bytes to copy
 * @return 0 on success, error code on failure
 */
int copy_to_device(void* dev_ptr, const void* host_ptr, size_t size);

/**
 * Copy data from device to host.
 *
 * @param host_ptr  Host destination pointer
 * @param dev_ptr   Device source pointer
 * @param size     Size in bytes to copy
 * @return 0 on success, error code on failure
 */
int copy_from_device(void* host_ptr, const void* dev_ptr, size_t size);

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
int finalize_runtime(RuntimeHandle runtime);

/**
 * Set device and create streams for memory operations.
 *
 * Must be called before init_runtime() to enable device tensor allocation.
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

/* Note: register_kernel() has been internalized into init_runtime().
 * Kernel binaries are now passed directly to init_runtime() which handles
 * registration and stores addresses in Runtime's func_id_to_addr_[] array.
 */

/**
 * Record a tensor pair for copy-back during finalize.
 *
 * Used by orchestration to track host-device memory mappings.
 * During finalize_runtime(), tensors with non-null host_ptr will be
 * copied back from device to host, then all device memory is freed.
 *
 * @param runtime   Runtime handle
 * @param host_ptr  Host memory pointer (NULL if no copy-back needed)
 * @param dev_ptr   Device memory pointer
 * @param size      Size of tensor in bytes
 */
void record_tensor_pair(RuntimeHandle runtime,
                       void* host_ptr,
                       void* dev_ptr,
                       size_t size);


/**
 * Enable or disable performance profiling for swimlane visualization.
 *
 * Must be called before init_runtime() to enable profiling.
 * When enabled, the runtime will record task execution timestamps on AICore/AICPU
 * and generate swim_time.json after finalize_runtime().
 *
 * @param runtime  Runtime handle
 * @param enabled  1 to enable profiling, 0 to disable
 * @return 0 on success, -1 on failure
 */
int enable_runtime_profiling(RuntimeHandle runtime, int enabled);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PTO_RUNTIME_C_API_H */
