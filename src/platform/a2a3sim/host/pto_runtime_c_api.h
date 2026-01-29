/**
 * PTO Runtime C API (Simulation)
 *
 * Pure C interface for Python ctypes bindings. This is the SAME API
 * as the real a2a3 platform, ensuring compatibility.
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
 */
typedef void* RuntimeHandle;

/* ===========================================================================
 * Runtime API
 * ===========================================================================
 */

/**
 * Get the size of Runtime structure for memory allocation.
 *
 * @return Size of Runtime structure in bytes
 */
size_t get_runtime_size(void);

/**
 * Initialize a runtime with dynamic orchestration.
 *
 * @param runtime           User-allocated memory of size GetRuntimeSize()
 * @param orch_so_binary    Orchestration shared library binary data
 * @param orch_so_size      Size of orchestration SO binary in bytes
 * @param orch_func_name    Name of the orchestration function to call
 * @param func_args         Arguments for orchestration
 * @param func_args_count   Number of arguments
 * @return 0 on success, -1 on failure
 */
int init_runtime(RuntimeHandle runtime,
                const uint8_t* orch_so_binary,
                size_t orch_so_size,
                const char* orch_func_name,
                uint64_t* func_args,
                int func_args_count);

/* ===========================================================================
 * Device Memory API
 * ===========================================================================
 */

/**
 * Allocate memory (host memory in simulation).
 *
 * @param size  Size in bytes to allocate
 * @return Pointer on success, NULL on failure
 */
void* device_malloc(size_t size);

/**
 * Free memory.
 *
 * @param dev_ptr  Pointer to free
 */
void device_free(void* dev_ptr);

/**
 * Copy data (memcpy in simulation).
 *
 * @param dev_ptr   Destination pointer
 * @param host_ptr  Source pointer
 * @param size      Size in bytes to copy
 * @return 0 on success
 */
int copy_to_device(void* dev_ptr, const void* host_ptr, size_t size);

/**
 * Copy data (memcpy in simulation).
 *
 * @param host_ptr  Destination pointer
 * @param dev_ptr   Source pointer
 * @param size      Size in bytes to copy
 * @return 0 on success
 */
int copy_from_device(void* host_ptr, const void* dev_ptr, size_t size);

/**
 * Execute a runtime using thread-based simulation.
 *
 * @param runtime           Initialized runtime handle
 * @param aicpu_thread_num  Number of AICPU threads
 * @param block_dim         Number of blocks (1 block = 1 AIC + 2 AIV)
 * @param device_id         Device ID (ignored in simulation)
 * @param aicpu_binary      AICPU binary (ignored in simulation)
 * @param aicpu_size        Size of AICPU binary
 * @param aicore_binary     AICore binary (ignored in simulation)
 * @param aicore_size       Size of AICore binary
 * @return 0 on success
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
 * @param runtime  Runtime handle to finalize
 * @return 0 on success
 */
int finalize_runtime(RuntimeHandle runtime);

/**
 * Set device (no-op in simulation).
 *
 * @param device_id  Device ID
 * @return 0 on success
 */
int set_device(int device_id);

/**
 * Register a kernel for a func_id.
 *
 * In simulation mode, bin_data should contain a function pointer.
 *
 * @param func_id   Function identifier
 * @param bin_data  Pointer to function pointer
 * @param bin_size  Size (should be sizeof(uint64_t))
 * @return 0 on success
 */
int register_kernel(int func_id, const uint8_t* bin_data, size_t bin_size);

#ifdef __cplusplus
}
#endif

#endif  // PTO_RUNTIME_C_API_H
