/**
 * PTO Runtime C API
 *
 * Pure C interface for Python ctypes bindings. Wraps C++ classes (Graph, DeviceRunner)
 * as opaque pointers and provides C functions to manipulate them.
 *
 * Key design:
 * - All functions use C linkage (extern "C")
 * - Opaque pointers hide C++ implementation details
 * - Error codes: 0 = success, negative = error
 * - Memory management: User allocates Graph with malloc(GetGraphSize())
 */

#ifndef PTO_RUNTIME_C_API_H
#define PTO_RUNTIME_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque pointer types for C interface.
 * These hide the C++ class implementations.
 */
typedef void* GraphHandle;

/* =========================================================================== */
/* Graph API */
/* =========================================================================== */

/**
 * Get the size of Graph structure for memory allocation.
 *
 * User should allocate: Graph* g = (Graph*)malloc(GetGraphSize());
 *
 * @return Size of Graph structure in bytes
 */
size_t GetGraphSize(void);

/**
 * Initialize a graph for the basic example.
 *
 * Uses placement new to construct Graph in user-allocated memory.
 * Builds the task graph, allocates device tensors, initializes data.
 * Does NOT initialize device runner - that happens in launch_graph().
 *
 * @param graph  User-allocated memory of size GetGraphSize()
 * @return 0 on success, -1 on failure
 */
int InitGraph(GraphHandle graph);

/**
 * Execute a graph on the device.
 *
 * Initializes DeviceRunner singleton (if first call), registers kernel
 * addresses, copies graph to device, launches kernels, synchronizes,
 * and copies graph back from device.
 *
 * @param graph            Initialized graph handle
 * @param aicpu_thread_num Number of AICPU scheduler threads
 * @param block_dim        Number of blocks (1 block = 1 AIC + 2 AIV)
 * @param device_id        Device ID (0-15)
 * @param aicpu_binary     AICPU shared object binary data
 * @param aicpu_size       Size of AICPU binary in bytes
 * @param aicore_binary    AICore kernel binary data
 * @param aicore_size      Size of AICore binary in bytes
 * @return 0 on success, error code on failure
 */
int launch_graph(GraphHandle graph,
                 int aicpu_thread_num, int block_dim,
                 int device_id,
                 const uint8_t* aicpu_binary, size_t aicpu_size,
                 const uint8_t* aicore_binary, size_t aicore_size);

/**
 * Finalize and cleanup a graph instance.
 *
 * Validates results, frees device tensors, calls Graph destructor.
 * After this call, user can free(graph).
 *
 * @param graph  Graph handle to finalize
 * @return 0 on success, -1 on failure
 */
int FinalizeGraph(GraphHandle graph);

/**
 * Set device and create streams for memory operations.
 *
 * Must be called before InitGraph() to enable device tensor allocation.
 * Only performs minimal initialization:
 * - rtSetDevice(device_id)
 * - Create AICPU and AICore streams
 *
 * Binary loading happens later in launch_graph().
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
 * and stores the GM address for later use by launch_graph().
 *
 * @param func_id   Function identifier (0, 1, 2, ...)
 * @param bin_data  Kernel .text section binary data
 * @param bin_size  Size of binary data in bytes
 * @return 0 on success, error code on failure
 */
int RegisterKernel(int func_id, const uint8_t* bin_data, size_t bin_size);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* PTO_RUNTIME_C_API_H */
