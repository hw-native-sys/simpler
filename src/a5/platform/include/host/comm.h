/**
 * Backend-neutral distributed communication C API.
 *
 * Provides five primitives for multi-rank communication: init, allocate
 * shared windows, query local window base, barrier, and destroy.
 *
 * Implementations:
 *   onboard/host/comm_hccl.cpp — HCCL backend (links CANN hccl/hccl_fwk)
 *   sim/host/comm_sim.cpp      — malloc-based simulation
 *
 * All functions are compiled into libhost_runtime.so. The linker selects
 * the implementation at build time (onboard vs sim), with no runtime
 * dispatch or virtual functions.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CommHandle_* CommHandle;

/**
 * Initialize a communicator for the given rank.
 *
 * On the HCCL backend this performs ACL init, RootInfo exchange (rank 0
 * writes the file, others wait), and HcclCommInitRootInfo.
 *
 * @param rank           This process's rank (0-based).
 * @param nranks         Total number of ranks.
 * @param device_id      Physical device ID used by this process.
 * @param rootinfo_path  Filesystem path used to exchange root info between
 *                       ranks (rank 0 writes, others read).
 * @return Opaque handle, or NULL on failure.
 */
CommHandle comm_init(int rank, int nranks, int device_id, const char* rootinfo_path);

/**
 * Allocate RDMA / shared-memory windows and populate the device context.
 *
 * On HCCL this calls HcclAllocComResourceByTiling and extracts per-rank
 * window addresses (MESH or RING topology).  On sim it mallocs a shared
 * region and partitions it.
 *
 * @param h               Handle from comm_init().
 * @param win_size        Window size hint (bytes per rank).  The backend
 *                        may allocate more; actual size is stored in the
 *                        returned device context.
 * @param device_ctx_out  Receives a device pointer to a CommDeviceContext
 *                        struct that can be passed to device kernels.
 * @return 0 on success, non-zero on failure.
 */
int comm_alloc_windows(CommHandle h, size_t win_size, uint64_t* device_ctx_out);

/**
 * Get the base address of this rank's local window.
 *
 * Window buffers allocated via comm_alloc_windows() are contiguous per
 * rank.  This returns the start of the local rank's region.
 *
 * @param h         Handle from comm_init().
 * @param base_out  Receives the device-pointer base address.
 * @return 0 on success, non-zero on failure.
 */
int comm_get_local_window_base(CommHandle h, uint64_t* base_out);

/**
 * Synchronize all ranks.
 *
 * Blocks until every rank in the communicator has called comm_barrier().
 *
 * @param h  Handle from comm_init().
 * @return 0 on success, non-zero on failure.
 */
int comm_barrier(CommHandle h);

/**
 * Destroy the communicator and release all resources.
 *
 * After this call the handle is invalid.
 *
 * @param h  Handle from comm_init().
 * @return 0 on success, non-zero on failure.
 */
int comm_destroy(CommHandle h);

#ifdef __cplusplus
}
#endif
