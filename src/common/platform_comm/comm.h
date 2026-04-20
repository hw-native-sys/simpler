/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
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

typedef struct CommHandle_ *CommHandle;

/**
 * Initialize a communicator for the given rank.
 *
 * The caller is responsible for ACL/device lifecycle before this call:
 *   - aclInit() must have been performed at least once in the process
 *     (DeviceRunner::ensure_acl_ready() is the canonical owner).
 *   - aclrtSetDevice() must be in effect on the current thread.
 *   - `stream` is a pre-created aclrtStream owned by the caller; this
 *     module does not create or destroy streams.
 *
 * On the HCCL backend this performs the RootInfo exchange (rank 0 writes
 * the file, others wait) and HcclCommInitRootInfo.
 *
 * @param rank           This process's rank (0-based).
 * @param nranks         Total number of ranks.
 * @param stream         Caller-owned aclrtStream (passed as void*) used for
 *                       HCCL operations like HcclBarrier and
 *                       HcclAllocComResourceByTiling.  Sim backend ignores it.
 * @param rootinfo_path  Filesystem path used to exchange root info between
 *                       ranks (rank 0 writes, others read).
 * @return Opaque handle, or NULL on failure.
 */
CommHandle comm_init(int rank, int nranks, void *stream, const char *rootinfo_path);

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
 * @param device_ctx_out  Receives a device pointer to a CommContext
 *                        struct that can be passed to device kernels.
 * @return 0 on success, non-zero on failure.
 */
int comm_alloc_windows(CommHandle h, size_t win_size, uint64_t *device_ctx_out);

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
int comm_get_local_window_base(CommHandle h, uint64_t *base_out);

/**
 * Get the actual per-rank window size allocated by the backend.
 *
 * @param h         Handle from comm_init().
 * @param size_out  Receives the actual per-rank window size in bytes.
 * @return 0 on success, non-zero on failure.
 */
int comm_get_window_size(CommHandle h, size_t *size_out);

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
