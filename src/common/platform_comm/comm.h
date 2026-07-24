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
 * Provides primitives for multi-rank communication: init, create
 * subcommunicators, allocate shared windows, query local window base, query
 * window size, derive domain contexts, barrier, and destroy.
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

#include "common/dma_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CommHandle_ *CommHandle;

#define COMM_GLOBAL_DOMAIN_VERSION 1U
#define COMM_GLOBAL_DOMAIN_HANDLE_BYTES 256U
#define COMM_GLOBAL_DOMAIN_DESCRIPTOR_BYTES 288U

typedef enum CommGlobalDomainProfile {
    COMM_GLOBAL_DOMAIN_PROFILE_SIM_SHM = 1,
    COMM_GLOBAL_DOMAIN_PROFILE_A3_FABRIC = 2,
} CommGlobalDomainProfile;

typedef struct CommGlobalDomainDescriptor {
    uint32_t version;
    uint32_t profile;
    uint32_t domain_rank;
    uint32_t rank_count;
    uint64_t mapping_size;
    uint32_t handle_size;
    uint32_t reserved;
    uint8_t handle[COMM_GLOBAL_DOMAIN_HANDLE_BYTES];
} CommGlobalDomainDescriptor;

#ifdef __cplusplus
static_assert(
    sizeof(CommGlobalDomainDescriptor) == COMM_GLOBAL_DOMAIN_DESCRIPTOR_BYTES,
    "CommGlobalDomainDescriptor wire size drift"
);
static_assert(offsetof(CommGlobalDomainDescriptor, version) == 0, "CommGlobalDomainDescriptor.version offset drift");
static_assert(offsetof(CommGlobalDomainDescriptor, profile) == 4, "CommGlobalDomainDescriptor.profile offset drift");
static_assert(
    offsetof(CommGlobalDomainDescriptor, domain_rank) == 8, "CommGlobalDomainDescriptor.domain_rank offset drift"
);
static_assert(
    offsetof(CommGlobalDomainDescriptor, rank_count) == 12, "CommGlobalDomainDescriptor.rank_count offset drift"
);
static_assert(
    offsetof(CommGlobalDomainDescriptor, mapping_size) == 16, "CommGlobalDomainDescriptor.mapping_size offset drift"
);
static_assert(
    offsetof(CommGlobalDomainDescriptor, handle_size) == 24, "CommGlobalDomainDescriptor.handle_size offset drift"
);
static_assert(offsetof(CommGlobalDomainDescriptor, reserved) == 28, "CommGlobalDomainDescriptor.reserved offset drift");
static_assert(offsetof(CommGlobalDomainDescriptor, handle) == 32, "CommGlobalDomainDescriptor.handle offset drift");
#endif

/** Bit mask of DmaWorkspaceKind values this platform can provision. */
uint32_t dma_workspace_supported_mask(void);

/**
 * Provision the async-DMA workspaces named in required_mask for the current
 * device (aclrtGetDevice) and write their device addresses into addr_out[kind]
 * (0 where unused), for kind in [0, count). Bit `1u << kind` requests that
 * engine. A requested workspace is mandatory: success guarantees a non-zero
 * address for every requested bit.
 *
 * Called once at Worker init when the Worker was created with SDMA enabled.
 * The provisioned addresses are stable for the Worker's life; the host copies
 * them onto every run's KernelArgs, and AICPU injects them into GlobalContext
 * (get_dma_workspace) — they are never threaded as user args. On success
 * *handle_out owns the provider resources and must be released with
 * dma_workspace_release(). Bits outside dma_workspace_supported_mask() are
 * rejected, so provisioning fails fast on a platform/runtime without SDMA.
 *
 * @return 0 on success, non-zero on invalid/unsupported input or provisioning
 *         failure.
 */
int dma_workspace_provision(uint32_t required_mask, uint64_t *addr_out, int count, void **handle_out);

/**
 * Release the resources returned by a successful dma_workspace_provision().
 * Ordinary teardown: destroys the provider's streams and workspace. A null
 * handle is a no-op.
 */
void dma_workspace_release(void *handle);

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
 *                       HCCL operations like HcclBarrier.  Sim backend
 *                       ignores it.
 * @param rootinfo_path  Filesystem path used to exchange root info between
 *                       ranks (rank 0 writes, others read).
 * @return Opaque handle, or NULL on failure.
 */
CommHandle comm_init(int rank, int nranks, void *stream, const char *rootinfo_path);

/**
 * Allocate RDMA / shared-memory windows and populate the device context.
 *
 * On HCCL this builds a per-rank symmetric pool via the public ACL IPC
 * primitives (aclrtMalloc + aclrtIpcMemGetExportKey / SetImportPid /
 * ImportByKey) and enables cross-card P2P via aclrtDeviceEnablePeerAccess.
 * On sim it mallocs a shared region and partitions it.
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
 * Derive a domain-local CommContext from an allocated base communicator.
 *
 * The base handle must already have completed comm_alloc_windows().  The
 * derived context remaps dense domain ranks onto base-communicator ranks and
 * adds window_offset to each selected base rank window.  The returned
 * device_ctx_out points to a backend-owned device CommContext that remains
 * valid until comm_destroy(base).
 *
 * @param h                Allocated base communicator handle.
 * @param rank_ids         Base-communicator rank ids in domain rank order.
 * @param rank_count       Number of domain ranks.
 * @param domain_rank      This caller's dense rank in the domain.
 * @param window_offset    Byte offset of this domain slice in the base window.
 * @param window_size      Visible per-rank window size for this domain.
 * @param device_ctx_out   Receives the derived device CommContext pointer.
 * @return 0 on success, non-zero on failure.
 */
int comm_derive_context(
    CommHandle h, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank, size_t window_offset,
    size_t window_size, uint64_t *device_ctx_out
);

/**
 * Allocate a fresh per-rank symmetric window pool for a subset of ranks.
 *
 * Unlike comm_alloc_windows() which allocates the single base pool once at
 * bootstrap, this allocates an additional pool for a dynamically-derived
 * domain (a subset of the base communicator).  Multiple concurrent
 * allocations are disambiguated by `allocation_id`, which is mixed into
 * every internal handshake / barrier filename so a second allocation
 * does not collide with the first.
 *
 * This is a collective operation across the subset only: every
 * participating chip must call this with matching arguments; non-members
 * of the subset do NOT call it.  Internal file barriers synchronise the
 * subset, so the parent (orchestrator) only needs to dispatch and wait
 * for completion — it does not need to broker the cross-rank handshake.
 *
 * On HCCL this performs aclrtMalloc + the same Path-D IPC pattern as
 * comm_alloc_windows but on a fresh per-allocation buffer.  On sim it
 * shm_opens a fresh POSIX shm scoped by `allocation_id`.
 *
 * Resources allocated here remain owned by the base handle; either an
 * explicit comm_release_domain_windows() or final comm_destroy(base)
 * frees them.
 *
 * @param h                       Base handle from comm_init().
 * @param allocation_id           Caller-assigned unique id; scopes the
 *                                handshake / barrier filenames.  Must be
 *                                unique among currently-live allocations
 *                                on this handle.
 * @param rank_ids                Base-communicator rank ids in dense
 *                                domain order (length rank_count).
 * @param rank_count              Number of subset members.
 * @param domain_rank             This caller's dense rank in the subset.
 * @param window_size             Bytes per rank.  Backend must allocate
 *                                exactly this size; no auto-rounding.
 * @param device_ctx_out          Receives a device pointer to a new
 *                                CommContext for the subset.
 * @param local_window_base_out   Receives this rank's local window start
 *                                address (for buffer carving on the
 *                                caller side).
 * @return 0 on success, non-zero on failure.
 */
int comm_alloc_domain_windows(
    CommHandle h, uint64_t allocation_id, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank,
    size_t window_size, uint64_t *device_ctx_out, uint64_t *local_window_base_out
);

/**
 * Collectively release a domain window pool allocated by
 * comm_alloc_domain_windows().
 *
 * Frees the per-rank buffer (HCCL: aclrtFree; sim: munmap + shm_unlink)
 * and the device-side CommContext.  Synchronises subset members via a
 * release barrier scoped by `allocation_id` + the caller's `domain_rank`
 * within the subset.  `rank_count` is the subset size — used only to
 * size the barrier wait, not the rank list (which was already established
 * during alloc).
 *
 * IPC import refs and EnablePeerAccess routes are NOT explicitly torn
 * down — they get reclaimed by aclrtResetDevice at DeviceRunner::finalize.
 * Mirrors the lifetime contract of comm_alloc_windows on HCCL.
 *
 * @param h               Base handle from comm_init().
 * @param allocation_id   Same id passed to comm_alloc_domain_windows.
 * @param rank_count      Number of subset members (for barrier sizing).
 * @param domain_rank     This caller's dense rank in the subset.
 * @return 0 on success, non-zero on failure.
 */
int comm_release_domain_windows(CommHandle h, uint64_t allocation_id, size_t rank_count, uint32_t domain_rank);

/**
 * Create one local window for a Global CommDomain and export its transport
 * descriptor. This operation has no
 * HCCL or filesystem bootstrap dependency.
 *
 * The caller relays the descriptor through its control plane. It later
 * passes
 * the complete rank-ordered table to comm_global_domain_import(). A live
 * domain_id is unique within one
 * ChipWorker process.
 */
int comm_global_domain_prepare(
    uint64_t domain_id, uint32_t domain_rank, uint32_t rank_count, size_t window_size, uint32_t profile,
    CommGlobalDomainDescriptor *descriptor_out, uint64_t *local_window_base_out
);

/**
 * Import every peer descriptor and publish a device CommContext.
 *
 * descriptors must contain exactly one entry
 * for every dense domain rank.
 * The returned context and all imported mappings remain live until
 *
 * comm_global_domain_release().
 */
int comm_global_domain_import(
    uint64_t domain_id, const CommGlobalDomainDescriptor *descriptors, size_t descriptor_count, uint64_t *device_ctx_out
);

/**
 * Release a prepared or imported Global CommDomain. This is local teardown.
 * The L4 owner is responsible for
 * draining all rank tasks before fanout.
 */
int comm_global_domain_release(uint64_t domain_id);

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
