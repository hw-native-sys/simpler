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
 * PTO Runtime C API — canonical header
 *
 * Declares all C-linkage functions exported by the host runtime .so.
 * Both the ChipWorker (consumer, resolves public symbols via dlsym) and the
 * platform implementations (producers, define all symbols) include this file.
 *
 * Public API — resolved by ChipWorker via dlsym:
 *   create_device_context, destroy_device_context,
 *   get_runtime_size, set_device, run_runtime, finalize_device,
 *   host_malloc, host_free, host_register_mapped, host_unregister_mapped
 *
 * Memory management: caller allocates a buffer of get_runtime_size() bytes
 * and passes it to run_runtime(). Error codes: 0 = success, negative = error.
 */

#ifndef SRC_COMMON_WORKER_PTO_RUNTIME_C_API_H_
#define SRC_COMMON_WORKER_PTO_RUNTIME_C_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *RuntimeHandle;
typedef void *DeviceContextHandle;

/* ===========================================================================
 * Public API (resolved by ChipWorker via dlsym)
 * =========================================================================== */

/**
 * Create a new device context (heap-allocated DeviceRunner).
 * Each ChipWorker should own one context for the lifetime of its init→finalize cycle.
 * @return Opaque handle on success, NULL on failure.
 */
DeviceContextHandle create_device_context(void);

/**
 * Destroy a device context created by create_device_context().
 * Calls finalize internally, then frees the underlying object.
 */
void destroy_device_context(DeviceContextHandle ctx);

/** Return sizeof(Runtime) for caller buffer allocation. */
size_t get_runtime_size(void);

/** Set the target device. Must be called before the first run_runtime(). */
int set_device(DeviceContextHandle ctx, int device_id);

/**
 * Allocate pinned host memory suitable for host-register flows.
 *
 * @param ctx   Device context from create_device_context()
 * @param size  Size in bytes
 * @return Host pointer on success, NULL on failure
 */
void *host_malloc(DeviceContextHandle ctx, size_t size);

/**
 * Free host memory allocated by host_malloc().
 *
 * @param ctx       Device context from create_device_context()
 * @param host_ptr  Pointer returned by host_malloc()
 */
void host_free(DeviceContextHandle ctx, void *host_ptr);

/**
 * Register host memory as a device-visible mapped address.
 *
 * @param ctx        Device context from create_device_context()
 * @param host_ptr   Host pointer to register
 * @param size       Size in bytes
 * @param device_id  Target device ID
 * @param dev_ptr    Output mapped device-visible address
 * @return 0 on success, negative on error
 */
int host_register_mapped(DeviceContextHandle ctx, void *host_ptr, size_t size, int device_id, void **dev_ptr);

/**
 * Unregister host memory previously registered by host_register_mapped().
 *
 * @param ctx        Device context from create_device_context()
 * @param host_ptr   Host pointer to unregister
 * @param device_id  Target device ID
 * @return 0 on success, negative on error
 */
int host_unregister_mapped(DeviceContextHandle ctx, void *host_ptr, int device_id);

/**
 * Build the task graph, execute on device, copy results back, and clean up.
 *
 * @param ctx               Device context from create_device_context()
 * @param runtime           Caller-allocated buffer (size from get_runtime_size())
 * @param callable          Opaque ChipCallable pointer (orchestration + kernel binaries)
 * @param args              Opaque ChipStorageTaskArgs pointer (tensor/scalar arguments)
 * @param block_dim         Number of AICore blocks
 * @param aicpu_thread_num  Number of AICPU scheduler threads
 * @param device_id         Target device
 * @param aicpu_binary      AICPU executor binary blob
 * @param aicpu_size        Size of AICPU binary
 * @param aicore_binary     AICore executor binary blob
 * @param aicore_size       Size of AICore binary
 * @param enable_profiling  1 to enable profiling, 0 to disable
 * @return 0 on success, negative on error
 */
int run_runtime(
    DeviceContextHandle ctx, RuntimeHandle runtime, const void *callable, const void *args, int block_dim,
    int aicpu_thread_num, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size, const uint8_t *aicore_binary,
    size_t aicore_size, int enable_profiling
);

/**
 * Release all device resources held by the context.
 * Must be called before destroy_device_context() / dlclose().
 */
int finalize_device(DeviceContextHandle ctx);

#ifdef __cplusplus
}
#endif

#endif  // SRC_COMMON_WORKER_PTO_RUNTIME_C_API_H_
