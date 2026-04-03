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
 *   get_runtime_size, set_device, run_runtime, finalize_device
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

/* ===========================================================================
 * Public API (resolved by ChipWorker via dlsym)
 * =========================================================================== */

/** Return sizeof(Runtime) for caller buffer allocation. */
size_t get_runtime_size(void);

/** Set the target device. Must be called before the first run_runtime(). */
int set_device(int device_id);

/**
 * Build the task graph, execute on device, copy results back, and clean up.
 *
 * Combines init_runtime + launch_runtime + finalize_runtime into one call.
 */
int run_runtime(
    RuntimeHandle runtime, const void *callable, const void *args, int block_dim, int aicpu_thread_num,
    int orch_thread_num, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size, const uint8_t *aicore_binary,
    size_t aicore_size, int enable_profiling
);

/**
 * Release all device resources.
 * Must be called before dlclose() to avoid static destruction order issues.
 */
int finalize_device(void);

#ifdef __cplusplus
}
#endif

#endif  // SRC_COMMON_WORKER_PTO_RUNTIME_C_API_H_
