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
 * @file cpu_sim_context.h
 * @brief Per-device CPU simulation context for CANN intrinsic emulation
 *
 * Each simulated device gets an isolated context (pipe shared state)
 * so multiple ChipWorkers can run concurrently without interference.
 *
 * All pto_sim_* functions operate on the context bound to the calling
 * thread's device_id (set via pto_cpu_sim_bind_device).
 *
 * Invariant: each simulated device_id has a single owner ChipWorker per
 * process. The owner calls acquire_device() at set_device() time and
 * release_device() at finalize_device() time, after all worker threads
 * for that device have been joined. Concurrent access from multiple
 * ChipWorkers to the same device_id is undefined behavior.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/** Bind the calling thread to a simulated device ID. */
void pto_cpu_sim_bind_device(int device_id);

/** Return the device ID bound to the calling thread, or -1 if unbound. */
int pto_cpu_sim_get_bound_device(void);

/** Ensure a context exists for the given device_id. */
void pto_cpu_sim_acquire_device(int device_id);

/** Release and destroy the context for the given device_id. */
void pto_cpu_sim_release_device(int device_id);

/** Return the current thread's AIV lane (0 or 1). Resolved by pto-isa via function pointer injection. */
uint32_t pto_sim_get_subblock_id(void);

/** Return per-device per-cluster per-dispatch pipe shared memory. Resolved by pto-isa via function pointer injection.
 */
void *pto_sim_get_pipe_shared_state(uint64_t pipe_key, size_t size);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

/**
 * Set the current thread's simulated subblock_id.
 * Called by aicore_execute_wrapper (via function pointer injection).
 */
void sim_context_set_subblock_id(uint32_t subblock_id);

/**
 * Set the current thread's simulated cluster_id.
 * Called by aicore_execute_wrapper (via function pointer injection).
 */
void sim_context_set_cluster_id(uint32_t cluster_id);

/** Clear pipe shared state for the current thread's device. */
void clear_cpu_sim_shared_storage();

#endif
