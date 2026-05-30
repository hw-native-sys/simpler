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
 * @file aicore_profiling_state.h
 * @brief AICore-side per-core profiling state set/get interface.
 *
 * Mirrors the AICPU-side `set_l2_swimlane_enabled` / `set_pmu_enabled` / etc.
 * setters: the platform owns a per-core slot for profiling state, populated
 * once by the AICore kernel entry from `KernelArgs`, and read by
 * `aicore_execute` via getters. Runtime never touches the underlying storage,
 * so adding profiling fields does not change `aicore_execute`'s signature or
 * the runtime's `Handshake` struct.
 *
 * Storage backend:
 *   - onboard: `[[block_local]]` static variables in aicore/kernel.cpp
 *   - sim:     pthread TLS in aicore/kernel.cpp
 *
 * Lifecycle:
 *   1. Host fills `KernelArgs::enable_profiling_flag` and
 *      `KernelArgs::aicore_ring_addr` (points to a per-core `AicoreRotation`
 *      device-address table). Host allocates the table bytes; AICPU populates
 *      the entries inside `l2_perf_aicpu_init`.
 *   2. AICore kernel entry stashes `&aicore_ring_addr[block_idx]` (the slot
 *      pointer — NOT the dereferenced rotation pointer yet) via
 *      `set_aicore_rotation_slot()`, and calls `set_aicore_profiling_flag()`,
 *      before invoking `aicore_execute`.
 *   3. `get_aicore_rotation()` lazily dereferences the slot the first time
 *      it is called. Callers must defer the call until AFTER AICPU has
 *      dispatched the first task (so AICPU init has had a chance to populate
 *      the table). The executor handles this by calling it inside the main
 *      loop's first-task branch.
 */

#ifndef PLATFORM_AICORE_AICORE_PROFILING_STATE_H_
#define PLATFORM_AICORE_AICORE_PROFILING_STATE_H_

#include <cstdint>

#include "aicore/aicore.h"
#include "common/l2_perf_profiling.h"

/**
 * Profiling enable bitmask (umbrella over dump_tensor / l2_swimlane / pmu).
 * Same layout as `KernelArgs::enable_profiling_flag`. AICore reads via
 * `GET_PROFILING_FLAG(get_aicore_profiling_flag(), PROFILING_FLAG_*)`.
 */
__aicore__ void set_aicore_profiling_flag(uint32_t flag);
__aicore__ uint32_t get_aicore_profiling_flag();

/**
 * Per-core AICore rotation channel.
 *
 * `set_aicore_rotation_slot(slot)` stashes the address of THIS core's slot
 * in the rotation-address table — `&((uint64_t*)k_args->aicore_ring_addr)[block_idx]`.
 * No dereference happens here, because at kernel entry the AICPU side may
 * not yet have populated the table (the host launches both kernels and
 * AICPU's init runs concurrently with AICore's entry).
 *
 * `get_aicore_rotation()` lazily dereferences the stashed slot on first use,
 * caches the result, and returns it on subsequent calls. Callers MUST defer
 * the first call until after AICPU has dispatched the first task — by then
 * AICPU's init has completed and the slot holds a valid device address.
 * The executor's main loop honours this by reading the rotation only inside
 * the first-task branch of the dispatch poll.
 */
__aicore__ void set_aicore_rotation_slot(__gm__ uint64_t *slot_ptr);
__aicore__ __gm__ AicoreRotation *get_aicore_rotation();

#endif  // PLATFORM_AICORE_AICORE_PROFILING_STATE_H_
