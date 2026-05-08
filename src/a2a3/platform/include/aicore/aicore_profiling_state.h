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
 *      `KernelArgs::aicore_ring_addr`.
 *   2. AICore kernel entry indexes `aicore_ring_addr[block_idx]` for this
 *      core's `L2PerfAicoreRing*` and calls `set_aicore_profiling_flag()` +
 *      `set_aicore_l2_perf_ring()` before invoking `aicore_execute`.
 *   3. `aicore_execute` and downstream profiling helpers read via getters.
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
 * Per-core L2Perf staging ring. Set once at kernel entry from
 * `((uint64_t*)k_args->aicore_ring_addr)[block_idx]`; nullptr when the L2
 * swimlane bit is off or the address table itself is null.
 */
__aicore__ void set_aicore_l2_perf_ring(__gm__ L2PerfAicoreRing *ring);
__aicore__ __gm__ L2PerfAicoreRing *get_aicore_l2_perf_ring();

#endif  // PLATFORM_AICORE_AICORE_PROFILING_STATE_H_
