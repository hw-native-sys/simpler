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
 * Mirrors the AICPU-side `set_l2_swimlane_enabled` / `set_pmu_enabled` /
 * etc. setters: the platform owns a per-core slot for profiling state,
 * populated once by the AICore kernel entry from `KernelArgs`, and read by
 * `aicore_execute` via getters. Runtime never touches the underlying
 * storage, so adding profiling fields does not change `aicore_execute`'s
 * signature or the runtime's `Handshake` struct.
 *
 * Storage backend:
 *   - onboard: `[[block_local]]` static variables in aicore/kernel.cpp
 *   - sim:     pthread TLS in aicore/kernel.cpp
 *
 * Lifecycle:
 *   1. Host fills `KernelArgs::enable_profiling_flag` and the three
 *      per-core address arrays (`aicore_l2_perf_ring_addrs`,
 *      `aicore_pmu_ring_addrs`, `pmu_reg_addrs`).
 *   2. AICore kernel entry indexes each array by `block_idx` and calls the
 *      matching setter before invoking `aicore_execute`.
 *   3. `aicore_execute` and downstream profiling helpers read via getters.
 *
 * a5 specifics: PMU on a5 is AICore-side (AICore reads MMIO + writes the
 * staging slot), so a5 carries an extra `pmu_ring` + `pmu_reg_base` pair
 * that a2a3 does not have.
 */

#ifndef PLATFORM_AICORE_AICORE_PROFILING_STATE_H_
#define PLATFORM_AICORE_AICORE_PROFILING_STATE_H_

#include <cstdint>

#include "aicore/aicore.h"
#include "common/l2_perf_profiling.h"
#include "common/pmu_profiling.h"

/**
 * Profiling enable bitmask (umbrella over dump_tensor / l2_swimlane / pmu).
 * Same layout as `KernelArgs::enable_profiling_flag`. AICore reads via
 * `GET_PROFILING_FLAG(get_aicore_profiling_flag(), PROFILING_FLAG_*)`.
 */
__aicore__ void set_aicore_profiling_flag(uint32_t flag);
__aicore__ uint32_t get_aicore_profiling_flag();

/**
 * Per-core L2Perf staging ring. Set once at kernel entry from
 * `((__gm__ uint64_t*)k_args->aicore_l2_perf_ring_addrs)[block_idx]`;
 * nullptr when the L2 swimlane bit is off or the address table is null.
 */
__aicore__ void set_aicore_l2_perf_ring(__gm__ L2PerfAicoreRing *ring);
__aicore__ __gm__ L2PerfAicoreRing *get_aicore_l2_perf_ring();

/**
 * Per-core PMU staging ring (a5-only — AICore writes the snapshot).
 */
__aicore__ void set_aicore_pmu_ring(__gm__ PmuAicoreRing *ring);
__aicore__ __gm__ PmuAicoreRing *get_aicore_pmu_ring();

/**
 * Per-core PMU MMIO base (a5-only). The host pre-allocates a table on
 * device and AICPU fills `table[block_idx] = regs[physical_core_ids[i]]`
 * during `handshake_all_cores` (between aicore_regs_ready and
 * aicpu_regs_ready). The setter stashes the *table* pointer at kernel
 * entry; the getter dereferences `table[block_idx]` lazily so the value
 * read in `aicore_execute` post-Phase-3 reflects the AICPU's fill.
 *
 * A literal "store the value" semantics (a2a3 form) cannot be used on a5
 * because the host does not know the block_idx → physical_core_id mapping
 * before the AICore kernel launches; only AICPU learns it via handshake.
 */
__aicore__ void set_aicore_pmu_reg_addrs(__gm__ uint64_t *table);
__aicore__ uint64_t get_aicore_pmu_reg_base();

#endif  // PLATFORM_AICORE_AICORE_PROFILING_STATE_H_
