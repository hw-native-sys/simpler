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
 *   1. Host fills `KernelArgs::enable_profiling_flag`, the two per-core
 *      ring address arrays (`aicore_l2_swimlane_ring_addrs`,
 *      `aicore_pmu_ring_addrs`), and `regs` (the per-physical-core
 *      register-base array — already required for AICPU).
 *   2. AICore kernel entry indexes the ring arrays by `block_idx` and
 *      resolves its own PMU MMIO base from `regs[physical_core_id]`,
 *      then calls the matching setters before invoking `aicore_execute`.
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
#include "common/l2_swimlane_profiling.h"
#include "common/pmu_profiling.h"

/**
 * Profiling enable bitmask (umbrella over dump_tensor / l2_swimlane / pmu).
 * Same layout as `KernelArgs::enable_profiling_flag`. AICore reads via
 * `GET_PROFILING_FLAG(get_aicore_profiling_flag(), PROFILING_FLAG_*)`.
 */
__aicore__ void set_aicore_profiling_flag(uint32_t flag);
__aicore__ uint32_t get_aicore_profiling_flag();

/**
 * Per-core L2Swimlane staging ring. Set once at kernel entry from
 * `((__gm__ uint64_t*)k_args->aicore_l2_swimlane_ring_addrs)[block_idx]`;
 * nullptr when the L2 swimlane bit is off or the address table is null.
 */
__aicore__ void set_aicore_l2_swimlane_ring(__gm__ L2SwimlaneAicoreRing *ring);
__aicore__ __gm__ L2SwimlaneAicoreRing *get_aicore_l2_swimlane_ring();

/**
 * Per-core PMU staging ring (a5-only — AICore writes the snapshot).
 */
__aicore__ void set_aicore_pmu_ring(__gm__ PmuAicoreRing *ring);
__aicore__ __gm__ PmuAicoreRing *get_aicore_pmu_ring();

/**
 * Per-core PMU MMIO base (a5-only). AICore at kernel entry resolves this
 * once from `((__gm__ uint64_t*)k_args->regs)[get_physical_core_id()]`
 * and stashes the value via the setter; the getter just returns the
 * stored value. `regs` is filled by the host before kernel launch, so the
 * resolved base is valid from Phase 1 onward without depending on any
 * AICPU-side init ordering.
 */
__aicore__ void set_aicore_pmu_reg_base(uint64_t reg_base);
__aicore__ uint64_t get_aicore_pmu_reg_base();

#endif  // PLATFORM_AICORE_AICORE_PROFILING_STATE_H_
