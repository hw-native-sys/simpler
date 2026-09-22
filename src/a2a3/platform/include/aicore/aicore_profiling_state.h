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
 * Mirrors the AICPU-side `set_chip_swimlane_enabled` / `set_pmu_enabled` / etc.
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
 *      `KernelArgs::chip_swimlane_aicore_rotation_table` (an array of per-core
 *      slots, each holding a device address of an `ChipSwimlaneActiveHead`).
 *      Host allocates the table bytes; AICPU populates the slot entries
 *      inside `chip_swimlane_aicpu_init` with `&pool.head` for each AicoreTask
 *      pool.
 *   2. AICore kernel entry stashes `&chip_swimlane_aicore_rotation_table[block_idx]`
 *      (the slot pointer — NOT the dereferenced head pointer yet) via
 *      `set_chip_swimlane_aicore_head_slot()`, and calls `set_aicore_profiling_flag()`,
 *      before invoking `aicore_execute`.
 *   3. `get_chip_swimlane_aicore_head()` dereferences the slot on first use and
 *      caches the result. The first call is valid after AICore observes the
 *      AICPU initialization publication point for the current launch. In the
 *      current handshake this is Phase 2 exit (`DATA_MAIN_BASE != 0`) after
 *      AICPU opens the register window. Executors may resolve the head there or
 *      defer it until the first dispatch.
 */

#pragma once

#include <cstdint>

#include "aicore/aicore.h"
#include "common/chip_swimlane_profiling.h"

/**
 * Profiling enable bitmask (umbrella over dump_args / chip_swimlane / pmu).
 * Same layout as `KernelArgs::enable_profiling_flag`. AICore reads via
 * `SIMPLER_GET_DFX_FLAG(get_aicore_profiling_flag(), SIMPLER_DFX_FLAG_*)`.
 */
__aicore__ void set_aicore_profiling_flag(uint32_t flag);
__aicore__ uint32_t get_aicore_profiling_flag();

/**
 * Per-core AICore head channel.
 *
 * `set_chip_swimlane_aicore_head_slot(slot)` stashes the address of THIS core's
 * slot in the head-address table —
 * `&((uint64_t*)k_args->chip_swimlane_aicore_rotation_table)[block_idx]`. No
 * dereference happens here, because at kernel entry the AICPU side may not
 * yet have populated the table (the host launches both kernels and AICPU's
 * init runs concurrently with AICore's entry).
 *
 * `get_chip_swimlane_aicore_head()` dereferences the stashed slot on first use,
 * caches the result, and returns the cached pointer on subsequent calls.
 * Callers MUST defer the first call until AICore has observed the AICPU
 * initialization publication point for the current launch. In the current
 * handshake this is Phase 2 exit (`DATA_MAIN_BASE != 0`) after AICPU opens the
 * register window. `chip_swimlane_aicpu_init` publishes the slot before any
 * register window is opened, so resolving at handshake exit and lazy resolution
 * on first dispatch are both valid.
 */
__aicore__ void set_chip_swimlane_aicore_head_slot(__gm__ uint64_t *slot_ptr);
__aicore__ __gm__ ChipSwimlaneActiveHead *get_chip_swimlane_aicore_head();

/**
 * This run's handshake report identity, from `AicoreLaunchArgs::report_epoch`.
 *
 * Non-zero on a native program launch: the executor commits it into
 * `Handshake::report_epoch` after the payload and its store barrier, and the
 * AICPU accepts only a report carrying exactly this value. Zero on a
 * kernel/persistent launch, which keeps the `aicore_done != 0` predicate.
 *
 * Carried on the per-core state rather than in `aicore_execute`'s signature,
 * for the same reason the profiling state is: the runtime executors' signature
 * and the `Handshake` layout stay independent of what the launch block grows.
 */
__aicore__ void set_aicore_report_epoch(uint64_t epoch);
__aicore__ uint64_t get_aicore_report_epoch();
