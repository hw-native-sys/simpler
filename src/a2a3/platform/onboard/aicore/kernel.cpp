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
 * Minimal AICore Kernel
 */
#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "common/core_type.h"
#include "common/kernel_args.h"
#include "common/chip_swimlane_profiling.h"

#ifdef __DAV_VEC__
#define KERNEL_ENTRY(x) \
    x##_0_mix_aiv  // Dynamically generate function name: KERNEL_ENTRY(my_kernel) ->
                   // my_kernel_0_mix_aiv
#define block_idx block_idx_aiv
#define core_type core_type_aiv
#else
#define KERNEL_ENTRY(x) x##_0_mix_aic
#define block_idx block_idx_aic
#define core_type core_type_aic
#endif

[[block_local]] int block_idx;
[[block_local]] CoreType core_type;

// Per-core profiling state. Populated once by KERNEL_ENTRY from KernelArgs;
// read by aicore_execute and profiling helpers via the getters below. This
// mirrors the AICPU-side set_chip_swimlane_enabled / set_pmu_enabled pattern,
// keeping profiling fields out of runtime's Handshake and out of
// aicore_execute's signature.
//
// The setters/getters are marked `weak` because kernel.cpp is compiled twice
// (AIC + AIV) and linked into a single AICore binary; weak linkage lets the
// linker dedup the otherwise-duplicate symbol definitions across the two
// compilation units.
[[block_local]] static uint32_t s_aicore_profiling_flag;
// Slot pointer (NOT the dereferenced head address) — see
// aicore_profiling_state.h for the lazy-deref contract.
[[block_local]] static __gm__ uint64_t *s_chip_swimlane_aicore_head_slot;
[[block_local]] static __gm__ ChipSwimlaneActiveHead *s_chip_swimlane_aicore_head;
[[block_local]] static uint64_t s_aicore_report_epoch;

__attribute__((weak)) __aicore__ void set_aicore_profiling_flag(uint32_t flag) { s_aicore_profiling_flag = flag; }
__attribute__((weak)) __aicore__ uint32_t get_aicore_profiling_flag() { return s_aicore_profiling_flag; }

__attribute__((weak)) __aicore__ void set_aicore_report_epoch(uint64_t epoch) { s_aicore_report_epoch = epoch; }
__attribute__((weak)) __aicore__ uint64_t get_aicore_report_epoch() { return s_aicore_report_epoch; }

__attribute__((weak)) __aicore__ void set_chip_swimlane_aicore_head_slot(__gm__ uint64_t *slot_ptr) {
    s_chip_swimlane_aicore_head_slot = slot_ptr;
    s_chip_swimlane_aicore_head = nullptr;  // force lazy resolution on next get
}
__attribute__((weak)) __aicore__ __gm__ ChipSwimlaneActiveHead *get_chip_swimlane_aicore_head() {
    // Lazy first-call resolve. AICPU publishes the slot before opening any
    // register window, so it is valid after AICore observes Phase 2 exit.
    if (s_chip_swimlane_aicore_head == nullptr && s_chip_swimlane_aicore_head_slot != nullptr) {
        s_chip_swimlane_aicore_head =
            reinterpret_cast<__gm__ ChipSwimlaneActiveHead *>(*s_chip_swimlane_aicore_head_slot);
    }
    return s_chip_swimlane_aicore_head;
}

extern __aicore__ void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type);

/**
 * Kernel entry point with control loop
 *
 * This function implements the AICore-side task execution protocol:
 * 1. Signal AICore is ready (aicore_done = block_idx + 1)
 * 2. Wait for AICPU to open the register window (DATA_MAIN_BASE != 0)
 * 3. Enter polling loop:
 *    - Poll DATA_MAIN_BASE for a task or exit command
 *    - Execute newly dispatched tasks and report completion via COND
 *    - Use DCCI to ensure cache coherency with AICPU
 *
 * Each core (AIC or AIV) gets its own handshake buffer indexed by block_idx.
 * Every value this entry needs arrives in the launch parameter block, so the
 * per-core state below is published before the executor runs and without any
 * GM read. The host builds that block after collector arming, which is what
 * makes these addresses this run's final ones.
 *
 * @param runtime_args Device address of this run's Runtime image
 * @param enable_profiling_flag Profiling umbrella bitmask for this run
 * @param ffts_base_addr FFTS base for cross-core sync inside user kernels
 * @param chip_swimlane_aicore_rotation_table Device address of the
 *        uint64_t[num_aicore] table of per-core active-head slots, or 0
 * @param report_epoch This run's handshake report identity; 0 selects the
 *        pre-existing kernel/persistent report protocol
 */
extern "C" __global__ __aicore__ void KERNEL_ENTRY(aicore_kernel)(
    uint64_t runtime_args, uint32_t enable_profiling_flag, uint64_t ffts_base_addr,
    uint64_t chip_swimlane_aicore_rotation_table, uint64_t report_epoch
) {
    // Calculate block_idx for this core
#ifdef __DAV_VEC__
    block_idx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
    core_type = CoreType::AIV;
#else
    block_idx = get_block_idx();
    core_type = CoreType::AIC;
#endif

    set_ffts_base_addr(ffts_base_addr);

    // Publish per-core profiling state into platform-owned slots before the
    // executor runs. AICore reads via get_aicore_profiling_flag() /
    // get_chip_swimlane_aicore_head() — never touches Handshake for profiling.
    set_aicore_profiling_flag(enable_profiling_flag);
    // Published unconditionally for the same reason as the head slot below:
    // [[block_local]] storage survives across launches on the same loaded
    // binary, so a native run followed by a kernel-mode one must not inherit
    // the native run's epoch.
    set_aicore_report_epoch(report_epoch);
    // Always publish the head slot (nullptr when this launch is disabled or
    // has no rotation table). [[block_local]] storage persists across launches
    // on the same loaded kernel binary, so without an explicit nullptr
    // publication a sequence like enabled(valid)→enabled(NULL table) or
    // enabled→disabled would leave `get_chip_swimlane_aicore_head()` returning
    // the prior launch's freed pointer. AICore call sites are additionally
    // flag-gated on this-launch `chip_swimlane_enabled`, so the disabled-launch
    // read path is currently unreachable through the executors; the
    // unconditional reset is defensive against future call sites that don't
    // carry that gate. Mirrors the publish-nullptr branch in
    // sim/aicore/kernel.cpp (sim keys only on the table pointer; onboard
    // additionally AND-gates on SIMPLER_DFX_FLAG_CHIP_SWIMLANE — intentional,
    // since the onboard table is shared across collectors).
    if (SIMPLER_GET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_CHIP_SWIMLANE) &&
        chip_swimlane_aicore_rotation_table != 0) {
        // Stash only the slot pointer. The slot CONTENTS are written by
        // AICPU's `chip_swimlane_aicpu_init`, which races with this entry but
        // publishes the slot before opening any register window. The executor
        // dereferences via `get_chip_swimlane_aicore_head()` only after it
        // observes Phase 2 exit.
        __gm__ uint64_t *head_table = reinterpret_cast<__gm__ uint64_t *>(chip_swimlane_aicore_rotation_table);
        set_chip_swimlane_aicore_head_slot(&head_table[block_idx]);
    } else {
        set_chip_swimlane_aicore_head_slot(nullptr);
    }

    aicore_execute(reinterpret_cast<__gm__ Runtime *>(runtime_args), block_idx, core_type);
}
