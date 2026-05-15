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
#include "common/l2_perf_profiling.h"

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
// mirrors the AICPU-side set_l2_swimlane_perf_level / set_pmu_enabled pattern,
// keeping profiling fields out of runtime's Handshake and out of
// aicore_execute's signature.
//
// The setters/getters are marked `weak` because kernel.cpp is compiled twice
// (AIC + AIV) and linked into a single AICore binary; weak linkage lets the
// linker dedup the otherwise-duplicate symbol definitions across the two
// compilation units.
[[block_local]] static uint32_t s_aicore_profiling_flag;
[[block_local]] static __gm__ L2PerfAicoreRing *s_aicore_l2_perf_ring;

__attribute__((weak)) __aicore__ void set_aicore_profiling_flag(uint32_t flag) { s_aicore_profiling_flag = flag; }
__attribute__((weak)) __aicore__ uint32_t get_aicore_profiling_flag() { return s_aicore_profiling_flag; }

__attribute__((weak)) __aicore__ void set_aicore_l2_perf_ring(__gm__ L2PerfAicoreRing *ring) {
    s_aicore_l2_perf_ring = ring;
}
__attribute__((weak)) __aicore__ __gm__ L2PerfAicoreRing *get_aicore_l2_perf_ring() { return s_aicore_l2_perf_ring; }

extern __aicore__ void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type);

/**
 * Kernel entry point with control loop
 *
 * This function implements the AICore-side task execution protocol:
 * 1. Wait for AICPU ready signal (handshake initialization)
 * 2. Signal AICore is ready (aicore_done = core_id + 1)
 * 3. Enter polling loop:
 *    - Check control flag (1 = quit, 0 = continue)
 *    - If task pointer is non-zero, execute task and mark as complete
 *    - Use DCCI to ensure cache coherency with AICPU
 *
 * Each core (AIC or AIV) gets its own handshake buffer indexed by block_idx.
 * Profiling state flows from KernelArgs into platform-owned per-core slots
 * via set_aicore_profiling_flag() / set_aicore_l2_perf_ring(); the runtime's
 * Handshake stays profiling-free and aicore_execute keeps its original
 * signature.
 *
 * @param runtime Address of Runtime structure in device memory
 */
extern "C" __global__ __aicore__ void KERNEL_ENTRY(aicore_kernel)(__gm__ KernelArgs *k_args) {
    // Calculate block_idx for this core
#ifdef __DAV_VEC__
    block_idx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
    core_type = CoreType::AIV;
#else
    block_idx = get_block_idx();
    core_type = CoreType::AIC;
#endif

    set_ffts_base_addr((uint64_t)k_args->ffts_base_addr);

    // Publish per-core profiling state into platform-owned slots before the
    // executor runs. AICore reads via get_aicore_profiling_flag() /
    // get_aicore_l2_perf_ring() — never touches Handshake for profiling.
    set_aicore_profiling_flag(k_args->enable_profiling_flag);
    if (GET_PROFILING_FLAG(k_args->enable_profiling_flag, PROFILING_FLAG_L2_SWIMLANE)) {
        __gm__ uint64_t *ring_table = reinterpret_cast<__gm__ uint64_t *>(k_args->aicore_ring_addr);
        set_aicore_l2_perf_ring(reinterpret_cast<__gm__ L2PerfAicoreRing *>(ring_table[block_idx]));
    }

    aicore_execute(k_args->runtime_args, block_idx, core_type);
}
