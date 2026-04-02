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
#include "common/core_type.h"
#include "common/kernel_args.h"

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
    aicore_execute(k_args->runtime_args, block_idx, core_type);
}
