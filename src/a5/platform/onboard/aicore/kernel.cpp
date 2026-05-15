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
#include "common/platform_config.h"
#include "simt_meta.h"

class Runtime;

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

// Derive the section name from the same KERNEL_ENTRY macro that mangles the
// entry symbol, so the meta section name cannot drift if the suffix scheme
// changes. STRINGIFY needs two levels to expand the macro before stringizing.
#define SIMPLER_STRINGIFY_(x) #x
#define SIMPLER_STRINGIFY(x) SIMPLER_STRINGIFY_(x)
#define KERNEL_META_SECTION(func) ".ascend.meta." SIMPLER_STRINGIFY(KERNEL_ENTRY(func))

#ifdef __DAV_VEC__
static const FuncLevelMeta func_simt_section __attribute__((used, section(KERNEL_META_SECTION(aicore_kernel)))) = {
    {{F_TYPE_COMPILER_ALLOC_UB_SIZE, sizeof(unsigned int)}, PLATFORM_AICORE_SHARE_MEM_SIZE},
    {{F_TYPE_AIV_TYPE_FLAG, sizeof(unsigned int)}, AIV_TYPE_SIMD_SIMT_MIX_VF},
};
#endif

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
extern "C" __global__ __aicore__ void KERNEL_ENTRY(aicore_kernel)(__gm__ Runtime *runtime) {
    // Calculate block_idx for this core
#ifdef __DAV_VEC__
    block_idx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
    core_type = CoreType::AIV;
#else
    block_idx = get_block_idx();
    core_type = CoreType::AIC;
#endif

    aicore_execute(runtime, block_idx, core_type);
}
