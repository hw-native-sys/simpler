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

// Include the executor header BEFORE the per-subcore #define block_idx /
// core_type renames below. The header body references `my_hank->core_type`
// and a parameter named `core_type`; if it saw the macro it would textually
// rename those tokens and the struct-field access would fail to compile.
#include "aicore_executor.h"

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

// SIMT metadata TLV injection.
//
// The legacy launch path (rtKernelLaunchWithHandleV2 + rtRegisterAllKernel)
// requires the kernel ELF to carry two TLV records that runtime reads at
// register time (see runtime/src/runtime/core/inc/kernel/elf.hpp:113,115 and
// elf.cc:576-586):
//   - FUNC_META_TYPE_COMPILER_ALLOC_UB_SIZE (7)  -> Kernel::shareMemSize_
//   - FUNC_META_TYPE_AIV_TYPE_FLAG          (12) -> Kernel::kernelVfType_
// bisheng emits these only when it can statically infer the kernel uses SIMT
// intrinsics. Our entry is a SU dispatcher (vector ops live in task .o files
// invoked through aicore_execute), so the compiler cannot tag it.
//
// The CMakeLists.txt build pass `-mllvm -cce-dyn-kernel-stack-size=false`
// stops bisheng from auto-emitting the per-function meta section, leaving
// only this hand-written record as the source of TLV 7 / TLV 12 for the
// AIV variant. Without the flag bisheng would emit a sibling section with
// kernelVfType=NO_VF (1) and shareMemSize=0; runtime's parser keys
// kernelInfoMap by section name and overwrites instead of merging, so the
// auto-emitted section would shadow ours and force the launch back onto
// the non-SIMT path.
enum FuncMetaType {
    F_TYPE_COMPILER_ALLOC_UB_SIZE = 7,
    F_TYPE_AIV_TYPE_FLAG = 12,
};

enum AIVType {
    AIV_TYPE_NO_VF = 1,
    AIV_TYPE_SIMD_VF_ONLY = 2,
    AIV_TYPE_SIMT_VF_ONLY = 3,
    AIV_TYPE_SIMD_SIMT_MIX_VF = 4,
};

struct _base_tlv {
    unsigned short type;
    unsigned short len;
};

struct fun_meta_compiler_ub_size {
    _base_tlv head;
    unsigned int ub_size;
};

struct fun_meta_aiv_type_flag {
    _base_tlv head;
    unsigned int aiv_type;
};

struct fun_level_meta {
    fun_meta_compiler_ub_size ub_size_meta;
    fun_meta_aiv_type_flag aiv_type_meta;
};

#ifdef __DAV_VEC__
static const fun_level_meta func_simt_section __attribute__((used, section(".ascend.meta.aicore_kernel_0_mix_aiv"))) = {
    {{F_TYPE_COMPILER_ALLOC_UB_SIZE, sizeof(unsigned int)}, 8192},
    {{F_TYPE_AIV_TYPE_FLAG, sizeof(unsigned int)}, AIV_TYPE_SIMT_VF_ONLY},
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
