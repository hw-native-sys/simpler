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
#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "common/core_type.h"
#include "common/kernel_args.h"
#include "simt_anchor.h"
#include "task_interface/tmr_kernel_context.h"
#include "task_interface/tmr_kernel_control.h"

using simpler::tmr::TmrKernelAicoreArgs;
using simpler::tmr::TmrKernelContextDescriptor;

// Kernel mode currently rejects DFX at prepare. This separate ELF therefore
// supplies stateless disabled getters; it must not import program kernel.cpp
// (including its ID-0 entry) merely to obtain profiling storage. Weak linkage
// coalesces the AIC/AIV definitions, as in the program profiling interface.
__attribute__((weak)) __aicore__ void set_aicore_profiling_flag(uint32_t) {}
__attribute__((weak)) __aicore__ uint32_t get_aicore_profiling_flag() { return 0; }
__attribute__((weak)) __aicore__ void set_chip_swimlane_aicore_head_slot(__gm__ uint64_t *) {}
__attribute__((weak)) __aicore__ __gm__ ChipSwimlaneActiveHead *get_chip_swimlane_aicore_head() { return nullptr; }
__attribute__((weak)) __aicore__ void set_aicore_pmu_ring(__gm__ PmuAicoreRing *) {}
__attribute__((weak)) __aicore__ __gm__ PmuAicoreRing *get_aicore_pmu_ring() { return nullptr; }
__attribute__((weak)) __aicore__ void set_aicore_pmu_reg_base(uint64_t) {}
__attribute__((weak)) __aicore__ uint64_t get_aicore_pmu_reg_base() { return 0; }

extern __aicore__ void aicore_execute_kernel(
    __gm__ Runtime *runtime, __gm__ const TmrKernelContextDescriptor *context, int block_idx, CoreType core_type
);

#ifdef __DAV_VEC__
extern "C" __global__ __aicore__ void aicore_kernel_mode_0_mix_aiv(__gm__ TmrKernelAicoreArgs *args) {
    const int worker = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
    const CoreType type = CoreType::AIV;
#else
extern "C" __global__ __aicore__ void aicore_kernel_mode_0_mix_aic(__gm__ TmrKernelAicoreArgs *args) {
    const int worker = get_block_idx();
    const CoreType type = CoreType::AIC;
#endif
    if (args == nullptr) return;
    dcci(args, SINGLE_CACHE_LINE);
    dsb(static_cast<mem_dsb_t>(0));
    if (args->context_descriptor == 0 || args->context_descriptor % alignof(TmrKernelContextDescriptor) != 0 ||
        args->resident_kernel_args == 0 || args->resident_kernel_args % alignof(KernelArgs) != 0)
        return;
    auto *context = reinterpret_cast<__gm__ const TmrKernelContextDescriptor *>(args->context_descriptor);
    auto *k_args = reinterpret_cast<__gm__ KernelArgs *>(args->resident_kernel_args);
    dcci(k_args, ENTIRE_DATA_CACHE);
    dsb(static_cast<mem_dsb_t>(0));
    if (context->version != simpler::tmr::kTmrKernelContextVersion ||
        context->bytes != sizeof(TmrKernelContextDescriptor) || context->context_generation == 0 ||
        context->self_address != args->context_descriptor ||
        context->resident_kernel_args != args->resident_kernel_args || context->resident_runtime == 0 ||
        context->resident_runtime != reinterpret_cast<uint64_t>(k_args->runtime_args) || context->flags != 0 ||
        context->reserved[0] != 0 || context->reserved[1] != 0 || context->reserved[2] != 0)
        return;

    set_aicore_profiling_flag(0);
    set_chip_swimlane_aicore_head_slot(nullptr);
    set_aicore_pmu_ring(nullptr);
    set_aicore_pmu_reg_base(0);
#ifdef __DAV_VEC__
    // The prepare-owned zero field keeps SIMT metadata in this AIV entry.
    if (k_args->force_simt_anchor) {
        simt_meta_anchor(reinterpret_cast<__gm__ uint32_t *>(k_args));
    }
#endif
    aicore_execute_kernel(k_args->runtime_args, context, worker, type);
}
