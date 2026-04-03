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
 * @file intrinsic.h
 * @brief SPMD execution context for AICore user kernels
 *
 * Topology data exposed to user kernels has two distinct lifetimes:
 *
 *   1. Global topology (per-core, fixed after runtime init):
 *      - sub_block_id : identifies the AIV lane within a cluster
 *        (0 = AIV0/left, 1 = AIV1/right).  Initialized once at runtime
 *        startup based on each core's cluster position; never changes.
 *        Only meaningful for AIV kernels in MIX tasks.
 *
 *   2. Local per-dispatch context (changes each dispatch):
 *      - block_idx : which logical block the current worker is executing
 *      - block_num : total number of blocks in this task (= block_dim)
 *      Written by build_payload() before each dispatch.
 *
 * Both categories are injected via two pointer slots appended at the tail
 * of the kernel args[] array:
 *
 *   args layout:
 *     [0 .. tensor_count-1]                 = tensor GM pointers
 *     [tensor_count .. +scalar_count-1]     = scalar values
 *     ...
 *     [SPMD_LOCAL_CONTEXT_INDEX]            = (uint64_t)&LocalContext   (per-dispatch)
 *     [SPMD_GLOBAL_CONTEXT_INDEX]           = (uint64_t)&GlobalContext  (per-core)
 *
 * The suffix positions are compile-time constants and do not depend on the
 * runtime tensor_count or scalar_count.
 *
 * Include this header in AICore kernel source files to use the Get* accessors.
 * Do NOT depend on the raw index constants; always use the accessor functions.
 *
 * On CCEC (real hardware), __gm__ and __aicore__ must be defined before
 * including this header (e.g. via <pto/pto-inst.hpp> or manual #define).
 * The #ifndef guards below provide fallbacks for non-kernel builds
 * (AICPU, HOST) where these qualifiers are not needed.
 */

#pragma once

#include <stdint.h>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__
#endif

/** Number of extra pointer slots appended to the args[] tail (LocalContext + GlobalContext). */
static constexpr int32_t PTO2_EXT_PARAMS_COUNT = 2;

/**
 * Args[] suffix indices for context pointers.
 * Derived from MAX_TENSOR_ARGS(16) + MAX_SCALAR_ARGS(128).
 * Users should not depend on these values; use the Get* functions below.
 */
static constexpr int32_t SPMD_LOCAL_CONTEXT_INDEX = 144;
static constexpr int32_t SPMD_GLOBAL_CONTEXT_INDEX = 145;

/**
 * Per-core global context, stored in PTO2DispatchPayload.
 * Initialized once at runtime startup (init_global_context) based on each
 * core's cluster position.  Never modified after initialization.
 */
struct GlobalContext {
    // AIV lane within cluster: 0=AIV0(left), 1=AIV1(right).
    // Used by AIV to select the correct intra-cluster hw instruction.
    // Not meaningful for AIC kernels or single-AIV tasks.
    int32_t sub_block_id;
};

/**
 * Per-dispatch local context, stored in PTO2DispatchPayload.
 * Written by build_payload() before each dispatch. Different blocks of the
 * same task receive different block_idx values but the same block_num.
 *
 */
struct LocalContext {
    int32_t block_idx;  // Logical block index within the task [0, block_num)
    int32_t block_num;  // How many logical blocks this task requires.
                        // Currently fixed to 1 (block_dim > 1 not yet implemented).
                        // NOT the same as RUNTIME_CONFIG.block_dim in kernel_config.py,
                        // which controls how many physical cores the runtime launches.
};

/**
 * Return the AIV lane index within the cluster.
 * In a MIX 1C2V task: AIV0(left)=0, AIV1(right)=1.
 *
 * This value is only meaningful for AIV kernels in MIX tasks.  It tells
 * the AIV whether it is the left lane or the right lane within the cluster,
 * which determines the correct hardware instruction for intra-cluster
 * communication.
 *
 * AIC kernels should NOT call this function.
 * Single-AIV tasks have no intra-cluster communication, so sub_block_id
 * has no meaning and should not be used.
 */
static __aicore__ inline int32_t get_sub_block_id(__gm__ int64_t *args) {
    __gm__ GlobalContext *ctx =
        reinterpret_cast<__gm__ GlobalContext *>(static_cast<uint64_t>(args[SPMD_GLOBAL_CONTEXT_INDEX]));
    return ctx->sub_block_id;
}

/**
 * Return the logical block index assigned to the current worker.
 * Range: [0, get_block_num(args)).
 * Within the same task, different blocks receive different indices.
 */
static __aicore__ inline int32_t get_block_idx(__gm__ int64_t *args) {
    __gm__ LocalContext *ctx =
        reinterpret_cast<__gm__ LocalContext *>(static_cast<uint64_t>(args[SPMD_LOCAL_CONTEXT_INDEX]));
    return ctx->block_idx;
}

/**
 * Return how many logical blocks the current task requires.
 * All blocks of the same task see the same value.
 * Currently always returns 1 (block_dim>1 not yet implemented).
 *
 * Note: this is NOT the same as RUNTIME_CONFIG.block_dim in
 * kernel_config.py, which controls how many physical cores are launched.
 */
static __aicore__ inline int32_t get_block_num(__gm__ int64_t *args) {
    __gm__ LocalContext *ctx =
        reinterpret_cast<__gm__ LocalContext *>(static_cast<uint64_t>(args[SPMD_LOCAL_CONTEXT_INDEX]));
    return ctx->block_num;
}
