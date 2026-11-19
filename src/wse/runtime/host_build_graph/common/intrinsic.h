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

#pragma once

#include <stdint.h>

#include "wse/operation_trace.h"
#include "wse/block_scope.h"

#include "aicore_completion_mailbox_types.h"
#include "host_build_graph/task_id.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__
#endif

/** Number of extra pointer slots appended to the args[] tail (LocalContext + GlobalContext). */
static constexpr int32_t EXT_PARAMS_COUNT = 2;

/**
 * Args[] suffix indices for context pointers.
 * Derived from MAX_TENSOR_ARGS(32) + MAX_SCALAR_ARGS(16).
 * Users should not depend on these values; use the Get* functions below.
 */
static constexpr int32_t SPMD_LOCAL_CONTEXT_INDEX = 48;
static constexpr int32_t SPMD_GLOBAL_CONTEXT_INDEX = 49;
static constexpr int32_t PAYLOAD_LOCAL_CONTEXT_INDEX = SPMD_LOCAL_CONTEXT_INDEX;
static constexpr int32_t PAYLOAD_GLOBAL_CONTEXT_INDEX = SPMD_GLOBAL_CONTEXT_INDEX;

// Kernel-call identity; the observation callback is valid until this invocation returns.
struct GlobalContext {
    int32_t sub_block_id;
    int32_t core_id;
    int32_t pe_id;
    wse::OperationTrace operation_trace;
    uint64_t sram_base;
    uint64_t sram_access_offset;
};

struct AsyncCtx {
    volatile __gm__ uint32_t *completion_count;
    volatile __gm__ int32_t *completion_error_code;
    volatile __gm__ DeferredCompletionEntry *completion_entries;
    uint32_t completion_capacity;
    TaskId task_token;

    static inline AsyncCtx make(TaskId task_token, volatile __gm__ DeferredCompletionSlab *buffer) {
        AsyncCtx ctx{};
        ctx.task_token = task_token;
        if (buffer == nullptr) {
            ctx.task_token = TaskId::invalid();
            return ctx;
        }
        ctx.completion_count = &buffer->count;
        ctx.completion_error_code = &buffer->error_code;
        ctx.completion_entries = &buffer->entries[0];
        ctx.completion_capacity = MAX_COMPLETIONS_PER_TASK;
        return ctx;
    }
};

/**
 * Per-block local context populated by the PE executor. Different blocks of the
 * same task receive different block_idx values but the same block_num.
 *
 */
struct LocalContext {
    int32_t block_idx;  // Logical block index within the task [0, block_num)
    int32_t block_num;  // Original logical block count, including multi-wave execution.
                        // NOT the same as RUNTIME_CONFIG.block_dim in kernel_config.py,
                        // which controls how many physical cores the runtime launches.
    int32_t thread_idx;
    int32_t thread_num;
    wse::BlockScope block_scope;
    AsyncCtx async_ctx;
};

// Compatibility accessor for legacy CPU-operation kernels. WSE PEs have no lane type.
static __aicore__ inline int32_t get_sub_block_id(__gm__ int64_t *args) {
    __gm__ GlobalContext *ctx =
        reinterpret_cast<__gm__ GlobalContext *>(static_cast<uint64_t>(args[SPMD_GLOBAL_CONTEXT_INDEX]));
    return ctx->sub_block_id;
}

static __aicore__ inline int32_t get_core_id(__gm__ int64_t *args) {
    return reinterpret_cast<GlobalContext *>(args[SPMD_GLOBAL_CONTEXT_INDEX])->core_id;
}
static __aicore__ inline int32_t get_pe_id(__gm__ int64_t *args) {
    return reinterpret_cast<GlobalContext *>(args[SPMD_GLOBAL_CONTEXT_INDEX])->pe_id;
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
 *
 * Note: this is NOT the same as RUNTIME_CONFIG.block_dim in
 * kernel_config.py, which controls how many physical cores are launched.
 */
static __aicore__ inline int32_t get_block_num(__gm__ int64_t *args) {
    __gm__ LocalContext *ctx =
        reinterpret_cast<__gm__ LocalContext *>(static_cast<uint64_t>(args[SPMD_LOCAL_CONTEXT_INDEX]));
    return ctx->block_num;
}

static inline wse::OperationTrace get_wse_operation_trace(int64_t *args) {
    return reinterpret_cast<GlobalContext *>(args[SPMD_GLOBAL_CONTEXT_INDEX])->operation_trace;
}

static inline int32_t get_thread_idx(int64_t *args) {
    return reinterpret_cast<LocalContext *>(args[SPMD_LOCAL_CONTEXT_INDEX])->thread_idx;
}
static inline int32_t get_thread_num(int64_t *args) {
    return reinterpret_cast<LocalContext *>(args[SPMD_LOCAL_CONTEXT_INDEX])->thread_num;
}
static inline wse::BlockScope get_wse_block_scope(int64_t *args) {
    return reinterpret_cast<LocalContext *>(args[SPMD_LOCAL_CONTEXT_INDEX])->block_scope;
}
