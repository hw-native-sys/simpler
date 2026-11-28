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

#include "invocation_context.h"
#include "host_build_graph/completion_types.h"
#include "host_build_graph/task_id.h"

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__
#endif

// Shared scheduler catalog types retain their fixed layout; kernel args contain no context suffix.
static constexpr int32_t EXT_PARAMS_COUNT = 2;
static constexpr int32_t SPMD_LOCAL_CONTEXT_INDEX = 48;
static constexpr int32_t SPMD_GLOBAL_CONTEXT_INDEX = 49;
static constexpr int32_t PAYLOAD_LOCAL_CONTEXT_INDEX = SPMD_LOCAL_CONTEXT_INDEX;
static constexpr int32_t PAYLOAD_GLOBAL_CONTEXT_INDEX = SPMD_GLOBAL_CONTEXT_INDEX;

#if defined(__CPU_SIM)
#include <cstdlib>

namespace simpler::percounter {
using SimContextBind = int (*)(void *);
using SimContextGet = void *(*)();

// Each loaded kernel and executor has one injected callback pair across its translation units.
__attribute__((visibility("hidden"))) inline SimContextBind sim_context_bind = nullptr;
__attribute__((visibility("hidden"))) inline SimContextGet sim_context_get = nullptr;

inline bool percounter_bind_sim_context(PercounterInvocationContext *context) {
    return sim_context_bind != nullptr && sim_context_bind(context) == 0;
}

inline volatile PercounterInvocationContext *percounter_current_context() {
    auto *context =
        sim_context_get == nullptr ? nullptr : static_cast<PercounterInvocationContext *>(sim_context_get());
    if (context == nullptr) std::abort();
    return context;
}
}  // namespace simpler::percounter

extern "C" __attribute__((used, visibility("default"))) inline void simpler_sim_install_invocation_context_hooks(
    simpler::percounter::SimContextBind bind, simpler::percounter::SimContextGet get
) {
    simpler::percounter::sim_context_bind = bind;
    simpler::percounter::sim_context_get = get;
}
#else
namespace simpler::percounter {
#if defined(__CCE_AICORE__)
using PercounterContextPointer = volatile __ssbuf__ PercounterInvocationContext *;

inline __aicore__ PercounterContextPointer percounter_current_context() {
#if defined(__DAV_VEC__)
    const uint32_t lane = 1 + static_cast<uint32_t>(__cce_scalar::get_subblockid());
#else
    const uint32_t lane = 0;
#endif
    return reinterpret_cast<PercounterContextPointer>(
        PERCOUNTER_CONTEXT_SSBUF_OFFSET + lane * PERCOUNTER_CONTEXT_SSBUF_STRIDE
    );
}
#else
volatile PercounterInvocationContext *percounter_current_context();
#endif
}  // namespace simpler::percounter
#endif

static __aicore__ inline int32_t get_block_idx(__gm__ int64_t *) {
    return simpler::percounter::percounter_current_context()->local.block_idx;
}

static __aicore__ inline int32_t get_block_num(__gm__ int64_t *) {
    return simpler::percounter::percounter_current_context()->local.block_num;
}

static __aicore__ inline int32_t get_sub_block_id(__gm__ int64_t *) {
    return simpler::percounter::percounter_current_context()->global.sub_block_id;
}

static __aicore__ inline GlobalContext load_global_context(__gm__ int64_t *) {
    const auto *context = simpler::percounter::percounter_current_context();
    return {
        context->global.sub_block_id,        context->global.async_present,
        context->global.async_image_address, context->global.async_image_bytes,
        context->global.async_run_epoch,     context->global.async_group_offset,
        context->global.async_group_index,   context->global.async_output_capture_refused,
    };
}
