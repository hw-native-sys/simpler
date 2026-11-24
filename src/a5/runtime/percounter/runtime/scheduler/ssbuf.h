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
// The scheduler owns the upper 1 KiB of each cluster's 3 KiB SSBUF.
#pragma once

#include <cstdint>

#include "scheduler/scheduler_types.h"
#include "scheduler/profiling_memory.h"

#if defined(__CCE_AICORE__)
#define PERCOUNTER_SSBUF __ssbuf__
#else
#define PERCOUNTER_SSBUF
#endif

constexpr uint32_t PERCOUNTER_SSBUF_REGION_OFFSET = 2 * 1024;
constexpr uint32_t PERCOUNTER_SSBUF_REGION_SIZE = 1024;
static_assert(PERCOUNTER_SSBUF_REGION_OFFSET + PERCOUNTER_SSBUF_REGION_SIZE == PLATFORM_SSBUF_SIZE);

template <typename T>
static inline __aicore__ T percounter_ssbuf_load_field(const PERCOUNTER_SSBUF volatile T *field) {
    return simpler::percounter::percounter_scalar_memory_result(
        simpler::percounter::ProfilePhase::SsbufLoad, [field]() __aicore__ {
#if defined(__CCE_AICORE__)
            return *field;
#else
            return __atomic_load_n(field, __ATOMIC_ACQUIRE);
#endif
        }
    );
}

template <typename T>
static inline __aicore__ void percounter_ssbuf_store_field(PERCOUNTER_SSBUF volatile T *field, T value) {
    simpler::percounter::percounter_scalar_memory_issue(
        simpler::percounter::ProfilePhase::SsbufStore, 1, [field, value]() __aicore__ {
#if defined(__CCE_AICORE__)
            *field = value;
#else
            __atomic_store_n(field, value, __ATOMIC_RELEASE);
#endif
        }
    );
}

/**
 * Publication tokens, not read-modify-write state: SSBUF has no atomics.
 *
 * The barrier falls *after* the load and *before* the store, which is what
 * orders a token against the payload it stands for. Two consecutive acquire
 * loads therefore carry a barrier between them already -- the edge GM has to
 * ask for explicitly.
 */
static inline __aicore__ uint32_t percounter_ssbuf_load_acquire(const PERCOUNTER_SSBUF volatile uint32_t *token) {
    const uint32_t value = percounter_ssbuf_load_field(token);
#if defined(__CCE_AICORE__)
    simpler::percounter::percounter_scalar_memory_issue(simpler::percounter::ProfilePhase::DsbAll, 1, []() __aicore__ {
        dsb((mem_dsb_t)0);
    });
#endif
    return value;
}

static inline __aicore__ void
percounter_ssbuf_store_release(PERCOUNTER_SSBUF volatile uint32_t *token, uint32_t value) {
#if defined(__CCE_AICORE__)
    simpler::percounter::percounter_scalar_memory_issue(simpler::percounter::ProfilePhase::DsbAll, 1, []() __aicore__ {
        dsb((mem_dsb_t)0);
    });
#endif
    percounter_ssbuf_store_field(token, value);
}
