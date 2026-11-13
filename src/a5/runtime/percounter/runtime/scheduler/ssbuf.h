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
 * The cluster-local SSBUF window, as this runtime uses it.
 *
 * **The barrier in these two is the MIX rendezvous's only ordering.** A lane
 * publishes its arrival with the release store and reads its peers' with the
 * acquire load, and nothing else separates the token from the payload it
 * stands for; `percounter_mix_wait` runs one of these per peer per spin.
 *
 * The region layout this runtime puts in the window is its own
 * (`PercounterSsbufRegion`), so what was left to share with host_build_graph
 * was the address space, where the window sits, and these two accessors.
 * They are kept here because when the sibling instrumented its own scheduler,
 * routing these two through the instrumentation faulted every device run of
 * this one -- bisected to that commit, with the tree before it passing on the
 * same machine. That commit changed more than these two accessors, so which
 * of its changes a lane cannot absorb is not established; owning the window
 * is what keeps the question from arising again. See
 * `docs/investigations/2026-11-percounter-shared-primitive-coupling.md`.
 *
 * The offset and size are a hardware-and-convention fact rather than either
 * runtime's choice: A5 gives each 1AIC+2AIV cluster 3 KiB of SSBUF, and the
 * scheduler takes the last KiB so that user kernels keep the low 2 KiB.
 */

#pragma once

#include <cstdint>

#include "scheduler/scheduler_types.h"

#if defined(__CCE_AICORE__)
#define PERCOUNTER_SSBUF __ssbuf__
#else
#define PERCOUNTER_SSBUF
#endif

constexpr uint32_t PERCOUNTER_SSBUF_REGION_OFFSET = 2 * 1024;
constexpr uint32_t PERCOUNTER_SSBUF_REGION_SIZE = 1024;
static_assert(PERCOUNTER_SSBUF_REGION_OFFSET + PERCOUNTER_SSBUF_REGION_SIZE == PLATFORM_SSBUF_SIZE);

// A CPU reader can overlap record reuse even when it discards the snapshot.
// Acquiring a newer field also orders its opening generation before the
// reader's closing generation check.
template <typename T>
inline __aicore__ T percounter_ssbuf_load_field(const PERCOUNTER_SSBUF volatile T *field) {
#if defined(__CCE_AICORE__)
    return *field;
#else
    return __atomic_load_n(field, __ATOMIC_ACQUIRE);
#endif
}

template <typename T>
inline __aicore__ void percounter_ssbuf_store_field(PERCOUNTER_SSBUF volatile T *field, T value) {
#if defined(__CCE_AICORE__)
    *field = value;
#else
    __atomic_store_n(field, value, __ATOMIC_RELEASE);
#endif
}

/**
 * Publication tokens, not read-modify-write state: SSBUF has no atomics.
 *
 * The barrier falls *after* the load and *before* the store, which is what
 * orders a token against the payload it stands for. Two consecutive acquire
 * loads therefore carry a barrier between them already -- the edge GM has to
 * ask for explicitly.
 */
inline __aicore__ uint32_t percounter_ssbuf_load_acquire(const PERCOUNTER_SSBUF volatile uint32_t *token) {
#if defined(__CCE_AICORE__)
    const uint32_t value = *token;
    dsb((mem_dsb_t)0);
    return value;
#else
    return __atomic_load_n(token, __ATOMIC_ACQUIRE);
#endif
}

inline __aicore__ void percounter_ssbuf_store_release(PERCOUNTER_SSBUF volatile uint32_t *token, uint32_t value) {
#if defined(__CCE_AICORE__)
    dsb((mem_dsb_t)0);
    *token = value;
#else
    __atomic_store_n(token, value, __ATOMIC_RELEASE);
#endif
}
