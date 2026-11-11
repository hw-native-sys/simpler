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

#include "scheduler_memory.h"

namespace scheduler_async_pending {

inline constexpr uint32_t COUNT_MAX = INT16_MAX;
inline constexpr uint64_t REGISTERED_UNIT = UINT64_C(1) << 16;
inline constexpr uint64_t COMPUTATION_CLOSED = UINT64_C(1) << 32;

struct alignas(64) Join {
    volatile uint64_t word{0};
};

static_assert(sizeof(Join) == 64 && alignof(Join) == 64);
static_assert(std::is_standard_layout_v<Join> && std::is_trivially_copyable_v<Join>);

enum class Result : uint8_t {
    PENDING,
    SYNCHRONOUS,
    FINISH_OWNER,
    INVALID_WORD,
    INVALID_TRANSITION,
    COUNT_OVERFLOW,
    COUNT_UNDERFLOW,
};

struct Outcome {
    Result result;
    uint64_t previous;
};

constexpr inline __aicore__ uint32_t pending(uint64_t word) { return static_cast<uint32_t>(word & UINT64_C(0xffff)); }
constexpr inline __aicore__ uint32_t registered(uint64_t word) {
    return static_cast<uint32_t>((word >> 16) & UINT64_C(0xffff));
}
constexpr inline __aicore__ bool closed(uint64_t word) { return (word & COMPUTATION_CLOSED) != 0; }
constexpr inline __aicore__ bool valid(uint64_t word) {
    return (word >> 33) == 0 && registered(word) <= COUNT_MAX && pending(word) <= registered(word);
}

// Readiness validates an already granted FINISH lease; reading it grants no ownership.
constexpr inline __aicore__ bool ready_to_finish(uint64_t word) {
    return valid(word) && closed(word) && pending(word) == 0 && registered(word) != 0;
}

namespace detail {

namespace {

struct GmAtomics {
    __gm__ volatile uint64_t &word;
    inline __aicore__ uint64_t fetch_add(uint64_t delta) { return scheduler_gm_fetch_add(word, delta); }
};

}  // namespace

// Registration precedes CHECK publication and the execution's release. The
// existing block/lane join authorizes exactly one close after every release.
// Error results can follow an invalid RMW and require failing the run.
template <typename Atomics>
static inline __aicore__ Outcome register_group(Atomics &atomics) {
    const uint64_t previous = atomics.fetch_add(REGISTERED_UNIT + 1);
    if (!valid(previous)) return {Result::INVALID_WORD, previous};
    if (closed(previous)) return {Result::INVALID_TRANSITION, previous};
    if (registered(previous) == COUNT_MAX) return {Result::COUNT_OVERFLOW, previous};
    return {Result::PENDING, previous};
}

template <typename Atomics>
static inline __aicore__ Outcome close_computation(Atomics &atomics) {
    const uint64_t previous = atomics.fetch_add(COMPUTATION_CLOSED);
    if (!valid(previous)) return {Result::INVALID_WORD, previous};
    if (closed(previous)) return {Result::INVALID_TRANSITION, previous};
    if (pending(previous) != 0) return {Result::PENDING, previous};
    return {registered(previous) == 0 ? Result::SYNCHRONOUS : Result::FINISH_OWNER, previous};
}

template <typename Atomics>
static inline __aicore__ Outcome complete_group(Atomics &atomics) {
    const uint64_t previous = atomics.fetch_add(UINT64_MAX);
    if (!valid(previous)) return {Result::INVALID_WORD, previous};
    if (pending(previous) == 0) return {Result::COUNT_UNDERFLOW, previous};
    return {closed(previous) && pending(previous) == 1 ? Result::FINISH_OWNER : Result::PENDING, previous};
}

}  // namespace detail

static inline __aicore__ Outcome register_group(__gm__ Join &join) {
    detail::GmAtomics atomics{join.word};
    return detail::register_group(atomics);
}

static inline __aicore__ Outcome close_computation(__gm__ Join &join) {
    detail::GmAtomics atomics{join.word};
    return detail::close_computation(atomics);
}

static inline __aicore__ Outcome complete_group(__gm__ Join &join) {
    detail::GmAtomics atomics{join.word};
    return detail::complete_group(atomics);
}

}  // namespace scheduler_async_pending
