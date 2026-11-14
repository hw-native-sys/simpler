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
#include "profiling_record.h"

namespace simpler::percounter {
namespace {
// The slot belongs to this physical core (or one CPU test thread). Binding is
// confined to the percounter lane lifetime and never survives its return.
static constexpr bool profile_compiled = ProfileCompilation<>::enabled;
#if defined(__CCE_AICORE__)
[[block_local]] static PercounterScalarTrace *percounter_scalar_active = nullptr;
// Primitive wrappers own the slot address calculation and load its current
// binding; no observer pointer survives initialization or teardown.
static inline __aicore__ __attribute__((always_inline)) auto percounter_scalar_slot() {
    auto *slot = &percounter_scalar_active;
    // The register identity preserves the core-local address through folding.
    asm("MOV %0, %0" : "+l"(slot));
    return slot;
}
static inline __aicore__ PercounterScalarTrace *percounter_scalar_current() {
    if constexpr (profile_compiled) return *percounter_scalar_slot();
    return nullptr;
}
static inline __aicore__ void percounter_scalar_bind(PercounterScalarTrace *trace) {
    if constexpr (profile_compiled) *percounter_scalar_slot() = trace;
}
#elif defined(__CPU_SIM)

struct PercounterScalarThreadKey {
    pthread_key_t key{};
    bool valid{pthread_key_create(&key, nullptr) == 0};
    ~PercounterScalarThreadKey() {
        if (valid) pthread_key_delete(key);
    }
};
inline PercounterScalarThreadKey &percounter_scalar_thread_key() {
    static PercounterScalarThreadKey key;
    return key;
}
inline PercounterScalarTrace *percounter_scalar_current() {
    if constexpr (profile_compiled) {
        auto &key = percounter_scalar_thread_key();
        return key.valid ? static_cast<PercounterScalarTrace *>(pthread_getspecific(key.key)) : nullptr;
    }
    return nullptr;
}
inline void percounter_scalar_bind([[maybe_unused]] PercounterScalarTrace *trace) {
    if constexpr (profile_compiled) {
        auto &key = percounter_scalar_thread_key();
        if (key.valid) (void)pthread_setspecific(key.key, trace);
    }
}
#else
inline PercounterScalarTrace *percounter_scalar_current() { return nullptr; }
inline void percounter_scalar_bind(PercounterScalarTrace *) {}
#endif

namespace {

class PercounterScalarBinding {
public:
    inline __aicore__ PercounterScalarBinding() = default;
    inline __aicore__ void attach(PercounterScalarTrace *trace) {
        percounter_scalar_bind(trace != nullptr && trace->buffer != nullptr ? trace : nullptr);
    }
    inline __aicore__ ~PercounterScalarBinding() { percounter_scalar_bind(nullptr); }
    PercounterScalarBinding(const PercounterScalarBinding &) = delete;
    PercounterScalarBinding &operator=(const PercounterScalarBinding &) = delete;
};

}  // namespace

// Observer helpers have internal linkage because AIC and AIV objects share
// the final ELF. Clocks and record submission stay inside the bounded
// primitive wrapper; scheduler functions call that wrapper as one unit.
template <typename Operation>
static inline __aicore__ __attribute__((always_inline)) void
percounter_scalar_observed_issue(PercounterScalarTrace *trace, ProfilePhase kind, uint32_t calls, Operation operation) {
    const uint64_t begin = percounter_scalar_clock();
    operation();
    const uint64_t end = percounter_scalar_clock();
    trace->append_memory(kind, begin, end, calls);
}

template <typename Operation>
static inline __aicore__ __attribute__((always_inline)) auto
percounter_scalar_observed_result(PercounterScalarTrace *trace, ProfilePhase kind, Operation operation) {
    const uint64_t begin = percounter_scalar_clock();
    auto result = operation();
    uint64_t end;
#if defined(__CCE_AICORE__)
    auto dependency = static_cast<uint64_t>(result);
    asm volatile("MOV %0, %0\nMOV %1, SYS_CNT\n" : "+l"(dependency), "=&l"(end) : : "memory");
#else
    end = percounter_scalar_clock();
#endif
    trace->append_memory(kind, begin, end);
    return result;
}

// Each callback contains exactly one primitive and cannot nest observation.
// Keep the enable check inside this boundary to bound scheduler optimization
// work without splitting the timed primitive across another function call.
template <typename Operation>
static inline __aicore__ __attribute__((always_inline)) void
percounter_scalar_memory_issue_runtime(ProfilePhase kind, uint32_t calls, Operation operation) {
    auto *trace = percounter_scalar_current();
    if (trace == nullptr) operation();
    else percounter_scalar_observed_issue(trace, kind, calls, operation);
}

template <typename Operation>
static inline __aicore__ __attribute__((always_inline)) auto
percounter_scalar_memory_result_runtime(ProfilePhase kind, Operation operation) {
    auto *trace = percounter_scalar_current();
    if (trace == nullptr) return operation();
    return percounter_scalar_observed_result(trace, kind, operation);
}

template <typename Operation>
static inline __aicore__ __attribute__((always_inline)) void percounter_scalar_memory_issue(
    [[maybe_unused]] ProfilePhase kind, [[maybe_unused]] uint32_t calls, Operation operation
) {
    if constexpr (profile_compiled) percounter_scalar_memory_issue_runtime(kind, calls, operation);
    else operation();
}

template <typename Operation>
static inline __aicore__ __attribute__((always_inline)) auto
percounter_scalar_memory_result([[maybe_unused]] ProfilePhase kind, Operation operation) {
    if constexpr (profile_compiled) return percounter_scalar_memory_result_runtime(kind, operation);
    else return operation();
}

}  // namespace
}  // namespace simpler::percounter
