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
#include <atomic>
#include <cstdint>
#include "ccu_native_abi.h"

namespace simpler::ccu {
enum class CompletionPollState : uint8_t { pending, ready, failed, protocol_error };
struct CompletionPollResult {
    CompletionPollState state{CompletionPollState::pending};
    int64_t native_error{};
};

struct EventIo {
    static uint64_t load(uint64_t address) {
        const auto value = *reinterpret_cast<volatile uint64_t *>(address);
#if defined(__aarch64__)
        __asm__ volatile("dsb sy" ::: "memory");
#else
        std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
        return value;
    }
    static void clear(uint64_t address) {
        *reinterpret_cast<volatile uint64_t *>(address) = 0;
#if defined(__aarch64__)
        __asm__ volatile("dsb sy" ::: "memory");
#else
        std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
    }
};

template <typename Io = EventIo>
inline CompletionPollResult poll_completion(uint64_t address, uint64_t mask) {
    if (address == 0 || address % alignof(uint64_t) != 0 || mask != SIMPLER_CCU_EVENT_MASK)
        return {CompletionPollState::protocol_error, 0};
    if ((Io::load(address) & mask) != mask) return {CompletionPollState::pending, 0};
    Io::clear(address);
    // The completion cannot release dependencies until the hardware clear is confirmed.
    if (Io::load(address) != 0) return {CompletionPollState::protocol_error, 0};
    return {CompletionPollState::ready, 0};
}
}  // namespace simpler::ccu
