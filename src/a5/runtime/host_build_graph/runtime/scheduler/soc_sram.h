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

#include <cstdint>

#include "scheduler/profiling_memory.h"

// The 32-bit SET/read, ADD and OR aliases address one SoC SRAM word.
// ADD and OR update that word atomically without returning its old value.
inline constexpr uint64_t SOC_SRAM_ADD_OFFSET = 8;
inline constexpr uint64_t SOC_SRAM_OR_OFFSET = 16;

inline __aicore__ uint64_t percounter_soc_sram_add_address(uint64_t read_address) {
    return read_address == 0 ? 0 : read_address + SOC_SRAM_ADD_OFFSET;
}

inline __aicore__ uint32_t percounter_soc_sram_read(__gm__ volatile uint32_t &word) {
    return simpler::percounter::percounter_scalar_soc_sram_read(
        reinterpret_cast<uint64_t>(&word), [&word]() __aicore__ {
#if defined(__CCE_AICORE__)
            return static_cast<uint32_t>(__builtin_cce_ld_dev(const_cast<__gm__ uint32_t *>(&word), 0));
#else
            return __atomic_load_n(&word, __ATOMIC_ACQUIRE);
#endif
        }
    );
}

// The ADD alias interprets 1 as one increment command.
inline __aicore__ void percounter_soc_sram_increment(uint64_t add_address) {
    using namespace simpler::percounter;
    percounter_scalar_soc_sram_issue(ProfilePhase::SocSramAddIssue, add_address, 1, [=]() __aicore__ {
#if defined(__CCE_AICORE__)
        __builtin_cce_st_dev(UINT32_C(1), reinterpret_cast<__gm__ uint32_t *>(add_address), 0);
#else
        auto *word = reinterpret_cast<uint32_t *>(add_address - SOC_SRAM_ADD_OFFSET);
        (void)__atomic_fetch_add(word, UINT32_C(1), __ATOMIC_RELEASE);
#endif
    });
}

inline __aicore__ void percounter_soc_sram_or(uint64_t read_address, uint32_t bits) {
    using namespace simpler::percounter;
    const uint64_t or_address = read_address + SOC_SRAM_OR_OFFSET;
    percounter_scalar_soc_sram_issue(ProfilePhase::SocSramOrIssue, or_address, bits, [=]() __aicore__ {
#if defined(__CCE_AICORE__)
        __builtin_cce_st_dev(bits, reinterpret_cast<__gm__ uint32_t *>(or_address), 0);
#else
        auto *word = reinterpret_cast<uint32_t *>(read_address);
        (void)__atomic_fetch_or(word, bits, __ATOMIC_RELEASE);
#endif
    });
}
