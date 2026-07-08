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
 * @file device_time.h
 * @brief AICPU Device Timestamping Interface
 *
 * Provides get_sys_cnt_aicpu() for AICPU-side timestamping on both
 * real hardware and simulation.
 *
 * Platform Support (same shape for both arches):
 * - onboard: Real Ascend hardware (reads CNTVCT_EL0)
 * - sim: Host-based simulation using std::chrono
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEVICE_TIME_H_
#define SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEVICE_TIME_H_

#if !defined(__aarch64__)
#include <chrono>
#endif
#include <cstdint>

#include "common/platform_config.h"

/**
 * AICPU system counter for performance profiling.
 *
 * Returns a monotonic counter value compatible with AICore's get_sys_cnt().
 * Implementation is platform-specific (hardware counter or chrono simulation).
 *
 * @return Counter ticks
 */
uint64_t get_sys_cnt_aicpu();

inline uint64_t device_time_now_ticks() {
#if defined(__aarch64__)
    uint64_t value = 0;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(value));
    return value;
#else
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
#endif
}

inline uint64_t device_time_frequency_hz() {
#if defined(__aarch64__)
    uint64_t value = 0;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(value));
    return value;
#else
    return 1'000'000'000ULL;
#endif
}

inline uint64_t get_sys_cnt_aicpu_frequency_hz() { return PLATFORM_PROF_SYS_CNT_FREQ; }

inline uint64_t sys_cnt_ticks_to_ns(uint64_t ticks, uint64_t frequency_hz) {
    if (frequency_hz == 0) {
        return UINT64_MAX;
    }
    __uint128_t elapsed_ns = static_cast<__uint128_t>(ticks) * 1'000'000'000ULL / frequency_hz;
    if (elapsed_ns > UINT64_MAX) {
        return UINT64_MAX;
    }
    return static_cast<uint64_t>(elapsed_ns);
}

inline uint64_t sys_cnt_elapsed_ns(uint64_t start, uint64_t end, uint64_t frequency_hz) {
    return sys_cnt_ticks_to_ns(end - start, frequency_hz);
}

#endif  // SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEVICE_TIME_H_
