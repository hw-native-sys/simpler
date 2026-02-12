/**
 * @file inner_aicpu.h
 * @brief Platform-specific AICPU definitions for simulation (a2a3sim)
 *
 * Provides get_sys_cnt_aicpu() for AICPU-side timestamping in simulation mode.
 * Uses std::chrono::high_resolution_clock::time_since_epoch() to match
 * the same epoch as AICore's get_sys_cnt() in inner_kernel.h.
 */

#ifndef PLATFORM_A2A3SIM_AICPU_INNER_AICPU_H_
#define PLATFORM_A2A3SIM_AICPU_INNER_AICPU_H_

#include <chrono>
#include <cstdint>
#include "common/platform_config.h"

/**
 * AICPU system counter for a2a3sim simulation.
 * Uses time_since_epoch() to share the same clock epoch as AICore's
 * get_sys_cnt(), converting nanoseconds to platform tick frequency.
 */
inline uint64_t get_sys_cnt_aicpu() {
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
    constexpr uint64_t kNsPerSec = std::nano::den;
    uint64_t seconds = elapsed_ns / kNsPerSec;
    uint64_t remaining_ns = elapsed_ns % kNsPerSec;
    uint64_t ticks = seconds * PLATFORM_PROF_SYS_CNT_FREQ +
                     (remaining_ns * PLATFORM_PROF_SYS_CNT_FREQ) / kNsPerSec;
    return ticks;
}

#endif  // PLATFORM_A2A3SIM_AICPU_INNER_AICPU_H_
