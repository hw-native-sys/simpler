/**
 * @file inner_aicpu.h
 * @brief Platform-specific AICPU definitions for real hardware (a2a3)
 *
 * Provides get_sys_cnt_aicpu() for AICPU-side timestamping on Ascend hardware.
 * Reads CNTVCT_EL0 — the same physical counter as AICore's get_sys_cnt().
 */

#ifndef PLATFORM_A2A3_AICPU_INNER_AICPU_H_
#define PLATFORM_A2A3_AICPU_INNER_AICPU_H_

#include <cstdint>

/**
 * AICPU system counter for a2a3 hardware.
 * Reads the Arm generic timer counter (CNTVCT_EL0), which is the same
 * physical counter that AICore's get_sys_cnt() reads on the Ascend SoC.
 */
inline uint64_t get_sys_cnt_aicpu() {
    uint64_t ticks;
    asm volatile("mrs %0, cntvct_el0" : "=r"(ticks));
    return ticks;
}

#endif  // PLATFORM_A2A3_AICPU_INNER_AICPU_H_
