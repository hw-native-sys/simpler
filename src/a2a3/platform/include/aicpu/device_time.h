/**
 * @file device_time.h
 * @brief AICPU Device Timestamping Interface
 *
 * Provides get_sys_cnt_aicpu() for AICPU-side timestamping on both
 * real hardware and simulation.
 *
 * Platform Support:
 * - a2a3: Real Ascend hardware (reads CNTVCT_EL0)
 * - a2a3sim: Host-based simulation using std::chrono
 */

#ifndef PLATFORM_AICPU_DEVICE_TIME_H_
#define PLATFORM_AICPU_DEVICE_TIME_H_

#include <cstdint>

/**
 * AICPU system counter for performance profiling.
 *
 * Returns a monotonic counter value compatible with AICore's get_sys_cnt().
 * Implementation is platform-specific (hardware counter or chrono simulation).
 *
 * @return Counter ticks
 */
uint64_t get_sys_cnt_aicpu();

#endif  // PLATFORM_AICPU_DEVICE_TIME_H_
