/**
 * @file platform_config.h
 * @brief Platform-specific configuration and architectural constraints
 *
 * This header defines platform architectural parameters that affect
 * both platform and runtime layers. These configurations are derived from
 * hardware capabilities and platform design decisions.
 *
 * Configuration Hierarchy:
 * - Base: PLATFORM_MAX_BLOCKDIM (platform capacity)
 * - Derived: All other limits calculated from base configuration
 */

#ifndef PLATFORM_COMMON_PLATFORM_CONFIG_H_
#define PLATFORM_COMMON_PLATFORM_CONFIG_H_

#include <cstdint>

// =============================================================================
// Base Platform Configuration
// =============================================================================

/**
 * Maximum block dimension supported by platform
 * This is the fundamental platform capacity constraint.
 */
constexpr int PLATFORM_MAX_BLOCKDIM = 24;

/**
 * Core composition per block dimension
 * Current architecture: 1 block = 1 AIC cube + 2 AIV cubes
 */
constexpr int PLATFORM_CORES_PER_BLOCKDIM = 3;
constexpr int PLATFORM_AIC_CORES_PER_BLOCKDIM = 1;
constexpr int PLATFORM_AIV_CORES_PER_BLOCKDIM = 2;

/**
 * Maximum AICPU scheduling threads
 * Determines parallelism level of the AICPU task scheduler.
 */
constexpr int PLATFORM_MAX_AICPU_THREADS = 4;

// =============================================================================
// Derived Platform Limits
// =============================================================================

/**
 * Maximum cores per AICPU thread
 *
 * When running with 1 AICPU thread and MAX_BLOCKDIM blocks,
 * one thread must manage all cores:
 * - MAX_AIC_PER_THREAD = MAX_BLOCKDIM * AIC_CORES_PER_BLOCKDIM = 24 * 1 = 24
 * - MAX_AIV_PER_THREAD = MAX_BLOCKDIM * AIV_CORES_PER_BLOCKDIM = 24 * 2 = 48
 */
constexpr int PLATFORM_MAX_AIC_PER_THREAD =
    PLATFORM_MAX_BLOCKDIM * PLATFORM_AIC_CORES_PER_BLOCKDIM;  // 24

constexpr int PLATFORM_MAX_AIV_PER_THREAD =
    PLATFORM_MAX_BLOCKDIM * PLATFORM_AIV_CORES_PER_BLOCKDIM;  // 48

constexpr int PLATFORM_MAX_CORES_PER_THREAD =
    PLATFORM_MAX_AIC_PER_THREAD + PLATFORM_MAX_AIV_PER_THREAD;  // 72

// =============================================================================
// Performance Profiling Configuration
// =============================================================================

/**
 * Maximum number of cores that can be profiled simultaneously
 * Calculated as: MAX_BLOCKDIM * CORES_PER_BLOCKDIM = 24 * 3 = 72
 */
constexpr int PLATFORM_MAX_CORES =
    PLATFORM_MAX_BLOCKDIM * PLATFORM_CORES_PER_BLOCKDIM;  // 72

/**
 * Performance buffer capacity for each double buffer
 * Number of PerfRecord entries per buffer (ping or pong)
 */
constexpr int PLATFORM_PROF_BUFFER_SIZE = 20;

/**
 * Ready queue capacity for performance data collection
 * Queue holds (core_index, buffer_id) entries for buffers ready to be read by Host.
 * Capacity = PLATFORM_MAX_CORES * 2 (each core has 2 buffers: ping and pong)
 */
constexpr int PLATFORM_PROF_READYQUEUE_SIZE = PLATFORM_MAX_CORES * 2;  // 144

/**
 * System counter frequency (get_sys_cnt)
 * Used to convert timestamps to microseconds.
 */
constexpr uint64_t PLATFORM_PROF_SYS_CNT_FREQ = 50000000;  // 50 MHz

/**
 * Timeout duration for performance data collection (seconds)
 */
constexpr int PLATFORM_PROF_TIMEOUT_SECONDS = 2;

/**
 * Number of empty polling iterations before checking timeout
 */
constexpr int PLATFORM_PROF_EMPTY_POLLS_CHECK_NUM = 1000;

inline double cycles_to_us(uint64_t cycles) {
    return (static_cast<double>(cycles) / PLATFORM_PROF_SYS_CNT_FREQ) * 1000000.0;
};

#endif  // PLATFORM_COMMON_PLATFORM_CONFIG_H_
