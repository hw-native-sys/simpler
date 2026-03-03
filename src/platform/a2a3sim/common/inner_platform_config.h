/**
 * @file inner_platform_config.h
 * @brief Platform-specific configuration for a2a3sim (Simulation)
 *
 * This file provides platform-specific constants for the a2a3sim simulation platform.
 * Uses the same register offsets and capacity as a2a3 hardware.
 */

#ifndef PLATFORM_INNER_CONFIG_H_
#define PLATFORM_INNER_CONFIG_H_

#include <cstdint>

// =============================================================================
// Platform Capacity Constraints
// =============================================================================

/**
 * Maximum block dimension supported by a2a3sim platform
 * Each block contains 1 AIC + 2 AIV cores (same as a2a3 hardware)
 */
constexpr int PLATFORM_MAX_BLOCKDIM = 24;

/**
 * Maximum AICPU scheduling threads for a2a3sim platform
 */
constexpr int PLATFORM_MAX_AICPU_THREADS = 4;

// =============================================================================
// Register Offsets (a2a3sim uses same offsets as a2a3)
// =============================================================================

/**
 * Task dispatch register offset (AICPU→AICore)
 */
constexpr uint32_t REG_SPR_DATA_MAIN_BASE_OFFSET = 0xA0;

/**
 * Status register offset (AICore→AICPU)
 */
constexpr uint32_t REG_SPR_COND_OFFSET = 0x4C8;

/**
 * Fast path control register offset
 */
constexpr uint32_t REG_SPR_FAST_PATH_ENABLE_OFFSET = 0x18;

// =============================================================================
// Fast Path Control Values
// =============================================================================

/**
 * Value to enable fast path
 */
constexpr uint32_t REG_SPR_FAST_PATH_OPEN = 0xE;

/**
 * Value to disable fast path
 */
constexpr uint32_t REG_SPR_FAST_PATH_CLOSE = 0xF;

// =============================================================================
// Chip-Specific Configuration (DAV_2201)
// =============================================================================

/**
 * Chip-specific constants for DAV_2201 simulation
 * Uses the same physical core count as a2a3 hardware.
 */
namespace DAV_2201 {
    /**
     * Maximum physical AICore count for DAV_2201 chip (simulation)
     * Includes all AIC and AIV sub-cores across all block dimensions.
     */
    constexpr uint32_t PLATFORM_MAX_PHYSICAL_CORES = 25;
}

#endif  // PLATFORM_INNER_CONFIG_H_
