/**
 * @file platform_regs.h
 * @brief Platform-level register address access for AICPU
 *
 * Provides set/get functions for the per-core register base address array.
 * The platform layer calls set_platform_regs() before aicpu_execute(),
 * and runtime code calls get_platform_regs() when register access is needed.
 *
 * Implementation: src/platform/src/aicpu/platform_regs.cpp (shared across all platforms)
 */

#ifndef PLATFORM_AICPU_PLATFORM_REGS_H_
#define PLATFORM_AICPU_PLATFORM_REGS_H_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Set the platform register base address array.
 * Called by the platform layer before aicpu_execute().
 *
 * @param regs  Pointer (as uint64_t) to per-core register base address array
 */
void set_platform_regs(uint64_t regs);

/**
 * Get the platform register base address array.
 * Called by runtime AICPU executor code that needs register access.
 *
 * @return Pointer (as uint64_t) to per-core register base address array
 */
uint64_t get_platform_regs();

#ifdef __cplusplus
}
#endif

#endif  // PLATFORM_AICPU_PLATFORM_REGS_H_
