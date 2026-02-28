/**
 * @file platform_regs.h
 * @brief Platform-level register access interface for AICPU
 *
 * Provides unified interface for:
 * 1. Platform register base address management (set/get_platform_regs)
 * 2. Register read/write operations (read_reg/write_reg)
 *
 * The platform layer calls set_platform_regs() before aicpu_execute(),
 * and runtime code calls get_platform_regs() and read_reg/write_reg()
 * for register communication with AICore.
 *
 * Implementation: src/platform/src/aicpu/platform_regs.cpp (shared across all platforms)
 */

#ifndef PLATFORM_AICPU_PLATFORM_REGS_H_
#define PLATFORM_AICPU_PLATFORM_REGS_H_

#include <cstdint>
#include "common/platform_config.h"

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

/**
 * Read a register value from an AICore's register block
 *
 * @param reg_base_addr  Base address of the AICore's register block
 * @param reg            Register identifier (C++ enum class)
 * @return Register value (zero-extended to uint64_t)
 */
uint64_t read_reg(uint64_t reg_base_addr, RegId reg);

/**
 * Write a value to an AICore's register
 *
 * @param reg_base_addr  Base address of the AICore's register block
 * @param reg            Register identifier (C++ enum class)
 * @param value          Value to write (truncated to register width)
 */
void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value);


#endif  // PLATFORM_AICPU_PLATFORM_REGS_H_
