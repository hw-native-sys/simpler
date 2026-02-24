/**
 * @file aicpu_regs.h
 * @brief AICPU-side register access interface
 *
 * Provides unified read_reg/write_reg for AICPU to access AICore registers.
 * On real hardware (a2a3): MMIO volatile pointer access with memory barriers.
 * In simulation (a2a3sim): volatile pointer access to host-allocated memory.
 */

#ifndef PLATFORM_AICPU_AICPU_REGS_H_
#define PLATFORM_AICPU_AICPU_REGS_H_

#include <cstdint>
#include "common/platform_config.h"

/**
 * Read a register value from an AICore's register block
 *
 * @param reg_base_addr  Base address of the AICore's register block
 * @param reg            Register identifier
 * @return Register value (zero-extended to uint64_t)
 */
uint64_t read_reg(uint64_t reg_base_addr, RegId reg);

/**
 * Write a value to an AICore's register
 *
 * @param reg_base_addr  Base address of the AICore's register block
 * @param reg            Register identifier
 * @param value          Value to write (truncated to register width)
 */
void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value);

#endif  // PLATFORM_AICPU_AICPU_REGS_H_
