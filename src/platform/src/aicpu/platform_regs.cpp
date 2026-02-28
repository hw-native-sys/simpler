/**
 * @file platform_regs.cpp
 * @brief Platform-level register access implementation for AICPU
 *
 * Provides unified interface for:
 * 1. Platform register base address management (set/get_platform_regs)
 * 2. Register read/write operations with optimized memory barriers
 *
 * Memory Barrier Strategy:
 * - read_reg: Full barriers (__sync_synchronize) to ensure store-load ordering
 * - write_reg: Full barriers (__sync_synchronize) to guarantee global visibility
 *
 * Platform Support:
 * - a2a3: MMIO volatile pointer access to real hardware registers
 * - a2a3sim: Volatile pointer access to host-allocated simulated registers
 */

#include <cstdint>
#include "aicpu/platform_regs.h"

static uint64_t g_platform_regs = 0;

void set_platform_regs(uint64_t regs) {
    g_platform_regs = regs;
}

uint64_t get_platform_regs() {
    return g_platform_regs;
}

uint64_t read_reg(uint64_t reg_base_addr, RegId reg) {
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(
        reg_base_addr + reg_offset(reg));

    __sync_synchronize();

    // Read the register value
    uint64_t value = static_cast<uint64_t>(*ptr);

    __sync_synchronize();

    return value;
}

void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value) {
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(
        reg_base_addr + reg_offset(reg));

    __sync_synchronize();

    // Write the register value
    *ptr = static_cast<uint32_t>(value);

    __sync_synchronize();
}
