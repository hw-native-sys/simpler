/**
 * @file aicpu_regs.cpp
 * @brief AICPU-side register access implementation (a2a3 real hardware)
 *
 * Uses volatile MMIO pointer access with memory barriers for
 * cross-core register communication.
 */

#include "aicpu/aicpu_regs.h"

uint64_t read_reg(uint64_t reg_base_addr, RegId reg) {
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(
        reg_base_addr + reg_offset(reg));
    __sync_synchronize();
    return static_cast<uint64_t>(*ptr);
}

void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value) {
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(
        reg_base_addr + reg_offset(reg));
    *ptr = static_cast<uint32_t>(value);
    __sync_synchronize();
}
