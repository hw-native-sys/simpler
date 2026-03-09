/**
 * @file inner_platform_regs.cpp
 * @brief AICPU register read/write for real hardware (a5)
 *
 * halResMap maps each AICore as 3MB of contiguous MMIO. Hardware offsets
 * (e.g. DATA_MAIN_BASE=0xD0, COND=0x5108) are applied directly to the
 * virtual address with no remapping.
 */

#include <cstdint>
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

uint64_t read_reg(uint64_t reg_base_addr, RegId reg) {
    uint32_t offset = reg_offset(reg);
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(reg_base_addr + offset);

    __sync_synchronize();
    uint64_t value = static_cast<uint64_t>(*ptr);
    __sync_synchronize();

    return value;
}

void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value) {
    uint32_t offset = reg_offset(reg);
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(reg_base_addr + offset);

    __sync_synchronize();
    *ptr = static_cast<uint32_t>(value);
    __sync_synchronize();
}
