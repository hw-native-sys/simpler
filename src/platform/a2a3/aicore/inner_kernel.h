/**
 * @file inner_kernel.h
 * @brief Platform-specific AICore definitions for real hardware (a2a3)
 *
 * This header provides platform-specific macro definitions for AICore kernels
 * running on real Ascend hardware with CANN compiler support.
 */

#ifndef PLATFORM_A2A3_AICORE_INNER_KERNEL_H_
#define PLATFORM_A2A3_AICORE_INNER_KERNEL_H_

#include <cstdint>
#include "common/platform_config.h"

// AICore function attribute for CANN compiler
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// dcci (Data Cache Clean and Invalidate) is provided by CANN headers
// No need to define it here - it's a hardware instruction

/**
 * Read an AICore register via SPR access
 *
 * @param reg  Register identifier
 * @return Register value (zero-extended to uint64_t)
 */
__aicore__ inline uint64_t read_reg(RegId reg) {
    switch (reg) {
        case RegId::DATA_MAIN_BASE: {
            uint32_t val;
            __asm__ volatile("MOV %0, DATA_MAIN_BASE\n" : "=l"(val));
            return static_cast<uint64_t>(val);
        }
        case RegId::COND:
        case RegId::FAST_PATH_ENABLE:
            return 0;
    }
}

/**
 * Write to an AICore register
 *
 * @param reg    Register identifier
 * @param value  Value to write
 */
__aicore__ inline void write_reg(RegId reg, uint64_t value) {
    switch (reg) {
        case RegId::COND:
            set_cond(static_cast<AICoreStatus>(static_cast<uint32_t>(value)));
            break;
        case RegId::DATA_MAIN_BASE:
        case RegId::FAST_PATH_ENABLE:
            break;
    }
}

/**
 * Get the physical core ID from hardware
 *
 * @return Physical core ID (masked to 12 bits)
 */
__aicore__ inline uint32_t get_physical_core_id() {
    return static_cast<uint32_t>(get_coreid()) & AICORE_COREID_MASK;
}

#endif  // PLATFORM_A2A3_AICORE_INNER_KERNEL_H_
