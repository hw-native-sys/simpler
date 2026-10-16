/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file inner_kernel.h
 * @brief Platform-specific AICore definitions for real hardware (a5)
 *
 * This header provides platform-specific macro definitions for AICore kernels
 * running on real Ascend hardware with CANN compiler support.
 */

#ifndef PLATFORM_A5_AICORE_INNER_KERNEL_H_
#define PLATFORM_A5_AICORE_INNER_KERNEL_H_

#include <cstdint>
#include "aicore_teardown.h"

#include "common/platform_config.h"

// AICore function attribute for CANN compiler
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// dcci (Data Cache Clean and Invalidate) is provided by CANN headers
// No need to define it here - it's a hardware instruction

// SPIN_WAIT_HINT - no-op on real hardware (AICore has dedicated polling support)
#define SPIN_WAIT_HINT() ((void)0)

// A kernel's output dcci issues a write-back; it does not wait for one. The task
// reports FIN right after, and a consumer that observes FIN then reads that
// output, so the flush has to have completed first. pto-isa documents the pair
// for this chip family and the runtime already pairs dsb with its own flushes
// when it publishes completion data (runtime/async_kernel_api.h).
#define OUT_OF_ORDER_STORE_BARRIER() dsb(DSB_DDR)

// OUT_OF_ORDER_LOAD_BARRIER - no-op on real hardware (dcci handles cache coherency)
#define OUT_OF_ORDER_LOAD_BARRIER() ((void)0)

// OUT_OF_ORDER_FULL_BARRIER - no-op on real hardware (dcci handles full cache coherency)
#define OUT_OF_ORDER_FULL_BARRIER() ((void)0)

// Initialize only this core's ACK line, and complete the reset before its
// report can make the core a retirement target. Zero never matches a request.
__aicore__ inline void initialize_aicore_exit_ack(__gm__ AicoreTeardownControl *control) {
    control->exit_ack = 0;
    dcci(&control->exit_ack, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb(DSB_DDR);
}

// COND alone can retain EXITED from an earlier launch. Echo the AICPU's request
// after all this core's writes have drained, then wait for its post-close gate.
// A5 may broadcast EXIT before a partition publishes its request; keep observing
// the request while parked so that ordering still makes progress.
__aicore__ inline void acknowledge_aicore_exit_and_wait(__gm__ AicoreTeardownControl *control) {
    dsb(DSB_DDR);
    uint32_t published_request = 0;
    while (true) {
        const uint32_t request = static_cast<uint32_t>(ld_dev(&control->exit_request, 0));
        if (request != 0 && request != published_request) {
            control->exit_ack = request;
            dcci(&control->exit_ack, SINGLE_CACHE_LINE, CACHELINE_OUT);
            dsb(DSB_DDR);
            published_request = request;
        }
        if (static_cast<uint32_t>(ld_dev(&control->post_close_release, 0)) == AICORE_POST_CLOSE_RELEASE) break;
        SPIN_WAIT_HINT();
    }
    dsb(DSB_DDR);
}

// EXITED acknowledges quiescence; only the AICPU's post-close gate permits return.
__aicore__ inline void wait_for_post_close_release(__gm__ uint32_t *release) {
    while (static_cast<uint32_t>(ld_dev(release, 0)) != AICORE_POST_CLOSE_RELEASE) {
        SPIN_WAIT_HINT();
    }
    dsb(DSB_DDR);
}

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
    case RegId::CTRL:
        return static_cast<uint64_t>(get_ctrl());
    case RegId::COND:
    default:
        // PMU MMIO regs are read via ld_dev(base, offset) — see
        // pmu_collector_aicore.h. This single-arg SPR form only covers
        // DATA_MAIN_BASE / CTRL / COND.
        return 0;
    }
}

// Resident startup can fail before AICPU has reset this run's return gate.
// Only its DMB EXIT, published after that reset, permits the EXITED ACK.
__aicore__ inline void wait_for_aicpu_exit_signal() {
    while (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) != AICORE_EXIT_SIGNAL) {
        SPIN_WAIT_HINT();
    }
}

/**
 * Read the high 32 bits of DATA_MAIN_BASE.
 *
 * AICore reads the full 64-bit SPR via MOV; the high half is the
 * early-dispatch doorbell written by AICPU (low half stays the dispatch
 * token). Read-only on the AICore side, so this is always valid (unlike writes
 * to DATA_MAIN_BASE, which the SPR-write port rejects).
 */
__aicore__ inline uint32_t read_dmb_high32() {
    uint64_t v;
    __asm__ volatile("MOV %0, DATA_MAIN_BASE\n" : "=l"(v));
    return static_cast<uint32_t>(v >> 32);
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
        set_cond(static_cast<uint32_t>(value));
        break;
    case RegId::CTRL:
        set_ctrl(value);
        break;
    default:
        // DATA_MAIN_BASE is write-only from AICPU; PMU MMIO regs are
        // written only by AICPU (init/finalize path). AICore does not
        // need a write path here.
        break;
    }
}

/**
 * Get the physical core ID from hardware
 *
 * @return Physical core ID (masked to 12 bits)
 */
__aicore__ inline uint32_t get_physical_core_id() { return static_cast<uint32_t>(get_coreid()) & AICORE_COREID_MASK; }

__aicore__ inline uint64_t platform_ssbuf_hardware_base_address() { return 0; }

// =============================================================================
// System Counter
// =============================================================================

/**
 * Get AICore system counter
 *
 * @return Hardware counter value (ticks)
 */
__aicore__ __attribute__((always_inline)) inline uint64_t get_sys_cnt_aicore() { return get_sys_cnt(); }

// A native retirement claimant publishes one request. After the nonce
// becomes observable, this core performs no more SPR writes.
__aicore__ inline void acknowledge_aicore_cond_nonce_and_wait(__gm__ AicoreTeardownControl *control) {
    dsb(DSB_DDR);
    while (true) {
        const uint32_t request = static_cast<uint32_t>(ld_dev(&control->exit_request, 0));
        if (request == AICORE_EXIT_NONCE_0 || request == AICORE_EXIT_NONCE_1) {
            set_cond(request);
            break;
        }
        SPIN_WAIT_HINT();
    }
    wait_for_post_close_release(&control->post_close_release);
}

#endif  // PLATFORM_A5_AICORE_INNER_KERNEL_H_
