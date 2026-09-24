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
 * @file platform_regs.cpp
 * @brief AICPU register interface - shared implementation
 *
 * Contains platform-agnostic functions shared across all platforms.
 * Platform-specific read_reg/write_reg are in:
 *   sim/aicpu/inner_platform_regs.cpp    -- sparse_reg_ptr mapping for simulation
 *   onboard/aicpu/inner_platform_regs.cpp -- direct MMIO offset for hardware
 */

#include <cstdint>
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "spin_hint.h"

static uint64_t g_platform_regs = 0;

void set_platform_regs(uint64_t regs) { g_platform_regs = regs; }

uint64_t get_platform_regs() { return g_platform_regs; }

void platform_init_aicore_regs(uint64_t reg_addr) {
    // Initialize task dispatch register to idle state
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
}

void platform_signal_aicore_exit(uint64_t reg_addr) { write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL); }

// Timeout is variant-specific (sim wider than onboard) — see
// inner_get_deinit_timeout_ticks declaration in platform_regs.h.
uint64_t platform_aicore_exit_deadline() { return get_sys_cnt_aicpu() + inner_get_deinit_timeout_ticks(); }

void platform_close_aicore_window(uint64_t reg_addr) {
    // Initialize task dispatch register to idle state
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
    // Complete the posted MMIO close. The store retires into the bus's
    // outstanding queue on its early write-ack and is not a device-write
    // completion fence on its own; a load to the same register cannot pass it,
    // so this readback is what drains it. The drain that pairs with this read is
    // the caller's, so several windows share one.
    (void)read_reg(reg_addr, RegId::DATA_MAIN_BASE);
}

int32_t platform_retire_aicore_group(const uint64_t *reg_addrs, size_t count, uint64_t deadline, bool *released) {
    // `released` is filled before anything can return, rejection included, so a
    // caller may read it without initializing the buffer. An over-large `count`
    // says nothing about how big that buffer is, so the fill stops at the one
    // size the contract guarantees.
    if (released != nullptr) {
        const size_t reportable = count < PLATFORM_MAX_CORES ? count : PLATFORM_MAX_CORES;
        for (size_t i = 0; i < reportable; ++i)
            released[i] = false;
    }
    if (count > PLATFORM_MAX_CORES || (count != 0 && reg_addrs == nullptr)) return -1;
    for (size_t i = 0; i < count; ++i) {
        if (reg_addrs[i] == 0) return -1;
    }

    // Broadcast to the whole group before waiting on any member, so the cores
    // drain concurrently and a dead core's wait does not serialize behind the
    // cores ahead of it.
    for (size_t i = 0; i < count; ++i) {
        platform_signal_aicore_exit(reg_addrs[i]);
    }
    wmb();

    // Round-robin rather than blocking on one core at a time. Blocking spends
    // the shared deadline on whichever core happens to come first, and every
    // core behind it is then judged on a peer's timeout instead of its own.
    // Sweeping non-blockingly gives each core the whole budget: a core is only
    // abandoned once the deadline passes with it still silent.
    // The sweep below reads an entry before it writes it, so the first `count`
    // must start false. Entries past `count` are never read.
    bool acknowledged[PLATFORM_MAX_CORES];
    for (size_t i = 0; i < count; ++i)
        acknowledged[i] = false;
    size_t remaining = count;
    while (remaining != 0) {
        for (size_t i = 0; i < count; ++i) {
            if (acknowledged[i]) continue;
            if (read_reg(reg_addrs[i], RegId::COND) == AICORE_EXITED_VALUE) {
                acknowledged[i] = true;
                --remaining;
            }
        }
        if (remaining == 0 || get_sys_cnt_aicpu() > deadline) break;
    }

    // No window closes until every ACK above is in, so a core is never quiesced
    // while a peer is still being waited on. COND is not re-read here: the
    // passes above already established it.
    int32_t rc = 0;
    for (size_t i = 0; i < count; ++i) {
        if (acknowledged[i]) {
            platform_close_aicore_window(reg_addrs[i]);
        } else {
            rc = -1;
        }
    }
    // One drain covers every readback the close pass issued, and it is what
    // orders every store below after the close it belongs to: a dsb blocks every
    // later instruction until it completes.
    rmb();
    if (released != nullptr) {
        for (size_t i = 0; i < count; ++i)
            released[i] = acknowledged[i];
    }
    return rc;
}

int32_t platform_deinit_aicore_regs(uint64_t reg_addr) {
    if (platform_retire_aicore_group(&reg_addr, 1, platform_aicore_exit_deadline()) != 0) {
        LOG_ERROR("Timed out waiting for AICore exit ack at reg_addr=0x%lx", static_cast<unsigned long>(reg_addr));
        return -1;
    }
    return 0;
}

uint32_t platform_get_physical_cores_count() {
    return DAV_3510::PLATFORM_MAX_PHYSICAL_CORES * PLATFORM_CORES_PER_BLOCKDIM;
}
