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
 * @brief Platform-level register access implementation for AICPU
 *
 * Provides unified interface for:
 * 1. Platform register base address management (set/get_platform_regs)
 * 2. Register read/write operations (volatile MMIO, no barrier)
 * 3. Platform-agnostic AICore register initialization/deinitialization
 *
 * Ordering: read_reg / write_reg emit only the volatile MMIO load/store.
 * The MMIO region is Device-nGnRE (Early-write-ack, no Gathering, no
 * Reordering) — see docs/hardware/mmio-performance.md for the driver
 * source trace. nR orders accesses within the same region; cross
 * Device <-> Normal-cacheable ordering is the caller's responsibility
 * (wmb() before a publishing register write, rmb() after observing a
 * register hand-off bit).
 *
 * Platform Support:
 * - a2a3: MMIO volatile pointer access to real hardware registers
 * - a2a3sim: Volatile pointer access to host-allocated simulated registers
 */

#include <cstdint>
#include "aicpu/platform_regs.h"
#include "aicpu/device_time.h"
#include "common/platform_config.h"
#include "common/memory_barrier.h"
#include "aicore_teardown.h"

static uint64_t g_platform_regs = 0;
static uint64_t g_platform_pmu_reg_addrs = 0;

void set_platform_regs(uint64_t regs) { g_platform_regs = regs; }

uint64_t get_platform_regs() { return g_platform_regs; }

void set_platform_pmu_reg_addrs(uint64_t pmu_regs) { g_platform_pmu_reg_addrs = pmu_regs; }

uint64_t get_platform_pmu_reg_addrs() { return g_platform_pmu_reg_addrs; }

volatile uint32_t *get_reg_ptr(uint64_t reg_base_addr, RegId reg) {
    return reinterpret_cast<volatile uint32_t *>(reg_base_addr + reg_offset(reg));
}

uint64_t read_reg(uint64_t reg_base_addr, RegId reg) {
    return static_cast<uint64_t>(reg_load_acquire(get_reg_ptr(reg_base_addr, reg)));
}

void platform_init_aicore_regs(uint64_t reg_addr) {
    // Both a2a3 and a2a3sim require fast path control to be enabled before use
    write_reg(reg_addr, RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);

    // Initialize task dispatch register to idle state
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
}

void platform_signal_aicore_exit(uint64_t reg_addr) { write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL); }

uint64_t platform_aicore_exit_deadline() { return get_sys_cnt_aicpu() + inner_get_deinit_timeout_ticks(); }

int32_t platform_retire_aicore_group(const AicoreExitTarget *targets, size_t count, uint64_t deadline, bool *released) {
    // `released` is filled before anything can return, rejection included, so a
    // caller may read it without initializing the buffer. An over-large `count`
    // says nothing about how big that buffer is, so the fill stops at the one
    // size the contract guarantees.
    if (released != nullptr) {
        const size_t reportable = count < PLATFORM_MAX_CORES ? count : PLATFORM_MAX_CORES;
        for (size_t i = 0; i < reportable; ++i)
            released[i] = false;
    }
    if (count > PLATFORM_MAX_CORES || (count != 0 && targets == nullptr)) return -1;
    for (size_t i = 0; i < count; ++i) {
        if (targets[i].reg_addr == 0 || targets[i].teardown == nullptr) return -1;
    }

    // COND can retain a prior launch's EXITED or nonce. Only this physical
    // core writes it; ordinary task tokens cannot use the reserved encodings.
    // The unique retirement owner selects a value different from that actual
    // retained value, so only a response to this request can acknowledge exit.
    uint32_t exit_requests[PLATFORM_MAX_CORES];
    for (size_t i = 0; i < count; ++i) {
        const uint32_t old_cond = static_cast<uint32_t>(read_reg(targets[i].reg_addr, RegId::COND));
        const uint32_t request = old_cond == AICORE_EXIT_NONCE_0 ? AICORE_EXIT_NONCE_1 : AICORE_EXIT_NONCE_0;
        exit_requests[i] = request;
        __atomic_store_n(&targets[i].teardown->exit_request, request, __ATOMIC_RELAXED);
    }
    // The request is Normal memory and EXIT is posted Device memory. Publish
    // every request before any core can observe the EXIT that asks for it.
    wmb();

    // Broadcast to the whole group before waiting on any member, so the cores
    // drain concurrently and a dead core's wait does not serialize behind the
    // cores ahead of it.
    for (size_t i = 0; i < count; ++i) {
        platform_signal_aicore_exit(targets[i].reg_addr);
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
            const uint32_t cond = static_cast<uint32_t>(read_reg(targets[i].reg_addr, RegId::COND));
            if (cond == exit_requests[i]) {
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
            write_reg(targets[i].reg_addr, RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
            write_reg(targets[i].reg_addr, RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
        } else {
            rc = -1;
        }
    }
    // Each acknowledged window retains its own completion readback before
    // the common drain and release pass. Unacknowledged windows stay open.
    for (size_t i = 0; i < count; ++i) {
        if (acknowledged[i]) {
            (void)read_reg(targets[i].reg_addr, RegId::FAST_PATH_ENABLE);
        }
    }
    // One drain covers every readback the close pass issued, and it is what
    // orders every store below after the CLOSE it belongs to: a dsb blocks
    // every later instruction until it completes, so the relaxed stores cannot
    // move ahead of it.
    rmb();
    // An open return gate is only ever paired with a closed window: releasing a
    // core whose window is still open is the ordering violation this protocol
    // exists to prevent. Unreleased cores are reported through `released` so
    // the caller can name them; the host recovery path owns them from here.
    for (size_t i = 0; i < count; ++i) {
        if (acknowledged[i]) {
            __atomic_store_n(&targets[i].teardown->post_close_release, AICORE_POST_CLOSE_RELEASE, __ATOMIC_RELAXED);
        }
        if (released != nullptr) released[i] = acknowledged[i];
    }
    return rc;
}

uint32_t platform_get_physical_cores_count() {
    return DAV_2201::PLATFORM_MAX_PHYSICAL_CORES * PLATFORM_CORES_PER_BLOCKDIM;
}
