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
 * Link-time stub for platform_regs.h's get_reg_ptr, plus the accessors a test
 * reads the stubbed register back through.
 *
 * A file of its own, rather than part of another stub group, because the two
 * arches keep this one definition in different files. a2a3 has a single copy
 * under platform/shared/aicpu/platform_regs.cpp, which a test compiling only
 * the sim variant does not link, so it needs this stub. a5 splits the
 * implementation per variant and so defines it in
 * platform/sim/aicpu/inner_platform_regs.cpp, which such a test does link —
 * there this stub would be a second definition.
 *
 * So a target that compiles a5's inner_platform_regs.cpp takes the other
 * members of <arch>_ut_support without this one. The archive is what makes
 * that possible: a member is pulled only to resolve a symbol nothing else has
 * defined, so no target has to name what it skips.
 */

#include <cstdint>

#include "aicpu/platform_regs.h"

// SchedulerState::ring_one_doorbell (scheduler.h, speculative early-dispatch)
// is an inline that resolves a register id to its MMIO pointer via get_reg_ptr
// and writes a 64-bit token through it. There is no MMIO on the host UT runner;
// hand back writable static storage (8 bytes — the doorbell is a 64-bit store)
// and retain the requested base address for ownership tests.
static volatile uint64_t g_test_reg = 0;
static uint64_t g_test_reg_base_addr = 0;

volatile uint32_t *get_reg_ptr(uint64_t reg_base_addr, RegId /* reg */) {
    g_test_reg_base_addr = reg_base_addr;
    return reinterpret_cast<volatile uint32_t *>(&g_test_reg);
}

void reset_test_reg_stub() {
    g_test_reg = 0;
    g_test_reg_base_addr = 0;
}

uint64_t get_test_reg_stub_value() { return g_test_reg; }

uint64_t get_test_reg_stub_base_addr() { return g_test_reg_base_addr; }
