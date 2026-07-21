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
 * @file inner_platform_regs.cpp
 * @brief Variant-specific platform_regs hooks for real hardware (a2a3)
 *
 * a2a3 sim and onboard share the same register layout, so read_reg /
 * write_reg live in the shared src/aicpu/platform_regs.cpp. This file holds
 * the variant-specific hooks: the reg_load_acquire / reg_store_release
 * handshake-gate accessors (plain MMIO here) and the deinit-timeout budget —
 * see platform_regs.h for the rationale of each.
 */

#include <cstdint>
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

// Plain Device-nGnRnE MMIO load/store (atomics are not valid on Device
// memory); cross Device<->Normal ordering is the caller's rmb()/wmb(). See
// platform_regs.h.
uint32_t reg_load_acquire(const volatile uint32_t *p) { return *p; }

void reg_store_release(volatile uint32_t *p, uint32_t v) { *p = v; }

/**
 * @brief Deinit ACK-wait budget on real hardware: 1 s.
 *
 * On silicon a non-response this long means the op was STARS-killed or the
 * core is wedged; aclrtResetDevice will clean up. The tight budget preserves
 * hardware hang detection. See the declaration in platform_regs.h for the
 * full rationale.
 *
 * @return Timeout in profiling system-counter ticks.
 */
uint64_t inner_get_deinit_timeout_ticks() { return PLATFORM_PROF_SYS_CNT_FREQ; }
