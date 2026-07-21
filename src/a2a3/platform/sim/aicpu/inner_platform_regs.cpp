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
 * @brief Variant-specific platform_regs hooks for simulation (a2a3sim)
 *
 * a2a3 sim and onboard share the same register layout, so read_reg /
 * write_reg live in the shared src/aicpu/platform_regs.cpp. This file holds
 * the variant-specific hooks: the reg_load_acquire / reg_store_release
 * handshake-gate accessors (atomic here) and the deinit-timeout budget — see
 * platform_regs.h for the rationale of each.
 */

#include <cstdint>
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

// Atomic acquire/release: simulated registers are plain host memory shared
// across the AICPU and AICore host threads, so the access itself must carry
// happens-before (and be visible to TSAN). See platform_regs.h.
uint32_t reg_load_acquire(const volatile uint32_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }

void reg_store_release(volatile uint32_t *p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }

/**
 * @brief Deinit ACK-wait budget on sim: 10 s.
 *
 * On sim "AICore" is a host CPU thread, so a missing exit ACK usually just
 * means the OS scheduler hasn't given that thread a slice on a CPU-starved CI
 * runner — not a wedged op. The wide budget tolerates that jitter. See the
 * declaration in platform_regs.h for the full rationale.
 *
 * @return Timeout in profiling system-counter ticks.
 */
uint64_t inner_get_deinit_timeout_ticks() { return 10 * PLATFORM_PROF_SYS_CNT_FREQ; }
