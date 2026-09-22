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
 * @file spin_hint.h
 * @brief Platform-specific spin-wait policy for AICPU (simulation)
 *
 * In simulation, all AICPU scheduler threads share a small number of host CPU
 * cores with AICore threads. Without explicit yielding, idle scheduler threads
 * in tight polling loops starve the AICore thread executing the actual kernel,
 * causing premature scheduler timeouts before the kernel can complete —
 * especially on resource-constrained CI runners (e.g., 2 cores running 13+
 * threads).
 *
 * The CPU hint (pause/yield) plus sched_yield() let the OS scheduler give time
 * slices to threads doing real work. The companion mitigation is the
 * no-progress budget PLATFORM_SCHEDULER_TIMEOUT_MS, which is sized to keep a
 * slow CPU-sim task (e.g. matmul-heavy kernels) making real progress from being
 * mistaken for a deadlock; it is one value across every platform variant and
 * lives in platform_config.h.
 */

#pragma once

#include <cstdint>
#include <sched.h>

#include "common/platform_config.h"

#if defined(__aarch64__)
#define SPIN_WAIT_HINT()                        \
    do {                                        \
        __asm__ volatile("yield" ::: "memory"); \
        sched_yield();                          \
    } while (0)
#elif defined(__x86_64__)
#define SPIN_WAIT_HINT()        \
    do {                        \
        __builtin_ia32_pause(); \
        sched_yield();          \
    } while (0)
#else
#define SPIN_WAIT_HINT() sched_yield()
#endif

constexpr int32_t PLATFORM_TENSOR_DATA_WAIT_TIMEOUT_MS = PLATFORM_SIM_TENSOR_DATA_WAIT_TIMEOUT_MS;
