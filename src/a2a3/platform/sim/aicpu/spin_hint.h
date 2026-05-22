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
 * @brief Platform-specific spin-wait hint for AICPU (simulation)
 *
 * In simulation, all AICPU scheduler threads share a small number of host CPU
 * cores with AICore threads. Without explicit yielding, idle scheduler threads
 * in tight polling loops starve the AICore thread executing the actual kernel,
 * causing iteration-based timeouts (MAX_IDLE_ITERATIONS) before the kernel can
 * complete — especially on resource-constrained CI runners (e.g., 2 cores
 * running 13+ threads).
 *
 * Two mitigations live here. The CPU hint (pause/yield) plus sched_yield() let
 * the OS scheduler give time slices to threads doing real work, and
 * PLATFORM_MAX_IDLE_ITERATIONS below raises the idle cap well above the onboard
 * value so a slow CPU-sim task (e.g. matmul-heavy kernels) making real progress
 * is not mistaken for a deadlock.
 */

#ifndef PLATFORM_A2A3SIM_AICPU_SPIN_HINT_H_
#define PLATFORM_A2A3SIM_AICPU_SPIN_HINT_H_

#include <cstdint>
#include <sched.h>

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

// Consecutive idle scheduler iterations (no task progress) before the dispatch
// loop aborts with PTO2_ERROR_SCHEDULER_TIMEOUT. CPU sim accumulates idle
// iterations far faster than real kernel progress, so the cap is raised well
// above the onboard value to avoid false timeouts on slow (e.g. matmul-heavy)
// kernels. Set to 10x the onboard cap as a conservative bump; raise further if
// a slow kernel still false-times-out. The runtime consumes it as
// MAX_IDLE_ITERATIONS (see scheduler_types.h).
constexpr int32_t PLATFORM_MAX_IDLE_ITERATIONS = 8000000;

#endif  // PLATFORM_A2A3SIM_AICPU_SPIN_HINT_H_
