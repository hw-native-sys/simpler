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
 * The CPU hint (pause/yield) reduces pipeline waste, and sched_yield() lets the
 * OS scheduler give time slices to threads doing real work.
 */

#ifndef PLATFORM_A2A3SIM_AICPU_SPIN_HINT_H_
#define PLATFORM_A2A3SIM_AICPU_SPIN_HINT_H_

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

#endif  // PLATFORM_A2A3SIM_AICPU_SPIN_HINT_H_
