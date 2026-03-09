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

#if defined(__x86_64__)
#define SPIN_WAIT_HINT() do { __builtin_ia32_pause(); sched_yield(); } while(0)
#elif defined(__aarch64__)
#define SPIN_WAIT_HINT() do { __asm__ volatile("yield" ::: "memory"); sched_yield(); } while(0)
#else
#define SPIN_WAIT_HINT() sched_yield()
#endif

#endif  // PLATFORM_A2A3SIM_AICPU_SPIN_HINT_H_
