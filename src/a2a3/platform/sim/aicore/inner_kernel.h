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
 * @brief Platform-specific AICore definitions for simulation (a2a3sim)
 *
 * This header provides platform-specific macro definitions for AICore kernels
 * running in host-based simulation environment.
 */

#ifndef PLATFORM_A2A3SIM_AICORE_INNER_KERNEL_H_
#define PLATFORM_A2A3SIM_AICORE_INNER_KERNEL_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <dlfcn.h>

#include "common/platform_config.h"

// AICore function attribute - no-op in simulation
#ifndef __aicore__
#define __aicore__
#endif

// dcci (Data Cache Clean and Invalidate) - full fence in simulation
// Hardware dcci has two roles:
//   - without CACHELINE_OUT: invalidate (read from memory) -> acquire semantics
//   - with CACHELINE_OUT: write-back/flush (write to memory) -> release semantics
// On aarch64, acquire-only fences do NOT prevent store-store reordering across the
// barrier, so using acquire for the flush direction causes a race: the AICPU can
// observe the COND register FIN signal before l2_swimlane_buf->count is visible.
// Using seq_cst (dmb ish / full barrier) covers both directions safely.
// Use variadic macro to support both 2-arg and 3-arg calls.
#define dcci(...) std::atomic_thread_fence(std::memory_order_seq_cst)

// dsb / mem_dsb_t — CANN provides these on real AICore; l2_swimlane_collector uses them after dcci flush.
// Simulation: full fence (same strength as dcci above) so AICPU ordering matches hardware intent.
typedef int mem_dsb_t;
#define dsb(_kind)                                           \
    do {                                                     \
        (void)(_kind);                                       \
        std::atomic_thread_fence(std::memory_order_seq_cst); \
    } while (0)

// Cache coherency constants (no-op in simulation)
#define ENTIRE_DATA_CACHE 0
#define SINGLE_CACHE_LINE 0
#define CACHELINE_OUT 0

// pipe_barrier - memory barrier in simulation (hardware pipeline synchronization)
#define PIPE_ALL 0
#define pipe_barrier(pipe) __sync_synchronize()

// SPIN_WAIT_HINT - CPU pause hint + OS yield for idle polling loops in simulation.
// In simulation, all AICore/AICPU threads share a small number of host CPU cores.
// The CPU hint (pause/yield) reduces pipeline waste, and sched_yield() lets the OS
// scheduler give time slices to threads doing real work (e.g., kernel execution),
// preventing starvation-induced timeouts on resource-constrained CI runners.
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

// OUT_OF_ORDER_STORE_BARRIER - store-store barrier preventing store reordering.
// Ensures stores preceding the barrier are visible before stores following it.
// Used in the AICore-AICPU handshake to ensure data fields (core_type) are
// visible before the signal field (aicore_done), and to flush kernel outputs
// before writing the FIN signal register.
#if defined(__aarch64__)
#define OUT_OF_ORDER_STORE_BARRIER() __asm__ volatile("dmb ishst" ::: "memory")
#elif defined(__x86_64__)
#define OUT_OF_ORDER_STORE_BARRIER() __asm__ volatile("" ::: "memory")
#else
#define OUT_OF_ORDER_STORE_BARRIER() std::atomic_thread_fence(std::memory_order_release)
#endif

// OUT_OF_ORDER_LOAD_BARRIER - load-acquire barrier preventing load reordering.
// Ensures loads following the barrier are not reordered before the load
// immediately preceding it. Used after reading a signal register to ensure
// subsequent payload reads observe AICPU's writes.
#if defined(__aarch64__)
#define OUT_OF_ORDER_LOAD_BARRIER() __asm__ volatile("dmb ishld" ::: "memory")
#elif defined(__x86_64__)
#define OUT_OF_ORDER_LOAD_BARRIER() __asm__ volatile("" ::: "memory")
#else
#define OUT_OF_ORDER_LOAD_BARRIER() std::atomic_thread_fence(std::memory_order_acquire)
#endif

// OUT_OF_ORDER_FULL_BARRIER - full memory barrier preventing all load/store reordering.
// Equivalent to dmb ish (aarch64) / mfence (x86).
#define OUT_OF_ORDER_FULL_BARRIER() __sync_synchronize()

// =============================================================================
// System Counter Simulation
// =============================================================================

/**
 * Get simulated AICore system counter
 *
 * @return Simulated counter value (ticks)
 */
inline uint64_t get_sys_cnt_aicore() {
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

    // Convert nanoseconds to counter ticks
    constexpr uint64_t kNsPerSec = std::nano::den;
    uint64_t seconds = elapsed_ns / kNsPerSec;
    uint64_t remaining_ns = elapsed_ns % kNsPerSec;

    uint64_t ticks = seconds * PLATFORM_PROF_SYS_CNT_FREQ + (remaining_ns * PLATFORM_PROF_SYS_CNT_FREQ) / kNsPerSec;

    return ticks;
}

// =============================================================================
// Register Access Simulation
// =============================================================================

/**
 * Per-thread simulated register base address and physical core ID.
 * Stored in pthread TLS (not C++ thread_local) to avoid glibc TLSDESC
 * issues when this SO is loaded with RTLD_LOCAL on aarch64.
 * Set by aicore_execute_wrapper, read by read_reg/write_reg.
 */
volatile uint8_t *sim_get_reg_base();
uint32_t sim_get_physical_core_id();

/**
 * Read an AICore register from simulated register memory
 *
 * @param reg  Register identifier
 * @return Register value (zero-extended to uint64_t)
 */
inline uint64_t read_reg(RegId reg) {
    uint32_t offset = reg_offset(reg);
    // The register cell is the AICPU<->AICore handshake gate (dispatch / COND).
    // In sim it is plain host memory shared across the AICore and AICPU host
    // threads, so the load itself must be an atomic acquire for happens-before
    // to hold against the writer's release (and for TSAN to see it). A bare
    // fence beside a non-atomic load does neither, so __atomic_load_n subsumes
    // the old OUT_OF_ORDER_LOAD_BARRIER().
    volatile uint32_t *ptr = reinterpret_cast<volatile uint32_t *>(sim_get_reg_base() + offset);
    return static_cast<uint64_t>(__atomic_load_n(ptr, __ATOMIC_ACQUIRE));
}

/**
 * Read the high 32 bits of DATA_MAIN_BASE (speculative-dispatch doorbell).
 * The high word lives one 32-bit slot above the dispatch token.
 */
inline uint32_t read_dmb_high32() {
    uint32_t offset = reg_offset(RegId::DATA_MAIN_BASE);
    uint32_t hi = *reinterpret_cast<volatile uint32_t *>(sim_get_reg_base() + offset + 4);
    OUT_OF_ORDER_LOAD_BARRIER();
    return hi;
}

/**
 * Write to an AICore register in simulated register memory
 *
 * @param reg    Register identifier
 * @param value  Value to write
 */
inline void write_reg(RegId reg, uint64_t value) {
    uint32_t offset = reg_offset(reg);
    // Atomic release store: publishes any prior stores (e.g. kernel outputs,
    // the COND task_id payload) before the gate becomes visible to the AICPU's
    // acquire load. Subsumes the old OUT_OF_ORDER_STORE_BARRIER().
    volatile uint32_t *ptr = reinterpret_cast<volatile uint32_t *>(sim_get_reg_base() + offset);
    __atomic_store_n(ptr, static_cast<uint32_t>(value), __ATOMIC_RELEASE);
}

/**
 * Get the physical core ID from simulation state
 *
 * @return Physical core ID for the current simulated core
 */
inline uint32_t get_physical_core_id() { return sim_get_physical_core_id(); }

#endif  // PLATFORM_A2A3SIM_AICORE_INNER_KERNEL_H_
