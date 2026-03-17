/**
 * @file inner_kernel.h
 * @brief Platform-specific AICore definitions for simulation (a5sim)
 *
 * This header provides platform-specific macro definitions for AICore kernels
 * running in host-based simulation environment.
 */

#ifndef PLATFORM_A5SIM_AICORE_INNER_KERNEL_H_
#define PLATFORM_A5SIM_AICORE_INNER_KERNEL_H_

#include <atomic>
#include <chrono>
#include <cstdint>

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
// observe the COND register FIN signal before perf_buf->count is visible.
// Using seq_cst (dmb ish / full barrier) covers both directions safely.
// Use variadic macro to support both 2-arg and 3-arg calls.
#define dcci(...) std::atomic_thread_fence(std::memory_order_seq_cst)

// Cache coherency constants (no-op in simulation)
#define ENTIRE_DATA_CACHE 0
#define SINGLE_CACHE_LINE 0
#define CACHELINE_OUT 0

// SPIN_WAIT_HINT - CPU pause hint + OS yield for idle polling loops in simulation.
// In simulation, all AICore/AICPU threads share a small number of host CPU cores.
// The CPU hint (pause/yield) reduces pipeline waste, and sched_yield() lets the OS
// scheduler give time slices to threads doing real work (e.g., kernel execution),
// preventing starvation-induced timeouts on resource-constrained CI runners.
#include <sched.h>

#if defined(__aarch64__)
#define SPIN_WAIT_HINT() do { __asm__ volatile("yield" ::: "memory"); sched_yield(); } while(0)
#elif defined(__x86_64__)
#define SPIN_WAIT_HINT() do { __builtin_ia32_pause(); sched_yield(); } while(0)
#else
#define SPIN_WAIT_HINT() sched_yield()
#endif

// STORE_RELEASE_FENCE - store-store barrier to prevent reordering of data writes
// (physical_core_id, core_type) past the signal write (aicore_done) in handshake.
// Without this fence, aarch64 can reorder stores to different addresses, causing
// AICPU to read stale physical_core_id after observing aicore_done != 0.
#if defined(__aarch64__)
#define STORE_RELEASE_FENCE() __asm__ volatile("dmb ishst" ::: "memory")
#elif defined(__x86_64__)
#define STORE_RELEASE_FENCE() __asm__ volatile("" ::: "memory")
#else
#define STORE_RELEASE_FENCE() std::atomic_thread_fence(std::memory_order_release)
#endif

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
    uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();

    // Convert nanoseconds to counter ticks
    constexpr uint64_t kNsPerSec = std::nano::den;
    uint64_t seconds = elapsed_ns / kNsPerSec;
    uint64_t remaining_ns = elapsed_ns % kNsPerSec;

    uint64_t ticks = seconds * PLATFORM_PROF_SYS_CNT_FREQ +
                     (remaining_ns * PLATFORM_PROF_SYS_CNT_FREQ) / kNsPerSec;

    return ticks;
}

// =============================================================================
// Register Access Simulation
// =============================================================================

/**
 * Per-thread simulated register base address for page 0 (0x0000-0x0FFF).
 * Set by the kernel wrapper before calling aicore_execute().
 * Points to the first 4KB page of the sparse register block.
 */
extern thread_local volatile uint8_t* g_sim_reg_base;

/**
 * Per-thread simulated physical core ID.
 * Set by the kernel wrapper before calling aicore_execute().
 */
extern thread_local uint32_t g_sim_physical_core_id;

/**
 * Read an AICore register from simulated register memory
 *
 * Supports sparse register mapping via sparse_reg_ptr() helper.
 *
 * @param reg  Register identifier
 * @return Register value (zero-extended to uint64_t)
 */
inline uint64_t read_reg(RegId reg) {
    uint32_t offset = reg_offset(reg);
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(
        sparse_reg_ptr(g_sim_reg_base, offset));

    __sync_synchronize();
    return static_cast<uint64_t>(*ptr);
}

/**
 * Write to an AICore register in simulated register memory
 *
 * Supports sparse register mapping via sparse_reg_ptr() helper.
 *
 * @param reg    Register identifier
 * @param value  Value to write
 */
inline void write_reg(RegId reg, uint64_t value) {
    uint32_t offset = reg_offset(reg);
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(
        sparse_reg_ptr(g_sim_reg_base, offset));

    *ptr = static_cast<uint32_t>(value);
    __sync_synchronize();
}

/**
 * Get the physical core ID from simulation state
 *
 * @return Physical core ID for the current simulated core
 */
inline uint32_t get_physical_core_id() {
    return g_sim_physical_core_id;
}

#endif  // PLATFORM_A5SIM_AICORE_INNER_KERNEL_H_
