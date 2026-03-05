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

#include "common/platform_config.h"

// AICore function attribute - no-op in simulation
#ifndef __aicore__
#define __aicore__
#endif

// dcci (Data Cache Clean and Invalidate) - acquire fence in simulation
// Hardware dcci invalidates cache to ensure fresh reads from shared memory.
// In simulation, an acquire fence provides the equivalent ordering guarantee.
// Use variadic macro to support both 2-arg and 3-arg calls.
#define dcci(...) std::atomic_thread_fence(std::memory_order_acquire)

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
#if defined(__x86_64__)
#define SPIN_WAIT_HINT() do { __builtin_ia32_pause(); sched_yield(); } while(0)
#elif defined(__aarch64__)
#define SPIN_WAIT_HINT() do { __asm__ volatile("yield" ::: "memory"); sched_yield(); } while(0)
#else
#define SPIN_WAIT_HINT() sched_yield()
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
 * Per-thread simulated register base address.
 * Set by the kernel wrapper before calling aicore_execute().
 * Points to a SIM_REG_BLOCK_SIZE-byte block allocated by DeviceRunner.
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
 * @param reg  Register identifier
 * @return Register value (zero-extended to uint64_t)
 */
inline uint64_t read_reg(RegId reg) {
    uint32_t offset = reg_offset(reg);
    __sync_synchronize();
    return static_cast<uint64_t>(
        *reinterpret_cast<volatile uint32_t*>(g_sim_reg_base + offset));
}

/**
 * Write to an AICore register in simulated register memory
 *
 * @param reg    Register identifier
 * @param value  Value to write
 */
inline void write_reg(RegId reg, uint64_t value) {
    uint32_t offset = reg_offset(reg);
    *reinterpret_cast<volatile uint32_t*>(g_sim_reg_base + offset) =
        static_cast<uint32_t>(value);
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

#endif  // PLATFORM_A2A3SIM_AICORE_INNER_KERNEL_H_
