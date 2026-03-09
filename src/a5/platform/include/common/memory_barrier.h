/**
 * @file memory_barrier.h
 * @brief Memory barrier definitions for shared memory synchronization
 *
 * This header provides platform-specific memory barrier macros for
 * synchronizing shared memory accesses between Host, AICPU, and AICore.
 *
 * Memory barriers ensure that:
 * - Read barriers (rmb): All reads before the barrier complete before any reads after
 * - Write barriers (wmb): All writes before the barrier complete before any writes after
 *
 * These are critical for correct operation of lock-free data structures
 * and shared memory protocols across different processing units.
 */

#ifndef PLATFORM_COMMON_MEMORY_BARRIER_H_
#define PLATFORM_COMMON_MEMORY_BARRIER_H_

// =============================================================================
// Memory Barrier Macros
// =============================================================================

#ifdef __aarch64__
    /**
     * Read memory barrier (ARM64)
     * Ensures all loads before this point complete before any loads after.
     */
    #define rmb() __asm__ __volatile__("dsb ld" ::: "memory")

    /**
     * Write memory barrier (ARM64)
     * Ensures all stores before this point complete before any stores after.
     */
    #define wmb() __asm__ __volatile__("dsb st" ::: "memory")
#else
    /**
     * Compiler barrier (fallback for non-ARM64 platforms)
     * Prevents compiler reordering but does not emit hardware barriers.
     */
    #define rmb() __asm__ __volatile__("" ::: "memory")
    #define wmb() __asm__ __volatile__("" ::: "memory")
#endif

#endif  // PLATFORM_COMMON_MEMORY_BARRIER_H_
