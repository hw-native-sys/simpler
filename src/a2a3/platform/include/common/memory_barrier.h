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

#ifndef SRC_A2A3_PLATFORM_INCLUDE_COMMON_MEMORY_BARRIER_H_
#define SRC_A2A3_PLATFORM_INCLUDE_COMMON_MEMORY_BARRIER_H_

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

/**
 * Store-store barrier (ARM64, inner shareable domain)
 * Ensures all stores before this barrier are globally visible before any
 * stores after.
 */
#define OUT_OF_ORDER_STORE_BARRIER() __asm__ __volatile__("dmb ishst" ::: "memory")
#else
/**
 * Compiler barrier (fallback for non-ARM64 platforms)
 * Prevents compiler reordering but does not emit hardware barriers.
 */
#define rmb() __asm__ __volatile__("" ::: "memory")
#define wmb() __asm__ __volatile__("" ::: "memory")
#define OUT_OF_ORDER_STORE_BARRIER() __asm__ __volatile__("" ::: "memory")
#endif

// =============================================================================
// Polling Acquire Barrier
// =============================================================================

/**
 * Polling acquire barrier
 *
 * Use after poll_reg() detects the awaited condition (e.g., task completion),
 * before accessing Normal memory whose correctness depends on the polled value.
 *
 * ARM64: dmb ish (data memory barrier, inner shareable, full)
 *   Ensures the Device-memory register read is ordered before all subsequent
 *   Normal-memory loads and stores in the completion path.
 *   Chosen over dmb ishld (load-only) for safety margin: negligible cost
 *   (executed once per completion, not per poll iteration) and protects
 *   against future stores that may be added to the completion path.
 *
 * x86_64: compiler barrier only (TSO provides implicit acquire on all loads)
 *
 * Other: full barrier fallback (__sync_synchronize)
 */
#if defined(__aarch64__)
#define poll_acquire_barrier() __asm__ __volatile__("dmb ish" ::: "memory")
#elif defined(__x86_64__)
#define poll_acquire_barrier() __asm__ __volatile__("" ::: "memory")
#else
#define poll_acquire_barrier() __sync_synchronize()
#endif

#endif  // SRC_A2A3_PLATFORM_INCLUDE_COMMON_MEMORY_BARRIER_H_
