/**
 * PTO CQ Kernel API — inline functions for AICore kernels.
 *
 * These are NOT AICPU function calls. They are structured GM writes
 * that the AICPU scheduler reads after the kernel returns.
 *
 * All overloads follow the (ENGINE, QUEUE, data...) parameter convention,
 * symmetric with pto2_send_request_entry(ENGINE, SQ_ID, desc) in the SQ API.
 *
 * Usage in kernel code:
 *
 *   auto* cq = pto2_cq_get(args[CQ_ARG_IDX]);
 *   pto2_save_expected_completion(PTO2_ENGINE_SDMA, cq, tag);  // flag: expected=1
 *   pto2_cq_flush();
 */

#ifndef PTO_CQ_KERNEL_API_H
#define PTO_CQ_KERNEL_API_H

#include "pto_cq_types.h"

// Requires __gm__ and __aicore__ to be defined before including this header.
// Kernel sources should define them (or include PTO-ISA headers) first.

// Unified engine constants — shared by SQ and CQ APIs.
// Must match PTO2AsyncEngine in pto_types.h.
#define PTO2_ENGINE_SDMA  0
#define PTO2_ENGINE_ROCE  1
#define PTO2_ENGINE_URMA  2
#define PTO2_ENGINE_CCU   3

// Completion type constants (must match PTO2CompletionType in pto_types.h)
#define PTO2_CQ_COMPLETION_COUNTER           0

inline __aicore__ void pto2_cq_writeback_gm_line(volatile __gm__ void* addr) {
    __gm__ int32_t* gm_addr = (__gm__ int32_t*)addr;
#if defined(SINGLE_CACHE_LINE) && defined(CACHELINE_OUT)
    dcci(gm_addr, SINGLE_CACHE_LINE, CACHELINE_OUT);
#elif defined(SINGLE_CACHE_LINE)
    dcci(gm_addr, SINGLE_CACHE_LINE);
#endif
#if defined(DSB_DDR)
    dsb(DSB_DDR);
#endif
}

/**
 * Obtain the completion queue pointer from a kernel scalar arg.
 */
inline __aicore__ volatile __gm__ PTO2CompletionQueue* pto2_cq_get(uint64_t addr) {
    return reinterpret_cast<volatile __gm__ PTO2CompletionQueue*>(
        static_cast<uintptr_t>(addr));
}

/**
 * Reset the CQ header before the kernel appends completion entries.
 *
 * Runtime-owned CQ buffers may be reused across tasks, so kernels should
 * explicitly republish an empty header before the first append.
 */
inline __aicore__ void pto2_cq_reset(volatile __gm__ PTO2CompletionQueue* cq) {
    // Republish the header line even when the queue was already zeroed in a
    // reused runtime buffer. Some hardware paths were observed to require an
    // explicit header-state transition before the subsequent count increment
    // became visible to the AICPU scheduler.
    cq->count = -1;
    pto2_cq_writeback_gm_line(&cq->count);
    cq->count = 0;
    pto2_cq_writeback_gm_line(&cq->count);
}

/**
 * Register one expected completion condition in the CQ.
 *
 * All completion conditions are COUNTER type: the scheduler polls
 * *addr >= expected_value.  Hardware flags (SDMA event flags) are
 * the special case where expected_value = 1 (flag goes 0 → non-zero).
 *
 * Parameter order: (ENGINE, QUEUE, addr, expected) — symmetric with SQ API.
 * Each call appends an entry and increments cq->count.
 * The caller must ensure total calls per task <= PTO2_CQ_MAX_ENTRIES.
 */
inline __aicore__ void pto2_save_expected_completion(
    uint32_t engine,
    volatile __gm__ PTO2CompletionQueue* cq,
    uint64_t addr,
    uint32_t expected_value)
{
    int32_t idx = cq->count;
    volatile __gm__ PTO2CQEntry* entry =
        const_cast<volatile __gm__ PTO2CQEntry*>(&cq->entries[idx]);
    entry->engine = engine;
    entry->completion_type = PTO2_CQ_COMPLETION_COUNTER;
    entry->addr = addr;
    entry->expected_value = expected_value;
    pto2_cq_writeback_gm_line(entry);

    cq->count = idx + 1;
    pto2_cq_writeback_gm_line(&cq->count);
}

/**
 * Simplified overload for hardware flags: (ENGINE, CQ, tag).
 *
 * Registers a COUNTER condition with expected_value=1.
 * Equivalent to polling *tag_addr >= 1 (i.e. flag != 0).
 * Symmetric with pto2_send_request_entry(ENGINE, SQ_ID, desc).
 */
inline __aicore__ void pto2_save_expected_completion(
    uint32_t engine,
    volatile __gm__ PTO2CompletionQueue* cq,
    uint64_t tag)
{
    pto2_save_expected_completion(engine, cq, tag, 1);
}

/**
 * Final flush before kernel returns. Ensures all CQ writes
 * are visible to the AICPU scheduler.
 *
 * Uses CCE compiler built-in enum constants (cache_line_t, dcci_dst_t,
 * dsb_mode_t, pipe_t) which are available when compiling for AICore
 * via the bisheng/CCE toolchain.  Previous #if-defined guards broke
 * because these are C++ enums, not preprocessor macros.
 */
inline __aicore__ void pto2_cq_flush() {
    pipe_barrier(PIPE_ALL);
}

inline __aicore__ void pto2_cq_flush(volatile __gm__ PTO2CompletionQueue* cq) {
    dcci((__gm__ int32_t*)cq, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
}

#endif  // PTO_CQ_KERNEL_API_H
