/**
 * PTO Completion Queue Types — shared between AICore kernels and AICPU runtime.
 *
 * This header must remain simple and C-compatible. AICore compilation
 * environments have restricted standard library access.
 */

#ifndef PTO_CQ_TYPES_H
#define PTO_CQ_TYPES_H

#include <stdint.h>

#define PTO2_CQ_MAX_ENTRIES 16

/**
 * Single CQ entry written by a kernel via pto2_save_expected_completion().
 * The scheduler reads these after the worker returns.
 */
struct PTO2CQEntry {
    uint32_t engine;            // PTO2AsyncEngine value
    int32_t  completion_type;   // PTO2CompletionType value
    uint64_t addr;              // completion token (flag/handle/counter GM address)
    uint32_t expected_value;    // for COUNTER completions
    uint32_t _pad;
};

/**
 * Per-task completion queue.
 *
 * Allocated by the runtime and passed to the kernel as a scalar arg.
 * The kernel calls pto2_save_expected_completion() to append entries
 * and increment `count`. The scheduler reads the CQ after all
 * subtasks have returned and creates completion conditions accordingly.
 *
 * Memory ordering contract:
 *   - Kernel writes entries[i] fields BEFORE incrementing count.
 *   - Kernel flushes caches (dcci+dsb on HW) before returning.
 *   - Scheduler reads only after detecting task_status==0,
 *     which implies all kernel writes are visible.
 */
struct PTO2CompletionQueue {
    volatile int32_t count;     // entries written so far (kernel increments)
    int32_t _pad;
    PTO2CQEntry entries[PTO2_CQ_MAX_ENTRIES];
};

#endif  // PTO_CQ_TYPES_H
