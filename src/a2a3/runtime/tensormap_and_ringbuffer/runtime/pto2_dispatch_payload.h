/**
 * @file pto2_dispatch_payload.h
 * @brief Per-core dispatch payload for AICore kernel execution
 *
 * PTO2DispatchPayload holds the kernel function address and a pointer to the
 * pre-built args[] array in the task payload. AICPU maintains a static array
 * of these (one per core) and writes both fields before each dispatch. AICore
 * caches a pointer to its per-core slot at startup and reads from it on each
 * dispatch. The struct is cache-line aligned to avoid false sharing across
 * concurrently dispatched cores.
 *
 * The args[] array (tensor GM pointers followed by scalar values) is built once
 * by the Orchestrator at submit time in PTO2TaskPayload::init().
 *
 * The DATA_MAIN_BASE register protocol is unchanged from the base runtime:
 * a monotonically increasing reg_task_id signals new work to AICore.
 */

#ifndef RT2_PTO2_DISPATCH_PAYLOAD_H_
#define RT2_PTO2_DISPATCH_PAYLOAD_H_

#include <stdint.h>
#include "pto_types.h"
#include "common/qualifier.h"

/** Max dispatch arguments: 128 scalars + up to 16 tensor pointers */
#ifndef PTO2_DISPATCH_MAX_ARGS
#define PTO2_DISPATCH_MAX_ARGS (PTO2_MAX_SCALAR_PARAMS + PTO2_MAX_TENSOR_PARAMS)
#endif

/**
 * Per-core dispatch payload: function address + args pointer.
 *
 * AICPU maintains a static array s_pto2_payload_per_core[RUNTIME_MAX_WORKER].
 * Before each dispatch, AICPU writes function_bin_addr (looked up from
 * func_id_to_addr_[kernel_id]) and args (pointer to pre-built dispatch_args[]
 * in PTO2TaskPayload). AICore caches a pointer to its slot at startup (via
 * Handshake.task) and reads both fields after each DATA_MAIN_BASE change.
 */
struct alignas(64) PTO2DispatchPayload {
    uint64_t function_bin_addr;    /**< Kernel entry address in GM (set by Scheduler) */
    __gm__ uint64_t* args;         /**< Pre-built args in task payload GM (set by Scheduler) */
};

#endif  // RT2_PTO2_DISPATCH_PAYLOAD_H_
