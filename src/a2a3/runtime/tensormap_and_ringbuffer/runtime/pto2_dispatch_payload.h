/**
 * @file pto2_dispatch_payload.h
 * @brief Dispatch descriptor and per-core payload for AICore kernel execution
 *
 * PTO2DispatchDesc is embedded in PTO2TaskPayload and built by the Orchestrator
 * at submit time. It contains per-slot function addresses and a unified args[]
 * array (tensor pointers + scalar values).
 *
 * PTO2DispatchPerCore wraps a PTO2DispatchDesc pointer with the active slot_idx.
 * AICPU maintains a static array of these (one per core) and writes the pointer
 * + slot_idx before each dispatch. AICore caches a pointer to its per-core slot
 * at startup and reads from it on each dispatch.
 *
 * The DATA_MAIN_BASE register protocol is unchanged from the base runtime:
 * a monotonically increasing reg_task_id signals new work to AICore.
 */

#ifndef RT2_PTO2_DISPATCH_PAYLOAD_H_
#define RT2_PTO2_DISPATCH_PAYLOAD_H_

#include <stdint.h>

#ifndef __gm__
#define __gm__
#endif

#include "pto_submit_types.h"

/** Max arguments per task; must match RUNTIME_MAX_ARGS and PTO2_MAX_OUTPUTS */
#ifndef PTO2_DISPATCH_MAX_ARGS
#define PTO2_DISPATCH_MAX_ARGS 128
#endif

// =============================================================================
// Dispatch Descriptor
// =============================================================================

/**
 * Dispatch descriptor: execution interface for AICore.
 *
 * Layout: per-slot function_bin_addrs[] followed by unified args[].
 * AICore reads function_bin_addrs[slot_idx], casts to UnifiedKernelFunc,
 * and calls with args (tensor GM pointers followed by scalar values).
 *
 * Built once by the Orchestrator during submit_mixed_task(); AICore reads
 * it in-place from the task payload via a pointer in PTO2DispatchPerCore.
 */
struct PTO2DispatchDesc {
    /** Per-slot kernel entry addresses in GM (AIC, AIV0, AIV1); 0 = inactive */
    uint64_t function_bin_addrs[PTO2_SUBTASK_SLOT_COUNT];
    /** Kernel arguments: tensor GM pointers first, then scalar values */
    uint64_t args[PTO2_DISPATCH_MAX_ARGS];
};

// =============================================================================
// Per-Core Dispatch Payload
// =============================================================================

/**
 * Per-core dispatch payload: descriptor pointer + active slot index.
 *
 * AICPU maintains a static array s_pto2_payload_per_core[RUNTIME_MAX_WORKER].
 * Before each dispatch, AICPU writes the PTO2DispatchDesc pointer and slot_idx.
 * AICore caches a pointer to its slot at startup (via Handshake.task) and
 * reads from it after each DATA_MAIN_BASE register change.
 */
struct PTO2DispatchPerCore {
    __gm__ PTO2DispatchDesc* dispatch;  /**< Pointer to dispatch desc in task payload (GM) */
    /** Active subtask slot (0=AIC, 1=AIV0, 2=AIV1) */
    uint32_t slot_idx;
};

#endif  // RT2_PTO2_DISPATCH_PAYLOAD_H_
