/**
 * @file pto2_dispatch_payload.h
 * @brief Handshake dispatch payload aligned with runtime2 PTO2TaskDescriptor
 *
 * Shared between AICPU (pack from PTO2TaskDescriptor) and AICore (unpack to run kernel).
 * When merging runtime2 into rt2, Handshake.task points to PTO2DispatchPayload.
 */

#ifndef RT2_PTO2_DISPATCH_PAYLOAD_H_
#define RT2_PTO2_DISPATCH_PAYLOAD_H_

#include <stdint.h>

#include "common/core_type.h"

/** Max arguments per task; must match RUNTIME_MAX_ARGS and PTO2_MAX_OUTPUTS */
#ifndef PTO2_DISPATCH_MAX_ARGS
#define PTO2_DISPATCH_MAX_ARGS 16
#endif

/**
 * Dispatch payload: execution-relevant fields from PTO2TaskDescriptor.
 * AICPU packs this from PTO2TaskDescriptor; AICore unpacks to run kernel.
 */
struct PTO2DispatchPayload {
    int32_t task_id;           /**< Task ID (for completion_queue) */
    int32_t kernel_id;         /**< InCore function id (debug/trace) */
    CoreType core_type;        /**< AIC or AIV */
    uint64_t function_bin_addr; /**< Kernel entry in GM: (UnifiedKernelFunc)function_bin_addr */
    int32_t num_args;          /**< Number of valid args[] */
    uint64_t args[PTO2_DISPATCH_MAX_ARGS]; /**< Kernel arguments (GM pointers) */
};

#endif  // RT2_PTO2_DISPATCH_PAYLOAD_H_
