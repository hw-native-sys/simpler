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
 * @file pto2_dispatch_payload.h
 * @brief Minimal dispatch payload for AICore kernel execution
 *
 * Shared between AICPU (builds in-place) and AICore (reads to run kernel).
 * Handshake.task points to PTO2DispatchPayload embedded in PTO2TaskPayload.
 *
 * Only contains fields AICore needs to execute: function address + arguments.
 * Metadata (task_id, kernel_id, core_type) lives in PTO2TaskDescriptor and
 * is accessed by AICPU when needed (profiling, diagnostics).
 */

#ifndef RT2_PTO2_DISPATCH_PAYLOAD_H_
#define RT2_PTO2_DISPATCH_PAYLOAD_H_

#include <stdint.h>

/** Max arguments per task; must match RUNTIME_MAX_ARGS and PTO2_MAX_OUTPUTS */
#ifndef PTO2_DISPATCH_MAX_ARGS
#define PTO2_DISPATCH_MAX_ARGS 128
#endif

/**
 * Dispatch payload: minimal execution interface for AICore.
 * Layout: function_bin_addr followed by args[].
 * AICore reads function_bin_addr, casts to UnifiedKernelFunc, calls with args.
 */
struct PTO2DispatchPayload {
    uint64_t function_bin_addr;            /**< Kernel entry in GM: (UnifiedKernelFunc)function_bin_addr */
    uint64_t args[PTO2_DISPATCH_MAX_ARGS]; /**< Kernel arguments (GM pointers + scalars) */
};

#endif  // RT2_PTO2_DISPATCH_PAYLOAD_H_
