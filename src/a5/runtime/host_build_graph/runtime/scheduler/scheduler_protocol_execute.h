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

#pragma once

#include "inner_kernel.h"
#include "scheduler_protocol_execution.h"

inline __aicore__ void scheduler_execute_protocol_task(
    __gm__ void *base, SchedulerLocalState *local, uint32_t pending_slot, __gm__ DispatchPayload *payload
) {
    scheduler_protocol_execution_entry(base, local, pending_slot, payload);
    if (payload == nullptr || payload->function_bin_addr == 0) return;
    using Kernel = void (*)(__gm__ int64_t *);
    const auto kernel = reinterpret_cast<Kernel>(payload->function_bin_addr);
    kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
    const uint64_t returned = scheduler_cycles();
    OUT_OF_ORDER_STORE_BARRIER();
    scheduler_protocol_execution_returned(base, local, pending_slot, returned, scheduler_cycles());
}
