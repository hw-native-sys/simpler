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

#include "completion_context.h"
#include "common/platform_config.h"

#if defined(__CPU_SIM)
extern "C" bool pto_cpu_sim_cancel_requested(uint32_t exit_signal);
#endif

// Call from bounded work chunks or a polling wait. On true, return from the
// kernel immediately. This publishes registered requests for scheduler drain;
// it neither cancels DMA nor makes a hardware blocking-wait interruptible.
inline __aicore__ bool kernel_cancel_requested(AsyncCtx &context) {
#if defined(__CPU_SIM)
    if (!pto_cpu_sim_cancel_requested(AICORE_EXIT_SIGNAL)) return false;
#else
    uint32_t dispatch;
    __asm__ volatile("MOV %0, DATA_MAIN_BASE\n" : "=l"(dispatch));
    if (dispatch != AICORE_EXIT_SIGNAL) return false;
#endif
    pto2::detail::defer_flush(context);
    return true;
}
