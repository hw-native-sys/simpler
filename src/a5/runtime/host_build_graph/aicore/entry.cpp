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
// The linked image shares this weak entry across AIC and AIV.
// Dispatch to a separate executor compiled for each core type.
#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "runtime.h"
#include "scheduler/gm.h"

template <CoreType Architecture>
__aicore__ void percounter_execute_unprofiled(__gm__ Runtime *runtime, int block_idx);

template <CoreType Architecture>
__aicore__ void percounter_execute_profiled(__gm__ Runtime *runtime, int block_idx);

template <CoreType Architecture>
__aicore__ void percounter_execute_task_profiled(__gm__ Runtime *runtime, int block_idx);

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    // The host's bootstrap and this worker's handshake are separate 64-byte
    // lines, so each needs its own observe; one observe covers a whole line.
    // The bootstrap carries the PercounterControl address every lane reads.
    __gm__ SchedulerBootstrapInputs *bootstrap = &runtime->dev.scheduler_bootstrap;
    percounter_observe_cache_line(bootstrap);
    __gm__ Handshake *handshake =
        (__gm__ Handshake *)(&reinterpret_cast<__gm__ RuntimeSharedState *>(runtime->dev.shared_state_addr)
                                  ->workers[block_idx]);
    percounter_observe_cache_line(handshake);
    if (runtime->dev.chip_swimlane_level >= 3) {
        if (core_type == CoreType::AIC) percounter_execute_profiled<CoreType::AIC>(runtime, block_idx);
        else percounter_execute_profiled<CoreType::AIV>(runtime, block_idx);
    } else if (runtime->dev.chip_swimlane_level == 2) {
        if (core_type == CoreType::AIC) percounter_execute_task_profiled<CoreType::AIC>(runtime, block_idx);
        else percounter_execute_task_profiled<CoreType::AIV>(runtime, block_idx);
    } else {
        if (core_type == CoreType::AIC) percounter_execute_unprofiled<CoreType::AIC>(runtime, block_idx);
        else percounter_execute_unprofiled<CoreType::AIV>(runtime, block_idx);
    }
}
