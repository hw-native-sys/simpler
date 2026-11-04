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
 * This runtime's AICore entry.
 *
 * host_build_graph's aicore_executor.cpp dispatches on the mode the host
 * published. Here the mode is settled before the binary exists, so this takes
 * the lane straight to the percounter executor and that file is not compiled
 * into this runtime at all — neither it nor the legacy executor beside it has
 * a path this runtime can reach.
 *
 * Weak like the one it stands in for: every AICore source is compiled once for
 * AIC and once for AIV, and the two objects are combined, so a strong
 * definition here would collide with its own other half.
 */

#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "percounter/percounter_executor.h"
#include "runtime.h"
#include "scheduler/scheduler_memory.h"

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    // The host's bootstrap and this worker's handshake are separate 64-byte
    // lines, so each needs its own observe; one observe covers a whole line.
    // The bootstrap carries the PercounterControl address every lane reads.
    __gm__ SchedulerBootstrapInputs *bootstrap = &runtime->dev.scheduler_bootstrap;
    scheduler_observe_cache_line(bootstrap);
    __gm__ Handshake *handshake = (__gm__ Handshake *)(&runtime->dev.workers[block_idx]);
    scheduler_observe_cache_line(handshake);
    simpler::hbg::percounter::percounter_aicore_execute(runtime, block_idx, core_type);
}
