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

#include <stdint.h>

#include "common/core_type.h"

struct SchedulerWorkerContext;
struct SchedulerRunControl;

namespace simpler::hbg::resident {

// The architecture parameter separates the AIC and AIV symbols in the combined image.
template <bool WithProfiling, bool WithNativeGraph, CoreType Architecture>
__aicore__ void
run(__gm__ SchedulerWorkerContext *context, __gm__ SchedulerRunControl *run_control, __gm__ void *scheduler_state_base,
    uint64_t profiling_level, uint64_t aicore_entry_cycles, uint64_t handshake_publish_cycles);

// Task/schedule timing retains trace storage without compiling Scalar observers.
// The dispatcher selects this entry only for L1/L2 or sampled timing at L0.
template <bool WithNativeGraph, CoreType Architecture>
__aicore__ void run_timing(
    __gm__ SchedulerWorkerContext *context, __gm__ SchedulerRunControl *run_control, __gm__ void *scheduler_state_base,
    uint64_t profiling_level, uint64_t aicore_entry_cycles, uint64_t handshake_publish_cycles
);

}  // namespace simpler::hbg::resident
