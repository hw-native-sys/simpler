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

#include "resident_scheduler_impl.h"

namespace simpler::hbg::resident {

// This TU keeps the default Scalar policy disabled. Do not call the public
// run<true,...> symbol: the full profiling TU owns that separate implementation.
template <bool WithNativeGraph, CoreType Architecture>
__aicore__ void run_timing(
    __gm__ SchedulerWorkerContext *context, __gm__ SchedulerRunControl *run_control, __gm__ void *scheduler_state_base,
    uint64_t profiling_level, uint64_t aicore_entry_cycles
) {
    static_assert(!kSchedulerScalarInstrumentation);
    run_resident_scheduler<true, WithNativeGraph>(
        context, run_control, scheduler_state_base, profiling_level, aicore_entry_cycles
    );
}

#if defined(__DAV_CUBE__)
template __aicore__ void run_timing<false, CoreType::AIC>(
    __gm__ SchedulerWorkerContext *, __gm__ SchedulerRunControl *, __gm__ void *, uint64_t, uint64_t
);
#elif defined(__DAV_VEC__)
template __aicore__ void run_timing<false, CoreType::AIV>(
    __gm__ SchedulerWorkerContext *, __gm__ SchedulerRunControl *, __gm__ void *, uint64_t, uint64_t
);
#else
template __aicore__ void run_timing<false, CoreType::AIC>(
    __gm__ SchedulerWorkerContext *, __gm__ SchedulerRunControl *, __gm__ void *, uint64_t, uint64_t
);
template __aicore__ void run_timing<false, CoreType::AIV>(
    __gm__ SchedulerWorkerContext *, __gm__ SchedulerRunControl *, __gm__ void *, uint64_t, uint64_t
);
#endif

}  // namespace simpler::hbg::resident
