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

#if defined(__DAV_CUBE__)
template __aicore__ void run<false, true, CoreType::AIC>(
    __gm__ SchedulerWorkerContext *, __gm__ SchedulerRunControl *, __gm__ void *, uint64_t, uint64_t, uint64_t
);
#elif defined(__DAV_VEC__)
template __aicore__ void run<false, true, CoreType::AIV>(
    __gm__ SchedulerWorkerContext *, __gm__ SchedulerRunControl *, __gm__ void *, uint64_t, uint64_t, uint64_t
);
#else
template __aicore__ void run<false, true, CoreType::AIC>(
    __gm__ SchedulerWorkerContext *, __gm__ SchedulerRunControl *, __gm__ void *, uint64_t, uint64_t, uint64_t
);
template __aicore__ void run<false, true, CoreType::AIV>(
    __gm__ SchedulerWorkerContext *, __gm__ SchedulerRunControl *, __gm__ void *, uint64_t, uint64_t, uint64_t
);
#endif

}  // namespace simpler::hbg::resident
