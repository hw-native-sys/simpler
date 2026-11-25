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
#include "aicore/aicore.h"
#include "scheduler/profiling_record.h"
namespace simpler::percounter {
namespace {
template <>
struct ProfileCompilation<> {
    static constexpr bool enabled = true;
};
}  // namespace
}  // namespace simpler::percounter
#include "lane_loop.h"

template <CoreType Architecture>
__aicore__ void percounter_execute_profiled(__gm__ Runtime *runtime, int block_idx) {
    simpler::percounter::percounter_scalar_bind(nullptr);
    simpler::percounter::percounter_aicore_execute(runtime, block_idx, Architecture);
}

#if defined(__DAV_CUBE__)
template __aicore__ void percounter_execute_profiled<CoreType::AIC>(__gm__ Runtime *, int);
#elif defined(__DAV_VEC__)
template __aicore__ void percounter_execute_profiled<CoreType::AIV>(__gm__ Runtime *, int);
#else
template __aicore__ void percounter_execute_profiled<CoreType::AIC>(__gm__ Runtime *, int);
template __aicore__ void percounter_execute_profiled<CoreType::AIV>(__gm__ Runtime *, int);
#endif
