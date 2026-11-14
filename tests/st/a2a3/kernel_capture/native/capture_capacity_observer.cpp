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

#include "capture_capacity_observer.h"
#include <dlfcn.h>
#include <cstdlib>
#include "runtime/kernel.h"
#include "runtime/rts/rts_kernel.h"
namespace {
thread_local simpler::capture_test::LaunchCounts counts;
template <typename F>
F next(const char *name) {
    const auto symbol = reinterpret_cast<F>(dlsym(RTLD_NEXT, name));
    if (symbol == nullptr) std::abort();
    return symbol;
}
}  // namespace
namespace simpler::capture_test {
LaunchCounts launch_counts() { return counts; }
}  // namespace simpler::capture_test
extern "C" rtError_t rtKernelLaunchWithHandleV2(
    void *handle, uint64_t key, uint32_t blocks, rtArgsEx_t *args, rtSmDesc_t *sm, rtStream_t stream,
    const rtTaskCfgInfo_t *config
) {
    static const auto real = next<decltype(&rtKernelLaunchWithHandleV2)>("rtKernelLaunchWithHandleV2");
    ++counts.core;
    return real(handle, key, blocks, args, sm, stream, config);
}
extern "C" rtError_t rtsLaunchCpuKernel(
    const rtFuncHandle function, uint32_t blocks, rtStream_t stream, const rtKernelLaunchCfg_t *config,
    rtCpuKernelArgs_t *args
) {
    static const auto real = next<decltype(&rtsLaunchCpuKernel)>("rtsLaunchCpuKernel");
    ++counts.cpu;
    return real(function, blocks, stream, config, args);
}
