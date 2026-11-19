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
#include <cstring>
#include "common/hbg_execution_image.h"
#include "common/kernel_args.h"
#include "runtime.h"
#include "runtime/kernel.h"
#include "runtime/rts/rts_kernel.h"
namespace {
thread_local simpler::capture_test::LaunchCounts counts;
thread_local uint64_t runtime_image_address;
thread_local uint64_t rts_execution_image_bytes;
template <typename F>
F next(const char *name) {
    const auto symbol = reinterpret_cast<F>(dlsym(RTLD_NEXT, name));
    if (symbol == nullptr) std::abort();
    return symbol;
}
}  // namespace
namespace simpler::capture_test {
LaunchCounts launch_counts() { return counts; }
uint64_t last_runtime_image_address() { return runtime_image_address; }
uint64_t last_rts_execution_image_bytes() { return rts_execution_image_bytes; }
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
    runtime_image_address = 0;
    rts_execution_image_bytes = 0;
    if (args != nullptr && args->baseArgs.args != nullptr && args->baseArgs.hostInputInfoPtr != nullptr) {
        for (uint16_t i = 0; i < args->baseArgs.hostInputInfoNum; ++i) {
            const auto &input = args->baseArgs.hostInputInfoPtr[i];
            if (input.addrOffset == offsetof(KernelArgs, execution_image) &&
                input.dataOffset <= args->baseArgs.argsSize &&
                sizeof(HbgExecutionImage) <= args->baseArgs.argsSize - input.dataOffset) {
                HbgExecutionImage image{};
                std::memcpy(&image, static_cast<const char *>(args->baseArgs.args) + input.dataOffset, sizeof(image));
                if (image.working_address != 0 &&
                    image.bytes <= args->baseArgs.argsSize - input.dataOffset - sizeof(image))
                    rts_execution_image_bytes = image.bytes;
            }
            const size_t field = static_cast<size_t>(input.dataOffset) + offsetof(DeviceRuntimeLaunchDesc, gm_sm_ptr_);
            if (input.addrOffset == offsetof(KernelArgs, runtime_args) && field <= args->baseArgs.argsSize &&
                sizeof(void *) <= args->baseArgs.argsSize - field) {
                void *image = nullptr;
                std::memcpy(&image, static_cast<const char *>(args->baseArgs.args) + field, sizeof(image));
                runtime_image_address = reinterpret_cast<uint64_t>(image);
            }
        }
    }
    return real(function, blocks, stream, config, args);
}
