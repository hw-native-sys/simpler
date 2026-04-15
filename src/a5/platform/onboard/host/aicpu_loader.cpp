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
 * AICPU Loader Implementation (Legacy Interface)
 *
 * Provides AICPU kernel launching via the legacy rtAicpuKernelLaunchExWithArgs API.
 * Used when BUILD_WITH_NEW_CANN is OFF. When BUILD_WITH_NEW_CANN is ON,
 * device_runner uses LoadAicpuOp (src/common/host/load_aicpu_op.h) instead.
 */

#include "aicpu_loader.h"

#include <cstring>

#include "common/unified_log.h"
#include "common/kernel_args.h"

int AicpuLoader::init_with_binary(
    const std::vector<uint8_t> &aicpu_binary, const std::vector<std::string> &kernel_names
) {
    // Legacy interface: No pre-loading needed
    (void)aicpu_binary;
    (void)kernel_names;
    LOG_INFO("AicpuLoader: Using legacy rtAicpuKernelLaunchExWithArgs interface");
    return 0;
}

int AicpuLoader::init(const std::string &so_path, const std::vector<std::string> &kernel_names) {
    // Legacy interface: No pre-loading needed
    (void)so_path;
    (void)kernel_names;
    LOG_INFO("AicpuLoader: Using legacy rtAicpuKernelLaunchExWithArgs interface");
    return 0;
}

int AicpuLoader::launch(rtStream_t stream, KernelArgs *k_args, const char *kernel_name, int aicpu_num) {
    // Legacy interface: rtAicpuKernelLaunchExWithArgs
    struct Args {
        KernelArgs k_args;
        char kernel_name[32];
        const char so_name[32] = {"libaicpu_extend_kernels.so"};
        const char op_name[32] = {""};
    } args;

    args.k_args = *k_args;
    std::strncpy(args.kernel_name, kernel_name, sizeof(args.kernel_name) - 1);
    args.kernel_name[sizeof(args.kernel_name) - 1] = '\0';

    rtAicpuArgsEx_t rt_args;
    std::memset(&rt_args, 0, sizeof(rt_args));
    rt_args.args = &args;
    rt_args.argsSize = sizeof(args);
    rt_args.kernelNameAddrOffset = offsetof(struct Args, kernel_name);
    rt_args.soNameAddrOffset = offsetof(struct Args, so_name);

    return rtAicpuKernelLaunchExWithArgs(
        rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", aicpu_num, &rt_args, nullptr, stream, 0
    );
}

void AicpuLoader::finalize() {
    // Legacy interface: No-op
}
