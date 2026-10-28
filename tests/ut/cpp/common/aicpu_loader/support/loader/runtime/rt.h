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
#include <cstddef>
#include <cstdint>
using rtStream_t = void *;
using rtFuncHandle = void *;
using rtError_t = int32_t;
constexpr int RT_ERROR_NONE = 0;
enum class rtKernelType_t { KERNEL_TYPE_AICPU_KFC };
struct rtAicpuArgsEx_t {
    void *args{};
    uint32_t argsSize{};
    uint32_t kernelNameAddrOffset{};
    uint32_t soNameAddrOffset{};
};
struct rtLaunchKernelAttr_t {};
struct rtKernelLaunchCfg_t {
    rtLaunchKernelAttr_t *attrs;
    uint32_t numAttrs;
};
struct rtCpuKernelArgs_t {
    struct {
        void *args{};
        uint32_t argsSize{};
    } baseArgs;
};
enum rtLoadBinaryOption { RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE };
struct rtLoadBinaryOption_t {
    rtLoadBinaryOption optionId{};
    union {
        int32_t cpuKernelMode;
    } value{};
};
struct rtLoadBinaryConfig_t {
    rtLoadBinaryOption_t *options{};
    size_t numOpt{};
};
extern "C" {
rtError_t
rtAicpuKernelLaunchExWithArgs(rtKernelType_t, const char *, uint32_t, rtAicpuArgsEx_t *, void *, rtStream_t, uint32_t);
rtError_t rtsBinaryLoadFromFile(const char *, const rtLoadBinaryConfig_t *, void **);
rtError_t rtsBinaryUnload(void *);
rtError_t rtsFuncGetByName(void *, const char *, rtFuncHandle *);
rtError_t rtsLaunchCpuKernel(rtFuncHandle, uint32_t, rtStream_t, const rtKernelLaunchCfg_t *, rtCpuKernelArgs_t *);
}
