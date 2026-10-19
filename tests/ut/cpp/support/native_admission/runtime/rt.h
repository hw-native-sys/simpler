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
constexpr int RT_MEMORY_HBM = 0;
constexpr int RT_DEV_BINARY_MAGIC_ELF = 0;
constexpr int RT_DEV_BINARY_MAGIC_ELF_AIVEC = 1;
constexpr int RT_SCHEM_MODE_BATCH = 0;
enum rtMemcpyKind_t { RT_MEMCPY_HOST_TO_DEVICE = 1, RT_MEMCPY_DEVICE_TO_HOST = 2, RT_MEMCPY_DEVICE_TO_DEVICE = 3 };
struct rtDevBinary_t {
    uint32_t magic;
    uint32_t version;
    const void *data;
    uint64_t length;
};
struct rtArgsEx_t {
    void *args;
    uint32_t argsSize;
};
struct rtTaskCfgInfo_t {
    uint32_t schemMode;
};
extern "C" {
int rtMalloc(void **ptr, uint64_t bytes, uint32_t type, uint16_t module_id);
int rtFree(void *ptr);
int rtMallocHost(void **ptr, uint64_t bytes, uint16_t module_id);
int rtFreeHost(void *ptr);
int rtMemcpyAsync(
    void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind, rtStream_t stream
);
int rtMemcpy(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind);
int rtSetDevice(int32_t device);
int rtDeviceReset(int32_t device);
int rtStreamCreate(rtStream_t *stream, int32_t priority);
int rtStreamDestroy(rtStream_t stream);
int rtStreamQuery(rtStream_t stream);
int rtStreamSynchronize(rtStream_t stream);
int rtRegisterAllKernel(const rtDevBinary_t *binary, void **handle);
int rtUnregisterAllKernel(void *handle);
int rtKernelLaunchWithHandleV2(void *, uint64_t, uint32_t, rtArgsEx_t *, void *, rtStream_t, rtTaskCfgInfo_t *);
}
