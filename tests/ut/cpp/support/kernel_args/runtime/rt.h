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
// Minimal RTS declarations for the CPU-only KernelArgsHelper tests.
#pragma once

#include <cstdint>

using rtStream_t = void *;
using rtError_t = int32_t;
constexpr rtError_t RT_ERROR_NONE = 0;
constexpr uint32_t RT_MEMORY_HBM = 0;
enum rtMemcpyKind_t { RT_MEMCPY_HOST_TO_DEVICE = 1 };

extern "C" {
rtError_t rtMalloc(void **ptr, uint64_t bytes, uint32_t type, uint16_t module_id);
rtError_t rtFree(void *ptr);
rtError_t rtMemcpy(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind);
rtError_t rtStreamQuery(rtStream_t stream);
}
