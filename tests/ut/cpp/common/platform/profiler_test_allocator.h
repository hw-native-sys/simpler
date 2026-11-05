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

#include <cstring>
#include "host/aligned_allocation.h"

inline void *profiler_test_malloc(size_t size) { return simpler::platform::aligned_host_alloc(size); }

inline void *profiler_test_calloc(size_t size) {
    void *ptr = profiler_test_malloc(size);
    if (ptr == nullptr) return nullptr;
    std::memset(ptr, 0, size);
    return ptr;
}
