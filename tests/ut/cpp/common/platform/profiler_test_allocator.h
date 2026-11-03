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

#include <cstdlib>
#include <cstring>

inline void *profiler_test_malloc(size_t size) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 64, size == 0 ? 1 : size) != 0) return nullptr;
    return ptr;
}

inline void *profiler_test_calloc(size_t size) {
    void *ptr = profiler_test_malloc(size);
    if (ptr == nullptr) return nullptr;
    std::memset(ptr, 0, size);
    return ptr;
}
