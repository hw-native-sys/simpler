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
#include <cstdlib>

namespace simpler::platform {

// Profiling buffers include 64-byte-aligned device structures. The returned
// pointer remains compatible with std::free, including on allocation failure.
inline void *aligned_host_alloc(size_t size) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 64, size == 0 ? 1 : size) != 0) return nullptr;
    return ptr;
}

}  // namespace simpler::platform
