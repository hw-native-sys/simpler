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

#ifndef SRC_COMMON_PLATFORM_INCLUDE_AICPU_CACHE_MAINTENANCE_H_
#define SRC_COMMON_PLATFORM_INCLUDE_AICPU_CACHE_MAINTENANCE_H_

#include <stddef.h>

namespace aicpu_cache_maintenance {

void invalidate_range_impl(const void *addr, size_t size);
void flush_range_impl(const void *addr, size_t size);

}  // namespace aicpu_cache_maintenance

inline void cache_invalidate_range(const void *addr, size_t size) {
    aicpu_cache_maintenance::invalidate_range_impl(addr, size);
}

inline void cache_flush_range(const void *addr, size_t size) { aicpu_cache_maintenance::flush_range_impl(addr, size); }

#endif  // SRC_COMMON_PLATFORM_INCLUDE_AICPU_CACHE_MAINTENANCE_H_
