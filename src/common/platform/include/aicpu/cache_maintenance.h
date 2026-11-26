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

#include <stddef.h>

namespace aicpu_cache_maintenance {

void invalidate_range_impl(const void *addr, size_t size);
void flush_range_impl(const void *addr, size_t size);
void clean_line_impl(const void *addr);
void complete_clean_impl();

}  // namespace aicpu_cache_maintenance

// A2/A3 onboard maintains cache lines and completes the maintenance.
// Coherent A5 and sim are no-ops; callers provide their protocol's barriers.
inline void cache_invalidate_range(const void *addr, size_t size) {
    aicpu_cache_maintenance::invalidate_range_impl(addr, size);
}

inline void cache_flush_range(const void *addr, size_t size) { aicpu_cache_maintenance::flush_range_impl(addr, size); }

// The calling thread must own each line. Leave the scope before publishing
// that the cleaned data is visible to other agents.
class CacheLineCleanBatch {
public:
    static constexpr size_t kLineSize = 64;

    CacheLineCleanBatch() = default;
    CacheLineCleanBatch(const CacheLineCleanBatch &) = delete;
    CacheLineCleanBatch &operator=(const CacheLineCleanBatch &) = delete;
    ~CacheLineCleanBatch() {
        if (pending_) aicpu_cache_maintenance::complete_clean_impl();
    }

    template <typename Line>
    void clean(const Line &line) {
        static_assert(sizeof(Line) == kLineSize && alignof(Line) == kLineSize);
        aicpu_cache_maintenance::clean_line_impl(&line);
        pending_ = true;
    }

private:
    bool pending_{false};
};
