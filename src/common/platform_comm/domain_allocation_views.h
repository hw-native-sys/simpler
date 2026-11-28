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

#include <algorithm>
#include <limits>
#include <vector>
#include "platform_comm/comm.h"

namespace simpler::comm {
inline int publish_domain_allocation_views(
    const std::vector<CommAllocationSpan> &views, CommAllocationSpan *spans, size_t capacity, size_t *count_out
) {
    if (count_out == nullptr) return -1;
    *count_out = 0;
    if (views.empty() || (spans == nullptr && capacity != 0)) return -1;
    for (const auto &view : views) {
        if (view.addr == 0 || view.bytes == 0 || view.bytes > std::numeric_limits<uint64_t>::max() - view.addr) {
            return -1;
        }
    }
    *count_out = views.size();
    if (spans == nullptr && capacity == 0) return 0;
    if (capacity < views.size()) return -1;
    std::copy(views.begin(), views.end(), spans);
    return 0;
}
}  // namespace simpler::comm
