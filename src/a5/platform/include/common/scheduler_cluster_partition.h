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

#include <cstdint>

struct SchedulerClusterRange {
    int32_t begin{0};
    int32_t end{0};
};

// Balanced contiguous partition: scheduler t owns [t*N/A, (t+1)*N/A).
constexpr SchedulerClusterRange scheduler_cluster_range(int32_t total_clusters, int32_t active, int32_t thread_idx) {
    if (total_clusters < 0 || active <= 0 || thread_idx < 0 || thread_idx >= active) return {};
    return {
        static_cast<int32_t>((static_cast<int64_t>(thread_idx) * total_clusters) / active),
        static_cast<int32_t>((static_cast<int64_t>(thread_idx + 1) * total_clusters) / active),
    };
}
