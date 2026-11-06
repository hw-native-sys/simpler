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
namespace sdma_series {
constexpr int64_t kMaxPosts = 66;
constexpr int64_t kQueueNum = 48;
constexpr int64_t kMetaWords = 256;
constexpr int64_t kReceiptBase = 64;
constexpr int64_t kReceiptEnd = kReceiptBase + 2 * kQueueNum;
constexpr int64_t kLargeElements = 1024 * 1024;
constexpr int64_t kGuardElements = 16;
constexpr int64_t kSmallStride = 48;
constexpr int64_t kAllocatedElements = kLargeElements + 2 * kGuardElements + (kMaxPosts - 1) * kSmallStride;
enum Field : int64_t {
    SUBMIT_VISITS = 0,
    RECEIPT_COUNT = 8,
    ERROR = 16,
    CONSUME_VISITS = 24,
    EARLY = 32,
    TASK_TOKEN = 40,
    PRIOR_INDEX = 48,
    PRIOR_RECORD = 49,
    PRIOR_VALUE = 50,
    PRIOR_EXPECTED = 51,
    PENDING_SAMPLE = 56
};
inline int64_t elements(int64_t index) { return index == 0 ? kLargeElements : 8; }
inline int64_t offset(int64_t index) {
    return index == 0 ? kGuardElements :
                        kLargeElements + 2 * kGuardElements + (index - 1) * kSmallStride + kGuardElements;
}
}  // namespace sdma_series
