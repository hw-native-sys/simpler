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
#include <type_traits>

constexpr uint32_t AICORE_POST_CLOSE_RELEASE = 1;

// Only AICPU writes this line. It clears both words before READY or EXIT,
// publishes one reserved request per exclusively claimed core, and releases
// only after the matching COND nonce and window-close/readback/drain.
// AICore bypass-loads both words and never caches or writes this line.
struct alignas(64) AicoreTeardownControl {
    uint32_t post_close_release;
    uint32_t exit_request;
};

static_assert(sizeof(AicoreTeardownControl) == 64);
static_assert(offsetof(AicoreTeardownControl, post_close_release) == 0);
static_assert(offsetof(AicoreTeardownControl, exit_request) == 4);
static_assert(alignof(AicoreTeardownControl) == 64);
static_assert(std::is_standard_layout_v<AicoreTeardownControl>);
static_assert(std::is_trivially_copyable_v<AicoreTeardownControl>);
