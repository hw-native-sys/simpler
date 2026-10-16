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

// The AICPU owns the first cache line: it resets both words before any EXIT,
// publishes a nonzero request, and releases only after ACK and window-close.
// AICore bypass-loads that line; its cached stores and dcci never touch it.
// AICore alone owns the second line. GM-ACK modes clear it before reporting
// and echo requests after quiescence; native COND-nonce modes leave it unused.
// The two lines must not share writers.
struct alignas(64) AicoreTeardownControl {
    uint32_t post_close_release;
    uint32_t exit_request;
    alignas(64) uint32_t exit_ack;
};

static_assert(sizeof(AicoreTeardownControl) == 128);
static_assert(offsetof(AicoreTeardownControl, post_close_release) == 0);
static_assert(offsetof(AicoreTeardownControl, exit_request) == 4);
static_assert(offsetof(AicoreTeardownControl, exit_ack) == 64);
static_assert(alignof(AicoreTeardownControl) == 64);
static_assert(std::is_standard_layout_v<AicoreTeardownControl>);
static_assert(std::is_trivially_copyable_v<AicoreTeardownControl>);
