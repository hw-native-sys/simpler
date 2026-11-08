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
#include <cstring>
#include <type_traits>

inline constexpr uint64_t HBG_ASYNC_RETIREMENT_MAGIC = UINT64_C(0x4842475245544952);
inline constexpr uint32_t HBG_ASYNC_RETIREMENT_VERSION = 1;
inline constexpr uint32_t HBG_ASYNC_CORES_RETIRED = 1;

// HBG-private bytes in the existing DeviceRunResultRegion failure payload.
// The producer records its real retirement acknowledgements independently of
// task success. This says nothing about independent DMA engines or other peers.
struct HbgAsyncRetirementPayload {
    uint64_t magic{HBG_ASYNC_RETIREMENT_MAGIC};
    uint64_t run_epoch{0};
    uint32_t version{HBG_ASYNC_RETIREMENT_VERSION};
    uint32_t flags{0};
};
static_assert(sizeof(HbgAsyncRetirementPayload) == 24);
static_assert(
    std::is_standard_layout_v<HbgAsyncRetirementPayload> && std::is_trivially_copyable_v<HbgAsyncRetirementPayload>
);

inline bool hbg_async_cores_retired(const void *payload, size_t bytes, uint64_t epoch) {
    if (payload == nullptr || bytes != sizeof(HbgAsyncRetirementPayload) || epoch == 0) return false;
    HbgAsyncRetirementPayload evidence{};
    std::memcpy(&evidence, payload, sizeof(evidence));
    return evidence.magic == HBG_ASYNC_RETIREMENT_MAGIC && evidence.version == HBG_ASYNC_RETIREMENT_VERSION &&
           evidence.run_epoch == epoch && evidence.flags == HBG_ASYNC_CORES_RETIRED;
}
