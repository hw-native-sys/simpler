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

namespace peer_dmb {
enum Mode : uint32_t { GM_EARLY, GM_ORDERED, PEER_DMB };
enum Fence : uint32_t { NONE, DDR, ALL };
constexpr unsigned kSlots = 108;  // DAV_3510, 2 dies with 18 AIC and 36 AIV IDs each.
constexpr uint32_t kUnset = UINT32_MAX;
struct Config {
    uint64_t set, inc, dmb[kSlots];
    uint32_t mode, fence;
};
struct alignas(128) Sender {
    uint32_t core, stage, doorbell;
};
struct alignas(128) Receiver {
    uint32_t core, initial, token, armed, first, sampled, eventual, observed, status;
};
struct State {
    Sender sender;
    Receiver receiver;
};
}  // namespace peer_dmb
