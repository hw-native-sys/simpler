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

namespace ccu_fault {
constexpr uint32_t kLineWords = 16;
constexpr uint32_t kMaxAiv = 160;
constexpr uint32_t kSubmitted = 0;
constexpr uint32_t kPeerReadySeen = 1;
constexpr uint32_t kPeerWidthSeen = 2;
constexpr uint32_t kConsumer = kLineWords;
constexpr uint32_t kSuccessorPublished = 2 * kLineWords;
constexpr uint32_t kWitnessWidth = 3 * kLineWords;
constexpr uint32_t kWitnessBase = 4 * kLineWords;
constexpr uint32_t kControlWords = kWitnessBase + kMaxAiv * kLineWords;
constexpr uint32_t kReady = 1;
constexpr uint32_t kRejected = 2;
// The deadline bounds a missing handshake; readiness depends only on published flags.
constexpr uint64_t kHandshakeTimeoutNs = 4000000000ULL;
}  // namespace ccu_fault
