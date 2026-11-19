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

namespace peer_counter {
constexpr uint32_t kMaxRows = 256;
constexpr uint32_t kDataWords = 32;
constexpr uint32_t kRowWords = 64;
constexpr uint32_t kSignalWords = 32;
constexpr uint32_t kReadyWords = 32;
constexpr uint32_t kDiagnosticWords = 16;
constexpr uint32_t kRegistrationWords = 16;
constexpr int32_t kGuard = 0x13579BDF;
constexpr int32_t kPoison = -1000000007;
constexpr int32_t kReadyVersion = 1;
constexpr int64_t kMaxTag = 20000;
constexpr uint64_t kGateTimeoutMs = 2000;

enum class Field : uint32_t {
    VISITS,
    BODY_DONE,
    ERROR,
    EARLY,
    SEEN_SIGNAL,
    SEEN_TAG,
    ROWS,
    CAPACITY,
    TIMEOUT,
    REGISTERED,
    CHECKSUM,
    PUBLISHED,
    ROLE,
    EPOCH,
    TAG,
    GUARD
};
enum class Role : uint32_t { WAITER, GATE, READY_SEND, PUBLISH, CONSUME };
enum class Error : int64_t {
    BAD_ARGUMENT = 1,
    MISSING_CONTEXT,
    REGISTRATION_FAILED,
    GATE_TIMEOUT,
    BAD_COMM_CONTEXT,
    BAD_READY_PACKET,
    REGISTRATION_MISMATCH,
    EARLY_SIGNAL,
    BAD_PAYLOAD,
    BAD_SOURCE,
    BAD_GUARD
};
enum class ReadyField : uint32_t { VERSION, EPOCH, TAG, ROWS, REGISTERED, RECEIVER_RANK, STATUS, CAPACITY };
}  // namespace peer_counter
