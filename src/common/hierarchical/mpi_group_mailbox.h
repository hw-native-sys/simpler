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

namespace mpi_group_mailbox {

inline constexpr uint8_t MAGIC[8] = {'S', 'M', 'P', 'I', 'B', 'O', 'X', '\0'};
inline constexpr uint32_t PROTOCOL_VERSION = 1;
inline constexpr size_t HEADER_BYTES = 256;
inline constexpr size_t PAYLOAD_BYTES = 16U * 1024U * 1024U;
inline constexpr size_t ERROR_BYTES = 64U * 1024U;
inline constexpr size_t REQUEST_OFFSET = HEADER_BYTES;
inline constexpr size_t RESPONSE_OFFSET = REQUEST_OFFSET + PAYLOAD_BYTES;
inline constexpr size_t ERROR_OFFSET = RESPONSE_OFFSET + PAYLOAD_BYTES;
inline constexpr size_t MAILBOX_BYTES = ERROR_OFFSET + ERROR_BYTES;

inline constexpr size_t OFF_MAGIC = 0;
inline constexpr size_t OFF_PROTOCOL_VERSION = 8;
inline constexpr size_t OFF_HEADER_BYTES = 12;
inline constexpr size_t OFF_MAILBOX_BYTES = 16;
inline constexpr size_t OFF_WORLD_SIZE = 24;
inline constexpr size_t OFF_GROUP_STATE = 28;
inline constexpr size_t OFF_REQUEST_STATE = 32;
inline constexpr size_t OFF_SEQUENCE_ID = 40;
inline constexpr size_t OFF_OPCODE = 48;
inline constexpr size_t OFF_TARGET = 52;
inline constexpr size_t OFF_TARGET_RANK = 56;
inline constexpr size_t OFF_REQUEST_COUNT = 60;
inline constexpr size_t OFF_REQUEST_BYTES = 64;
inline constexpr size_t OFF_RESPONSE_COUNT = 68;
inline constexpr size_t OFF_RESPONSE_BYTES = 72;
inline constexpr size_t OFF_ERROR_BYTES = 76;

enum class GroupState : int32_t {
    INITIALIZING = 0,
    READY = 1,
    TERMINAL = 2,
    CLOSED = 3,
};

enum class RequestState : int32_t {
    IDLE = 0,
    REQUEST_READY = 1,
    TASK_ACCEPTED = 2,
    TASK_DONE = 3,
    TASK_FAILED = 4,
    SHUTDOWN_READY = 5,
    SHUTDOWN_DONE = 6,
};

enum class Opcode : uint32_t {
    TASK = 1,
    CONTROL = 2,
    PING = 3,
    SHUTDOWN = 4,
};

enum class Target : uint32_t {
    GROUP = 1,
    RANK = 2,
    PER_RANK = 3,
};

}  // namespace mpi_group_mailbox
