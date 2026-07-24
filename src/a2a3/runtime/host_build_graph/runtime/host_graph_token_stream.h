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

#ifndef SRC_A2A3_RUNTIME_HOST_BUILD_GRAPH_RUNTIME_HOST_GRAPH_TOKEN_STREAM_H_
#define SRC_A2A3_RUNTIME_HOST_BUILD_GRAPH_RUNTIME_HOST_GRAPH_TOKEN_STREAM_H_

#include <stdint.h>

constexpr uint64_t PTO2_HOST_GRAPH_TOKEN_STREAM_MAGIC_VERSION = 0x4847535400010000ULL;
constexpr int32_t PTO2_HOST_GRAPH_TOKEN_STREAM_TRAILER_SCALARS = 14;

enum PTO2HostGraphTokenFlags : uint32_t {
    PTO2_HOST_GRAPH_TOKEN_FINAL = 1U << 0,
    PTO2_HOST_GRAPH_TOKEN_SYNTHETIC = 1U << 1,
};

struct PTO2HostGraphTokenPacket {
    uint64_t magic_version;
    uint64_t request_id;
    uint64_t token_seq;
    int64_t token_id;
    uint32_t flags;
    int32_t status;
};

static_assert(sizeof(PTO2HostGraphTokenPacket) == 40, "HostGraph token packet ABI drift");

#endif  // SRC_A2A3_RUNTIME_HOST_BUILD_GRAPH_RUNTIME_HOST_GRAPH_TOKEN_STREAM_H_
