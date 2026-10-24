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

#include "task_interface/kernel_invocation_header.h"

namespace hbg {

inline constexpr uint32_t GRAPH_PACKET_MAGIC = 0x50474248;  // HBGP
inline constexpr uint16_t GRAPH_PACKET_VERSION = 5;

enum class GraphImageKind : uint32_t { Runtime = 1, Definitions = 2, Scheduler = 3 };
enum class GraphPacketAddress : uint32_t { HostTemplate, DeviceCopy, RelativeLaunch };
enum class GraphPacketStatus : uint32_t {
    Ok,
    InvalidEnvelope,
    InvalidHeader,
    InvalidBinding,
    InvalidRegion,
    InvalidChecksum
};

struct GraphDestination {
    uint64_t address;
    uint64_t capacity;
};

// Offsets are relative to inline_payload_addr and the selected destination.
struct GraphImageRegion {
    GraphImageKind kind;
    uint32_t reserved;
    uint64_t source_offset;
    uint64_t bytes;
    uint64_t destination_offset;
};

// Follows the shared common invocation header. Device addresses name
// context-owned storage; they are claims to compare with the registry, not authority.
struct GraphPacketHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint64_t total_bytes;
    uint64_t payload_offset;  // From this header, past the region table and padding.
    uint64_t payload_bytes;
    uint64_t inline_payload_addr;  // Zero in the template; one CANN placeholder.
    uint64_t checksum;             // Accidental-corruption check, not authentication.
    uint64_t slot_generation;      // Independent of callable residency generation.
    uint64_t callable_hash;
    uint64_t argument_hash;
    uint64_t function_hash;
    uint64_t runtime_offset;
    uint64_t sm_offset;
    uint64_t task_window;
    uint32_t total_tasks;
    uint32_t region_count;
    GraphDestination destinations[4];  // heap, runtime/SM, Definitions, A5 scheduler
    int32_t device_id;
    uint32_t reserved;
    uint64_t runtime_binary_id;  // Runtime ABI/code identity, independent of callable identity.
    uint64_t heap_bytes;         // Live heap prefix for this invocation, within the frozen destination.
};

static_assert(std::is_trivially_copyable_v<GraphDestination> && std::is_standard_layout_v<GraphDestination>);
static_assert(std::is_trivially_copyable_v<GraphImageRegion> && std::is_standard_layout_v<GraphImageRegion>);
static_assert(std::is_trivially_copyable_v<GraphPacketHeader> && std::is_standard_layout_v<GraphPacketHeader>);
static_assert(sizeof(GraphDestination) == 16);
static_assert(sizeof(GraphImageRegion) == 32);
static_assert(sizeof(GraphPacketHeader) == 200);
static_assert(alignof(GraphPacketHeader) == 8);
static_assert(offsetof(GraphPacketHeader, inline_payload_addr) == 32);
static_assert(offsetof(GraphPacketHeader, checksum) == 40);
static_assert(offsetof(GraphPacketHeader, slot_generation) == 48);
static_assert(offsetof(GraphPacketHeader, destinations) == 112);
static_assert(offsetof(GraphPacketHeader, device_id) == 176);
static_assert(offsetof(GraphPacketHeader, runtime_binary_id) == 184);
static_assert(offsetof(GraphImageRegion, source_offset) == 8);

// Parsed headers are copied once; payload borrows immutable packet storage.
// This proves packet structure only, not current callable/resource authority or
// graph image semantics. Keep the source alive and unchanged through consumption.
struct ParsedGraphPacket {
    SimplerKernelInvocationHeader invocation{};
    GraphPacketHeader graph{};
    const std::byte *payload{nullptr};
    GraphImageRegion regions[4]{};
};

}  // namespace hbg
