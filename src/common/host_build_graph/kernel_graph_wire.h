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

#include "task_interface/kernel_invocation_header.h"
#include "task_interface/kernel_invocation_validation.h"
#include "utils/fnv1a_64.h"

namespace hbg {

inline constexpr uint32_t GRAPH_PACKET_MAGIC = 0x50474248;  // HBGP
inline constexpr uint16_t GRAPH_PACKET_VERSION = 5;

enum class GraphImageKind : uint32_t { Runtime = 1, Definitions = 2, Scheduler = 3 };
enum class GraphPacketAddress : uint32_t { HostTemplate, DeviceCopy };
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
    uint64_t slot_generation;      // Independent of callable residency generation in K9.
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

inline bool graph_span_fits(uint64_t offset, uint64_t size, uint64_t capacity) noexcept {
    return offset <= capacity && size <= capacity - offset;
}

// The only excluded bytes are the runtime-patched address and this checksum.
// The envelope, binding, identity, region descriptors and padding are covered.
inline uint64_t graph_packet_checksum(const void *packet, size_t size) noexcept {
    constexpr size_t skip = sizeof(SimplerKernelInvocationHeader) + offsetof(GraphPacketHeader, inline_payload_addr);
    if (packet == nullptr || size < skip + 16) return 0;
    // XXH64, seed zero, over the concatenation of the two covered spans.
    // https://github.com/Cyan4973/xxHash/blob/v0.8.3/doc/xxhash_spec.md
    static_assert(skip == 64);
    constexpr uint64_t p1 = 0x9e3779b185ebca87ULL;
    constexpr uint64_t p2 = 0xc2b2ae3d27d4eb4fULL;
    constexpr uint64_t p3 = 0x165667b19e3779f9ULL;
    constexpr uint64_t p4 = 0x85ebca77c2b2ae63ULL;
    constexpr uint64_t p5 = 0x27d4eb2f165667c5ULL;
    const auto rotate = [](uint64_t value, unsigned bits) {
        return (value << bits) | (value >> (64 - bits));
    };
    const auto read64 = [](const uint8_t *ptr) {
        uint64_t value;
        std::memcpy(&value, ptr, sizeof(value));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        value = __builtin_bswap64(value);
#endif
        return value;
    };
    const auto read32 = [](const uint8_t *ptr) {
        uint32_t value;
        std::memcpy(&value, ptr, sizeof(value));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        value = __builtin_bswap32(value);
#endif
        return value;
    };
    const auto round = [&](uint64_t acc, uint64_t lane) {
        return rotate(acc + lane * p2, 31) * p1;
    };
    uint64_t a = p1 + p2, b = p2, c = 0, d = uint64_t{0} - p1;
    const auto stripes = [&](const uint8_t *&ptr, size_t &remaining) {
        while (remaining >= 32) {
            a = round(a, read64(ptr));
            b = round(b, read64(ptr + 8));
            c = round(c, read64(ptr + 16));
            d = round(d, read64(ptr + 24));
            ptr += 32;
            remaining -= 32;
        }
    };
    const auto *bytes = static_cast<const uint8_t *>(packet);
    size_t head_bytes = skip;
    stripes(bytes, head_bytes);
    const auto *tail = static_cast<const uint8_t *>(packet) + skip + 16;
    size_t remaining = size - skip - 16;
    stripes(tail, remaining);
    uint64_t hash = rotate(a, 1) + rotate(b, 7) + rotate(c, 12) + rotate(d, 18);
    hash = (hash ^ round(0, a)) * p1 + p4;
    hash = (hash ^ round(0, b)) * p1 + p4;
    hash = (hash ^ round(0, c)) * p1 + p4;
    hash = (hash ^ round(0, d)) * p1 + p4;
    hash += size - 16;
    while (remaining >= 8) {
        hash = rotate(hash ^ round(0, read64(tail)), 27) * p1 + p4;
        tail += 8;
        remaining -= 8;
    }
    if (remaining >= 4) {
        hash = rotate(hash ^ (uint64_t{read32(tail)} * p1), 23) * p2 + p3;
        tail += 4;
        remaining -= 4;
    }
    while (remaining != 0) {
        hash = rotate(hash ^ (uint64_t{*tail++} * p5), 11) * p1;
        --remaining;
    }
    hash = (hash ^ (hash >> 33)) * p2;
    hash = (hash ^ (hash >> 29)) * p3;
    return hash ^ (hash >> 32);
}

// Allocation-free framing validation. No device destination is read or written.
// Registry identity checks and semantic validation of restored images precede
// execution separately; a checksum-valid packet alone never authorizes restore.
inline GraphPacketStatus
validate_graph_packet(const void *packet, size_t size, GraphPacketAddress address_mode) noexcept {
    constexpr size_t prefix = sizeof(SimplerKernelInvocationHeader);
    if (packet == nullptr || size < prefix + sizeof(GraphPacketHeader) || size > UINT32_MAX)
        return GraphPacketStatus::InvalidEnvelope;
    const auto *bytes = static_cast<const uint8_t *>(packet);
    SimplerKernelInvocationHeader invocation{};
    std::memcpy(&invocation, bytes, sizeof(invocation));
    if (invocation.mode != SIMPLER_MODE_KERNEL || invocation.callable_id < 0 ||
        invocation.callable_id >= MAX_REGISTERED_CALLABLE_IDS ||
        !simpler::kernel::valid_invocation_counts(invocation.tensor_count, invocation.scalar_count) ||
        invocation.host_copy_tensor_count != 0 || invocation.reserved_ != 0 ||
        invocation.payload_bytes != size - prefix)
        return GraphPacketStatus::InvalidEnvelope;
    GraphPacketHeader header{};
    std::memcpy(&header, bytes + prefix, sizeof(header));
    if (header.magic != GRAPH_PACKET_MAGIC || header.version != GRAPH_PACKET_VERSION ||
        header.header_bytes != sizeof(header) || header.total_bytes != size - prefix || header.region_count < 1 ||
        header.region_count > 3 || header.slot_generation == 0 || header.callable_hash == 0 ||
        header.argument_hash == 0 || header.function_hash == 0 || header.task_window == 0 ||
        header.task_window > 32768 || header.total_tasks >= header.task_window || header.reserved != 0 ||
        header.device_id < 0 || header.runtime_binary_id == 0 || header.heap_bytes > header.destinations[0].capacity)
        return GraphPacketStatus::InvalidHeader;
    const uint64_t table_end = sizeof(header) + header.region_count * sizeof(GraphImageRegion);
    const uint64_t payload_offset = (table_end + 63) & ~uint64_t{63};
    if (header.payload_offset != payload_offset || payload_offset > header.total_bytes ||
        header.payload_bytes != header.total_bytes - payload_offset)
        return GraphPacketStatus::InvalidHeader;
    const uint64_t base = reinterpret_cast<uintptr_t>(packet);
    if (!graph_span_fits(base, size, UINT64_MAX)) return GraphPacketStatus::InvalidHeader;
    const uint64_t expected_address = base + prefix + payload_offset;
    if ((address_mode == GraphPacketAddress::HostTemplate && header.inline_payload_addr != 0) ||
        (address_mode == GraphPacketAddress::DeviceCopy && header.inline_payload_addr != expected_address) ||
        (address_mode != GraphPacketAddress::HostTemplate && address_mode != GraphPacketAddress::DeviceCopy))
        return GraphPacketStatus::InvalidHeader;
    for (uint64_t i = table_end; i < payload_offset; ++i)
        if (bytes[prefix + i] != 0) return GraphPacketStatus::InvalidHeader;
    for (size_t i = 0; i < 4; ++i) {
        const auto &dst = header.destinations[i];
        if ((dst.capacity == 0) != (dst.address == 0) || (i < 2 && dst.capacity == 0) || dst.address % 1024 != 0 ||
            !graph_span_fits(dst.address, dst.capacity, UINT64_MAX))
            return GraphPacketStatus::InvalidBinding;
        for (size_t j = 0; j < i; ++j) {
            const auto &other = header.destinations[j];
            if (dst.capacity && other.capacity && dst.address < other.address + other.capacity &&
                other.address < dst.address + dst.capacity)
                return GraphPacketStatus::InvalidBinding;
        }
    }
    if (header.runtime_offset % 64 != 0 || header.sm_offset % 64 != 0 || header.runtime_offset >= header.sm_offset ||
        header.sm_offset >= header.destinations[1].capacity)
        return GraphPacketStatus::InvalidBinding;
    uint64_t cursor = 0;
    uint32_t previous_kind = 0;
    for (uint32_t index = 0; index < header.region_count; ++index) {
        GraphImageRegion region{};
        std::memcpy(&region, bytes + prefix + sizeof(header) + index * sizeof(region), sizeof(region));
        const uint32_t kind = static_cast<uint32_t>(region.kind);
        if (kind <= previous_kind || kind > 3 || (index == 0 && kind != 1)) return GraphPacketStatus::InvalidRegion;
        const uint64_t capacity = header.destinations[kind].capacity;
        const uint64_t aligned = (cursor + 63) & ~uint64_t{63};
        const uint64_t destination_offset = kind == 1 ? header.runtime_offset : 0;
        if (region.reserved != 0 || region.source_offset != aligned ||
            region.destination_offset != destination_offset || region.bytes == 0 ||
            !graph_span_fits(region.destination_offset, region.bytes, capacity) ||
            (kind == 1 && (region.bytes <= header.sm_offset - header.runtime_offset || region.bytes % 64 != 0)) ||
            !graph_span_fits(region.source_offset, region.bytes, header.payload_bytes))
            return GraphPacketStatus::InvalidRegion;
        for (; cursor < aligned; ++cursor)
            if (bytes[prefix + payload_offset + cursor] != 0) return GraphPacketStatus::InvalidRegion;
        cursor = region.source_offset + region.bytes;
        previous_kind = kind;
    }
    if (cursor != header.payload_bytes) return GraphPacketStatus::InvalidRegion;
    if (header.checksum != graph_packet_checksum(packet, size)) return GraphPacketStatus::InvalidChecksum;
    return GraphPacketStatus::Ok;
}

}  // namespace hbg
