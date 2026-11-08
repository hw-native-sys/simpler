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

namespace simpler::ccu::device {

// V7 positional transport. This is independent of the Host module ABI.
inline constexpr uint32_t PROTOCOL_VERSION = 2;
inline constexpr uint32_t MAX_TENSORS = 16;
inline constexpr uint32_t MAX_SCALARS = 16;

enum class Admission : uint32_t { OPEN = 0, CLOSED = 1, UNCONFIRMED = 2 };
enum class Slot : uint32_t { FREE = 0, RESERVED = 1, SUBMITTED = 2 };
enum class Outcome : uint32_t { PENDING = 0, SUCCEEDED = 1, FAILED = 2, UNCONFIRMED = 3 };
enum class ScalarKind : uint32_t { UNSIGNED = 0, SIGNED = 1, FLOAT = 2 };

struct TensorArgument {
    uint64_t addr;
    uint64_t bytes;
    uint64_t credential;
};
struct ScalarArgument {
    uint64_t bits;
    ScalarKind kind;
    uint32_t width;
};
struct Arguments {
    uint32_t tensor_count;
    uint32_t scalar_count;
    TensorArgument tensors[MAX_TENSORS];
    ScalarArgument scalars[MAX_SCALARS];
};
struct MemoryRange {
    uint64_t addr;
    uint64_t bytes;
    uint64_t credential;
};
struct ProgramBinding {
    int32_t callable_id;
    uint32_t ccu_id;
    uint64_t request_offset;
    uint64_t doorbell_addr;
    uint64_t doorbell_value;
    uint64_t platform_program;
};

struct InstanceEntry {
    int32_t callable_id;
    uint32_t reserved;
    uint64_t image_addr;  // Device address of a separately owned instance image.
    uint64_t image_bytes;
};

// CommContext's CCU workspace names this directory. Host may replace the table
// only after closing admission and draining all domain runs. Instance images
// never move when another callable is registered; old directories stay leased
// until all readers stop. Domain generation and context address cannot be reused.
struct alignas(128) DomainDirectory {
    uint32_t version;
    uint32_t header_bytes;
    uint64_t image_bytes;
    uint64_t domain_owner;
    uint64_t domain_id;
    uint64_t generation;
    uint64_t entries_offset;
    uint32_t entry_count;
    uint32_t rank;
    uint32_t rank_count;
    volatile Admission admission;
};

// Host publishes an immutable table before admission; offsets are relative to
// this image, never Host pointers. All functions of one prepared instance share
// one Request. The caller serializes that instance using an ordering Tensor.
// Domain context addresses must not be recycled while stale device handles can
// survive. A new generation gets a new image and a new CommContext address.
struct alignas(128) Workspace {
    uint32_t version;
    uint32_t header_bytes;
    uint64_t image_bytes;
    uint64_t domain_owner;
    uint64_t domain_id;
    uint64_t generation;
    uint64_t ranges_offset;
    uint64_t bindings_offset;
    uint64_t timeout_ns;
    uint32_t range_count;
    uint32_t binding_count;
    uint32_t rank;
    uint32_t rank_count;
    volatile Admission admission;
    // The platform owns the request snapshot and completion record in this image.
    // Zero selects the in-image transport used by controllable test backends.
    uint64_t platform_context;
    uint64_t platform_context_bytes;
};

struct alignas(128) Completion {
    // Backend publishes all metadata and data visibility before sequence LAST.
    volatile uint64_t sequence;
    volatile uint64_t generation;
    volatile uint32_t binding_index;
    volatile Outcome outcome;
    volatile int32_t native_status;
};

struct alignas(128) Request {
    volatile uint64_t sequence;
    uint64_t generation;
    uint64_t submitted_ns;
    uint32_t binding_index;
    volatile Slot slot;
    // Completion has its own cache lines: producer cache maintenance must not
    // write an old completion back over the platform's publication.
    alignas(128) Arguments arguments;
    Completion completion;
};

static_assert(sizeof(Workspace) == 128);
static_assert(sizeof(DomainDirectory) == 128 && sizeof(InstanceEntry) == 24);
static_assert(sizeof(Completion) == 128);
static_assert(offsetof(Request, arguments) == 128);
static_assert(offsetof(Request, completion) % 128 == 0);
static_assert(std::is_trivially_copyable_v<Workspace> && std::is_standard_layout_v<Workspace>);
static_assert(std::is_trivially_copyable_v<Request> && std::is_standard_layout_v<Request>);
static_assert(sizeof(ProgramBinding) == 40 && sizeof(TensorArgument) == 24 && sizeof(ScalarArgument) == 16);

}  // namespace simpler::ccu::device
