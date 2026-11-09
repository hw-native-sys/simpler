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
#include <type_traits>

namespace simpler::ccu {

inline constexpr uint32_t ABI_VERSION = 1;

struct DomainIdentity {
    uint64_t owner;
    uint64_t id;
    uint64_t generation;

    bool operator==(const DomainIdentity &other) const {
        return owner == other.owner && id == other.id && generation == other.generation;
    }
};

struct Window {
    uint64_t base;
    uint64_t size;
    uint64_t credential;
};

enum class StopResult : int32_t { CONFIRMED = 0, UNCONFIRMED = 1 };
enum class PollResult : int32_t { PENDING = 0, COMPLETE = 1, FAILED = 2 };

// This is a Host-only C calling convention. No pointer in these structs is uploaded.
struct BuildContext {
    uint32_t abi_version;
    uint32_t struct_size;
    DomainIdentity domain;
    uint32_t rank;
    uint32_t window_count;
    const Window *windows;
    void *platform_context;
};

struct Program {
    uint32_t abi_version;
    uint32_t struct_size;
    void *handle;
    uint64_t device_program;
    // After producers stop, drain quiesces the current run's accepted requests.
    // CONFIRMED means no request can still access the domain and the program supports a subsequent run.
    // Drain neither consumes handle nor permanently closes the program; permanent shutdown uses close.
    StopResult (*drain)(void *handle);
    // CONFIRMED close consumes handle. On every other result ownership remains with the caller.
    StopResult (*close)(void *handle);
    // Nonblocking. COMPLETE proves no request is still using this program or its domain.
    // A partial failed builder may omit poll; an executable program must provide it.
    PollResult (*poll)(void *handle);
};

// Exported as extern "C" uint32_t simpler_ccu_module_abi_version().
using ModuleVersion = uint32_t (*)();
// A failed builder must return any partially created handle and its cleanup callbacks.
// Builders and callbacks must never unwind an exception across the C boundary.
using Builder = int32_t (*)(const BuildContext *, Program *);

static_assert(std::is_trivially_copyable_v<DomainIdentity> && std::is_standard_layout_v<DomainIdentity>);
static_assert(std::is_trivially_copyable_v<Window> && std::is_standard_layout_v<Window>);
static_assert(std::is_trivially_copyable_v<BuildContext> && std::is_standard_layout_v<BuildContext>);
static_assert(std::is_trivially_copyable_v<Program> && std::is_standard_layout_v<Program>);
static_assert(sizeof(DomainIdentity) == 24);
static_assert(sizeof(Window) == 24);

}  // namespace simpler::ccu
