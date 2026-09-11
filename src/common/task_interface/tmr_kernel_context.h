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

namespace simpler::tmr {

inline constexpr uint32_t kTmrKernelContextVersion = 1;

// Published once by prepare. The owner retains every referenced allocation
// until all executions complete and all referring graphs are destroyed.
struct alignas(64) TmrKernelContextDescriptor {
    uint32_t version;
    uint32_t bytes;
    uint64_t context_generation;
    uint64_t self_address;
    uint64_t resident_runtime;
    uint64_t resident_kernel_args;
    uint64_t heap_base;
    uint64_t heap_capacity;
    uint64_t heap_required;
    uint64_t sm_base;
    uint64_t sm_capacity;
    uint64_t sm_required;
    uint64_t arena_base;
    uint64_t arena_capacity;
    uint64_t arena_required;
    uint64_t runtime_offset;
    uint64_t control_address;
    uint64_t control_bytes;
    uint64_t reports_address;
    uint64_t reports_bytes;
    int32_t launch_threads;
    int32_t execution_threads;
    int32_t worker_count;
    uint32_t flags;
    uint64_t reserved[3];
};

struct TmrContextRegistrationArgs {
    uint64_t descriptor_address;
    uint64_t context_generation;
    uint64_t callable_descriptor_base;
    uint32_t callable_descriptor_count;
    uint32_t callable_descriptor_stride;
};

struct TmrCallableRegistrationArgs {
    uint64_t context_generation;
    uint64_t residency_address;
    int32_t callable_id;
    uint32_t reserved;
    uint64_t slot_generation;
};

static_assert(
    std::is_trivially_copyable_v<TmrKernelContextDescriptor> && std::is_standard_layout_v<TmrKernelContextDescriptor>
);
static_assert(sizeof(TmrKernelContextDescriptor) == 192 && alignof(TmrKernelContextDescriptor) == 64);
static_assert(offsetof(TmrKernelContextDescriptor, version) == 0);
static_assert(offsetof(TmrKernelContextDescriptor, bytes) == 4);
static_assert(offsetof(TmrKernelContextDescriptor, context_generation) == 8);
static_assert(offsetof(TmrKernelContextDescriptor, self_address) == 16);
static_assert(offsetof(TmrKernelContextDescriptor, resident_runtime) == 24);
static_assert(offsetof(TmrKernelContextDescriptor, resident_kernel_args) == 32);
static_assert(offsetof(TmrKernelContextDescriptor, heap_base) == 40);
static_assert(offsetof(TmrKernelContextDescriptor, heap_capacity) == 48);
static_assert(offsetof(TmrKernelContextDescriptor, heap_required) == 56);
static_assert(offsetof(TmrKernelContextDescriptor, sm_base) == 64);
static_assert(offsetof(TmrKernelContextDescriptor, sm_capacity) == 72);
static_assert(offsetof(TmrKernelContextDescriptor, sm_required) == 80);
static_assert(offsetof(TmrKernelContextDescriptor, arena_base) == 88);
static_assert(offsetof(TmrKernelContextDescriptor, arena_capacity) == 96);
static_assert(offsetof(TmrKernelContextDescriptor, arena_required) == 104);
static_assert(offsetof(TmrKernelContextDescriptor, runtime_offset) == 112);
static_assert(offsetof(TmrKernelContextDescriptor, control_address) == 120);
static_assert(offsetof(TmrKernelContextDescriptor, control_bytes) == 128);
static_assert(offsetof(TmrKernelContextDescriptor, reports_address) == 136);
static_assert(offsetof(TmrKernelContextDescriptor, reports_bytes) == 144);
static_assert(offsetof(TmrKernelContextDescriptor, launch_threads) == 152);
static_assert(offsetof(TmrKernelContextDescriptor, execution_threads) == 156);
static_assert(offsetof(TmrKernelContextDescriptor, worker_count) == 160);
static_assert(offsetof(TmrKernelContextDescriptor, flags) == 164);
static_assert(offsetof(TmrKernelContextDescriptor, reserved) == 168);
static_assert(
    std::is_trivially_copyable_v<TmrContextRegistrationArgs> && std::is_standard_layout_v<TmrContextRegistrationArgs>
);
static_assert(sizeof(TmrContextRegistrationArgs) == 32);
static_assert(offsetof(TmrContextRegistrationArgs, descriptor_address) == 0);
static_assert(offsetof(TmrContextRegistrationArgs, context_generation) == 8);
static_assert(offsetof(TmrContextRegistrationArgs, callable_descriptor_base) == 16);
static_assert(offsetof(TmrContextRegistrationArgs, callable_descriptor_count) == 24);
static_assert(offsetof(TmrContextRegistrationArgs, callable_descriptor_stride) == 28);
static_assert(
    std::is_trivially_copyable_v<TmrCallableRegistrationArgs> && std::is_standard_layout_v<TmrCallableRegistrationArgs>
);
static_assert(sizeof(TmrCallableRegistrationArgs) == 32);
static_assert(offsetof(TmrCallableRegistrationArgs, context_generation) == 0);
static_assert(offsetof(TmrCallableRegistrationArgs, residency_address) == 8);
static_assert(offsetof(TmrCallableRegistrationArgs, callable_id) == 16);
static_assert(offsetof(TmrCallableRegistrationArgs, reserved) == 20);
static_assert(offsetof(TmrCallableRegistrationArgs, slot_generation) == 24);

}  // namespace simpler::tmr
