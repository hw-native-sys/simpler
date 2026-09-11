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

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "common/kernel_args.h"
#include "kernel_clear_plan.h"
#include "kernel_core_group.h"
#include "kernel_execution_inputs.h"
#include "task_interface/kernel_callable_residency.h"
#include "task_interface/tmr_kernel_context.h"

namespace simpler::tmr {

namespace prepared_context_detail {

struct Region {
    uint64_t base;
    uint64_t bytes;
};

inline bool valid_region(Region region, size_t alignment) noexcept {
    const uint64_t limit = std::numeric_limits<uintptr_t>::max();
    return region.base != 0 && region.base <= limit && region.base % alignment == 0 && region.bytes != 0 &&
           region.bytes <= limit - region.base;
}

// Both regions have passed valid_region before endpoint arithmetic is used.
inline bool disjoint(Region a, Region b) noexcept { return a.base + a.bytes <= b.base || b.base + b.bytes <= a.base; }

inline bool valid_residency_table(uint64_t base, uint32_t count, uint32_t stride) noexcept {
    return count != 0 && count <= MAX_REGISTERED_CALLABLE_IDS && stride >= sizeof(KernelCallableDeviceResidency) &&
           stride % alignof(KernelCallableDeviceResidency) == 0 &&
           valid_region({base, static_cast<uint64_t>(count) * stride}, alignof(KernelCallableDeviceResidency));
}

}  // namespace prepared_context_detail

// Read-only metadata, not an allocation owner. The registration caller pins
// all referenced regions and prevents replacement while any consumer borrows.
struct PreparedKernelContext {
    TmrKernelContextDescriptor descriptor{};
    KernelBindingView binding{};
    KernelHandshakeView handshake{};
    std::array<int32_t, MAX_GATE_THREADS> allowed_cpus{};
    uint64_t residency_base{0};
    uint32_t residency_count{0};
    uint32_t stride{0};
    uint64_t register_table{0};
    uint64_t arch_argument{0};  // a2a3 FFTS base; a5 has no extra mutable identity.
    int32_t ready_queue_shards{0};
    bool serial_orch_sched{false};

    bool contains_residency(uint64_t address) const noexcept {
        if (!prepared_context_detail::valid_residency_table(residency_base, residency_count, stride) ||
            address < residency_base)
            return false;
        const uint64_t offset = address - residency_base;
        return offset % stride == 0 && offset / stride < residency_count;
    }
};

// descriptor/resident are readable prepare-time snapshots; integer addresses
// are never dereferenced here. Only Runtime::dev crosses to the device, so its
// protected span excludes Runtime's Host-only vector tail. flags == 0 validates
// the descriptor version, not KernelArgs DFX state: the registration entry must
// separately validate its actual KernelArgs snapshot, including arch fields.
inline bool make_prepared_kernel_context(
    const TmrContextRegistrationArgs &registration, const TmrKernelContextDescriptor &descriptor,
    const Runtime &resident, PreparedKernelContext *out
) noexcept {
    using namespace prepared_context_detail;
    if (out == nullptr || descriptor.version != kTmrKernelContextVersion ||
        descriptor.bytes != sizeof(TmrKernelContextDescriptor) || descriptor.context_generation == 0 ||
        descriptor.context_generation != registration.context_generation ||
        descriptor.self_address != registration.descriptor_address || descriptor.flags != 0)
        return false;
    for (uint64_t reserved : descriptor.reserved)
        if (reserved != 0) return false;
    const auto &config = resident.dev;
    if (descriptor.execution_threads < 2 || descriptor.execution_threads > PLATFORM_MAX_AICPU_THREADS ||
        descriptor.execution_threads > descriptor.launch_threads || descriptor.launch_threads > MAX_GATE_THREADS ||
        descriptor.worker_count <= 0 || descriptor.worker_count > RUNTIME_MAX_WORKER ||
        descriptor.worker_count % 3 != 0 || config.aicpu_thread_num != descriptor.execution_threads ||
        config.aicpu_launch_count != descriptor.launch_threads ||
        config.aicpu_allowed_cpu_count != descriptor.execution_threads ||
        config.worker_count != descriptor.worker_count || config.ready_queue_shards <= 0 ||
        config.ready_queue_shards > PLATFORM_MAX_AICPU_THREADS)
        return false;
    for (int32_t i = 0; i < descriptor.execution_threads; ++i) {
        if (config.aicpu_allowed_cpus[i] < 0) return false;
        for (int32_t j = 0; j < i; ++j)
            if (config.aicpu_allowed_cpus[i] == config.aicpu_allowed_cpus[j]) return false;
    }
    if (config.active_callable_id_ != -1 || config.gm_sm_ptr_ != nullptr || config.prebuilt_arena_base_ != nullptr ||
        config.prebuilt_runtime_offset_ != 0 || config.orch_args_storage_.tensor_count() != 0 ||
        config.orch_args_storage_.scalar_count() != 0)
        return false;
    for (uint64_t function : config.func_id_to_addr_)
        if (function != 0) return false;

    const std::array<Region, 3> arenas{{
        {descriptor.heap_base, descriptor.heap_capacity},
        {descriptor.sm_base, descriptor.sm_capacity},
        {descriptor.arena_base, descriptor.arena_capacity},
    }};
    if (!valid_region(arenas[0], DeviceArena::kDefaultBaseAlign) ||
        !valid_region(arenas[1], alignof(SharedMemoryHeader)) ||
        !valid_region(arenas[2], DeviceArena::kDefaultBaseAlign) || descriptor.heap_required == 0 ||
        descriptor.heap_required > descriptor.heap_capacity || descriptor.sm_required == 0 ||
        descriptor.sm_required > descriptor.sm_capacity || descriptor.arena_required == 0 ||
        descriptor.arena_required > descriptor.arena_capacity)
        return false;
    for (size_t i = 0; i < arenas.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (!disjoint(arenas[i], arenas[j])) return false;

    if (!valid_residency_table(
            registration.callable_descriptor_base, registration.callable_descriptor_count,
            registration.callable_descriptor_stride
        ))
        return false;
    const TmrKernelClearBinding clear{
        descriptor.context_generation,
        {descriptor.control_address, descriptor.control_bytes},
        {descriptor.reports_address, descriptor.reports_bytes},
        descriptor.worker_count
    };
    if (!valid_tmr_clear_binding(clear)) return false;

    const std::array<Region, 6> auxiliary{{
        {descriptor.self_address, sizeof(TmrKernelContextDescriptor)},
        {descriptor.resident_runtime, sizeof(DeviceRuntimeLaunchDesc)},
        {descriptor.resident_kernel_args, sizeof(KernelArgs)},
        {registration.callable_descriptor_base,
         static_cast<uint64_t>(registration.callable_descriptor_count) * registration.callable_descriptor_stride},
        {descriptor.control_address, descriptor.control_bytes},
        {descriptor.reports_address, descriptor.reports_bytes},
    }};
    const std::array<size_t, auxiliary.size()> alignments{
        alignof(TmrKernelContextDescriptor),    alignof(Runtime),          alignof(KernelArgs),
        alignof(KernelCallableDeviceResidency), alignof(TmrLaunchControl), alignof(TmrCoreReport)
    };
    const Region image{descriptor.arena_base, descriptor.arena_required};
    for (size_t i = 0; i < auxiliary.size(); ++i) {
        const Region region = auxiliary[i];
        if (!valid_region(region, alignments[i]) || !disjoint(region, arenas[0]) || !disjoint(region, arenas[1]) ||
            !disjoint(region, image))
            return false;
        // Auxiliary regions may be separate allocations or wholly contained
        // in the arena's unused tail, never straddling its allocation boundary.
        if (!disjoint(region, arenas[2]) &&
            (region.base < image.base + image.bytes || region.base + region.bytes > arenas[2].base + arenas[2].bytes))
            return false;
        for (size_t j = 0; j < i; ++j)
            if (!disjoint(region, auxiliary[j])) return false;
    }

    PreparedKernelContext candidate{};
    candidate.descriptor = descriptor;
    candidate.binding = {
        {descriptor.self_address, descriptor.context_generation},
        reinterpret_cast<Runtime *>(descriptor.resident_runtime),
        {reinterpret_cast<void *>(descriptor.sm_base), static_cast<size_t>(descriptor.sm_capacity),
         static_cast<size_t>(descriptor.sm_required)},
        {reinterpret_cast<void *>(descriptor.arena_base), static_cast<size_t>(descriptor.arena_capacity),
         static_cast<size_t>(descriptor.arena_required)},
        static_cast<size_t>(descriptor.runtime_offset)
    };
    if (descriptor.runtime_offset > std::numeric_limits<size_t>::max() ||
        validate_execution_binding(candidate.binding) != InvocationStatus::Ok)
        return false;
    candidate.handshake = {
        reinterpret_cast<TmrLaunchControl *>(descriptor.control_address),
        reinterpret_cast<TmrCoreReport *>(descriptor.reports_address), descriptor.worker_count, 0
    };
    for (int32_t i = 0; i < descriptor.execution_threads; ++i)
        candidate.allowed_cpus[i] = config.aicpu_allowed_cpus[i];
    candidate.residency_base = registration.callable_descriptor_base;
    candidate.residency_count = registration.callable_descriptor_count;
    candidate.stride = registration.callable_descriptor_stride;
    candidate.ready_queue_shards = config.ready_queue_shards;
    candidate.serial_orch_sched = config.serial_orch_sched;
    *out = candidate;
    return true;
}

}  // namespace simpler::tmr
