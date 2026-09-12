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

#include <cstring>

#include "aicpu/args_dump_aicpu.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/dep_gen_collector_aicpu.h"
#include "aicpu/device_phase_aicpu.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/scope_stats_collector_aicpu.h"
#include "common/kernel_args.h"
#include "kernel_execution_round.h"
#include "kernel_native_status.h"
#include "kernel_prepared_callable.h"
#include "kernel_prepared_context.h"
#include "task_interface/kernel_dispatch_args.h"

namespace simpler::tmr {

// Prepare/close-only consumer hooks. One native thread, externally serialized
// with the entire DSO's program/kernel lifetime. The resource provider owns
// every referenced Device allocation, publishes it before registration, and
// retains it until native completion AND destruction of all referring graphs.
// Gate-idle alone is not a proof of that external quiescence.
template <typename Executor>
int register_kernel_context(Executor &executor, const void *arg) noexcept {
    if (arg == nullptr || !executor.kernel_gate_.idle() || executor.kernel_invocation_.active()) return -1;
    TmrContextRegistrationArgs registration{};
    std::memcpy(&registration, arg, sizeof(registration));
    if (registration.descriptor_address == 0 ||
        registration.descriptor_address % alignof(TmrKernelContextDescriptor) != 0 ||
        registration.descriptor_address > UINTPTR_MAX - sizeof(TmrKernelContextDescriptor))
        return -1;
    TmrKernelContextDescriptor descriptor{};
    const auto *address = reinterpret_cast<const void *>(registration.descriptor_address);
    cache_invalidate_range(address, sizeof(descriptor));
    std::memcpy(&descriptor, address, sizeof(descriptor));
    if (descriptor.resident_runtime == 0 || descriptor.resident_runtime % alignof(Runtime) != 0 ||
        descriptor.resident_runtime > UINTPTR_MAX - sizeof(DeviceRuntimeLaunchDesc) ||
        descriptor.resident_kernel_args == 0 || descriptor.resident_kernel_args % alignof(KernelArgs) != 0 ||
        descriptor.resident_kernel_args > UINTPTR_MAX - sizeof(KernelArgs))
        return -1;
    auto *resident = reinterpret_cast<Runtime *>(descriptor.resident_runtime);
    cache_invalidate_range(resident, sizeof(resident->dev));
    KernelArgs args;
    const auto *args_address = reinterpret_cast<const void *>(descriptor.resident_kernel_args);
    cache_invalidate_range(args_address, sizeof(args));
    std::memcpy(&args, args_address, sizeof(args));
    if (args.runtime_args != resident || args.regs == 0 || args.regs % alignof(uint64_t) != 0 ||
        args.enable_profiling_flag != 0 || args.dump_data_base != 0 || args.chip_swimlane_data_base != 0 ||
        args.pmu_data_base != 0 || args.dep_gen_data_base != 0 || args.scope_stats_data_base != 0 ||
        args.chip_swimlane_aicore_rotation_table != 0 || args.device_wall_data_base != 0)
        return -1;
    PreparedKernelContext candidate;
    if (!make_prepared_kernel_context(registration, descriptor, *resident, &candidate)) return -1;
    candidate.register_table = args.regs;
    if (!executor.kernel_arch_argument(args, &candidate.arch_argument)) return -1;
    if (executor.kernel_context_ready_) {
        const auto &old = executor.kernel_context_;
        return std::memcmp(&old.descriptor, &candidate.descriptor, sizeof(descriptor)) == 0 &&
                       old.residency_base == candidate.residency_base &&
                       old.residency_count == candidate.residency_count && old.stride == candidate.stride &&
                       old.allowed_cpus == candidate.allowed_cpus && old.register_table == candidate.register_table &&
                       old.arch_argument == candidate.arch_argument &&
                       old.ready_queue_shards == candidate.ready_queue_shards &&
                       old.serial_orch_sched == candidate.serial_orch_sched ?
                   0 :
                   -1;
    }
    // Per-DSO platform fields publish once, never concurrently from launch.
    set_platform_regs(args.regs);
    set_dump_args_enabled(false);
    set_platform_dump_base(0);
    set_chip_swimlane_enabled(false);
    set_platform_chip_swimlane_base(0);
    set_pmu_enabled(false);
    set_dep_gen_enabled(false);
    set_platform_dep_gen_base(0);
    set_scope_stats_enabled(false);
    set_platform_scope_stats_base(0);
    set_platform_phase_base(0);
    executor.kernel_context_ = candidate;
    executor.kernel_context_ready_ = true;
    return 0;
}

template <typename Executor>
int register_kernel_callable(Executor &executor, const void *arg) noexcept {
    if (arg == nullptr || !executor.kernel_context_ready_ || !executor.kernel_gate_.idle() ||
        executor.kernel_invocation_.active())
        return -1;
    TmrCallableRegistrationArgs registration{};
    std::memcpy(&registration, arg, sizeof(registration));
    const auto &context = executor.kernel_context_;
    if (registration.context_generation != context.descriptor.context_generation || registration.reserved != 0 ||
        registration.callable_id < 0 || registration.callable_id >= MAX_REGISTERED_CALLABLE_IDS ||
        registration.slot_generation == 0 || !context.contains_residency(registration.residency_address))
        return -1;
    KernelCallableDeviceResidency resident;
    const auto *address = reinterpret_cast<const void *>(registration.residency_address);
    cache_invalidate_range(address, sizeof(resident));
    std::memcpy(&resident, address, sizeof(resident));
    if (resident.callable_id != registration.callable_id || resident.generation != registration.slot_generation ||
        resident.reserved != 0 || resident.bytes > SIZE_MAX || resident.device_address == 0 ||
        resident.bytes > UINTPTR_MAX - resident.device_address ||
        !simpler::kernel::valid_kernel_callable_span(
            reinterpret_cast<const void *>(resident.device_address), static_cast<size_t>(resident.bytes)
        ))
        return -1;
    auto &slot = executor.orch_so_table_[registration.callable_id];
    if (slot.kernel.residency_address != 0) {
        const auto &old = slot.kernel;
        return old.residency_address == registration.residency_address &&
                       old.residency.device_address == resident.device_address &&
                       old.residency.bytes == resident.bytes && old.residency.generation == resident.generation ?
                   0 :
                   -1;
    }
    // Never replace an independently registered program SO under a kernel
    // identity. The provider must give kernel mode its own quiescent DSO.
    if (slot.in_use && !slot.kernel_owned) return -1;
    try {
        cache_invalidate_range(
            reinterpret_cast<const void *>(resident.device_address), static_cast<size_t>(resident.bytes)
        );
        PreparedKernelCallable candidate;
        if (!make_prepared_kernel_callable(registration.residency_address, resident, &candidate)) return -1;
        const auto *image = reinterpret_cast<const ChipCallable *>(resident.device_address);
        if (executor.load_orch_so(
                registration.callable_id, reinterpret_cast<uint64_t>(image->binary_data()), image->binary_size(),
                image->func_name(), image->config_name(), 0
            ) != 0)
            return -1;
        slot.kernel = std::move(candidate);
        slot.kernel_owned = true;
        return 0;
    } catch (...) {
        return -1;
    }
}

template <typename Executor>
int release_kernel_context(Executor &executor, const void *arg) noexcept {
    if (arg == nullptr || !executor.kernel_context_ready_ || !executor.kernel_gate_.idle() ||
        executor.kernel_invocation_.active())
        return -1;
    TmrContextRegistrationArgs registration{};
    std::memcpy(&registration, arg, sizeof(registration));
    const auto &context = executor.kernel_context_;
    if (registration.descriptor_address != context.descriptor.self_address ||
        registration.context_generation != context.descriptor.context_generation)
        return -1;
    // Existing SO handles stay cached, without any Device ownership change.
    // A later quiescent kernel registration may replace these now-unborrowed
    // handles through the existing loader, never through the launch path.
    for (auto &slot : executor.orch_so_table_)
        slot.kernel = {};
    executor.kernel_context_ = {};
    executor.kernel_context_ready_ = false;
    return 0;
}

// All native workers reach the same round, even on null/framing/stale input.
// CANN transports the full Host-adapter packet; no API exposes the physical
// readable byte count here. The embedded packet_bytes is NOT such an API.
template <typename Executor>
int dispatch_prepared_kernel_task(Executor &executor, void *arg, int32_t cpu) noexcept {
    if (!executor.kernel_context_ready_) return static_cast<int>(KernelDispatchStatus::UnsupportedPayload);
    const auto &context = executor.kernel_context_;
    KernelExecutionRequest request;
    request.binding = context.binding;
    request.handshake = context.handshake;
    request.allowed_cpus = context.allowed_cpus.data();
    request.execution_threads = context.descriptor.execution_threads;
    request.launched_threads = context.descriptor.launch_threads;
    request.admission_status = static_cast<int>(KernelDispatchStatus::InvalidArgs);
    SimplerKernelDispatchArgs dispatch{};
    if (arg != nullptr) {
        std::memcpy(&dispatch, arg, sizeof(dispatch));
        const auto &header = dispatch.invocation;
        if (dispatch.packet_bytes >= sizeof(dispatch) && dispatch.packet_bytes <= SIZE_MAX &&
            header.payload_bytes == dispatch.packet_bytes - sizeof(dispatch) && header.mode == SIMPLER_MODE_KERNEL &&
            header.generation != 0 &&
            simpler::kernel::valid_invocation_counts(header.tensor_count, header.scalar_count) &&
            header.host_copy_tensor_count == 0 && header.callable_id >= 0 &&
            header.callable_id < MAX_REGISTERED_CALLABLE_IDS) {
            const auto &slot = executor.orch_so_table_[header.callable_id].kernel;
            request.admission_status = static_cast<int>(KernelDispatchStatus::NotResident);
            if (slot.residency_address != 0 && dispatch.residency_address == slot.residency_address &&
                context.contains_residency(dispatch.residency_address)) {
                const auto *address = reinterpret_cast<const void *>(dispatch.residency_address);
                cache_invalidate_range(address, sizeof(KernelCallableDeviceResidency));
                KernelCallableDeviceResidency resident;
                std::memcpy(&resident, address, sizeof(resident));
                request.admission_status = static_cast<int>(KernelDispatchStatus::Stale);
                if (kernel_callable_residency_matches(header, resident) &&
                    resident.generation == slot.identity.slot_generation &&
                    resident.device_address == slot.residency.device_address &&
                    resident.bytes == slot.residency.bytes) {
                    constexpr size_t prefix = offsetof(SimplerKernelDispatchArgs, invocation);
                    request.packet = {
                        static_cast<const uint8_t *>(arg) + prefix, static_cast<size_t>(dispatch.packet_bytes) - prefix
                    };
                    request.callable = slot.view();
                    request.admission_status = 0;
                }
            }
        }
    }
    KernelFinalStatus result;
    const int status = execute_kernel_round_impl(executor, request, cpu, &result);
    return classify_kernel_dispatch_status(status, result.cleanup_status);
}

}  // namespace simpler::tmr
