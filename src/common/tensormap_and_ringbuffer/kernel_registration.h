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
#include "task_interface/tmr_kernel_revoke.h"

namespace simpler::tmr {

// The round leader refreshes the window bits before publishing initialization
// to the other AICPU threads. Host updates occur only between completed rounds.
inline void refresh_kernel_dfx(const PreparedKernelContext &context) noexcept {
    if (get_platform_chip_swimlane_base() == 0 && get_platform_dep_gen_base() == 0) return;
    const auto *args = reinterpret_cast<const KernelArgs *>(context.descriptor.resident_kernel_args);
    cache_invalidate_range(&args->enable_profiling_flag, sizeof(args->enable_profiling_flag));
    const uint32_t flags = args->enable_profiling_flag;
    set_chip_swimlane_enabled(SIMPLER_GET_DFX_FLAG(flags, SIMPLER_DFX_FLAG_CHIP_SWIMLANE));
    set_dep_gen_enabled(SIMPLER_GET_DFX_FLAG(flags, SIMPLER_DFX_FLAG_DEP_GEN));
}

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
    // Registration validates configured swimlane and dependency capabilities.
    // Each diagnostic's bit must agree with its addresses here; host window
    // boundaries can disable its bit after registration without releasing storage.
    const bool swimlane_requested = SIMPLER_GET_DFX_FLAG(args.enable_profiling_flag, SIMPLER_DFX_FLAG_CHIP_SWIMLANE);
    const bool swimlane_addressed = args.chip_swimlane_data_base != 0 && args.chip_swimlane_aicore_rotation_table != 0;
    const bool dep_gen_requested = SIMPLER_GET_DFX_FLAG(args.enable_profiling_flag, SIMPLER_DFX_FLAG_DEP_GEN);
    constexpr uint32_t kAdmittedDfxFlags =
        static_cast<uint32_t>(SIMPLER_DFX_FLAG_CHIP_SWIMLANE) | static_cast<uint32_t>(SIMPLER_DFX_FLAG_DEP_GEN);
    if (args.runtime_args != resident || args.regs == 0 || args.regs % alignof(uint64_t) != 0 ||
        (args.enable_profiling_flag & ~kAdmittedDfxFlags) != 0 || swimlane_requested != swimlane_addressed ||
        dep_gen_requested != (args.dep_gen_data_base != 0) || args.dump_data_base != 0 || args.pmu_data_base != 0 ||
        args.scope_stats_data_base != 0 || args.device_wall_data_base != 0)
        return -1;
    PreparedKernelContext candidate;
    if (!make_prepared_kernel_context(registration, descriptor, *resident, &candidate)) return -1;
    candidate.register_table = args.regs;
    if (!executor.kernel_arch_argument(args, &candidate.arch_argument)) return -1;
    if (executor.kernel_context_ready_) {
        const auto &old = executor.kernel_context_;
        return std::memcmp(&old.descriptor, &candidate.descriptor, sizeof(descriptor)) == 0 &&
                       old.allowed_cpus == candidate.allowed_cpus && old.register_table == candidate.register_table &&
                       old.arch_argument == candidate.arch_argument &&
                       old.ready_queue_shards == candidate.ready_queue_shards &&
                       old.serial_orch_sched == candidate.serial_orch_sched ?
                   0 :
                   -1;
    }
    // Per-DSO addresses publish once. The round leader refreshes enable bits.
    set_platform_regs(args.regs);
    set_dump_args_enabled(false);
    set_platform_dump_base(0);
    set_chip_swimlane_enabled(swimlane_requested);
    set_platform_chip_swimlane_base(args.chip_swimlane_data_base);
    set_platform_chip_swimlane_aicore_rotation_table(args.chip_swimlane_aicore_rotation_table);
    set_pmu_enabled(false);
    set_dep_gen_enabled(dep_gen_requested);
    set_platform_dep_gen_base(args.dep_gen_data_base);
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
        registration.bytes > SIZE_MAX ||
        !simpler::kernel::valid_kernel_callable_span(
            reinterpret_cast<const void *>(registration.device_address), static_cast<size_t>(registration.bytes)
        ))
        return -1;
    auto &slot = executor.orch_so_table_[registration.callable_id];
    if (slot.kernel.device_address != 0) {
        const auto &old = slot.kernel;
        return old.device_address == registration.device_address && old.bytes == registration.bytes ? 0 : -1;
    }
    // Never replace an independently registered program SO under a kernel
    // identity. The provider must give kernel mode its own quiescent DSO.
    if (slot.in_use && !slot.kernel_owned) return -1;
    try {
        cache_invalidate_range(
            reinterpret_cast<const void *>(registration.device_address), static_cast<size_t>(registration.bytes)
        );
        PreparedKernelCallable candidate;
        if (!make_prepared_kernel_callable(registration, &candidate)) return -1;
        slot.kernel = std::move(candidate);
        slot.kernel_owned = true;
        slot.needs_load = true;
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

// Establish residency for a callable the device has not seen, from the image
// span its launch packet names. The round leader is the only caller: it holds
// the gate, so this writes the shared table exactly once per round, and the
// packet's span was already checked against the resident one by every thread's
// admission. Returns false when the image is malformed or the slot belongs to
// a program-mode SO.
template <typename Executor>
bool ensure_kernel_residency(
    Executor &executor, int32_t callable_id, uint64_t image_address, uint64_t image_bytes, uint64_t generation
) noexcept {
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) return false;
    auto &entry = executor.orch_so_table_[callable_id];
    if (entry.kernel.device_address != 0) return true;
    if (entry.in_use && !entry.kernel_owned) return false;
    const TmrCallableRegistrationArgs from_packet{generation, image_address, image_bytes, callable_id, 0};
    try {
        cache_invalidate_range(reinterpret_cast<const void *>(image_address), static_cast<size_t>(image_bytes));
        PreparedKernelCallable candidate;
        if (!make_prepared_kernel_callable(from_packet, &candidate)) return false;
        entry.kernel = std::move(candidate);
        entry.kernel_owned = true;
        entry.needs_load = true;
        return true;
    } catch (...) {
        return false;
    }
}

// The owner closes submission admission and proves graph quiescence before
// this one-thread task. Busy/mismatched state leaves even the receipt untouched.
// An absent registration is revocable only when no kernel metadata borrows it.
template <typename Executor>
int revoke_kernel_context(Executor &executor, const void *arg) noexcept {
    if (arg == nullptr || !executor.kernel_gate_.idle() || executor.kernel_invocation_.active() ||
        executor.kernel_storage_attached_)
        return -1;
    TmrContextRevokeArgs args{};
    std::memcpy(&args, arg, sizeof(args));
    if (!valid_tmr_context_revoke_args(args)) return -1;
    const auto &context = executor.kernel_context_;
    const auto &descriptor = context.descriptor;
    if (executor.kernel_context_ready_) {
        if (descriptor.self_address != args.descriptor_address ||
            descriptor.context_generation != args.context_generation)
            return -1;
        using namespace prepared_context_detail;
        const Region receipt{args.receipt_address, args.receipt_bytes};
        const std::array<Region, 9> protected_regions{{
            {descriptor.self_address, sizeof(descriptor)},
            {descriptor.resident_runtime, sizeof(DeviceRuntimeLaunchDesc)},
            {descriptor.resident_kernel_args, sizeof(KernelArgs)},
            {descriptor.heap_base, descriptor.heap_capacity},
            {descriptor.sm_base, descriptor.sm_capacity},
            {descriptor.arena_base, descriptor.arena_capacity},
            {descriptor.control_address, descriptor.control_bytes},
            {descriptor.reports_address, descriptor.reports_bytes},
            {context.register_table, static_cast<uint64_t>(platform_get_physical_cores_count()) * sizeof(uint64_t)},
        }};
        for (const auto &region : protected_regions)
            if (!valid_region(region, 1) || !disjoint(receipt, region)) return -1;
        for (const auto &slot : executor.orch_so_table_) {
            if (slot.kernel.device_address != 0 && !disjoint(receipt, {slot.kernel.device_address, slot.kernel.bytes}))
                return -1;
        }
    } else {
        if (descriptor.self_address != 0 || descriptor.context_generation != 0 || context.binding.resident != nullptr ||
            context.binding.identity.device_binding_addr != 0 || context.binding.sm.base != nullptr ||
            context.binding.arena.base != nullptr || context.handshake.control != nullptr ||
            context.handshake.reports != nullptr || context.register_table != 0)
            return -1;
        for (const auto &slot : executor.orch_so_table_)
            if (slot.kernel.device_address != 0 || slot.kernel.bytes != 0 || !slot.kernel.functions.empty()) return -1;
    }
    auto *receipt = reinterpret_cast<TmrContextRevokeReceipt *>(args.receipt_address);
    cache_invalidate_range(receipt, sizeof(*receipt));
    TmrContextRevokeReceipt pending{};
    std::memcpy(&pending, receipt, sizeof(pending));
    if (!valid_tmr_context_revoke_receipt(pending, args, TmrRevokeCompletion::Pending)) return -1;
    if (executor.kernel_context_ready_) {
        const TmrContextRegistrationArgs registration{args.descriptor_address, args.context_generation};
        if (release_kernel_context(executor, &registration) != 0) return -1;
    }
    receipt->descriptor_address = args.descriptor_address;
    receipt->context_generation = args.context_generation;
    receipt->status = 0;
    __atomic_store_n(&receipt->complete, static_cast<uint32_t>(TmrRevokeCompletion::Complete), __ATOMIC_RELEASE);
    cache_flush_range(receipt, sizeof(*receipt));
    return 0;
}

// All native workers reach the same round, even on null or invalid framing.
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
    if (arg != nullptr && reinterpret_cast<uintptr_t>(arg) % alignof(SimplerKernelDispatchArgs) == 0) {
        std::memcpy(&dispatch, arg, sizeof(dispatch));
        const auto &header = dispatch.invocation;
        if (dispatch.packet_bytes >= sizeof(dispatch) && dispatch.packet_bytes <= SIZE_MAX &&
            dispatch.packet_bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(arg) &&
            header.payload_bytes == dispatch.packet_bytes - sizeof(dispatch) && header.mode == SIMPLER_MODE_KERNEL &&
            header.reserved_ == 0 &&
            simpler::kernel::valid_invocation_counts(header.tensor_count, header.scalar_count) &&
            header.host_copy_tensor_count == 0 && header.callable_id >= 0 &&
            header.callable_id < MAX_REGISTERED_CALLABLE_IDS) {
            // A callable's residency is established by whichever comes first:
            // the host's registration entry, or this packet. The packet carries
            // the image span, so a launch never depends on registration having
            // already executed — the two are unordered once a launch is
            // recorded into a graph.
            // Read-only here: every launched thread runs this, and the shared
            // table may be written only by the round leader.
            const auto &slot = executor.orch_so_table_[header.callable_id].kernel;
            request.admission_status = static_cast<int>(KernelDispatchStatus::InvalidBinding);
            const bool resident_matches =
                slot.device_address == 0 ||
                (dispatch.chip_callable_address == slot.device_address && dispatch.chip_callable_bytes == slot.bytes);
            if (resident_matches && dispatch.binding_address == context.descriptor.resident_kernel_args &&
                dispatch.context_generation == context.descriptor.context_generation &&
                dispatch.sm_bytes == context.descriptor.sm_capacity &&
                dispatch.arena_bytes == context.descriptor.arena_capacity) {
                constexpr size_t prefix = offsetof(SimplerKernelDispatchArgs, invocation);
                request.packet = {
                    static_cast<const uint8_t *>(arg) + prefix, static_cast<size_t>(dispatch.packet_bytes) - prefix
                };
                request.image_address = dispatch.chip_callable_address;
                request.image_bytes = dispatch.chip_callable_bytes;
                request.callable_id = header.callable_id;
                if (slot.device_address != 0) request.callable = slot.view();
                request.admission_status = 0;
            }
        }
    }
    KernelFinalStatus result;
    const int status = execute_kernel_round_impl(executor, request, cpu, &result);
    return classify_kernel_dispatch_status(status, result.cleanup_status);
}

}  // namespace simpler::tmr
