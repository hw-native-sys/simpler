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
#include "host_build_graph/kernel_graph_slot_registry.h"

#include <atomic>

#include "aicpu/args_dump_aicpu.h"
#include "aicpu/cache_maintenance.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/dep_gen_collector_aicpu.h"
#include "aicpu/device_phase_aicpu.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/scope_stats_collector_aicpu.h"
#include "common/kernel_args.h"
#include "host_build_graph/kernel_callable_registration.h"
#include "runtime.h"

namespace hbg {
namespace {
std::atomic<GraphSlotRegistry *> current_registry{nullptr};

struct KernelDfxBinding {
    const GraphSlotRegistry *registry{nullptr};
    Runtime *runtime{nullptr};
    uint64_t generation{0};
    uint64_t args_address{0};
    uint64_t regs{0};
    uint64_t swimlane{0};
    uint64_t rotation{0};
    uint64_t dependencies{0};
    uint32_t capabilities{0};
};

KernelDfxBinding kernel_dfx;

void publish_kernel_dfx(uint32_t flags) noexcept {
    set_chip_swimlane_enabled(SIMPLER_GET_DFX_FLAG(flags, SIMPLER_DFX_FLAG_CHIP_SWIMLANE));
    set_platform_chip_swimlane_base(kernel_dfx.swimlane);
    set_platform_chip_swimlane_aicore_rotation_table(kernel_dfx.rotation);
    set_dep_gen_enabled(SIMPLER_GET_DFX_FLAG(flags, SIMPLER_DFX_FLAG_DEP_GEN));
    set_platform_dep_gen_base(kernel_dfx.dependencies);
    // Program entry points share these globals with kernel mode.
    set_dump_args_enabled(false);
    set_platform_dump_base(0);
    set_pmu_enabled(false);
    set_platform_pmu_base(0);
    set_scope_stats_enabled(false);
    set_platform_scope_stats_base(0);
    set_platform_phase_base(0);
}

bool valid_registry(const GraphSlotRegistry *registry) {
    if (registry == nullptr || reinterpret_cast<uintptr_t>(registry) % 1024 != 0) return false;
    cache_invalidate_range(registry, offsetof(GraphSlotRegistry, registration));
    if (registry->magic != GRAPH_REGISTRY_MAGIC || registry->version != GRAPH_SLOT_VERSION ||
        registry->bytes != sizeof(*registry) || registry->device_id < 0 || registry->context_generation == 0 ||
        registry->runtime_binary_id == 0)
        return false;
    for (uint64_t word : registry->reserved)
        if (word != 0) return false;
    return true;
}

bool matches_registry(const GraphSlotRegistry &registry, const GraphSlotRegistration &registration) {
    return registration.registry.address == reinterpret_cast<uintptr_t>(&registry) &&
           registration.device_id == registry.device_id &&
           registration.slot_generation == registry.context_generation &&
           registration.runtime_binary_id == registry.runtime_binary_id;
}
}  // namespace

GraphSlotStatus initialize_graph_slot_registry(
    GraphSlotRegistry *registry, int device_id, uint64_t generation, uint64_t runtime_binary_id
) noexcept {
    if (registry == nullptr || reinterpret_cast<uintptr_t>(registry) % 1024 != 0 || device_id < 0 || generation == 0 ||
        runtime_binary_id == 0)
        return GraphSlotStatus::InvalidRegistry;
    if (current_registry.load(std::memory_order_acquire) == registry) return GraphSlotStatus::Conflict;
    GraphSlotRegistry initial{};
    initial.magic = GRAPH_REGISTRY_MAGIC;
    initial.version = GRAPH_SLOT_VERSION;
    initial.bytes = sizeof(initial);
    initial.phase = static_cast<uint32_t>(GraphSlotPhase::Empty);
    initial.device_id = device_id;
    initial.context_generation = generation;
    initial.runtime_binary_id = runtime_binary_id;
    std::memcpy(registry, &initial, sizeof(initial));
    cache_flush_range(registry, sizeof(*registry));
    return GraphSlotStatus::Ok;
}

GraphSlotStatus
register_graph_execution_slot(GraphSlotRegistry *registry, const void *registration, size_t bytes) noexcept {
    if (registration == nullptr || bytes != sizeof(GraphSlotRegistration)) return GraphSlotStatus::InvalidRegistration;
    GraphSlotRegistration candidate{};
    std::memcpy(&candidate, registration, sizeof(candidate));
    if (!valid_graph_slot_registration(candidate)) return GraphSlotStatus::InvalidRegistration;
    if (!valid_registry(registry)) return GraphSlotStatus::InvalidRegistry;
    if (!matches_registry(*registry, candidate)) return GraphSlotStatus::Conflict;
    for (;;) {
        uint32_t phase = __atomic_load_n(&registry->phase, __ATOMIC_ACQUIRE);
        if (phase == static_cast<uint32_t>(GraphSlotPhase::Poisoned)) return GraphSlotStatus::Poisoned;
        if (phase == static_cast<uint32_t>(GraphSlotPhase::Ready)) {
            cache_invalidate_range(&registry->registration, sizeof(registry->registration));
            return std::memcmp(&registry->registration, &candidate, sizeof(candidate)) == 0 ? GraphSlotStatus::Ok :
                                                                                              GraphSlotStatus::Conflict;
        }
        if (phase == static_cast<uint32_t>(GraphSlotPhase::Publishing)) return GraphSlotStatus::Publishing;
        if (phase != static_cast<uint32_t>(GraphSlotPhase::Empty)) return GraphSlotStatus::InvalidRegistry;
        if (!__atomic_compare_exchange_n(
                &registry->phase, &phase, static_cast<uint32_t>(GraphSlotPhase::Publishing), false, __ATOMIC_ACQ_REL,
                __ATOMIC_ACQUIRE
            ))
            continue;
        registry->registration = candidate;
        cache_flush_range(&registry->registration, sizeof(registry->registration));
        __atomic_store_n(&registry->phase, static_cast<uint32_t>(GraphSlotPhase::Ready), __ATOMIC_RELEASE);
        cache_flush_range(registry, offsetof(GraphSlotRegistry, registration));
        return GraphSlotStatus::Ok;
    }
}

GraphSlotStatus install_graph_execution_slot(const void *registration, size_t bytes) noexcept {
    if (registration == nullptr || bytes != sizeof(GraphSlotRegistration)) {
        return GraphSlotStatus::InvalidRegistration;
    }
    GraphSlotRegistration candidate{};
    std::memcpy(&candidate, registration, sizeof(candidate));
    if (!valid_graph_slot_registration(candidate)) return GraphSlotStatus::InvalidRegistration;

    auto *registry = reinterpret_cast<GraphSlotRegistry *>(candidate.registry.address);
    const auto *bound = current_registry.load(std::memory_order_acquire);
    if (bound != nullptr) {
        if (bound != registry) return GraphSlotStatus::Conflict;
        GraphSlotRegistration existing{};
        auto status =
            acquire_graph_execution_slot(registry, candidate.device_id, candidate.runtime_binary_id, existing);
        if (status != GraphSlotStatus::Ok) return status;
        status = register_graph_execution_slot(registry, &candidate, sizeof(candidate));
        if (status != GraphSlotStatus::Ok) return status;
        return bind_graph_slot_registry(registry, candidate.device_id, candidate.runtime_binary_id);
    }

    auto status = initialize_graph_slot_registry(
        registry, candidate.device_id, candidate.slot_generation, candidate.runtime_binary_id
    );
    if (status != GraphSlotStatus::Ok) return status;
    status = register_graph_execution_slot(registry, &candidate, sizeof(candidate));
    if (status != GraphSlotStatus::Ok) return status;
    return bind_graph_slot_registry(registry, candidate.device_id, candidate.runtime_binary_id);
}

GraphSlotStatus acquire_graph_execution_slot(
    const GraphSlotRegistry *registry, int device_id, uint64_t runtime_binary_id, GraphSlotRegistration &out
) noexcept {
    if (!valid_registry(registry)) return GraphSlotStatus::InvalidRegistry;
    if (registry->device_id != device_id) return GraphSlotStatus::DeviceMismatch;
    if (registry->runtime_binary_id != runtime_binary_id) return GraphSlotStatus::BinaryMismatch;
    const uint32_t phase = __atomic_load_n(&registry->phase, __ATOMIC_ACQUIRE);
    if (phase == static_cast<uint32_t>(GraphSlotPhase::Poisoned)) return GraphSlotStatus::Poisoned;
    if (phase == static_cast<uint32_t>(GraphSlotPhase::Empty)) return GraphSlotStatus::NotReady;
    if (phase == static_cast<uint32_t>(GraphSlotPhase::Publishing)) return GraphSlotStatus::Publishing;
    if (phase != static_cast<uint32_t>(GraphSlotPhase::Ready)) return GraphSlotStatus::InvalidRegistry;
    cache_invalidate_range(&registry->registration, sizeof(registry->registration));
    const GraphSlotRegistration candidate = registry->registration;
    if (!valid_graph_slot_registration(candidate) || !matches_registry(*registry, candidate))
        return GraphSlotStatus::InvalidRegistry;
    out = candidate;
    return GraphSlotStatus::Ok;
}

GraphSlotStatus
bind_graph_slot_registry(GraphSlotRegistry *registry, int device_id, uint64_t runtime_binary_id) noexcept {
    GraphSlotRegistration registration{};
    const auto status = acquire_graph_execution_slot(registry, device_id, runtime_binary_id, registration);
    if (status != GraphSlotStatus::Ok) return status;
    GraphSlotRegistry *expected = nullptr;
    if (!current_registry.compare_exchange_strong(
            expected, registry, std::memory_order_acq_rel, std::memory_order_acquire
        ) &&
        expected != registry)
        return GraphSlotStatus::Conflict;
    return GraphSlotStatus::Ok;
}

bool detach_graph_slot_registry(GraphSlotRegistry *registry) noexcept {
    if (registry == nullptr) return false;
    if (!current_registry.compare_exchange_strong(registry, nullptr, std::memory_order_acq_rel)) return false;
    kernel_dfx = {};
    publish_kernel_dfx(0);
    return true;
}

GraphSlotRegistry *current_graph_slot_registry() noexcept { return current_registry.load(std::memory_order_acquire); }

bool refresh_graph_kernel_dfx(
    const GraphSlotRegistry *registry, Runtime *runtime, const HbgCallableRegistration &registration
) noexcept {
    if (registry == nullptr || registry != current_graph_slot_registry() || runtime == nullptr ||
        registration.context_generation != registry->context_generation ||
        registration.runtime_address != reinterpret_cast<uintptr_t>(runtime))
        return false;
    const uint64_t address = *runtime->kernel_dfx_args_slot();
    if (address == 0 || address % alignof(KernelArgs) != 0 || address > UINTPTR_MAX - sizeof(KernelArgs)) return false;
    if (kernel_dfx.registry != nullptr) {
        if (kernel_dfx.registry != registry || kernel_dfx.runtime != runtime ||
            kernel_dfx.generation != registration.context_generation || kernel_dfx.args_address != address ||
            kernel_dfx.regs != registration.register_table_address)
            return false;
    } else {
        const auto *resident = reinterpret_cast<const KernelArgs *>(address);
        cache_invalidate_range(resident, sizeof(*resident));
        KernelArgs args{};
        std::memcpy(&args, resident, sizeof(args));
        const bool swimlane = args.chip_swimlane_data_base != 0;
        const bool rotation = args.chip_swimlane_aicore_rotation_table != 0;
        const auto valid_address = [](uint64_t value) {
            return value == 0 || (value % alignof(uint64_t) == 0 && value <= UINTPTR_MAX - sizeof(uint64_t));
        };
        uint32_t capabilities = swimlane ? SIMPLER_DFX_FLAG_CHIP_SWIMLANE : 0;
        if (args.dep_gen_data_base != 0) capabilities |= SIMPLER_DFX_FLAG_DEP_GEN;
        if (args.runtime_args != runtime || args.regs != registration.register_table_address || swimlane != rotation ||
            (args.enable_profiling_flag & ~capabilities) != 0 || !valid_address(args.chip_swimlane_data_base) ||
            !valid_address(args.chip_swimlane_aicore_rotation_table) || !valid_address(args.dep_gen_data_base) ||
            args.dump_data_base != 0 || args.pmu_data_base != 0 || args.scope_stats_data_base != 0 ||
            args.device_wall_data_base != 0)
            return false;
        kernel_dfx = {
            registry,
            runtime,
            registration.context_generation,
            address,
            args.regs,
            args.chip_swimlane_data_base,
            args.chip_swimlane_aicore_rotation_table,
            args.dep_gen_data_base,
            capabilities
        };
    }
    uint32_t flags = 0;
    if (kernel_dfx.capabilities != 0) {
        const auto *args = reinterpret_cast<const KernelArgs *>(address);
        cache_invalidate_range(&args->enable_profiling_flag, sizeof(args->enable_profiling_flag));
        flags = args->enable_profiling_flag;
        if ((flags & ~kernel_dfx.capabilities) != 0) return false;
    }
    publish_kernel_dfx(flags);
    return true;
}

GraphSlotStatus poison_graph_execution_slot(GraphSlotRegistry *registry) noexcept {
    if (!valid_registry(registry)) return GraphSlotStatus::InvalidRegistry;
    __atomic_store_n(&registry->phase, static_cast<uint32_t>(GraphSlotPhase::Poisoned), __ATOMIC_RELEASE);
    cache_flush_range(registry, offsetof(GraphSlotRegistry, registration));
    return GraphSlotStatus::Ok;
}

GraphSlotStatus admit_graph_packet_for_restore(
    const void *packet, size_t bytes, int device_id, uint64_t runtime_binary_id,
    const simpler::kernel::PreparedInvocationView &trusted_callable, GraphRestoreView &out,
    const GraphPacketReadOps &ops
) noexcept {
    const GraphSlotRegistry *registry = current_registry.load(std::memory_order_acquire);
    if (registry == nullptr) return GraphSlotStatus::NotReady;
    GraphSlotRegistration registration{};
    const auto status = acquire_graph_execution_slot(registry, device_id, runtime_binary_id, registration);
    if (status != GraphSlotStatus::Ok) return status;
    if (packet == nullptr || bytes > registration.max_packet_bytes) return GraphSlotStatus::InvalidPacket;
    const GraphDestination source{reinterpret_cast<uintptr_t>(packet), bytes};
    if (!graph_span_fits(source.address, source.capacity, UINT64_MAX)) return GraphSlotStatus::InvalidPacket;
    if (graph_windows_overlap(source, registration.registry)) return GraphSlotStatus::SourceOverlap;
    for (const auto &destination : registration.destinations)
        if (graph_windows_overlap(source, destination)) return GraphSlotStatus::SourceOverlap;
    if (bytes < sizeof(SimplerKernelInvocationHeader) + sizeof(GraphPacketHeader))
        return GraphSlotStatus::InvalidPacket;
    if (ops.invalidate) {
        if (!ops.invalidate(ops.context, packet, bytes)) return GraphSlotStatus::SourceUnavailable;
    } else {
        cache_invalidate_range(packet, bytes);
    }
    SimplerKernelInvocationHeader invocation{};
    const auto admission = simpler::kernel::validate_invocation_header(
        {static_cast<const uint8_t *>(packet), bytes}, trusted_callable, &invocation
    );
    if (admission == simpler::kernel::InvocationStatus::StaleCallable) return GraphSlotStatus::CallableMismatch;
    if (admission != simpler::kernel::InvocationStatus::Ok) return GraphSlotStatus::InvalidPacket;
    if (validate_graph_packet(packet, bytes, GraphPacketAddress::DeviceCopy) != GraphPacketStatus::Ok)
        return GraphSlotStatus::InvalidPacket;
    GraphPacketHeader header{};
    std::memcpy(
        &header, static_cast<const std::byte *>(packet) + sizeof(SimplerKernelInvocationHeader), sizeof(header)
    );
    if (header.device_id != registration.device_id) return GraphSlotStatus::DeviceMismatch;
    if (header.slot_generation != registration.slot_generation) return GraphSlotStatus::GenerationMismatch;
    if (header.runtime_binary_id != registration.runtime_binary_id) return GraphSlotStatus::BinaryMismatch;
    for (size_t i = 0; i < 4; ++i)
        if (header.destinations[i].address != registration.destinations[i].address ||
            header.destinations[i].capacity != registration.destinations[i].capacity)
            return GraphSlotStatus::BindingMismatch;
    out = {
        invocation, registration, header,
        static_cast<const std::byte *>(packet) + sizeof(SimplerKernelInvocationHeader) + header.payload_offset
    };
    for (uint32_t i = 0; i < header.region_count; ++i) {
        GraphImageRegion region{};
        std::memcpy(
            &region,
            static_cast<const std::byte *>(packet) + sizeof(SimplerKernelInvocationHeader) + sizeof(header) +
                i * sizeof(region),
            sizeof(region)
        );
        out.regions[static_cast<uint32_t>(region.kind)] = region;
    }
    return GraphSlotStatus::Ok;
}

}  // namespace hbg

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_hbg_register_execution_slot(void *arg) {
    const auto status = hbg::install_graph_execution_slot(arg, sizeof(hbg::GraphSlotRegistration));
    return status == hbg::GraphSlotStatus::Ok ? 0 : -1;
}

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_hbg_detach_execution_slot(void *arg) {
    if (arg == nullptr) return -1;
    hbg::GraphSlotDetach detach{};
    std::memcpy(&detach, arg, sizeof(detach));
    if (!hbg::valid_graph_slot_detach(detach)) return -1;
    auto *registry = reinterpret_cast<hbg::GraphSlotRegistry *>(detach.registry_address);
    if (registry != hbg::current_graph_slot_registry()) return -1;
    cache_invalidate_range(registry, offsetof(hbg::GraphSlotRegistry, registration));
    if (registry->magic != hbg::GRAPH_REGISTRY_MAGIC || registry->version != hbg::GRAPH_SLOT_VERSION ||
        registry->bytes != sizeof(*registry) || registry->device_id != detach.device_id ||
        registry->context_generation != detach.context_generation ||
        registry->runtime_binary_id != detach.runtime_binary_id)
        return -1;
    return hbg::detach_graph_slot_registry(registry) ? 0 : -1;
}
