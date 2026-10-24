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
#include "host_build_graph/kernel_graph_slot_control.h"

#include <atomic>

#include "aicpu/cache_maintenance.h"
#include "common/unified_log.h"

namespace hbg {
namespace {
std::atomic<GraphSlotRegistry *> current_registry{nullptr};

const char *graph_slot_status_name(GraphSlotStatus status) noexcept {
    switch (status) {
    case GraphSlotStatus::Ok:
        return "Ok";
    case GraphSlotStatus::InvalidRegistration:
        return "InvalidRegistration";
    case GraphSlotStatus::InvalidRegistry:
        return "InvalidRegistry";
    case GraphSlotStatus::NotReady:
        return "NotReady";
    case GraphSlotStatus::Publishing:
        return "Publishing";
    case GraphSlotStatus::Conflict:
        return "Conflict";
    case GraphSlotStatus::DeviceMismatch:
        return "DeviceMismatch";
    case GraphSlotStatus::GenerationMismatch:
        return "GenerationMismatch";
    case GraphSlotStatus::BinaryMismatch:
        return "BinaryMismatch";
    case GraphSlotStatus::InvalidPacket:
        return "InvalidPacket";
    case GraphSlotStatus::BindingMismatch:
        return "BindingMismatch";
    case GraphSlotStatus::SourceOverlap:
        return "SourceOverlap";
    case GraphSlotStatus::CallableMismatch:
        return "CallableMismatch";
    case GraphSlotStatus::Poisoned:
        return "Poisoned";
    case GraphSlotStatus::SourceUnavailable:
        return "SourceUnavailable";
    }
    return "Unknown";
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
    // Detachment does not retire the context identity or clear terminal failure.
    // A fresh allocation may reuse an address, but its owner must use a new generation.
    if (valid_registry(registry) && registry->context_generation == generation) {
        const auto phase = __atomic_load_n(&registry->phase, __ATOMIC_ACQUIRE);
        if (phase == static_cast<uint32_t>(GraphSlotRegistrationPhase::Poisoned)) return GraphSlotStatus::Poisoned;
        return GraphSlotStatus::Conflict;
    }
    GraphSlotRegistry initial{};
    initial.magic = GRAPH_REGISTRY_MAGIC;
    initial.version = GRAPH_SLOT_VERSION;
    initial.bytes = sizeof(initial);
    initial.phase = static_cast<uint32_t>(GraphSlotRegistrationPhase::Empty);
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
    if (!is_valid_graph_slot_registration(candidate)) return GraphSlotStatus::InvalidRegistration;
    if (!valid_registry(registry)) return GraphSlotStatus::InvalidRegistry;
    if (!matches_registry(*registry, candidate)) return GraphSlotStatus::Conflict;
    for (;;) {
        uint32_t phase = __atomic_load_n(&registry->phase, __ATOMIC_ACQUIRE);
        if (phase == static_cast<uint32_t>(GraphSlotRegistrationPhase::Poisoned)) return GraphSlotStatus::Poisoned;
        if (phase == static_cast<uint32_t>(GraphSlotRegistrationPhase::Ready)) {
            cache_invalidate_range(&registry->registration, sizeof(registry->registration));
            return std::memcmp(&registry->registration, &candidate, sizeof(candidate)) == 0 ? GraphSlotStatus::Ok :
                                                                                              GraphSlotStatus::Conflict;
        }
        if (phase == static_cast<uint32_t>(GraphSlotRegistrationPhase::Publishing)) return GraphSlotStatus::Publishing;
        if (phase != static_cast<uint32_t>(GraphSlotRegistrationPhase::Empty)) return GraphSlotStatus::InvalidRegistry;
        if (!__atomic_compare_exchange_n(
                &registry->phase, &phase, static_cast<uint32_t>(GraphSlotRegistrationPhase::Publishing), false,
                __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
            ))
            continue;
        registry->registration = candidate;
        cache_flush_range(&registry->registration, sizeof(registry->registration));
        __atomic_store_n(&registry->phase, static_cast<uint32_t>(GraphSlotRegistrationPhase::Ready), __ATOMIC_RELEASE);
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
    if (!is_valid_graph_slot_registration(candidate)) return GraphSlotStatus::InvalidRegistration;

    auto *registry = reinterpret_cast<GraphSlotRegistry *>(candidate.registry.address);
    const auto *bound = current_registry.load(std::memory_order_acquire);
    if (bound != nullptr && bound != registry) return GraphSlotStatus::Conflict;
    // An unbound registry can still belong to this context. Preserve its execution
    // counters and poison state instead of treating absence of a binding as freshness.
    if (bound != nullptr || (valid_registry(registry) && registry->context_generation == candidate.slot_generation)) {
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
    if (phase == static_cast<uint32_t>(GraphSlotRegistrationPhase::Poisoned)) return GraphSlotStatus::Poisoned;
    if (phase == static_cast<uint32_t>(GraphSlotRegistrationPhase::Empty)) return GraphSlotStatus::NotReady;
    if (phase == static_cast<uint32_t>(GraphSlotRegistrationPhase::Publishing)) return GraphSlotStatus::Publishing;
    if (phase != static_cast<uint32_t>(GraphSlotRegistrationPhase::Ready)) return GraphSlotStatus::InvalidRegistry;
    cache_invalidate_range(&registry->registration, sizeof(registry->registration));
    const GraphSlotRegistration candidate = registry->registration;
    if (!is_valid_graph_slot_registration(candidate) || !matches_registry(*registry, candidate))
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
    return current_registry.compare_exchange_strong(registry, nullptr, std::memory_order_acq_rel);
}

GraphSlotRegistry *current_graph_slot_registry() noexcept { return current_registry.load(std::memory_order_acquire); }

GraphSlotStatus poison_graph_execution_slot(GraphSlotRegistry *registry) noexcept {
    if (!valid_registry(registry)) return GraphSlotStatus::InvalidRegistry;
    __atomic_store_n(&registry->phase, static_cast<uint32_t>(GraphSlotRegistrationPhase::Poisoned), __ATOMIC_RELEASE);
    cache_flush_range(registry, offsetof(GraphSlotRegistry, registration));
    return GraphSlotStatus::Ok;
}

}  // namespace hbg

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_hbg_register_execution_slot(void *arg) {
    const auto status = hbg::install_graph_execution_slot(arg, sizeof(hbg::GraphSlotRegistration));
    if (status != hbg::GraphSlotStatus::Ok)
        LOG_ERROR(
            "HBG execution slot registration failed: %s (%u)", hbg::graph_slot_status_name(status),
            static_cast<unsigned>(status)
        );
    return status == hbg::GraphSlotStatus::Ok ? 0 : -1;
}

namespace {
hbg::GraphSlotDetachRequest last_detached{};
}  // namespace

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_hbg_detach_execution_slot(void *arg) {
    if (arg == nullptr) {
        LOG_ERROR("%s", "HBG execution slot detach failed: null control record");
        return -1;
    }
    hbg::GraphSlotDetachRequest detach{};
    std::memcpy(&detach, arg, sizeof(detach));
    if (!hbg::is_valid_graph_slot_detach_request(detach)) {
        LOG_ERROR("%s", "HBG execution slot detach failed: invalid control record");
        return -1;
    }
    auto *registry = reinterpret_cast<hbg::GraphSlotRegistry *>(detach.registry_address);
    auto *current = hbg::current_graph_slot_registry();
    if (current == nullptr) {
        cache_invalidate_range(&last_detached, sizeof(last_detached));
        return std::memcmp(&last_detached, &detach, sizeof(detach)) == 0 ? 0 : -1;
    }
    if (registry != current) {
        LOG_ERROR("%s", "HBG execution slot detach failed: registry is not currently bound");
        return -1;
    }
    cache_invalidate_range(registry, offsetof(hbg::GraphSlotRegistry, registration));
    if (registry->magic != hbg::GRAPH_REGISTRY_MAGIC || registry->version != hbg::GRAPH_SLOT_VERSION ||
        registry->bytes != sizeof(*registry) || registry->device_id != detach.device_id ||
        registry->context_generation != detach.context_generation ||
        registry->runtime_binary_id != detach.runtime_binary_id) {
        LOG_ERROR("%s", "HBG execution slot detach failed: registry identity mismatch");
        return -1;
    }
    if (!hbg::detach_graph_slot_registry(registry)) {
        LOG_ERROR("%s", "HBG execution slot detach failed: binding changed during control operation");
        return -1;
    }
    last_detached = detach;
    cache_flush_range(&last_detached, sizeof(last_detached));
    return 0;
}
