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
#include "host_build_graph/kernel_graph_execution_entry.h"

#include <atomic>
#include <cstring>

#include "aicpu/cache_maintenance.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/platform_entry_args.h"
#include "common/launch_entry_args.h"
#include "common/unified_log.h"
#include "host_build_graph/kernel_callable_registration.h"
#include "host_build_graph/kernel_graph_execution.h"
#include "host_build_graph/kernel_graph_execution_validation.h"
#include "host_build_graph/kernel_graph_packet_reader.h"
#include "host_build_graph/kernel_graph_slot_control.h"
#include "runtime.h"

namespace {
struct alignas(64) RegisteredCallable {
    std::atomic<uint64_t> generation{0};
    alignas(64) simpler::kernel::HbgCallableBinding value{};
};
RegisteredCallable callables[MAX_REGISTERED_CALLABLE_IDS];
std::atomic<uint64_t> restore_attempt{0};

bool valid(const simpler::kernel::HbgCallableBinding &r) {
    auto *registry = hbg::current_graph_slot_registry();
    return registry != nullptr && r.runtime_address != 0 && r.context_generation == registry->context_generation &&
           r.callable_generation != 0 && r.callable_id >= 0 && r.callable_id < MAX_REGISTERED_CALLABLE_IDS &&
           r.reserved == 0 && simpler::kernel::valid_invocation_counts(r.tensor_count, r.scalar_count);
}
}  // namespace

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_hbg_register_callable(void *args) {
    if (args == nullptr) return -1;
    simpler::kernel::HbgCallableBinding record{};
    std::memcpy(&record, args, sizeof(record));
    if (!valid(record)) return -1;
    auto &slot = callables[record.callable_id];
    cache_invalidate_range(&slot, sizeof(slot));
    if (slot.generation.load(std::memory_order_acquire) != 0)
        return std::memcmp(&slot.value, &record, sizeof(record)) == 0 ? 0 : -1;
    slot.value = record;
    cache_flush_range(&slot.value, sizeof(slot.value));
    slot.generation.store(record.callable_generation, std::memory_order_release);
    cache_flush_range(&slot.generation, sizeof(slot.generation));
    return 0;
}

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_hbg_revoke_callable(void *args) {
    if (args == nullptr) return -1;
    simpler::kernel::HbgCallableBinding record{};
    std::memcpy(&record, args, sizeof(record));
    if (!valid(record)) return -1;
    auto &slot = callables[record.callable_id];
    cache_invalidate_range(&slot, sizeof(slot));
    const uint64_t generation = slot.generation.load(std::memory_order_acquire);
    if (std::memcmp(&slot.value, &record, sizeof(record)) != 0) return -1;
    if (generation == 0) return 0;
    if (generation != record.callable_generation) return -1;
    slot.generation.store(0, std::memory_order_release);
    cache_flush_range(&slot.generation, sizeof(slot.generation));
    return 0;
}

int hbg::restore_kernel_graph(Runtime *runtime) noexcept {
    const auto args = get_platform_entry_args(platform_aicpu_affinity_thread_idx());
    if (args.source != static_cast<uint32_t>(EntryArgsSource::HbgKernelInvocation)) return 0;
    restore_attempt.store(0, std::memory_order_relaxed);
    auto *registry = current_graph_slot_registry();
    if (registry == nullptr || args.args_base == nullptr || args.offset != LAUNCH_ENVELOPE_HEADER_BYTES ||
        args.graph_offset != LAUNCH_ENVELOPE_HEADER_BYTES ||
        args.graph_source != static_cast<uint32_t>(GraphSectionSource::RestoredSlot) ||
        args.graph_bytes < sizeof(SimplerKernelInvocationHeader) + sizeof(GraphPacketHeader)) {
        LOG_ERROR("%s", "HBG kernel launch framing or slot is unavailable");
        return -1;
    }
    SimplerKernelInvocationHeader invocation{};
    std::memcpy(&invocation, static_cast<const std::byte *>(args.args_base) + args.offset, sizeof(invocation));
    // The packet id only selects a candidate. The cold record supplies the
    // authoritative Runtime address, generation and counts. Runtime's callable
    // id belongs to its host-only tail and must never be read on device.
    const int32_t id = invocation.callable_id;
    if (id < 0 || id >= MAX_REGISTERED_CALLABLE_IDS) return -1;
    auto &slot = callables[id];
    cache_invalidate_range(&slot, sizeof(slot));
    const uint64_t generation = slot.generation.load(std::memory_order_acquire);
    const auto record = slot.value;
    if (generation == 0 || record.callable_generation != generation ||
        record.runtime_address != reinterpret_cast<uint64_t>(runtime) || !valid(record)) {
        LOG_ERROR(
            "HBG kernel callable identity rejected: cid=%d generation=%llu", id,
            static_cast<unsigned long long>(generation)
        );
        return -1;
    }

    GraphSlotRegistration registration{};
    ParsedGraphPacket parsed{};
    const auto read_status = read_graph_execution_packet(
        static_cast<const std::byte *>(args.args_base) + args.offset, args.graph_bytes, registry->device_id,
        registry->runtime_binary_id, registration, parsed, {}, GraphPacketAddress::RelativeLaunch
    );
    if (read_status != GraphSlotStatus::Ok) {
        LOG_ERROR(
            "HBG kernel packet read rejected: status=%u packet_bytes=%u", static_cast<unsigned>(read_status),
            args.graph_bytes
        );
        return -1;
    }

    const simpler::kernel::PreparedInvocationView identity{id, record.tensor_count, record.scalar_count, generation};
    ValidatedGraphExecutionBinding binding{};
    const auto binding_status = validate_graph_execution_binding(parsed, registration, identity, binding);
    if (binding_status != GraphSlotStatus::Ok) {
        LOG_ERROR("HBG kernel packet binding rejected: status=%u", static_cast<unsigned>(binding_status));
        return -1;
    }

    cache_invalidate_range(&registry->execution, sizeof(registry->execution));
    const uint64_t attempt_before = __atomic_load_n(&registry->execution.attempt, __ATOMIC_ACQUIRE);
    GraphExecutionPreparation result{};
    const auto status = prepare_graph_execution(binding, result);
    cache_invalidate_range(&registry->execution, sizeof(registry->execution));
    const uint64_t attempt_after = __atomic_load_n(&registry->execution.attempt, __ATOMIC_ACQUIRE);
    // Only an admitted preparation can own a quarantined attempt. A rejection
    // before the Idle -> Preparing claim leaves retirement with nothing to do.
    restore_attempt.store(attempt_after != attempt_before ? attempt_after : 0, std::memory_order_release);
    if (status != GraphExecutionStatus::Ok) {
        LOG_ERROR(
            "HBG kernel execution preparation rejected: status=%u packet_bytes=%u attempt=%llu",
            static_cast<unsigned>(status), args.graph_bytes,
            static_cast<unsigned long long>(registry->execution.attempt)
        );
        return -1;
    }
    runtime->dev.host_total_tasks = result.total_tasks;
    runtime->dev.sm_image_bytes = result.sm_bytes;
    runtime->publish_graph_section(
        registration.destinations[2].address == 0 ? GraphSectionSource::None : GraphSectionSource::RestoredSlot,
        registration.destinations[2].address, static_cast<uint32_t>(registry->execution.live_bytes[2])
    );
    // The handshake region remains device-owned while cores are active.
    cache_flush_range(&runtime->dev, Runtime::device_image_bytes());
    return 0;
}

int hbg::retire_kernel_graph(Runtime *, int execution_status) noexcept {
    const auto args = get_platform_entry_args(platform_aicpu_affinity_thread_idx());
    if (args.source != static_cast<uint32_t>(EntryArgsSource::HbgKernelInvocation)) return 0;
    auto *registry = current_graph_slot_registry();
    const uint64_t attempt = restore_attempt.load(std::memory_order_acquire);
    if (registry == nullptr || attempt == 0) return execution_status;
    const GraphExecutionCompletion completion{
        execution_status == 0 ? GraphExecutionRetirement::Completed : GraphExecutionRetirement::FatalFailure,
        execution_status, 0
    };
    return retire_graph_execution(registry, attempt, completion) == GraphExecutionStatus::Ok ? 0 : -1;
}
