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
#if defined(SIMPLER_HBG_KERNEL_MODE)

#include "host_build_graph/kernel_callable_registration.h"
#include "host_build_graph/kernel_callable_binding.h"

#include <atomic>
#include <cstring>

#include "aicpu/kernel_invocation_consumer.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/cache_maintenance.h"
#include "aicpu/platform_regs.h"
#include "callable.h"
#include "common/unified_log.h"
#include "host_build_graph/kernel_graph_restore.h"
#include "host_build_graph/kernel_graph_dispatch.h"
#include "host_build_graph/kernel_graph_prepared.h"
#include "host_build_graph/kernel_graph_slot_registry.h"
#include "host_build_graph/kernel_graph_wire.h"
#include "host_build_graph/shared_memory.h"
#include "runtime.h"

int32_t aicpu_execute_restored_graph(Runtime *runtime);

namespace {

struct RegisteredCallable {
    std::atomic<uint64_t> generation{0};
    hbg::HbgCallableRegistration value{};
};

RegisteredCallable g_callables[MAX_REGISTERED_CALLABLE_IDS];
std::atomic<uint64_t> g_restore_epoch{0};
std::atomic<int32_t> g_restore_status{0};
std::atomic<int32_t> g_entered{0};
std::atomic<int32_t> g_finished{0};
std::atomic<int32_t> g_execution_status{0};
std::atomic<uint64_t> g_restore_attempt{0};

struct PreparedExecutionContext {
    const hbg::GraphSlotRegistry *registry{nullptr};
    hbg::GraphSlotRegistration slot{};
    Runtime *runtime{nullptr};
    uint64_t generation{0};
    int32_t threads{0};
    int32_t allowed_count{0};
    int32_t launch_count{0};
    int32_t allowed_cpus[MAX_GATE_THREADS]{};
};

PreparedExecutionContext g_prepared_context;
std::atomic<uint64_t> g_prepared_context_generation{0};

// These records stay outside every reset-DMA destination. A publication is
// immutable until all execution roles finish the corresponding cache phase.
hbg::GraphPreparedDispatch g_prepared_dispatch{};
std::atomic<uint64_t> g_prepared_start_epoch{0};
std::atomic<uint64_t> g_prepared_metadata_epoch{0};
std::atomic<int32_t> g_prepared_metadata_status{0};
std::atomic<int32_t> g_prepared_checked{0};
std::atomic<int32_t> g_prepared_cleaned{0};
std::atomic<int32_t> g_prepared_result{0};

bool load_callable(int32_t id, uint64_t generation, hbg::HbgCallableRegistration &out) noexcept {
    if (id < 0 || id >= MAX_REGISTERED_CALLABLE_IDS) return false;
    auto &slot = g_callables[id];
    if (slot.generation.load(std::memory_order_acquire) != generation) return false;
    cache_invalidate_range(&slot.value, sizeof(slot.value));
    std::memcpy(&out, &slot.value, sizeof(out));
    return slot.generation.load(std::memory_order_acquire) == generation && out.context_generation == generation &&
           hbg::valid_hbg_callable_registration(out);
}

hbg::KernelCallableBinding g_callable_binding;

bool prepared_context_is_current() {
    const uint64_t generation = g_prepared_context_generation.load(std::memory_order_acquire);
    const auto *registry = hbg::current_graph_slot_registry();
    if (generation == 0 || registry == nullptr || registry != g_prepared_context.registry) return false;
    cache_invalidate_range(registry, offsetof(hbg::GraphSlotRegistry, registration));
    return generation == registry->context_generation && generation == g_prepared_context.generation;
}

void maintain_prepared_image(const hbg::GraphPreparedDispatch &dispatch, uint64_t runtime_state_bytes = UINT64_MAX) {
    for (uint32_t i = 0; i < dispatch.graph.region_count; ++i) {
        const auto &region = dispatch.regions[i];
        const auto kind = static_cast<uint32_t>(region.kind);
        cache_invalidate_range(
            reinterpret_cast<const void *>(dispatch.graph.destinations[kind].address + region.destination_offset),
            static_cast<size_t>(
                region.kind == hbg::GraphImageKind::Runtime && runtime_state_bytes < region.bytes ?
                    runtime_state_bytes :
                    region.bytes
            )
        );
    }
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_hbg_register_callable(void *arg) {
    if (arg == nullptr) return -1;
    hbg::HbgCallableRegistration registration{};
    std::memcpy(&registration, arg, sizeof(registration));
    if (!hbg::valid_hbg_callable_registration(registration)) return -1;
    auto &slot = g_callables[registration.callable_id];
    const uint64_t old = slot.generation.load(std::memory_order_acquire);
    if (old == registration.context_generation)
        return std::memcmp(&slot.value, &registration, sizeof(registration)) == 0 ? 0 : -1;
    std::memcpy(&slot.value, &registration, sizeof(registration));
    cache_flush_range(&slot.value, sizeof(slot.value));
    // The context owns this table. The invocation leader publishes it before
    // graph execution; identical replays retain the published binding.
    set_platform_regs(registration.register_table_address);
    slot.generation.store(registration.context_generation, std::memory_order_release);
    return 0;
}

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_hbg_prepare_image_destinations(void *arg) {
    const auto *registry = hbg::current_graph_slot_registry();
    if (arg == nullptr || registry == nullptr) return -1;
    cache_invalidate_range(registry, offsetof(hbg::GraphSlotRegistry, registration));
    hbg::GraphSlotRegistration slot{};
    if (hbg::acquire_graph_execution_slot(registry, registry->device_id, registry->runtime_binary_id, slot) !=
            hbg::GraphSlotStatus::Ok ||
        !hbg::graph_dispatch_source_fits(arg, sizeof(slot), slot))
        return -1;
    cache_invalidate_range(arg, sizeof(slot));
    hbg::GraphSlotRegistration requested{};
    std::memcpy(&requested, arg, sizeof(requested));
    if (!hbg::valid_graph_slot_registration(requested) || std::memcmp(&requested, &slot, sizeof(slot)) != 0) return -1;
    // The Host serializes this one-time control with the first image DMA. It
    // never resets a slot that has admitted an execution or a failed attempt.
    cache_invalidate_range(&registry->restore, sizeof(registry->restore));
    if (__atomic_load_n(&registry->restore.phase, __ATOMIC_ACQUIRE) !=
            static_cast<uint32_t>(hbg::GraphRestorePhase::Idle) ||
        registry->restore.attempt != 0 || registry->restore.committed_generation != 0)
        return -1;
    prepare_kernel_aicpu_thread();
    for (size_t i = 1; i < 4; ++i)
        if (slot.destinations[i].capacity != 0)
            cache_invalidate_range(
                reinterpret_cast<const void *>(slot.destinations[i].address),
                static_cast<size_t>(slot.destinations[i].capacity)
            );
    return 0;
}

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_hbg_prepare_execution_context(void *arg) {
    const auto *registry = hbg::current_graph_slot_registry();
    if (arg == nullptr || registry == nullptr || g_entered.load(std::memory_order_acquire) != 0 ||
        g_finished.load(std::memory_order_acquire) != 0)
        return -1;
    cache_invalidate_range(registry, offsetof(hbg::GraphSlotRegistry, registration));
    hbg::GraphSlotRegistration slot{};
    if (hbg::acquire_graph_execution_slot(registry, registry->device_id, registry->runtime_binary_id, slot) !=
            hbg::GraphSlotStatus::Ok ||
        !hbg::graph_dispatch_source_fits(arg, sizeof(hbg::GraphPreparedContext), slot))
        return -1;
    cache_invalidate_range(arg, sizeof(hbg::GraphPreparedContext));
    hbg::GraphPreparedContext requested{};
    std::memcpy(&requested, arg, sizeof(requested));
    if (std::memcmp(&requested.slot, &slot, sizeof(slot)) != 0 || requested.runtime_address == 0 ||
        requested.runtime_address % alignof(Runtime) != 0 ||
        !hbg::graph_dispatch_source_fits(
            reinterpret_cast<const void *>(requested.runtime_address), sizeof(Runtime), slot
        ))
        return -1;
    auto *runtime = reinterpret_cast<Runtime *>(requested.runtime_address);
    cache_invalidate_range(runtime, sizeof(*runtime));
    PreparedExecutionContext context{};
    context.registry = registry;
    context.slot = slot;
    context.runtime = runtime;
    context.generation = slot.slot_generation;
    context.threads = runtime->get_aicpu_thread_num();
    context.allowed_count = runtime->get_aicpu_allowed_cpu_count();
    context.launch_count = runtime->get_aicpu_launch_count();
    if (context.threads < 1 || context.threads > PLATFORM_MAX_AICPU_THREADS ||
        context.allowed_count != context.threads || context.allowed_count > MAX_GATE_THREADS ||
        context.launch_count < context.threads || context.launch_count > PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH)
        return -1;
    for (int32_t i = 0; i < context.allowed_count; ++i) {
        context.allowed_cpus[i] = runtime->get_aicpu_allowed_cpus()[i];
        if (context.allowed_cpus[i] < 0) return -1;
        for (int32_t j = 0; j < i; ++j)
            if (context.allowed_cpus[j] == context.allowed_cpus[i]) return -1;
    }
    const uint64_t old_generation = g_prepared_context_generation.load(std::memory_order_acquire);
    if (old_generation == context.generation && g_prepared_context.registry == registry) {
        const auto &old = g_prepared_context;
        return old.runtime == context.runtime && std::memcmp(&old.slot, &context.slot, sizeof(slot)) == 0 &&
                       old.threads == context.threads && old.allowed_count == context.allowed_count &&
                       old.launch_count == context.launch_count &&
                       std::memcmp(old.allowed_cpus, context.allowed_cpus, sizeof(context.allowed_cpus)) == 0 ?
                   0 :
                   -1;
    }
    // Registration and detach are serialized Host control operations. A new
    // registry generation replaces the old context's borrowed addresses.
    g_prepared_context_generation.store(0, std::memory_order_release);
    g_prepared_context = context;
    g_prepared_context_generation.store(context.generation, std::memory_order_release);
    return 0;
}

// HBG owns the whole HostArgs packet, so it overrides the generic dispatcher
// which expects the fixed TMR envelope.
static int consume_prepared_graph_task(void *arg);

int consume_kernel_task(void *arg) {
    // The prepare control supplies the transport mode independently of the
    // invocation bytes, including when one thread receives a corrupt header.
    if (prepared_context_is_current()) return consume_prepared_graph_task(arg);
    if (arg == nullptr) {
        LOG_ERROR("%s", "HBG kernel invocation has null HostArgs");
        return -1;
    }
    // The independently bound context owns these fields; the canonical inner
    // packet cannot choose a registry or provide callable authority.
    const auto *registry = hbg::current_graph_slot_registry();
    if (registry == nullptr) return -1;
    cache_invalidate_range(registry, offsetof(hbg::GraphSlotRegistry, registration));
    hbg::GraphSlotRegistration slot{};
    if (hbg::acquire_graph_execution_slot(registry, registry->device_id, registry->runtime_binary_id, slot) !=
            hbg::GraphSlotStatus::Ok ||
        !hbg::graph_dispatch_source_fits(arg, sizeof(hbg::GraphDispatchHeader), slot))
        return -1;
    cache_invalidate_range(arg, sizeof(hbg::GraphDispatchHeader));
    uint32_t dispatch_magic = 0;
    std::memcpy(
        &dispatch_magic, static_cast<const uint8_t *>(arg) + offsetof(hbg::GraphDispatchHeader, magic),
        sizeof(dispatch_magic)
    );
    if (dispatch_magic == hbg::GRAPH_PREPARED_DISPATCH_MAGIC) return consume_prepared_graph_task(arg);
    hbg::GraphDispatchHeader dispatch{};
    if (!hbg::read_graph_dispatch_header(arg, sizeof(dispatch), dispatch) ||
        !hbg::graph_dispatch_matches_slot(arg, dispatch, slot))
        return -1;
    const auto registration = dispatch.registration;
    // A generation's existing authority is immutable. Do this comparison before
    // reading Runtime, including when the same callable ID has already replayed.
    if (g_callables[registration.callable_id].generation.load(std::memory_order_acquire) ==
        registration.context_generation) {
        hbg::HbgCallableRegistration existing{};
        if (!load_callable(registration.callable_id, registration.context_generation, existing) ||
            std::memcmp(&existing, &registration, sizeof(existing)) != 0)
            return -1;
    }
    auto *packet = static_cast<uint8_t *>(arg) + sizeof(dispatch);
    cache_invalidate_range(packet, sizeof(SimplerKernelInvocationHeader) + sizeof(hbg::GraphPacketHeader));
    SimplerKernelInvocationHeader invocation{};
    hbg::GraphPacketHeader graph{};
    std::memcpy(&invocation, packet, sizeof(invocation));
    std::memcpy(&graph, packet + sizeof(invocation), sizeof(graph));
    if (!hbg::graph_dispatch_matches_packet(dispatch, invocation, graph)) return -1;
    const size_t packet_bytes = static_cast<size_t>(dispatch.packet_bytes);
    auto *runtime = reinterpret_cast<Runtime *>(registration.runtime_address);
    if (runtime == nullptr) {
        LOG_ERROR("%s", "HBG callable registration has null Runtime address");
        return -1;
    }
    cache_invalidate_range(runtime, sizeof(*runtime));

    prepare_kernel_aicpu_thread();
    if (!platform_aicpu_affinity_gate_filter(
            runtime->get_aicpu_allowed_cpus(), runtime->get_aicpu_allowed_cpu_count(), runtime->get_aicpu_launch_count()
        ))
        return 0;
    const int32_t thread = platform_aicpu_affinity_thread_idx();
    const int32_t threads = runtime->get_aicpu_thread_num();
    if (thread < 0 || thread >= threads) {
        LOG_ERROR("HBG AICPU thread index %d is outside [0,%d)", thread, threads);
        return -1;
    }
    const uint64_t epoch = g_restore_epoch.load(std::memory_order_acquire);
    g_entered.fetch_add(1, std::memory_order_acq_rel);

    if (thread == 0) {
        while (g_entered.load(std::memory_order_acquire) != threads) {}
        hbg::GraphRestoreResult restored{};
        auto published = registration;
        const int registered = simpler_aicpu_l1_hbg_register_callable(&published);
        const auto status =
            registered == 0 && get_platform_regs() == registration.register_table_address &&
                    hbg::refresh_graph_kernel_dfx(registry, runtime, registration) ?
                hbg::restore_graph_packet(
                    packet, packet_bytes, registration.device_id, registration.runtime_binary_id,
                    {registration.callable_id, registration.tensor_count, registration.scalar_count}, restored
                ) :
                hbg::GraphRestoreStatus::Rejected;
        const auto binding = status == hbg::GraphRestoreStatus::Ok ? g_callable_binding.bind(runtime, registration) :
                                                                     hbg::CallableBindingResult::Rejected;
        const bool callable_bound = binding != hbg::CallableBindingResult::Rejected;
        int rc = callable_bound ? 0 : -1;
        if (rc != 0) {
            LOG_ERROR(
                "HBG graph restore failed: status=%u callable_bound=%d generation=%llu", static_cast<unsigned>(status),
                callable_bound ? 1 : 0, static_cast<unsigned long long>(restored.generation)
            );
        }
        if (rc == 0) {
            runtime->set_gm_sm_ptr(reinterpret_cast<void *>(graph.destinations[1].address + graph.sm_offset));
            runtime->set_prebuilt_arena(reinterpret_cast<void *>(graph.destinations[1].address), graph.runtime_offset);
            runtime->host_total_tasks = static_cast<int32_t>(restored.total_tasks);
            runtime->sm_image_bytes = restored.sm_bytes;

            for (const auto &span :
                 runtime->kernel_graph_publication_spans(binding == hbg::CallableBindingResult::Rebound))
                cache_flush_range(span.address, span.bytes);

            // READY is the release publication for the AICore prelaunch gate.
            // Flush it last so every control field above is visible first.
            runtime->kernel_prelaunch.state = HBG_KERNEL_PRELAUNCH_READY;
            cache_flush_range(&runtime->kernel_prelaunch, sizeof(runtime->kernel_prelaunch));
        } else {
            runtime->kernel_prelaunch.state = HBG_KERNEL_PRELAUNCH_CANCEL;
            cache_flush_range(&runtime->kernel_prelaunch, sizeof(runtime->kernel_prelaunch));
        }
        g_restore_attempt.store(restored.generation, std::memory_order_relaxed);
        g_restore_status.store(rc, std::memory_order_relaxed);
        g_restore_epoch.store(epoch + 1, std::memory_order_release);
    } else {
        while (g_restore_epoch.load(std::memory_order_acquire) == epoch) {}
    }
    int rc = g_restore_status.load(std::memory_order_acquire);
    if (rc == 0) rc = aicpu_execute_restored_graph(runtime);
    if (rc != 0) LOG_ERROR("HBG graph execution failed on thread %d: rc=%d", thread, rc);
    if (rc != 0) {
        int32_t expected = 0;
        g_execution_status.compare_exchange_strong(expected, rc, std::memory_order_acq_rel);
    }
    const int32_t finished = g_finished.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (finished == threads) {
        auto *registry = hbg::current_graph_slot_registry();
        rc = g_execution_status.load(std::memory_order_acquire);
        const hbg::GraphRestoreCompletion completion{
            rc == 0 ? hbg::GraphRestoreRetirement::Completed : hbg::GraphRestoreRetirement::FatalFailure,
            rc,
            0,
        };
        if (hbg::retire_graph_restore(registry, g_restore_attempt.load(std::memory_order_relaxed), completion) !=
            hbg::GraphRestoreStatus::Ok)
            rc = -1;
        g_finished.store(0, std::memory_order_relaxed);
        g_execution_status.store(0, std::memory_order_relaxed);
        g_entered.store(0, std::memory_order_release);
    } else {
        while (g_entered.load(std::memory_order_acquire) != 0) {}
    }
    return rc;
}

static int consume_prepared_graph_task(void *arg) {
    if (!prepared_context_is_current()) return -1;
    const auto context = g_prepared_context;
    const auto *registry = context.registry;
    auto *runtime = context.runtime;
    prepare_kernel_aicpu_thread();
    if (!platform_aicpu_affinity_gate_filter(context.allowed_cpus, context.allowed_count, context.launch_count))
        return 0;
    const int32_t thread = platform_aicpu_affinity_thread_idx();
    const int32_t threads = context.threads;
    if (thread < 0 || thread >= threads) return -1;
    const auto handshake = runtime->kernel_graph_handshake_span();
    cache_invalidate_range(handshake.address, handshake.bytes);
    const uint64_t epoch = g_restore_epoch.load(std::memory_order_acquire);
    if (thread == 0) {
        g_prepared_metadata_status.store(0, std::memory_order_relaxed);
        g_prepared_start_epoch.store(epoch + 1, std::memory_order_release);
    } else {
        while (g_prepared_start_epoch.load(std::memory_order_acquire) != epoch + 1) {}
    }
    hbg::GraphSlotRegistration slot{};
    bool metadata_ok =
        hbg::acquire_graph_execution_slot(registry, context.slot.device_id, context.slot.runtime_binary_id, slot) ==
            hbg::GraphSlotStatus::Ok &&
        std::memcmp(&slot, &context.slot, sizeof(slot)) == 0 &&
        hbg::graph_dispatch_source_fits(arg, sizeof(hbg::GraphPreparedDispatch), context.slot);
    hbg::GraphPreparedDispatch local{};
    if (metadata_ok) {
        cache_invalidate_range(arg, sizeof(local));
        std::memcpy(&local, arg, sizeof(local));
        metadata_ok = hbg::graph_prepared_dispatch_matches_slot(local, slot) &&
                      local.registration.runtime_address == reinterpret_cast<uintptr_t>(runtime);
    }
    if (metadata_ok) {
        const auto &registration = local.registration;
        if (g_callables[registration.callable_id].generation.load(std::memory_order_acquire) ==
            registration.context_generation) {
            hbg::HbgCallableRegistration existing{};
            metadata_ok = load_callable(registration.callable_id, registration.context_generation, existing) &&
                          std::memcmp(&existing, &registration, sizeof(existing)) == 0;
        }
    }
    if (thread == 0) {
        g_prepared_dispatch = local;
        g_prepared_metadata_epoch.store(epoch + 1, std::memory_order_release);
    } else {
        while (g_prepared_metadata_epoch.load(std::memory_order_acquire) != epoch + 1) {}
    }
    // Each CANN thread owns a different HostArgs copy. Compare their values,
    // then use the leader's immutable record for every reader's maintenance.
    if (!metadata_ok || std::memcmp(&local, &g_prepared_dispatch, sizeof(local)) != 0)
        g_prepared_metadata_status.store(-1, std::memory_order_relaxed);
    g_prepared_checked.fetch_add(1, std::memory_order_acq_rel);
    while (g_prepared_checked.load(std::memory_order_acquire) != threads) {}
    if (g_prepared_metadata_status.load(std::memory_order_acquire) == 0) maintain_prepared_image(g_prepared_dispatch);
    g_entered.fetch_add(1, std::memory_order_acq_rel);

    if (thread == 0) {
        while (g_entered.load(std::memory_order_acquire) != threads) {}
        hbg::GraphRestoreResult restored{};
        const auto &registration = g_prepared_dispatch.registration;
        auto published = registration;
        hbg::GraphRestoreStatus status = hbg::GraphRestoreStatus::Rejected;
        if (g_prepared_metadata_status.load(std::memory_order_acquire) == 0 &&
            simpler_aicpu_l1_hbg_register_callable(&published) == 0 &&
            get_platform_regs() == registration.register_table_address &&
            hbg::refresh_graph_kernel_dfx(registry, runtime, registration))
            status = hbg::attach_prepared_graph_image(g_prepared_dispatch, slot, restored);
        const auto binding = status == hbg::GraphRestoreStatus::Ok ? g_callable_binding.bind(runtime, registration) :
                                                                     hbg::CallableBindingResult::Rejected;
        const int rc = binding == hbg::CallableBindingResult::Rejected ? -1 : 0;
        if (rc == 0) {
            const auto &graph = g_prepared_dispatch.graph;
            runtime->set_gm_sm_ptr(reinterpret_cast<void *>(graph.destinations[1].address + graph.sm_offset));
            runtime->set_prebuilt_arena(reinterpret_cast<void *>(graph.destinations[1].address), graph.runtime_offset);
            runtime->host_total_tasks = static_cast<int32_t>(restored.total_tasks);
            runtime->sm_image_bytes = restored.sm_bytes;
            for (const auto &span :
                 runtime->kernel_graph_publication_spans(binding == hbg::CallableBindingResult::Rebound))
                cache_flush_range(span.address, span.bytes);
        } else {
            LOG_ERROR("HBG prepared graph attach failed: status=%u", static_cast<unsigned>(status));
        }
        runtime->kernel_prelaunch.state = rc == 0 ? HBG_KERNEL_PRELAUNCH_READY : HBG_KERNEL_PRELAUNCH_CANCEL;
        cache_flush_range(&runtime->kernel_prelaunch, sizeof(runtime->kernel_prelaunch));
        g_restore_attempt.store(restored.generation, std::memory_order_relaxed);
        g_restore_status.store(rc, std::memory_order_relaxed);
        g_restore_epoch.store(epoch + 1, std::memory_order_release);
    } else {
        while (g_restore_epoch.load(std::memory_order_acquire) == epoch) {}
    }
    int rc = g_restore_status.load(std::memory_order_acquire);
    if (rc == 0) rc = aicpu_execute_restored_graph(runtime);
    if (rc != 0) {
        int32_t expected = 0;
        g_execution_status.compare_exchange_strong(expected, rc, std::memory_order_acq_rel);
    }
    g_finished.fetch_add(1, std::memory_order_acq_rel);
    while (g_finished.load(std::memory_order_acquire) != threads) {}

    // All status reads, destruction and executor cleanup have returned. No
    // execution role touches the image again before the next ordered DMA.
    // A failed execution poisons the slot rather than assuming quiescence.
    if (g_execution_status.load(std::memory_order_acquire) == 0) {
        const auto &graph = g_prepared_dispatch.graph;
        // Live task state precedes the immutable fanin and argument pools.
        const auto offsets = sm_layout::segment_offsets(sm_layout::image_extents({graph.total_tasks, 0, 0, 0}));
        maintain_prepared_image(g_prepared_dispatch, graph.sm_offset - graph.runtime_offset + offsets.fanin_pool);
    }
    const int32_t cleaned = g_prepared_cleaned.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (cleaned == threads) {
        rc = g_execution_status.load(std::memory_order_acquire);
        const hbg::GraphRestoreCompletion completion{
            rc == 0 ? hbg::GraphRestoreRetirement::Completed : hbg::GraphRestoreRetirement::FatalFailure,
            rc,
            0,
        };
        if (hbg::retire_graph_restore(
                hbg::current_graph_slot_registry(), g_restore_attempt.load(std::memory_order_relaxed), completion
            ) != hbg::GraphRestoreStatus::Ok) {
            (void)hbg::poison_graph_execution_slot(hbg::current_graph_slot_registry());
            rc = -1;
        }
        g_prepared_result.store(rc, std::memory_order_relaxed);
        g_prepared_checked.store(0, std::memory_order_relaxed);
        g_prepared_cleaned.store(0, std::memory_order_relaxed);
        g_finished.store(0, std::memory_order_relaxed);
        g_execution_status.store(0, std::memory_order_relaxed);
        g_entered.store(0, std::memory_order_release);
    } else {
        while (g_entered.load(std::memory_order_acquire) != 0) {}
    }
    return g_prepared_result.load(std::memory_order_acquire);
}

#endif  // SIMPLER_HBG_KERNEL_MODE
