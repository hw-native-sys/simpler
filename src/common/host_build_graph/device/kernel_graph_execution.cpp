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

#include <atomic>
#include <cstring>

#include "aicpu/kernel_invocation_consumer.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/cache_maintenance.h"
#include "aicpu/platform_regs.h"
#include "callable.h"
#include "common/unified_log.h"
#include "host_build_graph/kernel_graph_restore.h"
#include "host_build_graph/kernel_graph_slot_registry.h"
#include "host_build_graph/kernel_graph_wire.h"
#include "runtime.h"

extern "C" int32_t aicpu_execute(Runtime *runtime);

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

bool load_callable(int32_t id, uint64_t generation, hbg::HbgCallableRegistration &out) noexcept {
    if (id < 0 || id >= MAX_REGISTERED_CALLABLE_IDS) return false;
    auto &slot = g_callables[id];
    if (slot.generation.load(std::memory_order_acquire) != generation) return false;
    cache_invalidate_range(&slot.value, sizeof(slot.value));
    std::memcpy(&out, &slot.value, sizeof(out));
    return slot.generation.load(std::memory_order_acquire) == generation && out.context_generation == generation &&
           hbg::valid_hbg_callable_registration(out);
}

bool bind_callable(Runtime *runtime, const hbg::HbgCallableRegistration &registration) noexcept {
    if (runtime == nullptr || registration.callable_bytes < sizeof(ChipCallable)) return false;
    auto *callable = reinterpret_cast<const ChipCallable *>(registration.callable_address);
    cache_invalidate_range(callable, registration.callable_bytes);
    const int32_t children = callable->child_count_;
    if (children < 0 || children > KERNEL_MAX_FUNC_ID) return false;
    runtime->clear_function_bin_addrs();
    for (int32_t i = 0; i < children; ++i) {
        const int32_t func_id = callable->child_func_ids_[i];
        const uint32_t offset = callable->child_offsets_[i];
        constexpr size_t header = offsetof(ChipCallable, storage_);
        if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID || offset > registration.callable_bytes - header ||
            sizeof(CoreCallable) > registration.callable_bytes - header - offset)
            return false;
        const auto *child = reinterpret_cast<const CoreCallable *>(
            reinterpret_cast<const uint8_t *>(callable) + offsetof(ChipCallable, storage_) + offset
        );
        if (child->resolved_addr_ == 0) return false;
        // HBG stores the device-side CoreCallable header in func_id_to_addr_.
        // SchedulerContext::build_payload() dereferences this header to obtain
        // resolved_addr(); storing the code address here would make it interpret
        // AICore instructions as a CoreCallable object.
        runtime->replay_function_bin_addr(func_id, reinterpret_cast<uint64_t>(child));
    }
    runtime->set_active_callable_id(registration.callable_id);
    return true;
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
    // The register table is a context-owned prepare resource. Publish it on the
    // serialized control path, as TMR does for its context registration, so a
    // captured/replayed launch never has to reconstruct platform state.
    set_platform_regs(registration.register_table_address);
    slot.generation.store(registration.context_generation, std::memory_order_release);
    return 0;
}

// HBG owns the whole HostArgs packet, so it overrides the generic dispatcher
// which expects the fixed TMR envelope.
int consume_kernel_task(void *arg) {
    if (arg == nullptr) {
        LOG_ERROR("%s", "HBG kernel invocation has null HostArgs");
        return -1;
    }
    SimplerKernelInvocationHeader invocation{};
    hbg::GraphPacketHeader graph{};
    std::memcpy(&invocation, arg, sizeof(invocation));
    std::memcpy(&graph, static_cast<const uint8_t *>(arg) + sizeof(invocation), sizeof(graph));
    if (graph.total_bytes > UINT32_MAX - sizeof(invocation)) {
        LOG_ERROR("HBG graph packet size is invalid: %llu", static_cast<unsigned long long>(graph.total_bytes));
        return -1;
    }
    const size_t packet_bytes = sizeof(invocation) + static_cast<size_t>(graph.total_bytes);

    hbg::HbgCallableRegistration registration{};
    if (!load_callable(invocation.callable_id, graph.slot_generation, registration)) {
        LOG_ERROR(
            "HBG callable lookup failed: id=%d generation=%llu", invocation.callable_id,
            static_cast<unsigned long long>(graph.slot_generation)
        );
        return -1;
    }
    if (get_platform_regs() != registration.register_table_address) {
        LOG_ERROR(
            "HBG register table mismatch: prepared=%llu resident=%llu",
            static_cast<unsigned long long>(registration.register_table_address),
            static_cast<unsigned long long>(get_platform_regs())
        );
        return -1;
    }
    if (registration.callable_hash != graph.callable_hash || registration.function_hash != graph.function_hash ||
        invocation.scalar_count != registration.scalar_count ||
        invocation.tensor_count - invocation.host_copy_tensor_count != registration.tensor_count) {
        LOG_ERROR(
            "HBG invocation identity mismatch: callable=%llu/%llu function=%llu/%llu tensors=%d/%d host_copies=%d "
            "scalars=%d/%d",
            static_cast<unsigned long long>(registration.callable_hash),
            static_cast<unsigned long long>(graph.callable_hash),
            static_cast<unsigned long long>(registration.function_hash),
            static_cast<unsigned long long>(graph.function_hash), invocation.tensor_count, registration.tensor_count,
            invocation.host_copy_tensor_count, invocation.scalar_count, registration.scalar_count
        );
        return -1;
    }
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
        const auto status = hbg::restore_graph_packet(
            arg, packet_bytes, registration.device_id, registration.runtime_binary_id,
            {registration.callable_id, registration.tensor_count, registration.scalar_count}, restored
        );
        const bool callable_bound = status == hbg::GraphRestoreStatus::Ok && bind_callable(runtime, registration);
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

            // Publish only AICPU-owned runtime control. AICore has already
            // written workers[] by this point; flushing the whole Runtime would
            // write stale cached handshake lines back over those reports and
            // strand AICPU initialization. teardown_gates[] is AICore-owned as
            // well, so start at the first field after both device-owned arrays.
            auto *runtime_bytes = reinterpret_cast<uint8_t *>(runtime);
            auto *control_begin = reinterpret_cast<uint8_t *>(&runtime->worker_count);
            cache_flush_range(control_begin, sizeof(*runtime) - static_cast<size_t>(control_begin - runtime_bytes));

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
    if (rc == 0) rc = aicpu_execute(runtime);
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

#endif  // SIMPLER_HBG_KERNEL_MODE
