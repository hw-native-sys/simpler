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
#include "host_build_graph/kernel_graph_restore.h"

#include <cstring>

#include "aicpu/cache_maintenance.h"
#include "host_build_graph/kernel_graph_image_validation.h"
#include "host_build_graph/kernel_graph_prepared.h"
#include "host_build_graph/runtime_core.h"

namespace hbg {
namespace {

bool copy_region(const GraphRestoreOps &ops, void *dst, const void *src, size_t bytes) noexcept {
    if (ops.copy) return ops.copy(ops.context, dst, src, bytes);
    std::memcpy(dst, src, bytes);
    return true;
}
bool flush_region(const GraphRestoreOps &ops, const void *dst, size_t bytes) noexcept {
    if (ops.flush) return ops.flush(ops.context, dst, bytes);
    cache_flush_range(dst, bytes);
    return true;
}
void publish(GraphRestoreControl &control, GraphRestorePhase phase, GraphRestoreStatus status) noexcept {
    control.status = static_cast<uint32_t>(status);
    cache_flush_range(&control, sizeof(control));
    __atomic_store_n(&control.phase, static_cast<uint32_t>(phase), __ATOMIC_RELEASE);
    cache_flush_range(&control, sizeof(control));
}

GraphRestoreStatus claim_restore(GraphRestoreControl &control) noexcept {
    cache_invalidate_range(&control, sizeof(control));
    uint32_t phase = __atomic_load_n(&control.phase, __ATOMIC_ACQUIRE);
    if (phase == static_cast<uint32_t>(GraphRestorePhase::Restoring) ||
        phase == static_cast<uint32_t>(GraphRestorePhase::Ready))
        return GraphRestoreStatus::Busy;
    if (phase == static_cast<uint32_t>(GraphRestorePhase::Failed)) return GraphRestoreStatus::Quarantined;
    if (phase != static_cast<uint32_t>(GraphRestorePhase::Idle)) return GraphRestoreStatus::Rejected;
    if (!__atomic_compare_exchange_n(
            &control.phase, &phase, static_cast<uint32_t>(GraphRestorePhase::Restoring), false, __ATOMIC_ACQ_REL,
            __ATOMIC_ACQUIRE
        ))
        return GraphRestoreStatus::Busy;
    if (control.attempt == UINT64_MAX) {
        __atomic_store_n(&control.phase, phase, __ATOMIC_RELEASE);
        cache_flush_range(&control, sizeof(control));
        return GraphRestoreStatus::Exhausted;
    }
    ++control.attempt;
    cache_flush_range(&control, sizeof(control));
    return GraphRestoreStatus::Ok;
}

GraphRestoreStatus fail_restore(GraphRestoreControl &control) noexcept {
    publish(control, GraphRestorePhase::Failed, GraphRestoreStatus::CopyFailed);
    return GraphRestoreStatus::CopyFailed;
}

RuntimeContext *initialize_restored_runtime(
    const GraphRestoreView &view, const RuntimeArenaLayout &layout, uint64_t sm_bytes, bool reuse_mailbox = false
) noexcept {
    const auto &working = view.slot.destinations[1];
    DeviceArena arena;
    arena.attach(reinterpret_cast<void *>(working.address));
    auto *runtime = reinterpret_cast<RuntimeContext *>(working.address + layout.off_runtime);
    runtime_wire_arena_pointers(arena, layout, runtime);
    void *sm = reinterpret_cast<void *>(working.address + view.graph.sm_offset);
    if (!runtime->sm_handle->attach_populated(
            sm, sm_bytes, view.graph.task_window, sm_layout::live_slot_pitch(view.graph.total_tasks), sm_bytes
        ) ||
        !runtime->scheduler->init_data_from_layout(layout.sched, arena, sm))
        return nullptr;
    runtime->scheduler->seed_queue_slots();
    if (reuse_mailbox) {
        auto &mailbox = *runtime->aicore_mailbox;
        mailbox.tail.store(mailbox.head.load(std::memory_order_acquire), std::memory_order_release);
    } else {
        runtime->aicore_mailbox->init_empty();
    }
    return runtime;
}

GraphRestoreStatus finish_restore(
    const GraphRestoreView &view, RuntimeContext *runtime, uint64_t sm_bytes, GraphRestoreControl &control,
    GraphRestoreResult &out, const GraphRestoreOps &ops
) noexcept {
    control.runtime_address = reinterpret_cast<uintptr_t>(runtime);
    control.sm_bytes = sm_bytes;
    control.total_tasks = view.graph.total_tasks;
    control.live_bytes[0] = view.graph.heap_bytes;
    for (size_t i = 1; i < 4; ++i)
        control.live_bytes[i] = view.regions[i].destination_offset + view.regions[i].bytes;
    if (!flush_region(ops, &control, sizeof(control))) return fail_restore(control);
    control.committed_generation = control.attempt;
    publish(control, GraphRestorePhase::Ready, GraphRestoreStatus::Ok);
    out = {runtime, control.attempt, sm_bytes, view.graph.total_tasks};
    return GraphRestoreStatus::Ok;
}
}  // namespace

GraphRestoreStatus restore_graph_packet(
    const void *packet, size_t bytes, int device_id, uint64_t runtime_binary_id,
    const simpler::kernel::PreparedInvocationView &trusted_callable, GraphRestoreResult &out, const GraphRestoreOps &ops
) noexcept {
    GraphRestoreView view{};
    const auto admission =
        admit_graph_packet_for_restore(packet, bytes, device_id, runtime_binary_id, trusted_callable, view);
    if (admission == GraphSlotStatus::Poisoned) return GraphRestoreStatus::Poisoned;
    if (admission != GraphSlotStatus::Ok) return GraphRestoreStatus::Rejected;
    RuntimeArenaLayout layout{};
    uint64_t sm_bytes = 0;
    if (!detail::validate_graph_images(view, layout, sm_bytes)) return GraphRestoreStatus::InvalidImage;
    auto *registry = reinterpret_cast<GraphSlotRegistry *>(view.slot.registry.address);
    auto &control = registry->restore;
    const auto claimed = claim_restore(control);
    if (claimed != GraphRestoreStatus::Ok) return claimed;
    for (size_t i = 1; i < 4; ++i) {
        const auto &dst = view.slot.destinations[i];
        const auto &region = view.regions[i];
        if (region.bytes == 0) continue;
        if (!copy_region(
                ops, reinterpret_cast<void *>(dst.address + region.destination_offset),
                view.payload + region.source_offset, region.bytes
            ))
            return fail_restore(control);
    }
    auto *runtime = initialize_restored_runtime(view, layout, sm_bytes);
    if (runtime == nullptr) return fail_restore(control);
    // Device-only queues and mailbox are shared through AICPU coherence and the Ready release/acquire.
    for (size_t i = 1; i < 4; ++i)
        if (view.regions[i].bytes &&
            !flush_region(
                ops, reinterpret_cast<void *>(view.slot.destinations[i].address + view.regions[i].destination_offset),
                view.regions[i].bytes
            ))
            return fail_restore(control);
    return finish_restore(view, runtime, sm_bytes, control, out, ops);
}

GraphRestoreStatus attach_prepared_graph_image(
    const GraphPreparedDispatch &prepared, const GraphSlotRegistration &slot, GraphRestoreResult &out
) noexcept {
    if (!graph_prepared_dispatch_matches_slot(prepared, slot)) return GraphRestoreStatus::Rejected;
    GraphRestoreView view{};
    view.invocation = prepared.invocation;
    view.slot = slot;
    view.graph = prepared.graph;
    for (uint32_t i = 0; i < prepared.graph.region_count; ++i) {
        const auto &region = prepared.regions[i];
        view.regions[static_cast<uint32_t>(region.kind)] = region;
    }
    DeviceArena reservations;
    uint64_t ready_capacity = 64;
    while (ready_capacity < view.graph.task_window)
        ready_capacity <<= 1;
    auto layout = runtime_reserve_layout(reservations, view.graph.task_window, ready_capacity);
    layout.sched.capacities =
        kernel_graph_queue_capacities(view.graph.total_tasks, view.graph.task_window, view.regions[2].bytes != 0);
    const auto &image = view.regions[1];
    if (view.graph.runtime_offset != layout.off_runtime || view.graph.sm_offset != layout.off_copied_end ||
        image.destination_offset != layout.off_runtime || view.graph.sm_offset < image.destination_offset ||
        !graph_span_fits(0, sizeof(RuntimeContext), image.bytes) ||
        !graph_span_fits(view.graph.sm_offset - image.destination_offset, sizeof(SharedMemoryHeader), image.bytes))
        return GraphRestoreStatus::InvalidImage;
    const uint64_t sm_bytes = image.bytes - (view.graph.sm_offset - image.destination_offset);
    const auto offsets = sm_layout::segment_offsets(sm_layout::image_extents({view.graph.total_tasks, 0, 0, 0}));
    if (sm_bytes > INT32_MAX || offsets.fanin_pool > sm_bytes) return GraphRestoreStatus::InvalidImage;
    auto *registry = reinterpret_cast<GraphSlotRegistry *>(slot.registry.address);
    auto &control = registry->restore;
    const auto claimed = claim_restore(control);
    if (claimed != GraphRestoreStatus::Ok) return claimed;
    const bool reuse_mailbox = control.committed_generation != 0 &&
                               control.committed_generation == control.attempt - 1 &&
                               control.runtime_address == slot.destinations[1].address + layout.off_runtime;
    auto *runtime = initialize_restored_runtime(view, layout, sm_bytes, reuse_mailbox);
    if (runtime == nullptr) return fail_restore(control);
    return finish_restore(view, runtime, sm_bytes, control, out, {});
}

GraphRestoreStatus
retire_graph_restore(GraphSlotRegistry *registry, uint64_t attempt, const GraphRestoreCompletion &completion) noexcept {
    if (registry == nullptr || attempt == 0 || reinterpret_cast<uintptr_t>(registry) % 1024 != 0)
        return GraphRestoreStatus::Rejected;
    GraphSlotRegistration slot{};
    const auto status = acquire_graph_execution_slot(registry, registry->device_id, registry->runtime_binary_id, slot);
    if (status == GraphSlotStatus::Poisoned) return GraphRestoreStatus::Poisoned;
    if (status != GraphSlotStatus::Ok) return GraphRestoreStatus::Rejected;
    auto &control = registry->restore;
    cache_invalidate_range(&control, sizeof(control));
    uint32_t phase = __atomic_load_n(&control.phase, __ATOMIC_ACQUIRE);
    if (phase == static_cast<uint32_t>(GraphRestorePhase::Restoring)) return GraphRestoreStatus::Busy;
    if (control.attempt != attempt || (phase != static_cast<uint32_t>(GraphRestorePhase::Ready) &&
                                       phase != static_cast<uint32_t>(GraphRestorePhase::Failed)))
        return GraphRestoreStatus::NotReady;
    const auto outcome = completion.outcome;
    if (completion.runtime_status != 0 || completion.unexpected_teardown_status != 0 ||
        outcome == GraphRestoreRetirement::FatalFailure) {
        return poison_graph_execution_slot(registry) == GraphSlotStatus::Ok ? GraphRestoreStatus::Poisoned :
                                                                              GraphRestoreStatus::Rejected;
    }
    const bool completed =
        outcome == GraphRestoreRetirement::Completed && phase == static_cast<uint32_t>(GraphRestorePhase::Ready) &&
        control.committed_generation == attempt && control.status == static_cast<uint32_t>(GraphRestoreStatus::Ok);
    const bool controlled = outcome == GraphRestoreRetirement::ControlledFailure &&
                            phase == static_cast<uint32_t>(GraphRestorePhase::Failed);
    if (!completed && !controlled) return GraphRestoreStatus::Rejected;
    if (!__atomic_compare_exchange_n(
            &control.phase, &phase, static_cast<uint32_t>(GraphRestorePhase::Idle), false, __ATOMIC_ACQ_REL,
            __ATOMIC_ACQUIRE
        ))
        return GraphRestoreStatus::Busy;
    cache_flush_range(&control, sizeof(control));
    return GraphRestoreStatus::Ok;
}

GraphRestoreStatus
acquire_graph_restore_result(const GraphSlotRegistry *registry, uint64_t generation, GraphRestoreResult &out) noexcept {
    if (registry == nullptr || generation == 0 || reinterpret_cast<uintptr_t>(registry) % 1024 != 0)
        return GraphRestoreStatus::NotReady;
    GraphSlotRegistration slot{};
    const auto status = acquire_graph_execution_slot(registry, registry->device_id, registry->runtime_binary_id, slot);
    if (status == GraphSlotStatus::Poisoned) return GraphRestoreStatus::Poisoned;
    if (status != GraphSlotStatus::Ok) return GraphRestoreStatus::NotReady;
    const auto &control = registry->restore;
    cache_invalidate_range(&control, sizeof(control));
    if (__atomic_load_n(&control.phase, __ATOMIC_ACQUIRE) != static_cast<uint32_t>(GraphRestorePhase::Ready) ||
        control.attempt != generation || control.committed_generation != generation ||
        control.status != static_cast<uint32_t>(GraphRestoreStatus::Ok) ||
        control.runtime_address < slot.destinations[1].address ||
        !graph_span_fits(
            control.runtime_address - slot.destinations[1].address, sizeof(RuntimeContext),
            slot.destinations[1].capacity
        ))
        return GraphRestoreStatus::NotReady;
    for (size_t i = 0; i < 4; ++i)
        if (control.live_bytes[i] > slot.destinations[i].capacity) return GraphRestoreStatus::NotReady;
    for (size_t i = 1; i < 4; ++i)
        if (control.live_bytes[i])
            cache_invalidate_range(reinterpret_cast<const void *>(slot.destinations[i].address), control.live_bytes[i]);
    out = {
        reinterpret_cast<RuntimeContext *>(control.runtime_address), generation, control.sm_bytes, control.total_tasks
    };
    return GraphRestoreStatus::Ok;
}
}  // namespace hbg
