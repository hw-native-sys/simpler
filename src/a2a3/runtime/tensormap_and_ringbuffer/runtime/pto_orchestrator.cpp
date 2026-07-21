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

/**
 * PTO Runtime2 - Orchestrator Implementation
 *
 * Implements orchestrator state management, scope handling, and task submission
 * for the polling completion design.
 */

#include "pto_orchestrator.h"

#include <assert.h>
#include <limits>

#include "aicpu/dep_gen_collector_aicpu.h"
#include "common/dep_gen.h"
#include "pto_dep_compute.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"
#include "tensor.h"

// -----------------------------------------------------------------------------
// File-local helpers
// -----------------------------------------------------------------------------

static int32_t orch_mark_fatal(PTO2OrchestratorState *orch, int32_t error_code);
static void orch_report_fatal_v(PTO2OrchestratorState *orch, int32_t error_code, const char *fmt, va_list args);
static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state);
static bool prepare_task(
    PTO2OrchestratorState *orch, const L0TaskArgs &args, int32_t total_output_size, ActiveMask active_mask,
    PTO2PreparedTask *out
);
static PTO2OutputLayout calculate_output_layout(const L0TaskArgs &args);
static bool append_fanin_or_fail(
    PTO2OrchestratorState *orch, PTO2TaskSlotState *prod_state, int32_t prod_local_id, PTO2FaninBuilder *fanin_builder
);
static bool check_scope_can_accept_task(PTO2OrchestratorState *orch, PTO2TaskAllocator &allocator);
static void prefetch_payload(PTO2TaskPayload *payload, int32_t tensor_count, int32_t scalar_count);
static TaskOutputTensors submit_task_common(
    PTO2OrchestratorState *orch, const L0TaskArgs &args, ActiveMask active_mask, int32_t aic_kernel_id,
    int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
);

static int32_t orch_mark_fatal(PTO2OrchestratorState *orch, int32_t error_code) {
    always_assert(orch != nullptr);
    orch->fatal = true;
    if (error_code == PTO2_ERROR_NONE || orch->sm_header == nullptr) return PTO2_ERROR_NONE;

    int32_t expected = PTO2_ERROR_NONE;
    std::atomic<int32_t> &orch_error_code = orch->sm_header->orch_error_code;
    if (orch_error_code.compare_exchange_strong(expected, error_code, std::memory_order_acq_rel)) return error_code;
    return expected;
}

static void orch_report_fatal_v(PTO2OrchestratorState *orch, int32_t error_code, const char *, va_list) {
    // fmt + args are accepted for future logging-sink wiring but are not yet
    // routed anywhere — the error_code is latched in shared memory via
    // orch_mark_fatal and that's what callers actually observe.
    orch_mark_fatal(orch, error_code);
}

static bool append_fanin_or_fail(
    PTO2OrchestratorState *orch, PTO2TaskSlotState *prod_state, int32_t prod_local_id, PTO2FaninBuilder *fanin_builder
) {
    if (fanin_builder->contains(prod_state)) return true;
    if (fanin_builder->count >= PTO2_MAX_FANIN) {
        orch_mark_fatal(orch, PTO2_ERROR_DEP_POOL_OVERFLOW);
        return false;
    }
    int32_t idx = fanin_builder->count++;
    fanin_builder->slots[idx] = prod_state;
    fanin_builder->local_ids[idx] = prod_local_id;
    fanin_builder->ring_ids[idx] = prod_state->ring_id;
    return true;
}

static PTO2OutputLayout calculate_output_layout(const L0TaskArgs &args) {
    PTO2OutputLayout layout;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        layout.offsets[i] = layout.total_output_size;
        layout.buffer_sizes[i] =
            PTO2_ALIGN_UP(args.tensor(i).create_info().buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
        layout.total_output_size += layout.buffer_sizes[i];
    }
    return layout;
}

static bool check_scope_can_accept_task(PTO2OrchestratorState *orch, PTO2TaskAllocator &allocator) {
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");

    int32_t scope_task_count = orch->scope_tasks_size - orch->scope_begins[orch->scope_stack_top];
    if (scope_task_count < allocator.window_size() - 1) return true;

    orch_mark_fatal(orch, PTO2_ERROR_SCOPE_DEADLOCK);
    return false;
}

static void prefetch_payload(PTO2TaskPayload *payload, int32_t tensor_count, int32_t scalar_count) {
    for (int32_t i = 0; i < tensor_count; i++) {
        __builtin_prefetch(&payload->tensors[i], 1, 3);
        __builtin_prefetch(reinterpret_cast<char *>(&payload->tensors[i]) + 64, 1, 3);
    }
    for (int32_t i = 0; i < scalar_count; i += 8)
        __builtin_prefetch(&payload->scalars[i], 1, 3);
    __builtin_prefetch(payload, 1, 3);
    __builtin_prefetch(reinterpret_cast<char *>(payload) + 64, 1, 3);
    __builtin_prefetch(reinterpret_cast<char *>(payload) + 128, 1, 3);
}

static bool prepare_task(
    PTO2OrchestratorState *orch, const L0TaskArgs &args, int32_t total_output_size, ActiveMask active_mask,
    PTO2PreparedTask *out
) {
    uint8_t ring_id = orch->current_ring_id();
    auto &allocator = orch->rings[ring_id].task_allocator;

    if (!check_scope_can_accept_task(orch, allocator)) return false;

    out->alloc_result = allocator.alloc(total_output_size);
    if (out->alloc_result.failed()) {
        orch_mark_fatal(orch, PTO2_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }

    out->task_id = PTO2TaskId::make(ring_id, static_cast<uint32_t>(out->alloc_result.task_id));
    out->slot_state = &orch->sm_header->rings[ring_id].get_slot_state_by_slot(out->alloc_result.slot);
    out->task = &orch->sm_header->rings[ring_id].task_descriptors[out->alloc_result.slot];
    out->payload = &orch->sm_header->rings[ring_id].task_payloads[out->alloc_result.slot];

    prefetch_payload(out->payload, args.tensor_count(), args.scalar_count());

    // Reset the fanout/wake-list/subtask bookkeeping for this reuse. The allocator
    // only returns a slot whose previous incarnation is fully consumed (alloc spins
    // until completed_watermark passes its last_consumer_local_id), and the slot is
    // not published to any scheduler thread until the wiring.queue.push at the end
    // of submit_task_common — so this reset is race-free. Doing it here (not relying
    // on the scheduler's eager reset-after-CONSUMED, which only covers the
    // contiguously-reclaimed tail within a single run) makes every reused slot
    // self-clean across runs, which lets the per-boot SM init skip its O(window)
    // per-slot loop. bind_ring is slot-invariant but cheap to re-assert on the
    // already-dirtied cache line. Mirrors upstream #1199.
    out->slot_state->bind_ring(ring_id);
    out->slot_state->reset_for_reuse();

    out->slot_state->bind_buffers(out->payload, out->task);

    // Clear the polling-fast completion byte for the newly-allocated slot.
    // The previous incarnation's completer set this byte to 1; we publish 0
    // before this task can be added as a fanin to any consumer (single-
    // orchestrator-thread guarantee) and before the wiring-queue push
    // (release-acquire) makes the slot visible to thread 0.
    orch->sm_header->rings[ring_id].completion_flags[out->alloc_result.slot].store(0, std::memory_order_relaxed);
    // Seed last_consumer_local_id to self — with no consumers, the slot is
    // safe to reclaim as soon as the watermark reaches this task itself.
    out->slot_state->last_consumer_local_id = out->alloc_result.task_id;
    int16_t block_num = args.launch_spec.block_num();
    out->slot_state->total_required_subtasks =
        static_cast<int16_t>(block_num * __builtin_popcount(active_mask.core_mask()));
    out->slot_state->logical_block_num = block_num;
    out->slot_state->active_mask = active_mask;
    scope_tasks_push(orch, out->slot_state);

    return true;
}

static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state) {
    if (orch->scope_tasks_size >= orch->scope_tasks_capacity) {
        orch->report_fatal(
            PTO2_ERROR_SCOPE_TASKS_OVERFLOW, __FUNCTION__,
            "scope_tasks buffer saturated at %d entries (all rings full)", orch->scope_tasks_capacity
        );
        return;
    }
    orch->scope_tasks[orch->scope_tasks_size++] = task_slot_state;
}

static TaskOutputTensors submit_task_common(
    PTO2OrchestratorState *orch, const L0TaskArgs &args, ActiveMask active_mask, int32_t aic_kernel_id,
    int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    TaskOutputTensors result;
    PTO2OutputLayout layout = calculate_output_layout(args);
    PTO2PreparedTask prepared;
    if (!prepare_task(orch, args, layout.total_output_size, active_mask, &prepared)) return result;
    uint8_t ring_id = prepared.task_id.ring();
    PTO2SchedulerState *sched = orch->scheduler;
    PTO2RingFlowControl &fc = orch->sm_header->rings[ring_id].fc;
    PTO2TaskId task_id = prepared.task_id;
    PTO2TaskSlotState &cur_slot_state = *prepared.slot_state;
    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;
    result.set_task_id(task_id);

    if (is_dep_gen_enabled()) {
        const void *tensor_ptrs[MAX_TENSOR_ARGS];
        uint8_t arg_types_u8[MAX_TENSOR_ARGS];
        const int tc_raw = args.tensor_count();
        const int tc = tc_raw > MAX_TENSOR_ARGS ? MAX_TENSOR_ARGS : tc_raw;
        for (int i = 0; i < tc; i++) {
            tensor_ptrs[i] = (args.tag(i) == TensorArgType::OUTPUT) ? nullptr : &args.tensor(i).ref();
            arg_types_u8[i] = static_cast<uint8_t>(args.tag(i));
        }
        const int32_t kernel_ids_capture[3] = {aic_kernel_id, aiv0_kernel_id, aiv1_kernel_id};
        dep_gen_aicpu_record_submit(
            task_id.raw, orch->in_manual_scope(), args.allow_early_resolve(), tc, tensor_ptrs, arg_types_u8,
            static_cast<int>(args.explicit_dep_count()), reinterpret_cast<const uint64_t *>(args.explicit_deps_data()),
            args.launch_spec.block_num(), kernel_ids_capture
        );
    }

    PTO2FaninBuilder fanin_builder;

    int32_t sm_last_task_alive = fc.last_task_alive.load(std::memory_order_acquire);
    orch->tensor_map.sync_tensormap(task_id, sm_last_task_alive);

    for (uint32_t i = 0; i < args.explicit_dep_count(); i++) {
        PTO2TaskId dep_task_id = args.explicit_dep(i);
        if (!dep_task_id.is_valid()) {
            orch->report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "Arg.set_dependencies(...) requires valid task ids"
            );
            return result;
        }
        PTO2SharedMemoryRingHeader &dep_ring = orch->sm_header->rings[dep_task_id.ring()];
        int32_t dep_local_task_id = static_cast<int32_t>(dep_task_id.local());
        int32_t dep_last_task_alive = dep_ring.fc.last_task_alive.load(std::memory_order_acquire);
        if (dep_local_task_id < dep_last_task_alive) continue;
        PTO2TaskSlotState *producer_slot_state = &dep_ring.get_slot_state_by_task_id(dep_local_task_id);
        if (!append_fanin_or_fail(orch, producer_slot_state, dep_local_task_id, &fanin_builder)) return result;
    }

    DepInputs dep_inputs{
        args.tensor_count(),       args.tensor_data(), args.tag_data(), static_cast<int32_t>(args.explicit_dep_count()),
        args.explicit_deps_data(),
    };

    auto runtime_emit = [&](PTO2TaskId producer_task_id) -> bool {
        int32_t prod_local = static_cast<int32_t>(producer_task_id.local());
        PTO2TaskSlotState *prod_state =
            &orch->sm_header->rings[producer_task_id.ring()].get_slot_state_by_task_id(prod_local);
        return append_fanin_or_fail(orch, prod_state, prod_local, &fanin_builder);
    };

    if (!compute_task_fanin(dep_inputs, orch->tensor_map, orch->in_manual_scope(), runtime_emit)) return result;

    register_task_outputs(dep_inputs, task_id, orch->tensor_map, orch->in_manual_scope());

    __builtin_prefetch(&task, 1, 1);
    task.task_id = task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = aic_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = aiv0_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = aiv1_kernel_id;
    task.task_timing_slot = args.task_timing_slot();
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    // Single pass over fanin_builder:
    //   - Copy local_id/ring_id into payload so the scheduler can index the
    //     producer's ring's completion_flags from the consumer side.
    //   - Push this consumer's local_id into each same-ring producer's
    //     last_consumer high-water-mark, replacing the per-completion
    //     fanout_refcount notification. Reclamation gates on the per-ring
    //     completed_watermark reaching this value. Only update for same-ring
    //     fanin: cross-ring consumers live in a different local_id space,
    //     so their id is meaningless to the producer's ring's watermark.
    //     Cross-ring producer slots reclaim on scope_end / ring wrap instead
    //     — acceptable since cross-ring fanin (e.g. alloc_tensors output)
    //     is sparse.
    // Use fanin_builder.ring_ids[i] (cache-warm SOA slice) for the same-ring
    // check so cross-ring iters skip the slot_state dereference entirely.
    const uint8_t self_ring = task_id.ring();
    const int32_t self_local = static_cast<int32_t>(task_id.local());
    payload.fanin_count = fanin_builder.count;
    for (int32_t i = 0; i < fanin_builder.count; i++) {
        const int32_t local = fanin_builder.local_ids[i];
        const uint8_t ring = fanin_builder.ring_ids[i];
        payload.fanin_local_ids[i] = local;
        payload.fanin_ring_ids[i] = ring;
        if (ring == self_ring) {
            PTO2TaskSlotState *prod = fanin_builder.slots[i];
            if (self_local > prod->last_consumer_local_id) prod->last_consumer_local_id = self_local;
        }
    }

    payload.init(args, result, prepared.alloc_result, layout);

    while (!sched->wiring.queue.push(&cur_slot_state))
        SPIN_WAIT_HINT();

    return result;
}

// -----------------------------------------------------------------------------
// PTO2OrchestratorState members
// -----------------------------------------------------------------------------

void PTO2OrchestratorState::report_fatal(int32_t error_code, [[maybe_unused]] const char *func, const char *fmt, ...) {
    auto *orch = this;
    va_list args;
    va_start(args, fmt);
    orch_report_fatal_v(orch, error_code, fmt, args);
    va_end(args);
}

void PTO2OrchestratorState::begin_scope(PTO2ScopeMode mode) {
    auto *orch = this;
    if (orch->fatal) return;
    assert(orch->scope_stack_top < static_cast<int32_t>(orch->scope_stack_capacity - 1) && "Scope stack overflow");
    if (mode == PTO2ScopeMode::AUTO && orch->in_manual_scope()) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "auto scope nested inside manual scope is not supported");
        return;
    }

    bool already_in_manual_scope = orch->in_manual_scope();
    ++orch->scope_stack_top;
    orch->scope_begins[orch->scope_stack_top] = orch->scope_tasks_size;
    if (mode == PTO2ScopeMode::MANUAL && !already_in_manual_scope) orch->manual_begin_depth = orch->scope_stack_top;
}

void PTO2OrchestratorState::end_scope() {
    auto *orch = this;
    if (orch->fatal) return;
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

    bool ending_manual_scope = orch->scope_stack_top == orch->manual_begin_depth;
    int32_t begin = orch->scope_begins[orch->scope_stack_top--];
    if (ending_manual_scope) orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;

    // Watermark-based reclamation: scope-end has no work to do — consumers
    // no longer need to notify producers.
    orch->scope_tasks_size = begin;
}

TaskOutputTensors PTO2OrchestratorState::submit_task(const MixedKernels &mixed_kernels, const L0TaskArgs &args) {
    auto *orch = this;

    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) return TaskOutputTensors{};

    // Validate Arg construction (errors recorded by add_input/add_output/etc.)
    if (args.has_error) {
        orch_mark_fatal(orch, PTO2_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);
    // === Validate submit inputs ===
    ActiveMask active_mask = mixed_kernels.to_active_mask();
    always_assert(static_cast<bool>(active_mask) && "MixedKernels must have at least one active slot");

    int16_t block_num = args.launch_spec.block_num();
    always_assert(block_num >= 1 && "block_num must be >= 1");

    MixedKernels normalized = mixed_kernels;
    bool has_aic = active_mask.has_mask(PTO2_SUBTASK_MASK_AIC);
    bool has_aiv0 = active_mask.has_mask(PTO2_SUBTASK_MASK_AIV0);
    bool has_aiv1 = active_mask.has_mask(PTO2_SUBTASK_MASK_AIV1);
    if (!has_aic && has_aiv1 && !has_aiv0) {
        normalized.aiv0_kernel_id = normalized.aiv1_kernel_id;
        normalized.aiv1_kernel_id = INVALID_KERNEL_ID;
        active_mask = normalized.to_active_mask();
    }

    // Encode require_sync_start into active_mask bit 3 (only meaningful for tasks with block_num > 1)
    if (block_num > 1 && args.launch_spec.require_sync_start()) {
        PTO2ResourceShape shape = active_mask.to_shape();
        int32_t limit = (shape == PTO2ResourceShape::AIV) ? orch->total_aiv_count : orch->total_cluster_count;
        if (limit > 0 && block_num > limit) {
            report_fatal(
                PTO2_ERROR_REQUIRE_SYNC_START_INVALID, __FUNCTION__,
                "require_sync_start block_num=%d > limit=%d (deadlock guaranteed)", block_num, limit
            );
            return TaskOutputTensors{};
        }
        active_mask.set_sync_start();
    }

    return submit_task_common(
        orch, args, active_mask, normalized.aic_kernel_id, normalized.aiv0_kernel_id, normalized.aiv1_kernel_id
    );
}

TaskOutputTensors PTO2OrchestratorState::submit_dummy_task(const L0TaskArgs &args) {
    auto *orch = this;

    if (orch->fatal) return TaskOutputTensors{};

    if (args.has_error) {
        orch_mark_fatal(orch, PTO2_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);

    return submit_task_common(orch, args, ActiveMask{}, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID);
}

TaskOutputTensors PTO2OrchestratorState::alloc_tensors(const L0TaskArgs &args) {
    auto *orch = this;
    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) return TaskOutputTensors{};

    if (args.tensor_count() <= 0) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors requires at least one TensorCreateInfo");
        return TaskOutputTensors{};
    }
    if (args.scalar_count() != 0) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args");
        return TaskOutputTensors{};
    }
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args"
            );
            return TaskOutputTensors{};
        }
    }

    if (args.has_error) {
        report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }

    PTO2OutputLayout layout = calculate_output_layout(args);
    PTO2PreparedTask prepared;
    if (!prepare_task(orch, args, layout.total_output_size, ActiveMask{}, &prepared)) return TaskOutputTensors{};

    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;

    task.task_id = prepared.task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = INVALID_KERNEL_ID;
    // Kernel-less alloc descriptor never dispatches; keep the slot untagged so a
    // recycled ring slot cannot leak a stale task-timing tag.
    task.task_timing_slot = TASK_TIMING_SLOT_NONE;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    TaskOutputTensors outputs;
    outputs.set_task_id(prepared.task_id);
    payload.init(args, outputs, prepared.alloc_result, layout);
    payload.fanin_count = 0;

    if (prepared.slot_state != nullptr) {
        // (m) Inline completion uses completion_flags only.
        uint8_t ring_id = prepared.task_id.ring();
        auto &ring = orch->sm_header->rings[ring_id];
        const int32_t my_id = static_cast<int32_t>(prepared.task_id.local());
        const int32_t mask = ring.task_window_mask;
        ring.completion_flags[prepared.alloc_result.slot].store(1, std::memory_order_release);
        // Inline-completed slots never reach on_mixed_task_complete, so
        // CAS-advance the per-ring completed_watermark here. Without this,
        // wait_for_tensor_ready(wait_for_consumers=true) on an alloc'd slot
        // (e.g. set_tensor_data on its output) hangs because the watermark
        // gate target (slot's own local_id) is never reached if no real
        // task with local_id > my_id completes.
        int32_t w = ring.completed_watermark.load(std::memory_order_acquire);
        while (w < my_id) {
            int32_t next = w + 1;
            if (ring.completion_flags[next & mask].load(std::memory_order_acquire) == 0) break;
            if (ring.completed_watermark.compare_exchange_weak(
                    w, next, std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                w = next;
            }
        }
    }
    orch->inline_completed_tasks++;

    return outputs;
}

void PTO2OrchestratorState::mark_done() {
    auto *orch = this;
    orch->sm_header->orchestrator_done.store(1, std::memory_order_release);
    orch->scope_tasks_size = 0;
    orch->scope_stack_top = -1;
    orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
}
