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

#ifndef PTO_ORCHESTRATOR_H
#define PTO_ORCHESTRATOR_H

#include "common/l2_swimlane_profiling.h"
#include "utils/device_arena.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_submit_types.h"
#include "pto_scheduler.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"

#include <stdarg.h>
#include <stdio.h>
#include "aicpu/dep_gen_collector_aicpu.h"
#include "common/dep_gen.h"
#include "pto_dep_compute.h"
#include "tensor.h"

struct PTO2OrchestratorState;

// Full definitions of helper aggregate types that the inline methods on
// PTO2OrchestratorState (and the helpers below) construct by value.
struct PTO2PreparedTask
{
    PTO2TaskId task_id = PTO2TaskId::invalid();
    PTO2TaskAllocResult alloc_result = {-1, 0, nullptr, nullptr};
    PTO2TaskDescriptor *task = nullptr;
    PTO2TaskPayload *payload = nullptr;
    PTO2TaskSlotState *slot_state = nullptr;
};

struct PTO2FaninBuilder
{
    int32_t count{0};
    PTO2TaskSlotState *slots[PTO2_MAX_FANIN];
    int32_t local_ids[PTO2_MAX_FANIN];

    bool contains(PTO2TaskSlotState *prod_state) const
    {
        for (int32_t i = 0; i < count; i++)
            if (slots[i] == prod_state) return true;
        return false;
    }
};

// Forward declarations of helpers defined below — needed because the inline
// methods on PTO2OrchestratorState reference them.
inline int32_t orch_mark_fatal(PTO2OrchestratorState *orch, int32_t error_code);
inline void orch_report_fatal_v(PTO2OrchestratorState *orch, int32_t error_code, const char *fmt, va_list args);
inline void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state);
inline bool prepare_task(PTO2OrchestratorState *orch, const Arg &args, int32_t total_output_size, ActiveMask active_mask, PTO2PreparedTask *out);
inline PTO2OutputLayout calculate_output_layout(const Arg &args);
inline bool append_fanin_or_fail(PTO2OrchestratorState *orch, PTO2TaskSlotState *prod_state, int32_t prod_local_id, PTO2FaninBuilder *fanin_builder);
inline bool check_scope_can_accept_task(PTO2OrchestratorState *orch, PTO2TaskAllocator &allocator);
inline void prefetch_payload(PTO2TaskPayload *payload, int32_t tensor_count, int32_t scalar_count);
inline TaskOutputTensors submit_task_common(PTO2OrchestratorState *orch, const Arg &args, ActiveMask active_mask, int32_t aic_kernel_id, int32_t aiv0_kernel_id, int32_t aiv1_kernel_id);

struct PTO2OrchestratorLayout
{
    size_t off_scope_tasks;
    size_t off_scope_begins;
    PTO2TensorMapLayout tensor_map;
    int32_t dep_pool_capacity;
    int32_t scope_tasks_cap;
    uint64_t scope_stack_capacity;
};

struct PTO2OrchestratorState
{
    // === SHARED MEMORY ACCESS ===
    PTO2SharedMemoryHeader *sm_header;

    // === PER-RING RESOURCES ===
    PTO2RingSet rings[PTO2_MAX_RING_DEPTH];
    uint32_t *fanin_seen_epoch[PTO2_MAX_RING_DEPTH];
    uint32_t fanin_seen_current_epoch{1};

    // === TENSOR MAP (Private) ===
    PTO2TensorMap tensor_map;  // Producer lookup

    PTO2TaskSlotState **scope_tasks;  // Flat buffer of taskSlotState (all scopes concatenated)
    int32_t scope_tasks_size;         // Number of task IDs currently in the buffer
    int32_t scope_tasks_capacity;     // Allocated capacity of scope_tasks
    int32_t *scope_begins;            // scope_begins[i] = start index of scope i in scope_tasks
    int32_t scope_stack_top;          // Current top of stack (-1 = no scope open)
    uint64_t scope_stack_capacity;    // Max nesting depth (PTO2_MAX_SCOPE_DEPTH)
    int32_t manual_begin_depth{PTO2_MAX_SCOPE_DEPTH};

    PTO2SchedulerState *scheduler;  // For simulated mode only

    // Total core counts set once at executor init; used for submit-time deadlock detection.
    int32_t total_cluster_count{0};  // AIC cores = MIX clusters
    int32_t total_aiv_count{0};      // AIV cores (= 2 × clusters on standard hardware)

    // === GM HEAP (for output buffers) ===
    void *gm_heap_base;     // Base address of GM heap
    uint64_t gm_heap_size;  // Total size of GM heap (all rings)

    bool fatal;

    int64_t inline_completed_tasks{0};

    // === STATISTICS ===

    uint8_t current_ring_id() const
    {
        int32_t depth = scope_stack_top;
        if (depth < 0) depth = 0;
        return depth < PTO2_MAX_RING_DEPTH ? static_cast<uint8_t>(depth) : PTO2_MAX_RING_DEPTH - 1;
    }

    bool in_manual_scope() const
    {
        return scope_stack_top >= manual_begin_depth;
    }

    // === Cold-path API ===

    static PTO2OrchestratorLayout reserve_layout(DeviceArena &arena, const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH], int32_t dep_pool_capacity)
    {
        PTO2OrchestratorLayout layout{};
        layout.dep_pool_capacity = dep_pool_capacity;
        layout.scope_tasks_cap = PTO2_SCOPE_TASKS_CAP;
        layout.scope_stack_capacity = PTO2_MAX_SCOPE_DEPTH;

        layout.off_scope_tasks = arena.reserve(static_cast<size_t>(layout.scope_tasks_cap) * sizeof(PTO2TaskSlotState *), alignof(PTO2TaskSlotState *));
        layout.off_scope_begins = arena.reserve(static_cast<size_t>(layout.scope_stack_capacity) * sizeof(int32_t), alignof(int32_t));
        layout.tensor_map = PTO2TensorMap::reserve_layout_default(arena, task_window_sizes);
        return layout;
    }

    bool init_data_from_layout(const PTO2OrchestratorLayout &layout, DeviceArena &arena, void *sm_dev_base, void *gm_heap, uint64_t heap_size, uint64_t task_window_size)
    {
        auto *orch = this;
        *orch = PTO2OrchestratorState{};

        orch->sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);
        orch->gm_heap_base = gm_heap;
        orch->gm_heap_size = heap_size * PTO2_MAX_RING_DEPTH;
        orch->fatal = false;

        // Mirror the SM API's per-ring window-size shape so a future per-ring
        // SM layout cannot silently disagree with the addresses we compute here.
        uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) task_window_sizes[r] = task_window_size;

        auto *orch_err = pto2_sm_layout::orch_error_code_addr(sm_dev_base);
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        {
            void *ring_heap_base = reinterpret_cast<char *>(gm_heap) + r * heap_size;
            auto *task_descs_dev = pto2_sm_layout::ring_task_descriptors_addr(sm_dev_base, task_window_sizes, r);
            auto *cur_idx_dev = pto2_sm_layout::ring_current_task_index_addr(sm_dev_base, r);
            auto *last_alive_dev = pto2_sm_layout::ring_last_task_alive_addr(sm_dev_base, r);

            orch->rings[r].task_allocator.init(task_descs_dev, static_cast<int32_t>(task_window_size), cur_idx_dev, last_alive_dev, ring_heap_base, heap_size, orch_err);
        }

        if (!orch->tensor_map.init_data_from_layout(layout.tensor_map, arena)) return false;

        orch->scope_tasks_size = 0;
        orch->scope_tasks_capacity = layout.scope_tasks_cap;
        orch->scope_stack_top = -1;
        orch->scope_stack_capacity = layout.scope_stack_capacity;
        orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;

        return true;
    }

    void wire_arena_pointers(const PTO2OrchestratorLayout &layout, DeviceArena &arena, PTO2SchedulerState *scheduler_arg)
    {
        auto *orch = this;
        orch->tensor_map.wire_arena_pointers(layout.tensor_map, arena);
        orch->scope_tasks = static_cast<PTO2TaskSlotState **>(arena.region_ptr(layout.off_scope_tasks));
        orch->scope_begins = static_cast<int32_t *>(arena.region_ptr(layout.off_scope_begins));
        orch->scheduler = scheduler_arg;
    }

    // Forget pointers; arena owns the backing buffers.
    void destroy()
    {
        auto *orch = this;
        orch->tensor_map.destroy();
        orch->scope_tasks = nullptr;
        orch->scope_begins = nullptr;
    }
    void set_scheduler(PTO2SchedulerState *scheduler)
    {
        this->scheduler = scheduler;
    }
    void report_fatal(int32_t error_code, [[maybe_unused]] const char *func, const char *fmt, ...)
    {
        auto *orch = this;
        va_list args;
        va_start(args, fmt);
        orch_report_fatal_v(orch, error_code, fmt, args);
        va_end(args);
    }
    void begin_scope(PTO2ScopeMode mode)
    {
        auto *orch = this;
        if (orch->fatal) return;
        assert(orch->scope_stack_top < static_cast<int32_t>(orch->scope_stack_capacity - 1) && "Scope stack overflow");
        if (mode == PTO2ScopeMode::AUTO && orch->in_manual_scope())
        {
            report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "auto scope nested inside manual scope is not supported");
            return;
        }

        bool already_in_manual_scope = orch->in_manual_scope();
        ++orch->scope_stack_top;
        orch->scope_begins[orch->scope_stack_top] = orch->scope_tasks_size;
        if (mode == PTO2ScopeMode::MANUAL && !already_in_manual_scope) orch->manual_begin_depth = orch->scope_stack_top;
    }
    void end_scope()
    {
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
    TaskOutputTensors submit_task(const MixedKernels &mixed_kernels, const Arg &args)
    {
        auto *orch = this;

        // Orchestration API should short-circuit after fatal, but keep this entry
        // robust as a no-op in case a caller reaches it directly.
        if (orch->fatal) return TaskOutputTensors{};

        // Validate Arg construction (errors recorded by add_input/add_output/etc.)
        if (args.has_error)
        {
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
        if (!has_aic && has_aiv1 && !has_aiv0)
        {
            normalized.aiv0_kernel_id = normalized.aiv1_kernel_id;
            normalized.aiv1_kernel_id = INVALID_KERNEL_ID;
            active_mask = normalized.to_active_mask();
        }

        // Encode require_sync_start into active_mask bit 3 (only meaningful for tasks with block_num > 1)
        if (block_num > 1 && args.launch_spec.require_sync_start())
        {
            PTO2ResourceShape shape = active_mask.to_shape();
            int32_t limit = (shape == PTO2ResourceShape::AIV) ? orch->total_aiv_count : orch->total_cluster_count;
            if (limit > 0 && block_num > limit)
            {
                report_fatal(PTO2_ERROR_REQUIRE_SYNC_START_INVALID, __FUNCTION__, "require_sync_start block_num=%d > limit=%d (deadlock guaranteed)", block_num, limit);
                return TaskOutputTensors{};
            }
            active_mask.set_sync_start();
        }

        return submit_task_common(orch, args, active_mask, normalized.aic_kernel_id, normalized.aiv0_kernel_id, normalized.aiv1_kernel_id);
    }
    TaskOutputTensors submit_dummy_task(const Arg &args)
    {
        auto *orch = this;

        if (orch->fatal) return TaskOutputTensors{};

        if (args.has_error)
        {
            orch_mark_fatal(orch, PTO2_ERROR_INVALID_ARGS);
            return TaskOutputTensors{};
        }
        always_assert(orch->scheduler != nullptr);

        return submit_task_common(orch, args, ActiveMask{}, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID);
    }
    TaskOutputTensors alloc_tensors(const Arg &args)
    {
        auto *orch = this;
        // Orchestration API should short-circuit after fatal, but keep this entry
        // robust as a no-op in case a caller reaches it directly.
        if (orch->fatal) return TaskOutputTensors{};

        if (args.tensor_count() <= 0)
        {
            report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors requires at least one TensorCreateInfo");
            return TaskOutputTensors{};
        }
        if (args.scalar_count() != 0)
        {
            report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args");
            return TaskOutputTensors{};
        }
        for (int32_t i = 0; i < args.tensor_count(); i++)
        {
            if (args.tag(i) != TensorArgType::OUTPUT)
            {
                report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args");
                return TaskOutputTensors{};
            }
        }

        if (args.has_error)
        {
            report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "%s", args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg");
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
        task.packed_buffer_base = prepared.alloc_result.packed_base;
        task.packed_buffer_end = prepared.alloc_result.packed_end;

        TaskOutputTensors outputs;
        outputs.set_task_id(prepared.task_id);
        payload.init(args, outputs, prepared.alloc_result, layout);
        payload.fanin_count = 0;

        if (prepared.slot_state != nullptr)
        {
            // (m) Inline completion uses completion_flags only.
            uint8_t ring_id = prepared.task_id.ring();
            orch->sm_header->rings[ring_id].completion_flags[prepared.alloc_result.slot].store(1, std::memory_order_release);
        }
        orch->inline_completed_tasks++;

        return outputs;
    }
    void mark_done()
    {
        auto *orch = this;
        orch->sm_header->orchestrator_done.store(1, std::memory_order_release);
        orch->scope_tasks_size = 0;
        orch->scope_stack_top = -1;
        orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
    }
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

inline int32_t orch_mark_fatal(PTO2OrchestratorState *orch, int32_t error_code)
{
    always_assert(orch != nullptr);
    orch->fatal = true;
    if (error_code == PTO2_ERROR_NONE || orch->sm_header == nullptr) return PTO2_ERROR_NONE;

    int32_t expected = PTO2_ERROR_NONE;
    std::atomic<int32_t> &orch_error_code = orch->sm_header->orch_error_code;
    if (orch_error_code.compare_exchange_strong(expected, error_code, std::memory_order_acq_rel)) return error_code;
    return expected;
}

inline void orch_report_fatal_v(PTO2OrchestratorState *orch, int32_t error_code, const char *, va_list)
{
    // fmt + args are accepted for future logging-sink wiring but are not yet
    // routed anywhere — the error_code is latched in shared memory via
    // orch_mark_fatal and that's what callers actually observe.
    orch_mark_fatal(orch, error_code);
}

inline bool append_fanin_or_fail(PTO2OrchestratorState *orch, PTO2TaskSlotState *prod_state, int32_t prod_local_id, PTO2FaninBuilder *fanin_builder)
{
    if (fanin_builder->contains(prod_state)) return true;
    if (fanin_builder->count >= PTO2_MAX_FANIN)
    {
        orch_mark_fatal(orch, PTO2_ERROR_DEPENDENCY_OVERFLOW);
        return false;
    }
    int32_t idx = fanin_builder->count++;
    fanin_builder->slots[idx] = prod_state;
    fanin_builder->local_ids[idx] = prod_local_id;
    return true;
}

inline PTO2OutputLayout calculate_output_layout(const Arg &args)
{
    PTO2OutputLayout layout;
    for (int32_t i = 0; i < args.tensor_count(); i++)
    {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        layout.offsets[i] = layout.total_output_size;
        layout.buffer_sizes[i] = PTO2_ALIGN_UP(args.tensor(i).create_info->buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
        layout.total_output_size += layout.buffer_sizes[i];
    }
    return layout;
}

inline bool check_scope_can_accept_task(PTO2OrchestratorState *orch, PTO2TaskAllocator &allocator)
{
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");

    int32_t scope_task_count = orch->scope_tasks_size - orch->scope_begins[orch->scope_stack_top];
    if (scope_task_count < allocator.window_size() - 1) return true;

    orch_mark_fatal(orch, PTO2_ERROR_SCOPE_DEADLOCK);
    return false;
}

inline void prefetch_payload(PTO2TaskPayload *payload, int32_t tensor_count, int32_t scalar_count)
{
    for (int32_t i = 0; i < tensor_count; i++)
    {
        __builtin_prefetch(&payload->tensors[i], 1, 3);
        __builtin_prefetch(reinterpret_cast<char *>(&payload->tensors[i]) + 64, 1, 3);
    }
    for (int32_t i = 0; i < scalar_count; i += 8) __builtin_prefetch(&payload->scalars[i], 1, 3);
    __builtin_prefetch(payload, 1, 3);
    __builtin_prefetch(reinterpret_cast<char *>(payload) + 64, 1, 3);
    __builtin_prefetch(reinterpret_cast<char *>(payload) + 128, 1, 3);
}

inline bool prepare_task(PTO2OrchestratorState *orch, const Arg &args, int32_t total_output_size, ActiveMask active_mask, PTO2PreparedTask *out)
{
    uint8_t ring_id = orch->current_ring_id();
    auto &allocator = orch->rings[ring_id].task_allocator;

    if (!check_scope_can_accept_task(orch, allocator)) return false;

    out->alloc_result = allocator.alloc(total_output_size);
    if (out->alloc_result.failed())
    {
        orch_mark_fatal(orch, PTO2_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }

    out->task_id = PTO2TaskId::make(ring_id, static_cast<uint32_t>(out->alloc_result.task_id));
    out->slot_state = &orch->sm_header->rings[ring_id].get_slot_state_by_slot(out->alloc_result.slot);
    out->task = &orch->sm_header->rings[ring_id].task_descriptors[out->alloc_result.slot];
    out->payload = &orch->sm_header->rings[ring_id].task_payloads[out->alloc_result.slot];

    prefetch_payload(out->payload, args.tensor_count(), args.scalar_count());

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
    out->slot_state->total_required_subtasks = static_cast<int16_t>(block_num * __builtin_popcount(active_mask.core_mask()));
    out->slot_state->logical_block_num = block_num;
    out->slot_state->active_mask = active_mask;
    scope_tasks_push(orch, out->slot_state);

    return true;
}

inline void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state)
{
    if (orch->scope_tasks_size >= orch->scope_tasks_capacity)
    {
        orch->report_fatal(PTO2_ERROR_SCOPE_TASKS_OVERFLOW, __FUNCTION__, "scope_tasks buffer saturated at %d entries (all rings full)", orch->scope_tasks_capacity);
        return;
    }
    orch->scope_tasks[orch->scope_tasks_size++] = task_slot_state;
}

inline TaskOutputTensors submit_task_common(PTO2OrchestratorState *orch, const Arg &args, ActiveMask active_mask, int32_t aic_kernel_id, int32_t aiv0_kernel_id, int32_t aiv1_kernel_id)
{
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

    if (is_dep_gen_enabled())
    {
        const void *tensor_ptrs[MAX_TENSOR_ARGS];
        uint8_t arg_types_u8[MAX_TENSOR_ARGS];
        const int tc_raw = args.tensor_count();
        const int tc = tc_raw > MAX_TENSOR_ARGS ? MAX_TENSOR_ARGS : tc_raw;
        for (int i = 0; i < tc; i++)
        {
            tensor_ptrs[i] = (args.tag(i) == TensorArgType::OUTPUT) ? nullptr : args.tensor(i).ptr;
            arg_types_u8[i] = static_cast<uint8_t>(args.tag(i));
        }
        const int32_t kernel_ids_capture[3] = {aic_kernel_id, aiv0_kernel_id, aiv1_kernel_id};
        dep_gen_aicpu_record_submit(task_id.raw, orch->in_manual_scope(), tc, tensor_ptrs, arg_types_u8, static_cast<int>(args.explicit_dep_count()), reinterpret_cast<const uint64_t *>(args.explicit_deps_data()), kernel_ids_capture);
    }

    PTO2FaninBuilder fanin_builder;

    int32_t sm_last_task_alive = fc.last_task_alive.load(std::memory_order_acquire);
    orch->tensor_map.sync_tensormap(task_id, sm_last_task_alive);

    for (uint32_t i = 0; i < args.explicit_dep_count(); i++)
    {
        PTO2TaskId dep_task_id = args.explicit_dep(i);
        if (!dep_task_id.is_valid())
        {
            orch->report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "Arg.set_dependencies(...) requires valid task ids");
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
        args.tensor_count(), args.tensor_data(), args.tag_data(), static_cast<int32_t>(args.explicit_dep_count()), args.explicit_deps_data(),
    };

    auto runtime_emit = [&](PTO2TaskId producer_task_id) -> bool {
        int32_t prod_local = static_cast<int32_t>(producer_task_id.local());
        PTO2TaskSlotState *prod_state = &orch->sm_header->rings[producer_task_id.ring()].get_slot_state_by_task_id(prod_local);
        return append_fanin_or_fail(orch, prod_state, prod_local, &fanin_builder);
    };

    if (!compute_task_fanin(dep_inputs, orch->tensor_map, orch->in_manual_scope(), runtime_emit)) return result;

    register_task_outputs(dep_inputs, task_id, orch->tensor_map, orch->in_manual_scope());

    __builtin_prefetch(&task, 1, 1);
    task.task_id = task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = aic_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = aiv0_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = aiv1_kernel_id;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    // Push this consumer's local_id into each producer's last_consumer high-
    // water-mark, replacing the per-completion fanout_refcount notification.
    // Reclamation gates on the global completed_watermark reaching this value.
    const int32_t self_local = static_cast<int32_t>(task_id.local());
    for (int32_t i = 0; i < fanin_builder.count; i++)
    {
        PTO2TaskSlotState *prod = fanin_builder.slots[i];
        if (self_local > prod->last_consumer_local_id) prod->last_consumer_local_id = self_local;
    }

    payload.fanin_count = fanin_builder.count;
    for (int32_t i = 0; i < fanin_builder.count; i++) payload.fanin_local_ids[i] = fanin_builder.local_ids[i];

    payload.init(args, result, prepared.alloc_result, layout);

    while (!sched->wiring.queue.push(&cur_slot_state)) SPIN_WAIT_HINT();

    return result;
}

#endif  // PTO_ORCHESTRATOR_H
