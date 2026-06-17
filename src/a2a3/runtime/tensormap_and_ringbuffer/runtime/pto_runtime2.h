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

#include "utils/device_arena.h"
#include "pto_runtime2_types.h"
#include "pto_submit_types.h"
#include "pto_shared_memory.h"
#include "pto_ring_buffer.h"
#include "pto_tensormap.h"
#include "pto_scheduler.h"
#include "pto_orchestrator.h"
#include "aicore_completion_mailbox.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "aicpu/device_time.h"

__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu();

enum PTO2RuntimeMode
{
    PTO2_MODE_EXECUTE = 0,    // Execute tasks on workers
    PTO2_MODE_SIMULATE = 1,   // Simulate task execution with cycle counting
    PTO2_MODE_GRAPH_ONLY = 2  // Build graph only, no execution
};

typedef struct PTO2Runtime PTO2Runtime;  // forward declare for ops signatures

struct PTO2RuntimeOps
{
    TaskOutputTensors (*submit_task)(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const Arg &args);
    void (*scope_begin)(PTO2Runtime *rt);
    void (*scope_end)(PTO2Runtime *rt);
    void (*orchestration_done)(PTO2Runtime *rt);
    bool (*is_fatal)(PTO2Runtime *rt);
    void (*report_fatal)(PTO2Runtime *rt, int32_t error_code, const char *func, const char *fmt, ...);

    // Logging (populated by runtime, called by orchestration)
    // INFO with explicit verbosity tier (v ∈ [0, 9]; gating done inside).

    // Cross-layer data access (orchestration reads/writes tensor values via runtime)
    // Placed after logging to avoid shifting hot-path field offsets.
    uint64_t (*get_tensor_data)(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[]);
    void (*set_tensor_data)(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value);
    TaskOutputTensors (*alloc_tensors)(PTO2Runtime *rt, const Arg &args);
    TaskOutputTensors (*submit_dummy_task)(PTO2Runtime *rt, const Arg &args);
    void (*scope_set_site)(const char *file, int line);
};

struct PTO2RuntimeArenaLayout
{
    size_t off_sm_handle{0};
    PTO2OrchestratorLayout orch;
    PTO2SchedulerLayout sched;
    size_t off_runtime{0};
    size_t off_mailbox{0};

    // Cached parameters (re-used by init_data + wire stages).
    uint64_t task_window_size{0};
    uint64_t heap_size{0};
    int32_t dep_pool_capacity{0};

    // Total arena byte size post-commit. Used by host to size the prebuilt
    // image buffer and as the rtMemcpy length.
    size_t arena_size{0};
};

struct PTO2Runtime
{
    // Ops table (first field — used by orchestration .so via function pointers)
    const PTO2RuntimeOps *ops;
    PTO2ScopeMode pending_scope_mode;

    // Components
    PTO2SharedMemoryHandle *sm_handle;
    PTO2OrchestratorState orchestrator;
    PTO2SchedulerState scheduler;
    AICoreCompletionMailbox *aicore_mailbox;

    // GM Heap for output buffers
    void *gm_heap;
    uint64_t gm_heap_size;
    bool gm_heap_owned;  // True if we allocated it

    // Mode
    PTO2RuntimeMode mode;

    // Statistics
    int64_t total_cycles;

    PTO2RuntimeArenaLayout prebuilt_layout;
};

inline PTO2RuntimeArenaLayout runtime_reserve_layout(DeviceArena &arena, uint64_t task_window_size, int32_t dep_pool_capacity)
{
    PTO2RuntimeArenaLayout layout{};
    layout.task_window_size = task_window_size;
    layout.dep_pool_capacity = dep_pool_capacity;

    int32_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) task_window_sizes[r] = static_cast<int32_t>(task_window_size);

    layout.off_sm_handle = arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
    layout.orch = PTO2OrchestratorState::reserve_layout(arena, task_window_sizes, dep_pool_capacity);
    layout.sched = PTO2SchedulerState::reserve_layout(arena, dep_pool_capacity);
    layout.off_runtime = arena.reserve(sizeof(PTO2Runtime), PTO2_ALIGN_SIZE);
    layout.off_mailbox = arena.reserve(sizeof(AICoreCompletionMailbox), alignof(AICoreCompletionMailbox));

    layout.arena_size = arena.total_size();
    return layout;
}

inline PTO2Runtime *runtime_init_data_from_layout(DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2RuntimeMode mode, void *sm_dev_base, uint64_t, void *gm_heap_dev_base, uint64_t heap_size)
{
    PTO2Runtime *rt = static_cast<PTO2Runtime *>(arena.region_ptr(layout.off_runtime));
    memset(rt, 0, sizeof(*rt));

    auto *sm_wrap = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(layout.off_sm_handle));
    memset(sm_wrap, 0, sizeof(*sm_wrap));

    // rt->ops is filled by the AICPU at boot.
    rt->mode = mode;
    rt->gm_heap = gm_heap_dev_base;
    rt->gm_heap_size = heap_size > 0 ? heap_size * PTO2_MAX_RING_DEPTH : 0;
    rt->gm_heap_owned = false;
    rt->total_cycles = 0;

    if (!rt->orchestrator.init_data_from_layout(layout.orch, arena, sm_dev_base, gm_heap_dev_base, heap_size, layout.task_window_size)) return nullptr;
    if (!rt->scheduler.init_data_from_layout(layout.sched, arena, sm_dev_base)) return nullptr;

    auto *mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.off_mailbox));
    memset(mailbox, 0, sizeof(*mailbox));

    return rt;
}

inline void runtime_wire_arena_pointers(DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2Runtime *rt)
{
    rt->sm_handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(layout.off_sm_handle));
    rt->aicore_mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.off_mailbox));
    rt->orchestrator.wire_arena_pointers(layout.orch, arena, &rt->scheduler);
    rt->scheduler.wire_arena_pointers(layout.sched, arena);
}

inline void runtime_destroy(PTO2Runtime *rt)
{
    // Arena buffer is pooled across runs by DeviceRunner — never freed here.
    if (!rt) return;
    rt->scheduler.destroy();
    rt->orchestrator.destroy();
    rt->aicore_mailbox = nullptr;
    rt->sm_handle = nullptr;
}

inline void runtime_set_mode(PTO2Runtime *rt, PTO2RuntimeMode mode)
{
    if (rt) rt->mode = mode;
}

inline void rt_scope_begin(PTO2Runtime *rt)
{
    PTO2ScopeMode mode = rt->pending_scope_mode;
    rt->pending_scope_mode = PTO2ScopeMode::AUTO;
    rt->orchestrator.begin_scope(mode);
}

inline void rt_scope_end(PTO2Runtime *rt)
{
    rt->orchestrator.end_scope();
}

inline void rt_orchestration_done(PTO2Runtime *rt)
{
    rt->orchestrator.mark_done();
}

inline void rt_report_fatal(PTO2Runtime *rt, int32_t error_code, const char *func, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    if (fmt == nullptr || fmt[0] == '\0')
    {
        rt->orchestrator.report_fatal(error_code, func, nullptr);
    }
    else
    {
        char message[1024];
        vsnprintf(message, sizeof(message), fmt, args);
        rt->orchestrator.report_fatal(error_code, func, "%s", message);
    }
    va_end(args);
}

MAYBE_UNINITIALIZED_BEGIN
inline bool wait_for_tensor_ready(PTO2Runtime *rt, const Tensor &tensor, bool wait_for_consumers, const char *caller)
{
    PTO2TaskId owner = tensor.owner_task_id;
    PTO2OrchestratorState &orch = rt->orchestrator;

    constexpr int kSegmentCap = 64;
    const PTO2TaskSlotState *seg[kSegmentCap];
    int seg_count = 0;
    bool signaled = false;
    bool failed = false;

    auto wait_one_producer = [&](const PTO2TaskSlotState &slot) {
        uint8_t ring_id = slot.ring_id;
        int32_t local_id = static_cast<int32_t>(slot.task->task_id.local());
        uint64_t t0 = get_sys_cnt_aicpu();
        int32_t spin_count = 0;
        while (slot.task_state.load(std::memory_order_acquire) < PTO2_TASK_COMPLETED)
        {
            SPIN_WAIT_HINT();
            if ((++spin_count & 1023) == 0 && get_sys_cnt_aicpu() - t0 > PTO2_TENSOR_DATA_TIMEOUT_CYCLES)
            {
                orch.report_fatal(PTO2_ERROR_TENSOR_WAIT_TIMEOUT, caller, "Timeout (%llu cycles): producer (ring=%d, local=%d) not completed", (unsigned long long)PTO2_TENSOR_DATA_TIMEOUT_CYCLES, ring_id, local_id);
                failed = true;
                return;
            }
        }
    };

    auto wait_one_consumers = [&](const PTO2TaskSlotState &slot) {
        uint8_t ring_id = slot.ring_id;
        int32_t local_id = slot.task->task_id.local();
        // With watermark-based reclamation, "all consumers done" means the
        // per-ring completed_watermark has reached this slot's recorded
        // last_consumer_local_id.
        PTO2SharedMemoryRingHeader &ring_hdr = rt->orchestrator.sm_header->rings[ring_id];
        int32_t target = slot.last_consumer_local_id;
        uint64_t t0 = get_sys_cnt_aicpu();
        int32_t spin_count = 0;
        while (ring_hdr.completed_watermark.load(std::memory_order_acquire) < target)
        {
            SPIN_WAIT_HINT();
            if ((++spin_count & 1023) == 0 && get_sys_cnt_aicpu() - t0 > PTO2_TENSOR_DATA_TIMEOUT_CYCLES)
            {
                orch.report_fatal(PTO2_ERROR_TENSOR_WAIT_TIMEOUT, caller, "Timeout (%llu cycles): consumers of producer (ring=%d, local=%d) not done", (unsigned long long)PTO2_TENSOR_DATA_TIMEOUT_CYCLES, ring_id, local_id);
                failed = true;
                return;
            }
        }
    };

    auto flush_segment = [&]() {
        for (int i = 0; i < seg_count; i++)
        {
            wait_one_producer(*seg[i]);
            if (failed) return;
            if (!wait_for_consumers) continue;
            wait_one_consumers(*seg[i]);
            if (failed) return;
        }
        seg_count = 0;
    };

    auto try_push = [&](const PTO2TaskSlotState &s) {
        for (int j = 0; j < seg_count; j++)
            if (seg[j] == &s) return;
        if (seg_count == kSegmentCap)
        {
            flush_segment();
            if (failed) return;
        }
        seg[seg_count++] = &s;
        if (!signaled)
        {
            orch.scheduler->wiring.orch_needs_drain.store(true, std::memory_order_release);
            signaled = true;
        }
    };

    auto do_wait = [&]() {
        if (owner.is_valid())
        {
            auto &s = orch.sm_header->rings[owner.ring()].get_slot_state_by_task_id(owner.local());
            try_push(s);
            if (failed) return;
        }

        orch.tensor_map.lookup(tensor, [&](PTO2TensorMapEntry &entry, OverlapStatus) -> bool {
            PTO2TaskId pid = entry.producer_task_id;
            auto &s = orch.sm_header->rings[pid.ring()].get_slot_state_by_task_id(pid.local());
            try_push(s);
            return !failed;
        });
        if (failed) return;
        flush_segment();
    };

    do_wait();
    if (signaled) orch.scheduler->wiring.orch_needs_drain.store(false, std::memory_order_release);
    return !failed;
}
MAYBE_UNINITIALIZED_END

inline uint64_t get_tensor_data(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[])
{
    if (tensor.buffer.addr == 0) return 0;

    if (!wait_for_tensor_ready(rt, tensor, false, __FUNCTION__)) return 0;

    uint64_t flat_offset = tensor.compute_flat_offset(indices, ndims);
    uint64_t elem_size = get_element_size(tensor.dtype);
    const void *ptr = reinterpret_cast<const void *>(tensor.buffer.addr + flat_offset * elem_size);
    uint64_t result = 0;
    memcpy(&result, ptr, elem_size);
    return result;
}

inline void set_tensor_data(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value)
{
    if (tensor.buffer.addr == 0) return;

    // Wait for producer + all consumers before writing (WAW + WAR safety)
    if (!wait_for_tensor_ready(rt, tensor, true, __FUNCTION__)) return;

    uint64_t flat_offset = tensor.compute_flat_offset(indices, ndims);
    uint64_t elem_size = get_element_size(tensor.dtype);
    void *ptr = reinterpret_cast<void *>(tensor.buffer.addr + flat_offset * elem_size);
    memcpy(ptr, &value, elem_size);
}

// Function-pointer ops table backing — moved from pto_runtime2.cpp so that
// the inline runtime_finalize_after_wire above can refer to it.

inline TaskOutputTensors submit_task_impl(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const Arg &args)
{
    return rt->orchestrator.submit_task(mixed_kernels, args);
}

inline TaskOutputTensors alloc_tensors_impl(PTO2Runtime *rt, const Arg &args)
{
    return rt->orchestrator.alloc_tensors(args);
}

inline TaskOutputTensors submit_dummy_task_impl(PTO2Runtime *rt, const Arg &args)
{
    return rt->orchestrator.submit_dummy_task(args);
}

inline bool is_fatal_impl(PTO2Runtime *rt)
{
    return rt->orchestrator.fatal;
}

inline const PTO2RuntimeOps s_runtime_ops = {
    .submit_task = submit_task_impl,
    .scope_begin = rt_scope_begin,
    .scope_end = rt_scope_end,
    .orchestration_done = rt_orchestration_done,
    .is_fatal = is_fatal_impl,
    .report_fatal = rt_report_fatal,
    .get_tensor_data = get_tensor_data,
    .set_tensor_data = set_tensor_data,
    .alloc_tensors = alloc_tensors_impl,
    .submit_dummy_task = submit_dummy_task_impl,
    .scope_set_site = nullptr,
};

inline void runtime_finalize_after_wire(PTO2Runtime *rt, int32_t aic_count, int32_t aiv_count)
{
    rt->ops = &s_runtime_ops;
    rt->orchestrator.total_cluster_count = aic_count;
    rt->orchestrator.total_aiv_count = aiv_count;
}

#ifndef PTO2_ORCHESTRATION_CONFIG_DEFINED
#define PTO2_ORCHESTRATION_CONFIG_DEFINED
struct PTO2OrchestrationConfig
{
    int expected_arg_count;
};
#endif
