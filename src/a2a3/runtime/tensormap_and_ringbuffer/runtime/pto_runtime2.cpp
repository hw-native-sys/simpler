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
 * PTO Runtime2 - Main Implementation
 *
 * Implements the unified runtime API that combines orchestrator and scheduler.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_runtime2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

#include "aicpu/device_time.h"
#include "common/unified_log.h"

// Weak fallback for HOST .so builds (never called, but satisfies linker).
// The AICPU build links the strong symbol from platform/.../device_time.cpp.
// Hidden visibility prevents HOST .so from polluting global symbol table.
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }

// =============================================================================
// Orchestration Ops Table (function-pointer dispatch for orchestration .so)
// =============================================================================

static TaskOutputTensors submit_task_impl(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const Arg &args) {
    return pto2_submit_mixed_task(&rt->orchestrator, mixed_kernels, args);
}

static TaskOutputTensors alloc_tensors_impl(PTO2Runtime *rt, const Arg &args) {
    return pto2_alloc_tensors(&rt->orchestrator, args);
}

PTO2ManualSubmitResult pto2_rt_submit_task_manual(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const Arg &args) {
    return pto2_submit_mixed_task_manual(&rt->orchestrator, mixed_kernels, args);
}

PTO2ManualSubmitResult pto2_rt_submit_task_manual_with_deps(
    PTO2Runtime *rt, const MixedKernels &mixed_kernels, const Arg &args, const PTO2TaskId explicit_producer_ids[],
    int32_t explicit_producer_count
) {
    return pto2_submit_mixed_task_manual_with_deps(
        &rt->orchestrator, mixed_kernels, args, explicit_producer_ids, explicit_producer_count
    );
}

void pto2_rt_add_dependency(PTO2Runtime *rt, PTO2TaskId producer, PTO2TaskId consumer) {
    pto2_add_dependency(&rt->orchestrator, producer, consumer);
}

void pto2_rt_scope_begin(PTO2Runtime *rt, PTO2ScopeMode mode) {
    pto2_scope_begin(&rt->orchestrator, mode);
}

void pto2_rt_scope_end(PTO2Runtime *rt) { pto2_scope_end(&rt->orchestrator); }

void pto2_rt_orchestration_done(PTO2Runtime *rt) { pto2_orchestrator_done(&rt->orchestrator); }

static bool is_fatal_impl(PTO2Runtime *rt) { return rt->orchestrator.fatal; }

static bool in_manual_scope_runtime(PTO2Runtime *rt) {
    return rt->orchestrator.manual_scope_active;
}

static void fail_manual_tensor_access(PTO2Runtime *rt, const char *caller) {
    PTO2OrchestratorState &orch = rt->orchestrator;
    orch.sm_handle->header->orch_error_code.store(PTO2_ERROR_INVALID_ARGS, std::memory_order_release);
    orch.fatal = true;
    unified_log_error(
        caller,
        "blocking tensor data access is not supported inside PTO2_SCOPE(PTO2ScopeMode::MANUAL); exit the manual scope first"
    );
}

// Wait for all producers of this tensor to be safe for data access.
// Checks owner metadata (lifecycle anchor) and OverlapMap (modifier writers).
// For reads: wait until each producer COMPLETED (done writing).
// For writes: also wait until all consumers done reading
//   (fanout_refcount >= fanout_count - 1, excluding scope reference).
// Uses cycle-based timeout (checked every 1024 spins).
// Returns false on timeout (sets orch.fatal).
MAYBE_UNINITIALIZED_BEGIN
static bool wait_for_tensor_ready(PTO2Runtime *rt, const Tensor &tensor, bool wait_for_consumers, const char *caller) {
    PTO2OrchestratorState &orch = rt->orchestrator;

    // Collect producer slot states from both maps, deduplicated by pointer.
    // +1: one creator slot + up to PTO2_LOOKUP_MAX_RESULTS modifier slots.
    constexpr int kMaxWait = PTO2_LOOKUP_MAX_RESULTS + 1;
    PTO2TaskSlotState *slots[kMaxWait];
    int slot_count = 0;

    // Step A: creator retention — read owner directly from tensor metadata
    PTO2TaskId owner = tensor.owner_task_id;
    if (owner.is_valid()) {
        slots[slot_count++] = &rt->scheduler.ring_sched_states[owner.ring()].get_slot_state_by_task_id(owner.local());
    }

    // Step B: modifier writer lookup (OverlapMap)
    PTO2LookupResult lookup_result;
    orch.tensor_map.lookup(tensor, lookup_result);
    for (int r = 0; r < lookup_result.count; r++) {
        PTO2TaskId pid = lookup_result.entries[r].entry->producer_task_id;
        PTO2TaskSlotState *s = &rt->scheduler.ring_sched_states[pid.ring()].get_slot_state_by_task_id(pid.local());
        bool already = false;
        for (int j = 0; j < slot_count; j++) {
            if (slots[j] == s) {
                already = true;
                break;
            }
        }
        if (!already && slot_count < kMaxWait) {
            slots[slot_count++] = s;
        }
    }

    // Wait for each producer
    for (int p = 0; p < slot_count; p++) {
        PTO2TaskSlotState &slot = *slots[p];
        uint8_t ring_id = slot.ring_id;
        int32_t local_id = static_cast<int32_t>(slot.task->task_id.local());

        uint64_t t0 = get_sys_cnt_aicpu();
        int32_t spin_count = 0;
        while (slot.task_state.load(std::memory_order_acquire) < PTO2_TASK_COMPLETED) {
            SPIN_WAIT_HINT();
            if ((++spin_count & 1023) == 0 && get_sys_cnt_aicpu() - t0 > PTO2_TENSOR_DATA_TIMEOUT_CYCLES) {
                orch.fatal = true;
                unified_log_error(
                    caller, "Timeout (%llu cycles): producer (ring=%d, local=%d) not completed",
                    (unsigned long long)PTO2_TENSOR_DATA_TIMEOUT_CYCLES,  // NOLINT(runtime/int)
                    ring_id, local_id
                );
                return false;
            }
        }

        if (wait_for_consumers) {
            t0 = get_sys_cnt_aicpu();
            spin_count = 0;
            while (slot.fanout_refcount.load(std::memory_order_acquire) < slot.fanout_count - 1) {
                SPIN_WAIT_HINT();
                if ((++spin_count & 1023) == 0 && get_sys_cnt_aicpu() - t0 > PTO2_TENSOR_DATA_TIMEOUT_CYCLES) {
                    orch.fatal = true;
                    unified_log_error(
                        caller, "Timeout (%llu cycles): consumers of producer (ring=%d, local=%d) not done",
                        (unsigned long long)PTO2_TENSOR_DATA_TIMEOUT_CYCLES,  // NOLINT(runtime/int)
                        ring_id, local_id
                    );
                    return false;
                }
            }
        }
    }
    return true;
}
MAYBE_UNINITIALIZED_END

uint64_t pto2_get_tensor_data(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[]) {
    if (in_manual_scope_runtime(rt)) {
        fail_manual_tensor_access(rt, __FUNCTION__);
        return 0;
    }
    if (tensor.buffer.addr == 0) {
        unified_log_error(
            __FUNCTION__, "get_tensor_data: buffer not allocated (addr=0). "
                          "Use the Tensor returned by add_output(TensorCreateInfo) after submit returns."
        );
        return 0;
    }

    if (!wait_for_tensor_ready(rt, tensor, false, __FUNCTION__)) {
        return 0;
    }

    uint64_t flat_offset = tensor.compute_flat_offset(indices, ndims);
    uint64_t elem_size = get_element_size(tensor.dtype);
    const void *ptr = reinterpret_cast<const void *>(tensor.buffer.addr + flat_offset * elem_size);
    uint64_t result = 0;
    memcpy(&result, ptr, elem_size);
    return result;
}

void pto2_set_tensor_data(
    PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
) {
    if (in_manual_scope_runtime(rt)) {
        fail_manual_tensor_access(rt, __FUNCTION__);
        return;
    }
    if (tensor.buffer.addr == 0) {
        unified_log_error(
            __FUNCTION__, "set_tensor_data: buffer not allocated (addr=0). "
                          "Use the Tensor returned by add_output(TensorCreateInfo) after submit returns."
        );
        return;
    }

    // Wait for producer + all consumers before writing (WAW + WAR safety)
    if (!wait_for_tensor_ready(rt, tensor, true, __FUNCTION__)) {
        return;
    }

    uint64_t flat_offset = tensor.compute_flat_offset(indices, ndims);
    uint64_t elem_size = get_element_size(tensor.dtype);
    void *ptr = reinterpret_cast<void *>(tensor.buffer.addr + flat_offset * elem_size);
    memcpy(ptr, &value, elem_size);
}

static const PTO2RuntimeOps s_runtime_ops = {
    .submit_task = submit_task_impl,
    .submit_task_manual = pto2_rt_submit_task_manual,
    .submit_task_manual_with_deps = pto2_rt_submit_task_manual_with_deps,
    .add_dependency = pto2_rt_add_dependency,
    .scope_begin = pto2_rt_scope_begin,
    .scope_end = pto2_rt_scope_end,
    .orchestration_done = pto2_rt_orchestration_done,
    .is_fatal = is_fatal_impl,
    .log_error = unified_log_error,
    .log_warn = unified_log_warn,
    .log_info = unified_log_info,
    .log_debug = unified_log_debug,
    .log_always = unified_log_always,
    .get_tensor_data = pto2_get_tensor_data,
    .set_tensor_data = pto2_set_tensor_data,
    .alloc_tensors = alloc_tensors_impl,
};

// =============================================================================
// Runtime Creation and Destruction
// =============================================================================

PTO2Runtime *pto2_runtime_create(PTO2RuntimeMode mode) {
    return pto2_runtime_create_custom(mode, PTO2_TASK_WINDOW_SIZE, PTO2_HEAP_SIZE);
}

PTO2Runtime *pto2_runtime_create_custom(
    PTO2RuntimeMode mode, uint64_t task_window_size, uint64_t heap_size, int32_t dep_pool_capacity
) {
    // Allocate runtime context
    PTO2Runtime *rt = static_cast<PTO2Runtime *>(calloc(1, sizeof(PTO2Runtime)));
    if (!rt) {
        return NULL;
    }

    rt->ops = &s_runtime_ops;
    rt->mode = mode;
    rt->sm_handle = pto2_sm_create(task_window_size, heap_size);
    if (!rt->sm_handle) {
        free(rt);
        return NULL;
    }

    // Allocate GM heap for output buffers (all rings combined)
    uint64_t total_heap_size = heap_size * PTO2_MAX_RING_DEPTH;
    rt->gm_heap_size = total_heap_size;
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    if (posix_memalign(&rt->gm_heap, PTO2_ALIGN_SIZE, total_heap_size) != 0) {
        pto2_sm_destroy(rt->sm_handle);
        free(rt);
        return NULL;
    }
#else
    rt->gm_heap = aligned_alloc(PTO2_ALIGN_SIZE, total_heap_size);
    if (!rt->gm_heap) {
        pto2_sm_destroy(rt->sm_handle);
        free(rt);
        return NULL;
    }
#endif
    rt->gm_heap_owned = true;

    // Initialize orchestrator
    if (!pto2_orchestrator_init(&rt->orchestrator, rt->sm_handle, rt->gm_heap, heap_size, dep_pool_capacity)) {
        free(rt->gm_heap);
        pto2_sm_destroy(rt->sm_handle);
        free(rt);
        return NULL;
    }

    // Initialize scheduler (heap_size = per-ring heap size)
    if (!pto2_scheduler_init(&rt->scheduler, rt->sm_handle)) {
        pto2_orchestrator_destroy(&rt->orchestrator);
        free(rt->gm_heap);
        pto2_sm_destroy(rt->sm_handle);
        free(rt);
        return NULL;
    }

    // Connect orchestrator to scheduler (for simulated mode)
    pto2_orchestrator_set_scheduler(&rt->orchestrator, &rt->scheduler);

    return rt;
}

PTO2Runtime *pto2_runtime_create_from_sm(
    PTO2RuntimeMode mode, PTO2SharedMemoryHandle *sm_handle, void *gm_heap, uint64_t heap_size,
    int32_t dep_pool_capacity
) {
    if (!sm_handle) return NULL;

    PTO2Runtime *rt = static_cast<PTO2Runtime *>(calloc(1, sizeof(PTO2Runtime)));
    if (!rt) return NULL;

    rt->ops = &s_runtime_ops;
    rt->mode = mode;
    rt->sm_handle = sm_handle;
    rt->gm_heap = gm_heap;
    rt->gm_heap_size = heap_size > 0 ? heap_size * PTO2_MAX_RING_DEPTH : 0;
    rt->gm_heap_owned = false;

    if (!pto2_orchestrator_init(&rt->orchestrator, rt->sm_handle, rt->gm_heap, heap_size, dep_pool_capacity)) {
        free(rt);
        return NULL;
    }

    // Initialize scheduler (heap_size = per-ring heap size)
    if (!pto2_scheduler_init(&rt->scheduler, rt->sm_handle)) {
        pto2_orchestrator_destroy(&rt->orchestrator);
        free(rt);
        return NULL;
    }

    pto2_orchestrator_set_scheduler(&rt->orchestrator, &rt->scheduler);

    return rt;
}

void pto2_runtime_destroy(PTO2Runtime *rt) {
    if (!rt) return;

    pto2_scheduler_destroy(&rt->scheduler);
    pto2_orchestrator_destroy(&rt->orchestrator);

    if (rt->gm_heap_owned && rt->gm_heap) {
        free(rt->gm_heap);
    }

    if (rt->sm_handle) {
        pto2_sm_destroy(rt->sm_handle);
    }

    free(rt);
}

void pto2_runtime_set_mode(PTO2Runtime *rt, PTO2RuntimeMode mode) {
    if (rt) {
        rt->mode = mode;
    }
}
