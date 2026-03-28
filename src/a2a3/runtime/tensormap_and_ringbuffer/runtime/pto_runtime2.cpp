/**
 * PTO Runtime2 - Main Implementation
 *
 * Implements the unified runtime API that combines orchestrator and scheduler.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_runtime2.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "common/unified_log.h"
#include "aicpu/device_time.h"

// Weak fallback for HOST .so builds (never called, but satisfies linker).
// The AICPU build links the strong symbol from platform/.../device_time.cpp.
// Hidden visibility prevents HOST .so from polluting global symbol table.
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }

// =============================================================================
// Thread-local orchestrator index for multi-orchestrator dispatch
// =============================================================================

thread_local int pto2_current_orch_idx = 0;

void pto2_set_orch_thread_idx(int idx) {
    pto2_current_orch_idx = idx;
}

// =============================================================================
// Orchestration Ops Table (function-pointer dispatch for orchestration .so)
// =============================================================================

static TaskOutputTensors submit_task_impl(PTO2Runtime* rt, const MixedKernels& mixed_kernels,
                             const Arg& args) {
    return pto2_submit_mixed_task(&rt->orchestrators[pto2_current_orch_idx], mixed_kernels,
                           args);
}

void pto2_rt_scope_begin(PTO2Runtime* rt) {
    pto2_scope_begin(&rt->orchestrators[pto2_current_orch_idx]);
}

void pto2_rt_scope_end(PTO2Runtime* rt) {
    pto2_scope_end(&rt->orchestrators[pto2_current_orch_idx]);
}

void pto2_rt_orchestration_done(PTO2Runtime* rt) {
    pto2_orchestrator_done(&rt->orchestrators[pto2_current_orch_idx]);
}

static bool is_fatal_impl(PTO2Runtime* rt) {
    return rt->orchestrators[pto2_current_orch_idx].fatal;
}

// Wait for TensorMap producers of this tensor to be safe for data access.
// For reads: wait until producer COMPLETED (done writing).
// For writes: also wait until all consumers done reading
//   (fanout_refcount >= fanout_count - 1, excluding scope reference).
// Uses cycle-based timeout (checked every 1024 spins).
// Returns false on timeout (sets orch.fatal).
static bool wait_for_tensor_ready(PTO2Runtime* rt, const Tensor& tensor,
                                  bool wait_for_consumers, const char* caller) {
    PTO2OrchestratorState& orch = rt->orchestrators[pto2_current_orch_idx];
    PTO2LookupResult lookup_result;
    orch.tensor_map.lookup(tensor, lookup_result);

    for (int r = 0; r < lookup_result.count; r++) {
        PTO2TensorMapEntry& entry = *lookup_result.entries[r].entry;
        PTO2TaskId producer_id = entry.producer_task_id;
        uint8_t ring_id = producer_id.ring();
        int32_t local_id = producer_id.local();
        PTO2TaskSlotState& slot =
            rt->scheduler.ring_sched_states[ring_id].get_slot_state_by_task_id(local_id);

        // Wait for producer to complete (WAW safety)
        uint64_t t0 = get_sys_cnt_aicpu();
        int32_t spin_count = 0;
        while (slot.task_state.load(std::memory_order_acquire) < PTO2_TASK_COMPLETED) {
            SPIN_WAIT_HINT();
            if ((++spin_count & 1023) == 0 &&
                get_sys_cnt_aicpu() - t0 > PTO2_TENSOR_DATA_TIMEOUT_CYCLES) {
                orch.fatal = true;
                unified_log_error(caller,
                    "Timeout (%llu cycles): producer (ring=%d, local=%d) not completed",
                    (unsigned long long)PTO2_TENSOR_DATA_TIMEOUT_CYCLES, ring_id, local_id);
                return false;
            }
        }

        // For writes: also wait for all consumers to finish reading (WAR safety).
        // fanout_count includes 1 scope reference that won't release until scope_end,
        // so wait until fanout_refcount >= fanout_count - 1.
        if (wait_for_consumers) {
            t0 = get_sys_cnt_aicpu();
            spin_count = 0;
            while (slot.fanout_refcount.load(std::memory_order_acquire)
                   < slot.fanout_count - 1) {
                SPIN_WAIT_HINT();
                if ((++spin_count & 1023) == 0 &&
                    get_sys_cnt_aicpu() - t0 > PTO2_TENSOR_DATA_TIMEOUT_CYCLES) {
                    orch.fatal = true;
                    unified_log_error(caller,
                        "Timeout (%llu cycles): consumers of producer (ring=%d, local=%d) not done",
                        (unsigned long long)PTO2_TENSOR_DATA_TIMEOUT_CYCLES, ring_id, local_id);
                    return false;
                }
            }
        }
    }
    return true;
}

uint64_t pto2_get_tensor_data(PTO2Runtime* rt, const Tensor& tensor,
                              uint32_t ndims, const uint32_t indices[]) {
    if (tensor.buffer.addr == 0) {
        unified_log_error(__FUNCTION__,
            "get_tensor_data: buffer not allocated (addr=0). "
            "Use the Tensor returned by add_output(TensorCreateInfo) after submit returns.");
        return 0;
    }

    if (!wait_for_tensor_ready(rt, tensor, false, __FUNCTION__)) {
        return 0;
    }

    uint64_t flat_offset = tensor.compute_flat_offset(indices, ndims);
    uint64_t elem_size = get_element_size(tensor.dtype);
    const void* ptr = reinterpret_cast<const void*>(
        tensor.buffer.addr + flat_offset * elem_size);
    uint64_t result = 0;
    memcpy(&result, ptr, elem_size);
    return result;
}

void pto2_set_tensor_data(PTO2Runtime* rt, const Tensor& tensor,
                          uint32_t ndims, const uint32_t indices[],
                          uint64_t value) {
    if (tensor.buffer.addr == 0) {
        unified_log_error(__FUNCTION__,
            "set_tensor_data: buffer not allocated (addr=0). "
            "Use the Tensor returned by add_output(TensorCreateInfo) after submit returns.");
        return;
    }

    // Wait for producer + all consumers before writing (WAW + WAR safety)
    if (!wait_for_tensor_ready(rt, tensor, true, __FUNCTION__)) {
        return;
    }

    uint64_t flat_offset = tensor.compute_flat_offset(indices, ndims);
    uint64_t elem_size = get_element_size(tensor.dtype);
    void* ptr = reinterpret_cast<void*>(
        tensor.buffer.addr + flat_offset * elem_size);
    memcpy(ptr, &value, elem_size);
}

static const PTO2RuntimeOps s_runtime_ops = {
    .submit_task          = submit_task_impl,
    .scope_begin          = pto2_rt_scope_begin,
    .scope_end            = pto2_rt_scope_end,
    .orchestration_done   = pto2_rt_orchestration_done,
    .is_fatal             = is_fatal_impl,
    .log_error            = unified_log_error,
    .log_warn             = unified_log_warn,
    .log_info             = unified_log_info,
    .log_debug            = unified_log_debug,
    .log_always           = unified_log_always,
    .get_tensor_data      = pto2_get_tensor_data,
    .set_tensor_data      = pto2_set_tensor_data,
};

// =============================================================================
// Runtime Creation and Destruction
// =============================================================================

PTO2Runtime* pto2_runtime_create(PTO2RuntimeMode mode) {
    return pto2_runtime_create_custom(mode,
                                       PTO2_TASK_WINDOW_SIZE,
                                       PTO2_HEAP_SIZE);
}

PTO2Runtime* pto2_runtime_create_custom(PTO2RuntimeMode mode,
                                         uint64_t task_window_size,
                                         uint64_t heap_size,
                                         int32_t dep_pool_capacity) {
    // Allocate runtime context
    PTO2Runtime* rt = (PTO2Runtime*)calloc(1, sizeof(PTO2Runtime));
    if (!rt) {
        return NULL;
    }

    rt->ops = &s_runtime_ops;
    rt->mode = mode;
    rt->orch_count = 1;
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

    // Initialize first orchestrator
    if (!pto2_orchestrator_init(&rt->orchestrators[0], rt->sm_handle,
                                 rt->gm_heap, heap_size, dep_pool_capacity)) {
        free(rt->gm_heap);
        pto2_sm_destroy(rt->sm_handle);
        free(rt);
        return NULL;
    }

    // Initialize scheduler (heap_size = per-ring heap size)
    if (!pto2_scheduler_init(&rt->scheduler, rt->sm_handle)) {
        pto2_orchestrator_destroy(&rt->orchestrators[0]);
        free(rt->gm_heap);
        pto2_sm_destroy(rt->sm_handle);
        free(rt);
        return NULL;
    }

    // Connect orchestrator to scheduler (for simulated mode)
    pto2_orchestrator_set_scheduler(&rt->orchestrators[0], &rt->scheduler);

    return rt;
}

PTO2Runtime* pto2_runtime_create_from_sm(PTO2RuntimeMode mode,
                                          PTO2SharedMemoryHandle* sm_handle,
                                          void* gm_heap,
                                          uint64_t heap_size,
                                          int orch_count,
                                          int32_t dep_pool_capacity) {
    if (!sm_handle) return NULL;
    if (orch_count < 1) orch_count = 1;
    if (orch_count > PTO2_MAX_ORCH_THREADS) orch_count = PTO2_MAX_ORCH_THREADS;

    PTO2Runtime* rt = (PTO2Runtime*)calloc(1, sizeof(PTO2Runtime));
    if (!rt) return NULL;

    rt->ops = &s_runtime_ops;
    rt->mode = mode;
    rt->sm_handle = sm_handle;
    rt->gm_heap = gm_heap;
    rt->gm_heap_size = heap_size > 0 ? heap_size * PTO2_MAX_RING_DEPTH : 0;
    rt->gm_heap_owned = false;
    rt->orch_count = orch_count;

    // Initialize all orchestrator states
    for (int i = 0; i < orch_count; i++) {
        if (!pto2_orchestrator_init(&rt->orchestrators[i], rt->sm_handle,
                                    rt->gm_heap, heap_size, dep_pool_capacity)) {
            for (int j = 0; j < i; j++) {
                pto2_orchestrator_destroy(&rt->orchestrators[j]);
            }
            free(rt);
            return NULL;
        }
    }

    // Initialize scheduler (heap_size = per-ring heap size)
    if (!pto2_scheduler_init(&rt->scheduler, rt->sm_handle)) {
        for (int i = 0; i < orch_count; i++) {
            pto2_orchestrator_destroy(&rt->orchestrators[i]);
        }
        free(rt);
        return NULL;
    }

    // Connect all orchestrators to scheduler
    for (int i = 0; i < orch_count; i++) {
        pto2_orchestrator_set_scheduler(&rt->orchestrators[i], &rt->scheduler);
    }

    return rt;
}

void pto2_runtime_destroy(PTO2Runtime* rt) {
    if (!rt) return;

    pto2_scheduler_destroy(&rt->scheduler);
    for (int i = 0; i < rt->orch_count; i++) {
        pto2_orchestrator_destroy(&rt->orchestrators[i]);
    }

    if (rt->gm_heap_owned && rt->gm_heap) {
        free(rt->gm_heap);
    }

    if (rt->sm_handle) {
        pto2_sm_destroy(rt->sm_handle);
    }

    free(rt);
}

void pto2_runtime_set_mode(PTO2Runtime* rt, PTO2RuntimeMode mode) {
    if (rt) {
        rt->mode = mode;
    }
}
