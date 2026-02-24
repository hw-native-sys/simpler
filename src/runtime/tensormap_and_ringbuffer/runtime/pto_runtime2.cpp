/**
 * PTO Runtime2 - Main Implementation
 *
 * Implements the unified runtime API that combines orchestrator and scheduler.
 *
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_runtime2.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// =============================================================================
// Ops Table (function-pointer dispatch for orchestration .so)
// =============================================================================

static void submit_task_impl(PTO2Runtime* rt, int32_t kernel_id,
                              PTO2WorkerType worker_type,
                              PTOParam* params, int32_t num_params) {
    pto2_submit_task(&rt->orchestrator, kernel_id, worker_type,
                     params, num_params);
}

static void scope_begin_impl(PTO2Runtime* rt) {
    pto2_scope_begin(&rt->orchestrator);
}

static void scope_end_impl(PTO2Runtime* rt) {
    pto2_scope_end(&rt->orchestrator);
}

static void orchestration_done_impl(PTO2Runtime* rt) {
    pto2_orchestrator_done(&rt->orchestrator);
}

static const PTO2RuntimeOps s_runtime_ops = {
    .submit_task        = submit_task_impl,
    .scope_begin        = scope_begin_impl,
    .scope_end          = scope_end_impl,
    .orchestration_done = orchestration_done_impl,
};

// =============================================================================
// Runtime Creation and Destruction
// =============================================================================

PTO2Runtime* pto2_runtime_create(PTO2RuntimeMode mode) {
    return pto2_runtime_create_custom(mode,
                                       PTO2_TASK_WINDOW_SIZE,
                                       PTO2_HEAP_SIZE,
                                       PTO2_DEP_LIST_POOL_SIZE);
}

PTO2Runtime* pto2_runtime_create_custom(PTO2RuntimeMode mode,
                                         int32_t task_window_size,
                                         int32_t heap_size,
                                         int32_t dep_list_size) {
    // Allocate runtime context
    PTO2Runtime* rt = (PTO2Runtime*)calloc(1, sizeof(PTO2Runtime));
    if (!rt) {
        return NULL;
    }

    rt->ops = &s_runtime_ops;
    rt->mode = mode;
    rt->sm_handle = pto2_sm_create(task_window_size, heap_size, dep_list_size);
    if (!rt->sm_handle) {
        free(rt);
        return NULL;
    }

    // Allocate GM heap for output buffers
    rt->gm_heap_size = heap_size;
    #if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
        if (posix_memalign(&rt->gm_heap, PTO2_ALIGN_SIZE, heap_size) != 0) {
            pto2_sm_destroy(rt->sm_handle);
            free(rt);
            return NULL;
        }
    #else
        rt->gm_heap = aligned_alloc(PTO2_ALIGN_SIZE, heap_size);
        if (!rt->gm_heap) {
            pto2_sm_destroy(rt->sm_handle);
            free(rt);
            return NULL;
        }
    #endif
    rt->gm_heap_owned = true;

    // Initialize orchestrator
    if (!pto2_orchestrator_init(&rt->orchestrator, rt->sm_handle,
                                 rt->gm_heap, heap_size)) {
        free(rt->gm_heap);
        pto2_sm_destroy(rt->sm_handle);
        free(rt);
        return NULL;
    }

    // Initialize scheduler
    if (!pto2_scheduler_init(&rt->scheduler, rt->sm_handle,
                              &rt->orchestrator.dep_pool)) {
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

PTO2Runtime* pto2_runtime_create_from_sm(PTO2RuntimeMode mode,
                                          PTO2SharedMemoryHandle* sm_handle,
                                          void* gm_heap,
                                          int32_t heap_size) {
    if (!sm_handle) return NULL;

    PTO2Runtime* rt = (PTO2Runtime*)calloc(1, sizeof(PTO2Runtime));
    if (!rt) return NULL;

    rt->ops = &s_runtime_ops;
    rt->mode = mode;
    rt->sm_handle = sm_handle;
    rt->gm_heap = gm_heap;
    rt->gm_heap_size = heap_size > 0 ? heap_size : 0;
    rt->gm_heap_owned = false;

    if (!pto2_orchestrator_init(&rt->orchestrator, rt->sm_handle,
                                rt->gm_heap, rt->gm_heap_size)) {
        free(rt);
        return NULL;
    }

    if (!pto2_scheduler_init(&rt->scheduler, rt->sm_handle,
                             &rt->orchestrator.dep_pool)) {
        pto2_orchestrator_destroy(&rt->orchestrator);
        free(rt);
        return NULL;
    }

    pto2_orchestrator_set_scheduler(&rt->orchestrator, &rt->scheduler);
    return rt;
}

void pto2_runtime_destroy(PTO2Runtime* rt) {
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

void pto2_runtime_reset(PTO2Runtime* rt) {
    if (!rt) return;

    pto2_orchestrator_reset(&rt->orchestrator);
    pto2_scheduler_reset(&rt->scheduler);
    pto2_sm_reset(rt->sm_handle);

    rt->total_cycles = 0;
}

void pto2_runtime_set_mode(PTO2Runtime* rt, PTO2RuntimeMode mode) {
    if (rt) {
        rt->mode = mode;
    }
}

// =============================================================================
// Orchestration API
// =============================================================================

void pto2_rt_scope_begin(PTO2Runtime* rt) {
    scope_begin_impl(rt);
}

void pto2_rt_scope_end(PTO2Runtime* rt) {
    scope_end_impl(rt);
}

void pto2_rt_orchestration_done(PTO2Runtime* rt) {
    orchestration_done_impl(rt);
}
