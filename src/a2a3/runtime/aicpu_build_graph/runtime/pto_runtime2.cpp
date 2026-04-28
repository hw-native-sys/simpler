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

#include "common/unified_log.h"

// =============================================================================
// Orchestration Ops Table (function-pointer dispatch for orchestration .so)
// =============================================================================

static SubmitResult submit_task_impl(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const Arg &args) {
    return pto2_submit_mixed_task(&rt->orchestrator, mixed_kernels, args);
}

static void add_dependency_impl(PTO2Runtime *rt, PTO2TaskId producer, PTO2TaskId consumer) {
    pto2_add_dependency(&rt->orchestrator, producer, consumer);
}

void rt_scope_begin(PTO2Runtime *rt) { pto2_scope_begin(&rt->orchestrator); }

void rt_scope_end(PTO2Runtime *rt) { pto2_scope_end(&rt->orchestrator); }

void rt_orchestration_done(PTO2Runtime *rt) { pto2_orchestrator_done(&rt->orchestrator); }

static bool is_fatal_impl(PTO2Runtime *rt) { return rt->orchestrator.fatal; }

static const PTO2RuntimeOps s_runtime_ops = {
    .submit_task = submit_task_impl,
    .add_dependency = add_dependency_impl,
    .scope_begin = rt_scope_begin,
    .scope_end = rt_scope_end,
    .orchestration_done = rt_orchestration_done,
    .is_fatal = is_fatal_impl,
    .log_error = unified_log_error,
    .log_warn = unified_log_warn,
    .log_info = unified_log_info,
    .log_debug = unified_log_debug,
    .log_always = unified_log_always,
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
    PTO2Runtime *rt = reinterpret_cast<PTO2Runtime *>(calloc(1, sizeof(PTO2Runtime)));
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
    if (!pto2_scheduler_init(&rt->scheduler, rt->sm_handle, rt->gm_heap, heap_size)) {
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

    PTO2Runtime *rt = reinterpret_cast<PTO2Runtime *>(calloc(1, sizeof(PTO2Runtime)));
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
    if (!pto2_scheduler_init(&rt->scheduler, rt->sm_handle, rt->gm_heap, heap_size)) {
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
