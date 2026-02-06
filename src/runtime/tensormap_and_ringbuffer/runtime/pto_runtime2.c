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
    
    rt->mode = mode;
    
    // Create shared memory
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
    pto2_scope_begin(&rt->orchestrator);
}

void pto2_rt_scope_end(PTO2Runtime* rt) {
    pto2_scope_end(&rt->orchestrator);
}

int32_t pto2_rt_submit_task(PTO2Runtime* rt,
                             int32_t kernel_id,
                             PTO2WorkerType worker_type,
                             void* func_ptr,
                             const char* func_name,
                             PTO2TaskParam* params,
                             int32_t num_params) {
    return pto2_submit_task(&rt->orchestrator, kernel_id, worker_type,
                            func_ptr, func_name, params, num_params);
}

int32_t pto2_rt_submit(PTO2Runtime* rt,
                        const char* func_name,
                        void* func_ptr,
                        PTO2TaskParam* params,
                        int32_t num_params) {
    // Auto-detect worker type based on function name
    PTO2WorkerType worker_type = PTO2_WORKER_VECTOR;  // Default
    
    if (func_name) {
        if (strstr(func_name, "gemm") || strstr(func_name, "matmul") ||
            strstr(func_name, "conv") || strstr(func_name, "cube")) {
            worker_type = PTO2_WORKER_CUBE;
        } else if (strstr(func_name, "dma") || strstr(func_name, "copy")) {
            worker_type = PTO2_WORKER_ACCELERATOR;
        } else if (strstr(func_name, "cpu") || strstr(func_name, "scalar")) {
            worker_type = PTO2_WORKER_AI_CPU;
        }
    }
    
    return pto2_submit_task(&rt->orchestrator, 0, worker_type,
                            func_ptr, func_name, params, num_params);
}

void pto2_rt_orchestration_done(PTO2Runtime* rt) {
    pto2_orchestrator_done(&rt->orchestrator);
}

void* pto2_rt_get_output(PTO2Runtime* rt, int32_t task_id, int32_t output_idx) {
    return pto2_task_get_output(&rt->orchestrator, task_id, output_idx);
}

// =============================================================================
// Execution API
// =============================================================================

void pto2_runtime_execute(PTO2Runtime* rt) {
    if (rt->mode == PTO2_MODE_GRAPH_ONLY) {
        return;  // Nothing to execute
    }
    
    // In simulated mode, process tasks in dependency order
    // For real execution, would dispatch to actual workers
    
    int max_iterations = rt->orchestrator.tasks_submitted * 10 + 1000;
    int iterations = 0;
    
    while (!pto2_scheduler_is_done(&rt->scheduler) && iterations < max_iterations) {
        iterations++;
        bool dispatched = false;
        
        // Try to dispatch ready tasks for each worker type
        for (int wtype = 0; wtype < PTO2_NUM_WORKER_TYPES; wtype++) {
            int32_t task_id = pto2_scheduler_get_ready_task(&rt->scheduler, (PTO2WorkerType)wtype);
            
            if (task_id >= 0) {
                dispatched = true;
                
                // Mark as running
                pto2_scheduler_mark_running(&rt->scheduler, task_id);
                
                // Execute task (in real mode, would dispatch to worker)
                PTO2TaskDescriptor* task = pto2_sm_get_task(rt->sm_handle, task_id);
                
                if (rt->mode == PTO2_MODE_SIMULATE) {
                    // Simulate execution with cycle counting
                    // TODO: Use cycle cost function
                    int64_t cycles = 1000;  // Placeholder
                    rt->total_cycles += cycles;
                }
                
                // Call the function if provided
                if (task->func_ptr) {
                    // Build args array from task parameters
                    void* args[PTO2_MAX_OUTPUTS];
                    int num_args = 0;
                    
                    // Add output pointers
                    for (int i = 0; i < task->num_outputs; i++) {
                        args[num_args++] = (char*)task->packed_buffer_base + 
                                          task->output_offsets[i];
                    }
                    
                    // Call function
                    PTO2InCoreFunc func = (PTO2InCoreFunc)task->func_ptr;
                    func(args, num_args);
                }
                
                // Mark task complete
                pto2_scheduler_on_task_complete(&rt->scheduler, task_id);
            }
        }
        
        if (!dispatched) {
            // No tasks ready, brief pause to avoid busy-waiting
            PTO2_SPIN_PAUSE();
        }
    }
}

void pto2_rt_task_complete(PTO2Runtime* rt, int32_t task_id) {
    pto2_scheduler_on_task_complete(&rt->scheduler, task_id);
}

int32_t pto2_rt_get_ready_task(PTO2Runtime* rt, PTO2WorkerType worker_type) {
    return pto2_scheduler_get_ready_task(&rt->scheduler, worker_type);
}

bool pto2_runtime_is_done(PTO2Runtime* rt) {
    return pto2_scheduler_is_done(&rt->scheduler);
}

// =============================================================================
// Statistics and Debug API
// =============================================================================

void pto2_runtime_print_stats(PTO2Runtime* rt) {
    printf("\n========== PTO Runtime2 Statistics ==========\n\n");
    
    // Shared memory layout
    pto2_sm_print_layout(rt->sm_handle);
    printf("\n");
    
    // Orchestrator stats
    pto2_orchestrator_print_stats(&rt->orchestrator);
    printf("\n");
    
    // Scheduler stats
    pto2_scheduler_print_stats(&rt->scheduler);
    printf("\n");
    
    // Ready queues
    pto2_scheduler_print_queues(&rt->scheduler);
    printf("\n");
    
    // TensorMap stats
    pto2_tensormap_print_stats(&rt->orchestrator.tensor_map);
    printf("\n");
    
    // Overall stats
    printf("=== Overall ===\n");
    printf("Mode:          %s\n", 
           rt->mode == PTO2_MODE_EXECUTE ? "EXECUTE" :
           rt->mode == PTO2_MODE_SIMULATE ? "SIMULATE" : "GRAPH_ONLY");
    printf("Total cycles:  %lld\n", (long long)rt->total_cycles);
    printf("===============\n");
    
    printf("\n================================================\n");
}

int64_t pto2_runtime_get_cycles(PTO2Runtime* rt) {
    return rt->total_cycles;
}

int pto2_runtime_dump_graph(PTO2Runtime* rt, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        return -1;
    }
    
    fprintf(f, "# PTO Runtime2 Task Graph\n");
    fprintf(f, "# Tasks: %lld\n\n", (long long)rt->orchestrator.tasks_submitted);
    
    // Dump task descriptors
    int32_t current_index = rt->sm_handle->header->current_task_index;
    int32_t last_alive = rt->sm_handle->header->last_task_alive;
    
    for (int32_t task_id = last_alive; task_id < current_index; task_id++) {
        PTO2TaskDescriptor* task = pto2_sm_get_task(rt->sm_handle, task_id);
        
        fprintf(f, "task %d: %s (kernel=%d, worker=%d)\n",
                task_id, 
                task->func_name ? task->func_name : "unknown",
                task->kernel_id,
                task->worker_type);
        
        // Dump fanin (dependencies)
        fprintf(f, "  fanin (%d): ", task->fanin_count);
        int32_t current = task->fanin_head;
        while (current > 0) {
            PTO2DepListEntry* entry = pto2_dep_pool_get(&rt->orchestrator.dep_pool, current);
            if (!entry) break;
            fprintf(f, "%d ", entry->task_id);
            current = entry->next_offset;
        }
        fprintf(f, "\n");
        
        // Dump fanout (consumers)
        fprintf(f, "  fanout (%d): ", task->fanout_count);
        current = task->fanout_head;
        while (current > 0) {
            PTO2DepListEntry* entry = pto2_dep_pool_get(&rt->orchestrator.dep_pool, current);
            if (!entry) break;
            fprintf(f, "%d ", entry->task_id);
            current = entry->next_offset;
        }
        fprintf(f, "\n\n");
    }
    
    fclose(f);
    return 0;
}

bool pto2_runtime_validate(PTO2Runtime* rt) {
    if (!rt) return false;
    
    // Validate shared memory
    if (!pto2_sm_validate(rt->sm_handle)) {
        fprintf(stderr, "Validation failed: shared memory\n");
        return false;
    }
    
    // Validate task ring consistency
    int32_t current_index = rt->sm_handle->header->current_task_index;
    int32_t last_alive = rt->sm_handle->header->last_task_alive;
    
    if (current_index < last_alive) {
        fprintf(stderr, "Validation failed: current_index < last_alive\n");
        return false;
    }
    
    if (current_index - last_alive > PTO2_TASK_WINDOW_SIZE) {
        fprintf(stderr, "Validation failed: task window overflow\n");
        return false;
    }
    
    // Validate heap ring consistency
    int32_t heap_top = rt->sm_handle->header->heap_top;
    int32_t heap_tail = rt->sm_handle->header->heap_tail;
    
    if (heap_top < 0 || heap_top > rt->gm_heap_size) {
        fprintf(stderr, "Validation failed: heap_top out of range\n");
        return false;
    }
    
    if (heap_tail < 0 || heap_tail > rt->gm_heap_size) {
        fprintf(stderr, "Validation failed: heap_tail out of range\n");
        return false;
    }
    
    return true;
}
