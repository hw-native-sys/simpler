/**
 * PTO Runtime2 - Orchestrator Implementation
 * 
 * Implements orchestrator state management, scope handling, and task submission.
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_orchestrator.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


// =============================================================================
// Per-Task Spinlock Implementation
// =============================================================================

/**
 * Acquire spinlock for task's fanout fields
 */
static inline void task_fanout_lock(PTO2TaskDescriptor* task) {
    while (PTO2_EXCHANGE(&task->fanout_lock, 1) != 0) {
        PTO2_SPIN_PAUSE();
    }
}

/**
 * Release spinlock for task's fanout fields
 */
static inline void task_fanout_unlock(PTO2TaskDescriptor* task) {
    PTO2_STORE_RELEASE(&task->fanout_lock, 0);
}

// =============================================================================
// Orchestrator Initialization
// =============================================================================

bool pto2_orchestrator_init(PTO2OrchestratorState* orch,
                             PTO2SharedMemoryHandle* sm_handle,
                             void* gm_heap,
                             int32_t heap_size) {
    memset(orch, 0, sizeof(PTO2OrchestratorState));
    
    orch->sm_handle = sm_handle;
    orch->gm_heap_base = gm_heap;
    orch->gm_heap_size = heap_size;
    
    // Initialize heap ring buffer
    pto2_heap_ring_init(&orch->heap_ring, gm_heap, heap_size,
                        &sm_handle->header->heap_tail);
    
    // Initialize task ring buffer
    pto2_task_ring_init(&orch->task_ring, sm_handle->task_descriptors,
                        sm_handle->header->task_window_size,
                        &sm_handle->header->last_task_alive);
    
    // Initialize dependency list pool
    pto2_dep_pool_init(&orch->dep_pool, sm_handle->dep_list_pool,
                       sm_handle->header->dep_list_pool_size);
    
    // Initialize TensorMap
    if (!pto2_tensormap_init_default(&orch->tensor_map)) {
        return false;
    }
    orch->tensormap_last_cleanup = 0;
    
    // Initialize scope stack
    orch->scope_stack = (int32_t*)malloc(PTO2_MAX_SCOPE_DEPTH * sizeof(int32_t));
    if (!orch->scope_stack) {
        pto2_tensormap_destroy(&orch->tensor_map);
        return false;
    }
    orch->scope_stack_top = -1;
    orch->scope_stack_capacity = PTO2_MAX_SCOPE_DEPTH;
    
    return true;
}

void pto2_orchestrator_destroy(PTO2OrchestratorState* orch) {
    pto2_tensormap_destroy(&orch->tensor_map);
    
    if (orch->scope_stack) {
        free(orch->scope_stack);
        orch->scope_stack = NULL;
    }
}

void pto2_orchestrator_reset(PTO2OrchestratorState* orch) {
    pto2_heap_ring_reset(&orch->heap_ring);
    pto2_task_ring_reset(&orch->task_ring);
    pto2_dep_pool_reset(&orch->dep_pool);
    pto2_tensormap_reset(&orch->tensor_map);
    
    orch->tensormap_last_cleanup = 0;
    orch->scope_stack_top = -1;
    
    orch->tasks_submitted = 0;
    orch->buffers_allocated = 0;
    orch->bytes_allocated = 0;
    orch->scope_depth_max = 0;
    
    // Reset shared memory header
    orch->sm_handle->header->current_task_index = 0;
    orch->sm_handle->header->heap_top = 0;
    orch->sm_handle->header->orchestrator_done = 0;
}

void pto2_orchestrator_set_scheduler(PTO2OrchestratorState* orch,
                                      PTO2SchedulerState* scheduler) {
    orch->scheduler = scheduler;
    orch->init_task_on_submit = true;  // Default: initialize task on submit
}

void pto2_orchestrator_set_scheduler_mode(PTO2OrchestratorState* orch,
                                           PTO2SchedulerState* scheduler,
                                           bool init_on_submit) {
    orch->scheduler = scheduler;
    orch->init_task_on_submit = init_on_submit;
}

// =============================================================================
// Scope Management
// =============================================================================

void pto2_scope_begin(PTO2OrchestratorState* orch) {
    // Check for stack overflow
    if (orch->scope_stack_top >= orch->scope_stack_capacity - 1) {
        fprintf(stderr, "ERROR: Scope stack overflow\n");
        return;
    }
    
    // Push current task index to scope stack
    int32_t current_pos = orch->task_ring.current_index;
    orch->scope_stack[++orch->scope_stack_top] = current_pos;
    
    // Update max depth tracking
    int32_t depth = orch->scope_stack_top + 1;
    if (depth > orch->scope_depth_max) {
        orch->scope_depth_max = depth;
    }
}

void pto2_scope_end(PTO2OrchestratorState* orch) {
    // Check for stack underflow
    if (orch->scope_stack_top < 0) {
        fprintf(stderr, "ERROR: Scope stack underflow\n");
        return;
    }
    
    // Pop scope stack to get begin position
    int32_t scope_begin_pos = orch->scope_stack[orch->scope_stack_top--];
    int32_t scope_end_pos = orch->task_ring.current_index;
    
    // Notify scheduler to release scope references
    // In simulated mode, we can call scheduler directly
    if (orch->scheduler) {
        pto2_scheduler_on_scope_end(orch->scheduler, scope_begin_pos, scope_end_pos);
    }
    // In real mode, scope_end would be communicated via shared memory or message
}

// =============================================================================
// TensorMap Synchronization
// =============================================================================

void pto2_orchestrator_sync_tensormap(PTO2OrchestratorState* orch) {
    // Read current last_task_alive from shared memory
    int32_t new_last_task_alive = PTO2_LOAD_ACQUIRE(
        &orch->sm_handle->header->last_task_alive);
    
    // Update TensorMap validity threshold
    pto2_tensormap_sync_validity(&orch->tensor_map, new_last_task_alive);
    
    // Periodically cleanup TensorMap to remove stale entries from bucket chains
    if (new_last_task_alive - orch->tensormap_last_cleanup >= 
        PTO2_TENSORMAP_CLEANUP_INTERVAL) {
        pto2_tensormap_cleanup_retired(&orch->tensor_map,
                                        orch->tensormap_last_cleanup,
                                        new_last_task_alive);
        orch->tensormap_last_cleanup = new_last_task_alive;
    }
}

// =============================================================================
// Task Submission
// =============================================================================

void pto2_add_consumer_to_producer(PTO2OrchestratorState* orch,
                                    PTO2TaskDescriptor* producer,
                                    int32_t producer_id,
                                    int32_t consumer_id) {
    // Acquire per-task spinlock
    // This synchronizes with scheduler's on_task_complete_threadsafe
    task_fanout_lock(producer);
    
    // Prepend consumer to producer's fanout list
    producer->fanout_head = pto2_dep_list_prepend(&orch->dep_pool,
                                                   producer->fanout_head,
                                                   consumer_id);
    producer->fanout_count++;
    
    // Check if producer has already completed
    // If so, we need to update consumer's fanin_refcount directly
    // because on_task_complete_threadsafe has already run and won't see this consumer
    if (orch->scheduler) {
        PTO2SchedulerState* sched = orch->scheduler;
        int32_t prod_slot = pto2_task_slot(sched, producer_id);
        int32_t prod_state = __atomic_load_n(&sched->task_state[prod_slot], __ATOMIC_ACQUIRE);
        
        if (prod_state >= PTO2_TASK_COMPLETED) {
            // Producer already completed - update consumer's fanin_refcount directly
            int32_t cons_slot = pto2_task_slot(sched, consumer_id);
            __atomic_fetch_add(&sched->fanin_refcount[cons_slot], 1, __ATOMIC_SEQ_CST);
        }
    }
    
    // Release spinlock
    task_fanout_unlock(producer);
}

void* pto2_alloc_packed_buffer(PTO2OrchestratorState* orch, int32_t total_size) {
    if (total_size <= 0) {
        return NULL;
    }
    
    void* buffer = pto2_heap_ring_alloc(&orch->heap_ring, total_size);
    
    orch->buffers_allocated++;
    orch->bytes_allocated += total_size;
    
    // Update shared memory with new heap top
    PTO2_STORE_RELEASE(&orch->sm_handle->header->heap_top, orch->heap_ring.top);
    
    return buffer;
}

static inline void pto2_param_set(PTO2TaskParam* p, int32_t type, void* buf, int32_t tile_index, int32_t size_bytes) {
    p->type = type;
    memset(p->_pad, 0, sizeof(p->_pad));
    p->buffer = buf;
    p->tile_index = tile_index;
    p->size = size_bytes;
}

void pto2_param_set_input(PTO2TaskParam* p, void* buf, int32_t tile_index, int32_t size_bytes) {
    pto2_param_set(p, (int32_t)PTO2_PARAM_INPUT, buf, tile_index, size_bytes);
}

void pto2_param_set_output(PTO2TaskParam* p, void* buf, int32_t tile_index, int32_t size_bytes) {
    pto2_param_set(p, (int32_t)PTO2_PARAM_OUTPUT, buf, tile_index, size_bytes);
}

void pto2_param_set_inout(PTO2TaskParam* p, void* buf, int32_t tile_index, int32_t size_bytes) {
    pto2_param_set(p, (int32_t)PTO2_PARAM_INOUT, buf, tile_index, size_bytes);
}

int32_t pto2_submit_task(PTO2OrchestratorState* orch,
                          int32_t kernel_id,
                          PTO2WorkerType worker_type,
                          const char* func_name,
                          PTO2TaskParam* params,
                          int32_t num_params) {

    // === STEP 0: Sync TensorMap validity and optional cleanup ===
    pto2_orchestrator_sync_tensormap(orch);

    // === STEP 1: Allocate task slot from Task Ring (blocks until available) ===
    int32_t task_id = pto2_task_ring_alloc(&orch->task_ring);

    PTO2TaskDescriptor* task = pto2_task_ring_get(&orch->task_ring, task_id);

    // Initialize task descriptor
    task->task_id = task_id;
    task->kernel_id = kernel_id;
    task->worker_type = worker_type;
    task->scope_depth = pto2_get_scope_depth(orch);
    task->func_name = func_name;
    task->fanin_head = 0;
    task->fanin_count = 0;
    task->fanout_head = 0;
    task->fanout_lock = 0;
    // Initial fanout_count = scope_depth (number of enclosing scopes that reference this task)
    // WARNING: If task_window_size is too small, this can cause deadlock:
    //   - Orchestrator waits for task ring space (flow control)
    //   - scope_end() needs orchestrator to continue execution
    //   - Tasks can't become CONSUMED without scope_end releasing references
    // Solution: Increase task_window_size to accommodate all tasks in scope
    task->fanout_count = task->scope_depth;
    task->packed_buffer_base = NULL;
    task->packed_buffer_end = NULL;
    task->num_outputs = 0;
    task->num_inputs = 0;
    task->is_active = true;
    
    // Temporary storage for collecting output sizes
    int32_t output_sizes[PTO2_MAX_OUTPUTS];
    int32_t num_outputs = 0;
    int32_t total_output_size = 0;
    
    // Temporary storage for fanin
    int32_t fanin_temp[PTO2_MAX_INPUTS];
    int32_t fanin_count = 0;
    
    // === STEP 2: First pass - collect output sizes and process inputs ===
    for (int i = 0; i < num_params; i++) {
        PTO2TaskParam* p = &params[i];
        int32_t param_size = p->size;
        PTO2TensorRegion region = {
            .base_ptr = p->buffer,
            .tile_index = p->tile_index,
            .offset = 0,
            .size = param_size
        };
        
        switch (p->type) {
            case PTO2_PARAM_INPUT: {
                // Look up producer via TensorMap
                int32_t producer_id = pto2_tensormap_lookup(&orch->tensor_map, &region);
                
                if (producer_id >= 0) {
                    // Check if this producer is already in fanin list (avoid duplicates)
                    bool already_added = false;
                    for (int j = 0; j < fanin_count; j++) {
                        if (fanin_temp[j] == producer_id) {
                            already_added = true;
                            break;
                        }
                    }
                    
                    if (!already_added) {
                        // Add to fanin list (this task depends on producer)
                        if (fanin_count < PTO2_MAX_INPUTS) {
                            fanin_temp[fanin_count++] = producer_id;
                        }
                        
                        // Add this task to producer's fanout list (with spinlock)
                        PTO2TaskDescriptor* producer = pto2_task_ring_get(
                            &orch->task_ring, producer_id);
                        pto2_add_consumer_to_producer(orch, producer, producer_id, task_id);
                    }
                }
                task->num_inputs++;
                break;
            }
            
            case PTO2_PARAM_OUTPUT: {
                // Collect output size for packed buffer allocation
                if (num_outputs < PTO2_MAX_OUTPUTS) {
                    output_sizes[num_outputs++] = param_size;
                    total_output_size += PTO2_ALIGN_UP(param_size, PTO2_PACKED_OUTPUT_ALIGN);
                }
                break;
            }
            
            case PTO2_PARAM_INOUT: {
                // INOUT = INPUT + OUTPUT
                
                // Handle as input (get dependency on previous writer)
                int32_t producer_id = pto2_tensormap_lookup(&orch->tensor_map, &region);
                if (producer_id >= 0) {
                    // Check if this producer is already in fanin list (avoid duplicates)
                    bool already_added = false;
                    for (int j = 0; j < fanin_count; j++) {
                        if (fanin_temp[j] == producer_id) {
                            already_added = true;
                            break;
                        }
                    }
                    
                    if (!already_added) {
                        if (fanin_count < PTO2_MAX_INPUTS) {
                            fanin_temp[fanin_count++] = producer_id;
                        }
                        PTO2TaskDescriptor* producer = pto2_task_ring_get(
                            &orch->task_ring, producer_id);
                        pto2_add_consumer_to_producer(orch, producer, producer_id, task_id);
                    }
                }
                task->num_inputs++;
                
                // Collect output size for packed buffer
                if (num_outputs < PTO2_MAX_OUTPUTS) {
                    output_sizes[num_outputs++] = param_size;
                    total_output_size += PTO2_ALIGN_UP(param_size, PTO2_PACKED_OUTPUT_ALIGN);
                }
                break;
            }
        }
    }
    
    /* Debug: each output tensor size at submit */
    fprintf(stderr, "[PTO2 submit] task_id=%d num_outputs=%d output_sizes:", task_id, num_outputs);
    for (int i = 0; i < num_outputs; i++)
        fprintf(stderr, " %d", output_sizes[i]);
    fprintf(stderr, "\n");
    
    // === STEP 3: Allocate packed buffer from Heap Ring (may stall) ===
    // Each output slot is aligned to PTO2_PACKED_OUTPUT_ALIGN (1024B); gap after data is padding.
    // Copy-back: use (packed_buffer_base + output_offsets[i]) as pointer, actual tensor size as length.
    if (total_output_size > 0) {
        task->packed_buffer_base = pto2_alloc_packed_buffer(orch, total_output_size);
        task->packed_buffer_end = (char*)task->packed_buffer_base + total_output_size;
        
        // Offsets: each output at 1024B-aligned slot; slot size = ALIGN_UP(output_sizes[i], 1024)
        int32_t offset = 0;
        for (int i = 0; i < num_outputs; i++) {
            task->output_offsets[i] = offset;
            offset += PTO2_ALIGN_UP(output_sizes[i], PTO2_PACKED_OUTPUT_ALIGN);
        }
        /* Debug: each output ptr and size after packed buffer alloc */
        fprintf(stderr, "[PTO2 packed] task_id=%d base=%p total=%d\n", task_id,
                (void*)task->packed_buffer_base, total_output_size);
        for (int i = 0; i < num_outputs; i++) {
            void* out_ptr = (char*)task->packed_buffer_base + task->output_offsets[i];
            fprintf(stderr, "  output[%d] ptr=%p offset=%d size=%d\n",
                    i, out_ptr, task->output_offsets[i], output_sizes[i]);
        }
    }
    task->num_outputs = num_outputs;
    
    // === STEP 4: Second pass - register outputs in TensorMap ===
    int32_t output_idx = 0;
    for (int i = 0; i < num_params; i++) {
        PTO2TaskParam* p = &params[i];
        
        if (p->type == PTO2_PARAM_OUTPUT || p->type == PTO2_PARAM_INOUT) {
            // IMPORTANT: Use the ORIGINAL buffer address (p->buffer) for TensorMap,
            // not the packed buffer address. This ensures that consumers looking up
            // dependencies using the original tensor address will find this producer.
            PTO2TensorRegion region = {
                .base_ptr = p->buffer,        // Use original tensor address
                .tile_index = p->tile_index,
                .offset = 0,
                .size = p->size
            };
            
            // Register in TensorMap: this region is produced by task_id
            pto2_tensormap_insert(&orch->tensor_map, &region, task_id);
            output_idx++;
        }
    }
    
    // === STEP 5: Finalize fanin list ===
    // First build the fanin list
    for (int i = 0; i < fanin_count; i++) {
        task->fanin_head = pto2_dep_list_prepend(&orch->dep_pool,
                                                  task->fanin_head,
                                                  fanin_temp[i]);
    }
    // Use release semantics to ensure fanin list is visible before fanin_count
    __atomic_store_n(&task->fanin_count, fanin_count, __ATOMIC_RELEASE);
    
    // === STEP 6: Initialize task in scheduler ===
    // In multi-threaded mode, scheduler thread handles task initialization via polling
    if (orch->scheduler && orch->init_task_on_submit) {
        pto2_scheduler_init_task(orch->scheduler, task_id, task);
    }
    
    // === STEP 7: Update shared memory with current task index ===
    PTO2_STORE_RELEASE(&orch->sm_handle->header->current_task_index,
                       orch->task_ring.current_index);
    
    orch->tasks_submitted++;

    return task_id;
}

void* pto2_task_get_output(PTO2OrchestratorState* orch, 
                            int32_t task_id, 
                            int32_t output_idx) {
    PTO2TaskDescriptor* task = pto2_task_ring_get(&orch->task_ring, task_id);
    
    if (output_idx < 0 || output_idx >= task->num_outputs) {
        return NULL;
    }
    
    return (char*)task->packed_buffer_base + task->output_offsets[output_idx];
}

// =============================================================================
// Flow Control
// =============================================================================

void pto2_orchestrator_done(PTO2OrchestratorState* orch) {
    int32_t total_tasks = orch->task_ring.current_index;
    fprintf(stdout, "=== [Orchestrator] total_tasks=%d ===\n", total_tasks);
    fflush(stdout);
    PTO2_STORE_RELEASE(&orch->sm_handle->header->orchestrator_done, 1);
}

void pto2_orchestrator_wait_all(PTO2OrchestratorState* orch) {
    if (!orch->scheduler) {
        return;  // Can't wait without scheduler reference
    }
    
    // Spin-wait until scheduler reports all tasks done
    while (!pto2_scheduler_is_done(orch->scheduler)) {
        PTO2_SPIN_PAUSE();
    }
}

bool pto2_orchestrator_has_space(PTO2OrchestratorState* orch) {
    return pto2_task_ring_has_space(&orch->task_ring);
}

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_orchestrator_print_stats(PTO2OrchestratorState* orch) {
    printf("=== Orchestrator Statistics ===\n");
    printf("Tasks submitted:     %lld\n", (long long)orch->tasks_submitted);
    printf("Buffers allocated:   %lld\n", (long long)orch->buffers_allocated);
    printf("Bytes allocated:     %lld\n", (long long)orch->bytes_allocated);
    printf("Max scope depth:     %lld\n", (long long)orch->scope_depth_max);
    printf("Current scope depth: %d\n", pto2_get_scope_depth(orch));
    printf("Task ring active:    %d\n", pto2_task_ring_active_count(&orch->task_ring));
    printf("Heap ring used:      %d / %d\n", 
           orch->heap_ring.top, orch->heap_ring.size);
    printf("Dep pool used:       %d / %d\n",
           pto2_dep_pool_used(&orch->dep_pool),
           orch->dep_pool.capacity);
    printf("TensorMap valid:     %d\n", 
           pto2_tensormap_valid_count(&orch->tensor_map));
    printf("===============================\n");
}

void pto2_orchestrator_print_scope_stack(PTO2OrchestratorState* orch) {
    printf("=== Scope Stack ===\n");
    printf("Depth: %d\n", pto2_get_scope_depth(orch));
    
    for (int i = 0; i <= orch->scope_stack_top; i++) {
        printf("  [%d] begin_pos = %d\n", i, orch->scope_stack[i]);
    }
    
    printf("==================\n");
}
