/**
 * PTO Runtime2 - Orchestrator Implementation
 *
 * Implements orchestrator state management, scope handling, and task submission.
 *
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_orchestrator.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pto_tensormap.h"
#include "tensor.h"

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
static inline void task_fanout_unlock(PTO2TaskDescriptor* task) { PTO2_STORE_RELEASE(&task->fanout_lock, 0); }

// =============================================================================
// Orchestrator Initialization
// =============================================================================

bool pto2_orchestrator_init(
    PTO2OrchestratorState* orch, PTO2SharedMemoryHandle* sm_handle, void* gm_heap, int32_t heap_size) {
    memset(orch, 0, sizeof(PTO2OrchestratorState));

    orch->sm_handle = sm_handle;
    orch->gm_heap_base = gm_heap;
    orch->gm_heap_size = heap_size;

    // Initialize heap ring buffer
    pto2_heap_ring_init(&orch->heap_ring, gm_heap, heap_size, &sm_handle->header->heap_tail);

    // Initialize task ring buffer
    pto2_task_ring_init(&orch->task_ring,
        sm_handle->task_descriptors,
        sm_handle->header->task_window_size,
        &sm_handle->header->last_task_alive);

    // Initialize dependency list pool
    pto2_dep_pool_init(&orch->dep_pool, sm_handle->dep_list_pool, sm_handle->header->dep_list_pool_size);

    // Initialize TensorMap
    if (!pto2_tensormap_init_default(&orch->tensor_map)) {
        return false;
    }
    orch->tensor_map.orch = orch;
    orch->tensormap_last_cleanup = 0;

    // Initialize scope stack: one flat buffer for task IDs + one array for begin offsets
    int32_t max_depth = PTO2_MAX_SCOPE_DEPTH;
    int32_t init_cap = PTO2_SCOPE_TASKS_INIT_CAP;
    orch->scope_tasks = (int32_t*)malloc(init_cap * sizeof(int32_t));
    orch->scope_begins = (int32_t*)malloc(max_depth * sizeof(int32_t));
    if (!orch->scope_tasks || !orch->scope_begins) {
        free(orch->scope_tasks);
        free(orch->scope_begins);
        pto2_tensormap_destroy(&orch->tensor_map);
        return false;
    }
    orch->scope_tasks_size = 0;
    orch->scope_tasks_capacity = init_cap;
    orch->scope_stack_top = -1;
    orch->scope_stack_capacity = max_depth;

    return true;
}

void pto2_orchestrator_destroy(PTO2OrchestratorState* orch) {
    pto2_tensormap_destroy(&orch->tensor_map);

    free(orch->scope_tasks);
    orch->scope_tasks = NULL;
    free(orch->scope_begins);
    orch->scope_begins = NULL;
}

void pto2_orchestrator_reset(PTO2OrchestratorState* orch) {
    pto2_heap_ring_reset(&orch->heap_ring);
    pto2_task_ring_reset(&orch->task_ring);
    pto2_dep_pool_reset(&orch->dep_pool);
    pto2_tensormap_reset(&orch->tensor_map);

    orch->tensormap_last_cleanup = 0;
    orch->scope_stack_top = -1;
    orch->scope_tasks_size = 0;

    orch->tasks_submitted = 0;
    orch->buffers_allocated = 0;
    orch->bytes_allocated = 0;

    // Reset shared memory header
    orch->sm_handle->header->current_task_index = 0;
    orch->sm_handle->header->heap_top = 0;
    orch->sm_handle->header->orchestrator_done = 0;
}

void pto2_orchestrator_set_scheduler(PTO2OrchestratorState* orch, PTO2SchedulerState* scheduler) {
    orch->scheduler = scheduler;
    orch->init_task_on_submit = true;  // Default: initialize task on submit
}

void pto2_orchestrator_set_scheduler_mode(
    PTO2OrchestratorState* orch, PTO2SchedulerState* scheduler, bool init_on_submit) {
    orch->scheduler = scheduler;
    orch->init_task_on_submit = init_on_submit;
}

// =============================================================================
// Scope Management
// =============================================================================

static void scope_tasks_push(PTO2OrchestratorState* orch, int32_t task_id) {
    if (orch->scope_tasks_size >= orch->scope_tasks_capacity) {
        int32_t new_cap = orch->scope_tasks_capacity * 2;
        int32_t* new_buf = (int32_t*)realloc(orch->scope_tasks, new_cap * sizeof(int32_t));
        assert(new_buf && "Failed to grow scope task buffer");
        orch->scope_tasks = new_buf;
        orch->scope_tasks_capacity = new_cap;
    }
    orch->scope_tasks[orch->scope_tasks_size++] = task_id;
}

void pto2_scope_begin(PTO2OrchestratorState* orch) {
    assert(orch->scope_stack_top < orch->scope_stack_capacity - 1 && "Scope stack overflow");

    ++orch->scope_stack_top;
    orch->scope_begins[orch->scope_stack_top] = orch->scope_tasks_size;
}

void pto2_scope_end(PTO2OrchestratorState* orch) {
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

    int32_t begin = orch->scope_begins[orch->scope_stack_top--];
    int32_t count = orch->scope_tasks_size - begin;

    if (orch->scheduler && count > 0) {
        pto2_scheduler_on_scope_end(orch->scheduler, &orch->scope_tasks[begin], count);
    }

    // Rewind the task buffer — these entries are no longer needed
    orch->scope_tasks_size = begin;
}

// =============================================================================
// Task Submission
// =============================================================================

void pto2_add_consumer_to_producer(
    PTO2OrchestratorState* orch, PTO2TaskDescriptor* producer, int32_t producer_id, int32_t consumer_id) {
    // Acquire per-task spinlock
    // This synchronizes with scheduler's on_task_complete_threadsafe
    task_fanout_lock(producer);

    // Prepend consumer to producer's fanout list
    producer->fanout_head = pto2_dep_list_prepend(&orch->dep_pool, producer->fanout_head, consumer_id);
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

void pto2_submit_task(PTO2OrchestratorState* orch,
    int32_t kernel_id,
    PTO2WorkerType worker_type,
    const char* func_name,
    PTOParam* params,
    int32_t num_params) {
    // === STEP 0: Sync TensorMap validity and optional cleanup ===
    pto2_orchestrator_sync_tensormap(&orch->tensor_map);

    // Submission without an open scope is illegal
    assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");

    // === STEP 1: Allocate task slot from Task Ring (blocks until available) ===
    int32_t task_id = pto2_task_ring_alloc(&orch->task_ring);

    PTO2TaskDescriptor* task = pto2_task_ring_get(&orch->task_ring, task_id);

    // Initialize task descriptor
    task->task_id = task_id;
    task->kernel_id = kernel_id;
    task->worker_type = worker_type;
    task->func_name = func_name;
    task->fanin_head = 0;
    task->fanin_count = 0;
    task->fanout_head = 0;
    task->fanout_lock = 0;
    // Initial fanout_count = 1 (the owning scope holds one reference)
    task->fanout_count = 1;
    task->packed_buffer_base = NULL;
    task->packed_buffer_end = NULL;
    task->num_outputs = 0;
    task->is_active = true;

    // Register this task in its owning scope
    scope_tasks_push(orch, task_id);

    // Temporary storage for collecting output sizes
    int32_t total_output_size = 0;

    // Temporary storage for fanin
    int32_t fanin_temp[PTO2_MAX_INPUTS];
    int32_t fanin_count = 0;

    task->param_count = num_params;
    for (int i = 0; i < task->param_count; i++) {
        task->params[i] = params[i];
        // Copy tensor data into task-owned storage; redirect task's pointer to it
        if (params[i].tensor) {
            task->tensor_copies[i] = *params[i].tensor;
            task->params[i].tensor = &task->tensor_copies[i];
        }
    }

    // === STEP 2: First pass - collect output sizes and process inputs ===

    for (int i = 0; i < num_params; i++) {
        PTOParam* p = &params[i];

        switch (p->type) {
            case PTOParamType::INOUT:
            case PTOParamType::INPUT: {
                // Look up producer via TensorMap
                auto dependency_task_ids = pto2_tensormap_lookup(&orch->tensor_map, params[i].tensor);

                for (auto [entry, overlap_status] : dependency_task_ids) {
                    // Check if this producer is already in fanin list (avoid duplicates)
                    int producer_task_id = entry->producer_task_id;
                    bool already_added = false;
                    for (int j = 0; j < fanin_count; j++) {
                        if (fanin_temp[j] == producer_task_id) {
                            already_added = true;
                            break;
                        }
                    }

                    if (!already_added) {
                        // Add to fanin list (this task depends on producer)
                        if (fanin_count < PTO2_MAX_INPUTS) {
                            fanin_temp[fanin_count++] = producer_task_id;
                        }

                        // Add this task to producer's fanout list (with spinlock)
                        PTO2TaskDescriptor* producer = pto2_task_ring_get(&orch->task_ring, producer_task_id);
                        pto2_add_consumer_to_producer(orch, producer, producer_task_id, task_id);
                    }
                    if (p->type == PTOParamType::INOUT && overlap_status == OverlapStatus::COVERED) {
                        // inout因为会再次insert进tensor map，
                        // 因此为了尽量减少依赖构建个数（尽可能构造链式依赖），当该tensor完全覆盖前面的tensor时，
                        // 应将前面的tensor从tensor map中剔除。
                        // 但是最开始的tensor除外，因为必须建立和最开始的task的依赖关系以保证tensor生命周期的正确管理
                        if (!entry->with_alloc) {
                            pto2_tensormap_remove_entry(orch->tensor_map, entry);
                        }
                    }
                }
                break;
            }

            case PTOParamType::OUTPUT: {
                // Only allocate from ring buffer when caller did not provide an address
                if (params[i].tensor->buffer.addr == 0) {
                    total_output_size += PTO2_ALIGN_UP(params[i].tensor->buffer.size, PTO2_PACKED_OUTPUT_ALIGN);
                }
                break;
            }
            default:
                break;
        }
    }

    // === STEP 3: Allocate packed buffer from Heap Ring (may stall) ===
    // Each output slot is aligned to PTO2_PACKED_OUTPUT_ALIGN (1024B); gap after data is padding.
    if (total_output_size > 0) {
        task->packed_buffer_base = pto2_alloc_packed_buffer(orch, total_output_size);
        task->packed_buffer_end = (char*)task->packed_buffer_base + total_output_size;

        // Offsets: each output at 1024B-aligned slot; slot size = ALIGN_UP(size, 1024)
        int32_t offset = 0;
        for (int i = 0; i < task->param_count; i++) {
            if (task->params[i].type == PTOParamType::OUTPUT) {
                if (task->tensor_copies[i].buffer.addr == 0) {
                    uint64_t alloc_addr = reinterpret_cast<uint64_t>((char*)task->packed_buffer_base + offset);
                    task->tensor_copies[i].buffer.addr = alloc_addr;
                    // Write back through caller's pointer (implicit update)
                    params[i].tensor->buffer.addr = alloc_addr;
                    offset += PTO2_ALIGN_UP(task->tensor_copies[i].buffer.size, PTO2_PACKED_OUTPUT_ALIGN);
                }
                task->output_index[task->num_outputs++] = i;
            }
        }
    }

    // === STEP 4: Second pass - register outputs in TensorMap ===
    int32_t output_idx = 0;
    for (int i = 0; i < num_params; i++) {
        PTOParam* p = &params[i];

        if (p->type == PTOParamType::OUTPUT || p->type == PTOParamType::INOUT) {
            // Register in TensorMap: this tensor is produced by task_id
            // Use task's tensor_copies (which has the heap-allocated address for outputs)
            pto2_tensormap_insert(&orch->tensor_map, &task->tensor_copies[i], task_id, p->type == PTOParamType::OUTPUT);
            output_idx++;
        }
    }

    // === STEP 5: Finalize fanin list ===
    // First build the fanin list
    for (int i = 0; i < fanin_count; i++) {
        task->fanin_head = pto2_dep_list_prepend(&orch->dep_pool, task->fanin_head, fanin_temp[i]);
    }
    // Use release semantics to ensure fanin list is visible before fanin_count
    __atomic_store_n(&task->fanin_count, fanin_count, __ATOMIC_RELEASE);

    // === STEP 6: Initialize task in scheduler ===
    // In multi-threaded mode, scheduler thread handles task initialization via polling
    if (orch->scheduler && orch->init_task_on_submit) {
        pto2_scheduler_init_task(orch->scheduler, task_id, task);
    }

    // === STEP 7: Update shared memory with current task index ===
    PTO2_STORE_RELEASE(&orch->sm_handle->header->current_task_index, orch->task_ring.current_index);

    orch->tasks_submitted++;
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

bool pto2_orchestrator_has_space(PTO2OrchestratorState* orch) { return pto2_task_ring_has_space(&orch->task_ring); }

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_orchestrator_print_stats(PTO2OrchestratorState* orch) {
    printf("=== Orchestrator Statistics ===\n");
    printf("Tasks submitted:     %lld\n", (long long)orch->tasks_submitted);
    printf("Buffers allocated:   %lld\n", (long long)orch->buffers_allocated);
    printf("Bytes allocated:     %lld\n", (long long)orch->bytes_allocated);
    printf("Current scope depth: %d\n", orch->scope_stack_top + 1);
    printf("Task ring active:    %d\n", pto2_task_ring_active_count(&orch->task_ring));
    printf("Heap ring used:      %d / %d\n", orch->heap_ring.top, orch->heap_ring.size);
    printf("Dep pool used:       %d / %d\n", pto2_dep_pool_used(&orch->dep_pool), orch->dep_pool.capacity);
    printf("TensorMap valid:     %d\n", pto2_tensormap_valid_count(&orch->tensor_map));
    printf("===============================\n");
}

void pto2_orchestrator_print_scope_stack(PTO2OrchestratorState* orch) {
    printf("=== Scope Stack ===\n");
    printf("Depth: %d\n", orch->scope_stack_top + 1);

    for (int i = 0; i <= orch->scope_stack_top; i++) {
        int32_t begin = orch->scope_begins[i];
        int32_t end = (i < orch->scope_stack_top) ? orch->scope_begins[i + 1] : orch->scope_tasks_size;
        printf("  [%d] tasks_owned = %d\n", i, end - begin);
    }

    printf("==================\n");
}
