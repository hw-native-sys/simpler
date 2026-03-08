/**
 * PTO Runtime2 - Orchestrator Implementation
 *
 * Implements orchestrator state management, scope handling, and task submission.
 *
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_orchestrator.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/unified_log.h"
#include "pto_runtime2_types.h"
#include "pto_tensormap.h"
#include "pto_types.h"
#include "tensor.h"

// =============================================================================
// Orchestrator Profiling (compile-time toggle)
// =============================================================================
#if PTO2_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/performance_collector_aicpu.h"
// Weak fallback for builds that don't link device_time.cpp (e.g. host).
// The strong symbol from platform/.../device_time.cpp wins in the AICPU build.
__attribute__((weak)) uint64_t get_sys_cnt_aicpu() { return 0; }
// Weak fallback for builds that don't link performance_collector_aicpu.cpp.
// The strong symbol from the AICPU build wins when profiling is available.
__attribute__((weak)) void perf_aicpu_record_orch_phase(
    AicpuPhaseId, uint64_t, uint64_t, uint32_t, uint32_t) {}
// Accumulated nanoseconds per sub-step
static uint64_t g_orch_sync_cycle = 0;       // tensormap sync
static uint64_t g_orch_alloc_cycle = 0;      // task ring alloc
static uint64_t g_orch_params_cycle = 0;     // param copy
static uint64_t g_orch_lookup_cycle = 0;     // tensormap lookup + dep building
static uint64_t g_orch_heap_cycle = 0;       // heap alloc + output assign
static uint64_t g_orch_insert_cycle = 0;     // tensormap insert
static uint64_t g_orch_fanin_cycle = 0;      // fanin list + early-return check
static uint64_t g_orch_finalize_cycle = 0;   // scheduler init + SM update
static uint64_t g_orch_scope_end_cycle = 0;  // scope_end overhead
static int64_t  g_orch_submit_count = 0;
static uint32_t g_orch_submit_idx = 0;
#if PTO2_ORCH_PROFILING
uint64_t g_orch_alloc_wait_cycle = 0;
uint64_t g_orch_heap_wait_cycle = 0;
uint64_t g_orch_fanin_wait_cycle = 0;
uint64_t g_orch_finalize_wait_cycle = 0;
uint64_t g_orch_alloc_atomic_count = 0;
uint64_t g_orch_params_atomic_count = 0;
uint64_t g_orch_heap_atomic_count = 0;
uint64_t g_orch_fanin_atomic_count = 0;
uint64_t g_orch_finalize_atomic_count = 0;
uint64_t g_orch_scope_end_atomic_count = 0;
#elif PTO2_SCHED_PROFILING
// When only PTO2_SCHED_PROFILING is enabled, shared methods still need
// orch counters as targets for orchestrator-context calls.
uint64_t g_orch_fanin_atomic_count = 0;
uint64_t g_orch_fanin_wait_cycle = 0;
uint64_t g_orch_finalize_atomic_count = 0;
uint64_t g_orch_finalize_wait_cycle = 0;
uint64_t g_orch_scope_end_atomic_count = 0;
#endif
#define CYCLE_COUNT_START() uint64_t _t0 = get_sys_cnt_aicpu(), _t1
#define CYCLE_COUNT_LAP(acc) do { _t1 = get_sys_cnt_aicpu(); acc += (_t1 - _t0); _t0 = _t1; } while(0)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid) do { \
    _t1 = get_sys_cnt_aicpu(); \
    acc += (_t1 - _t0); \
    perf_aicpu_record_orch_phase((phase_id), _t0, _t1, g_orch_submit_idx, (tid)); \
    _t0 = _t1; \
} while(0)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)
#endif

// =============================================================================
// Orchestrator Initialization
// =============================================================================

bool pto2_orchestrator_init(
    PTO2OrchestratorState* orch, PTO2SharedMemoryHandle* sm_handle, void* gm_heap, uint64_t heap_size) {
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
    pto2_dep_pool_init(&orch->dep_pool, sm_handle->dep_list_pool, (int32_t)sm_handle->header->dep_list_pool_size);

    // Initialize TensorMap
    if (!orch->tensor_map.init_default()) {
        return false;
    }
    orch->tensor_map.orch = orch;
    orch->tensormap_last_cleanup = 0;

    // Initialize scope stack: one flat buffer for task IDs + one array for begin offsets
    uint64_t max_depth = PTO2_MAX_SCOPE_DEPTH;
    int32_t init_cap = PTO2_SCOPE_TASKS_INIT_CAP;
    orch->scope_tasks = (int32_t*)malloc(init_cap * sizeof(int32_t));
    orch->scope_begins = (int32_t*)malloc(max_depth * sizeof(int32_t));
    if (!orch->scope_tasks || !orch->scope_begins) {
        free(orch->scope_tasks);
        free(orch->scope_begins);
        orch->tensor_map.destroy();
        return false;
    }
    orch->scope_tasks_size = 0;
    orch->scope_tasks_capacity = init_cap;
    orch->scope_stack_top = -1;
    orch->scope_stack_capacity = max_depth;

    orch->tensor_pool.init();
    TensorPool::set_instance(&orch->tensor_pool);

    return true;
}

void pto2_orchestrator_destroy(PTO2OrchestratorState* orch) {
    TensorPool::set_instance(nullptr);
    orch->tensor_map.destroy();

    free(orch->scope_tasks);
    orch->scope_tasks = NULL;
    free(orch->scope_begins);
    orch->scope_begins = NULL;
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
    assert(orch->scope_stack_top < (int32_t)(orch->scope_stack_capacity - 1) && "Scope stack overflow");

    ++orch->scope_stack_top;
    orch->scope_begins[orch->scope_stack_top] = orch->scope_tasks_size;
}

void pto2_scope_end(PTO2OrchestratorState* orch) {
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

#if PTO2_PROFILING
    uint64_t _se0 = get_sys_cnt_aicpu();
#endif

    int32_t begin = orch->scope_begins[orch->scope_stack_top--];
    int32_t count = orch->scope_tasks_size - begin;

    if (orch->scheduler && count > 0) {
        orch->scheduler->on_scope_end(&orch->scope_tasks[begin], count);
    }

    // Rewind the task buffer — these entries are no longer needed
    orch->scope_tasks_size = begin;

#if PTO2_PROFILING
    uint64_t _se1 = get_sys_cnt_aicpu();
    g_orch_scope_end_cycle += (_se1 - _se0);
    perf_aicpu_record_orch_phase(AicpuPhaseId::ORCH_SCOPE_END, _se0, _se1, g_orch_submit_idx, -1);
#endif
}

// =============================================================================
// Task Submission
// =============================================================================
void pto2_submit_task(
    PTO2OrchestratorState* orch, int32_t kernel_id, PTO2WorkerType worker_type, PTOParam* params, int32_t num_params) {
    CYCLE_COUNT_START();

    // === STEP 0: Sync TensorMap validity and optional cleanup ===
    orch->tensor_map.sync_tensormap();

    CYCLE_COUNT_LAP_RECORD(g_orch_sync_cycle, AicpuPhaseId::ORCH_SYNC, -1);

    // Submission without an open scope is illegal
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");

    // === STEP 1: Allocate task slot from Task Ring (blocks until available) ===
    int32_t task_id = orch->task_ring.pto2_task_ring_alloc();

    CYCLE_COUNT_LAP_RECORD(g_orch_alloc_cycle, AicpuPhaseId::ORCH_ALLOC, task_id);

    PTO2TaskDescriptor* task = pto2_task_ring_get(&orch->task_ring, task_id);

    // Initialize task descriptor
    task->task_id = task_id;
    task->kernel_id = kernel_id;
    task->worker_type = worker_type;
    task->fanin_head = nullptr;
    task->fanin_count = 0;
    task->fanout_head = nullptr;
    task->fanout_lock.store(0, std::memory_order_relaxed);
    // Initial fanout_count = 1 (the owning scope holds one reference)
    task->fanout_count.store(1, std::memory_order_relaxed);
    task->packed_buffer_base = NULL;
    task->packed_buffer_end = NULL;
    task->is_active = true;

    // Register this task in its owning scope
    scope_tasks_push(orch, task_id);

    // Temporary storage for fanin
    int32_t fanin_temp[PTO2_MAX_INPUTS];
    int32_t fanin_count = 0;

    task->param_count = num_params;
    for (int i = 0; i < num_params; i++) {
        task->params[i].type = params[i].type;
        if (params[i].type == PTOParamType::SCALAR) {
            task->params[i].scalar_value = params[i].scalar_value;
        } else {
            task->params[i].tensor = std::move(params[i].tensor);
        }
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_params_cycle, AicpuPhaseId::ORCH_PARAMS, task_id);
#if PTO2_ORCH_PROFILING
    g_orch_params_atomic_count += 2;  // fanout_lock.store + fanout_count.store
#endif

    // Temporary storage for collecting output sizes
    int32_t total_output_size = 0;
    for (int i = 0; i < num_params; i++) {
        PTOParam& p = task->params[i];
        if (p.type != PTOParamType::OUTPUT) {
            continue;
        }
        auto& tensor_data = p.tensor.data();
        // Only allocate from ring buffer when caller did not provide an address
        if (tensor_data.buffer.addr == 0) {
            total_output_size += PTO2_ALIGN_UP(tensor_data.buffer.size, PTO2_PACKED_OUTPUT_ALIGN);
        }
    }

    if (total_output_size > 0) {
        task->packed_buffer_base = orch->pto2_alloc_packed_buffer(total_output_size);
        task->packed_buffer_end = (char*)task->packed_buffer_base + total_output_size;
    }
    CYCLE_COUNT_LAP_RECORD(g_orch_heap_cycle, AicpuPhaseId::ORCH_HEAP, task_id);
#if PTO2_ORCH_PROFILING
    if (total_output_size > 0) {
        g_orch_heap_atomic_count += 1;  // heap_top.store in pto2_alloc_packed_buffer
    }
#endif

    // === STEP 2: First pass - set output addr and process tensor ===
    int32_t offset = 0;
    for (int i = 0; i < num_params; i++) {
        PTOParam& p = task->params[i];

        switch (p.type) {
            case PTOParamType::INOUT:
            case PTOParamType::INPUT: {
                // Look up producer via TensorMap
                PTO2LookupResult lookup_result;
                orch->tensor_map.lookup(p.tensor, lookup_result);

                for (int r = 0; r < lookup_result.count; r++) {
                    PTO2TensorMapEntry& entry = *lookup_result.entries[r].entry;
                    auto overlap_status = lookup_result.entries[r].overlap_status;
                    // Check if this producer is already in fanin list (avoid duplicates)
                    int producer_task_id = entry.producer_task_id;
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
                    }
                    if (p.type == PTOParamType::INOUT && overlap_status == OverlapStatus::COVERED) {
                        // inout因为会再次insert进tensor map，
                        // 因此为了尽量减少依赖构建个数（尽可能构造链式依赖），当该tensor完全覆盖前面的tensor时，
                        // 应将前面的tensor从tensor map中剔除。
                        // 但是最开始的tensor除外，因为必须建立和最开始的task的依赖关系以保证tensor生命周期的正确管理
                        if (!entry.with_alloc) {
                            orch->tensor_map.remove_entry(entry);
                        }
                    }
                }
                break;
            }

            case PTOParamType::OUTPUT: {
                auto& tensor_data = p.tensor.data();
                // Offsets: each output at 1024B-aligned slot; slot size = ALIGN_UP(size, 1024)
                // Allocation happens here only; no memcpy of buffer content. Caller's tensor gets addr written back.
                if (tensor_data.buffer.addr == 0) {
                    uint64_t alloc_addr = reinterpret_cast<uint64_t>((char*)task->packed_buffer_base + offset);
                    tensor_data.buffer.addr = alloc_addr;
                    offset += PTO2_ALIGN_UP(tensor_data.buffer.size, PTO2_PACKED_OUTPUT_ALIGN);

                }
                break;
            }
            default:
                break;
        }
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_lookup_cycle, AicpuPhaseId::ORCH_LOOKUP, task_id);


    // === STEP 4: Second pass - register outputs in TensorMap ===
    for (int i = 0; i < num_params; i++) {
        PTOParam& p = task->params[i];
        if (p.type == PTOParamType::OUTPUT || p.type == PTOParamType::INOUT) {
            // Register in TensorMap: this tensor is produced by task_id
            // Use task's tensor_copies (which has the heap-allocated address for outputs)
            orch->tensor_map.insert(p.tensor, task_id, p.type == PTOParamType::OUTPUT);
        }
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_insert_cycle, AicpuPhaseId::ORCH_INSERT, task_id);

    // === STEP 5: Finalize fanin list ===
    // First build the fanin list
    if (orch->scheduler) {
        PTO2SchedulerState* sched = orch->scheduler;
        int32_t slot = sched->pto2_task_slot(task_id);

        auto &dep_pool = orch->dep_pool;

        int32_t early_finished = 0;
        task->fanin_count = fanin_count + 1;  // +1 redundance for not being ready too early
        for (int i = 0; i < fanin_count; i++) {
            int32_t producer_task_id = fanin_temp[i];
            task->fanin_head = dep_pool.pto2_dep_list_prepend(task->fanin_head, producer_task_id);
        }
        for (int i = 0; i < fanin_count; i++) {
            int32_t producer_task_id = fanin_temp[i];
            // Add this task to producer's fanout list (with spinlock)
            PTO2TaskDescriptor* producer = pto2_task_ring_get(&orch->task_ring, producer_task_id);
            producer->fanout_count.fetch_add(1, std::memory_order_release);
            int32_t prod_slot = sched->pto2_task_slot(producer_task_id);
#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
            pto2_fanout_lock(producer, g_orch_fanin_atomic_count, g_orch_fanin_wait_cycle);
#else
            pto2_fanout_lock(producer);
#endif
            // Normal path: prepend consumer to producer's fanout list
            int32_t prod_state = sched->task_state[prod_slot].load(std::memory_order_acquire);
            if (prod_state >= PTO2_TASK_COMPLETED) {
                // Early return optimization: if producer already completed, we can skip adding dependency and directly
                // decrement fanin_count
                early_finished++;
            } else {
                producer->fanout_head = dep_pool.pto2_dep_list_prepend(producer->fanout_head, task_id);
            }
            pto2_fanout_unlock(producer);
        }
        if (early_finished > 0) {
            sched->fanin_refcount[slot].fetch_add(early_finished, std::memory_order_acq_rel);
        }
#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
        // Per producer: fetch_add(fanout_count) + load(task_state) + store(unlock) = 3 atomics
        // Lock atomics (loads + CAS) are counted inside pto2_fanout_lock
        g_orch_fanin_atomic_count += fanin_count * 3;
        if (early_finished > 0) {
            g_orch_fanin_atomic_count += 1;  // fanin_refcount.fetch_add
        }
#endif
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_fanin_cycle, AicpuPhaseId::ORCH_FANIN, task_id);


    // === STEP 6: Initialize task in scheduler ===
    // In multi-threaded mode, scheduler thread handles task initialization via polling
    if (orch->scheduler && orch->init_task_on_submit) {
        orch->scheduler->init_task(task_id, task);
    }

    // === STEP 7: Update shared memory with current task index ===
    orch->sm_handle->header->current_task_index.store(orch->task_ring.current_index, std::memory_order_release);

    CYCLE_COUNT_LAP_RECORD(g_orch_finalize_cycle, AicpuPhaseId::ORCH_FINALIZE, task_id);
#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    // task_state.store + fanout_refcount.store + fanin_refcount.fetch_add
    // + current_task_index.store = 4
    // Conditional CAS(task_state PENDING→READY) and push() atomics counted inside push()
    g_orch_finalize_atomic_count += 4;
#endif

#if PTO2_PROFILING
    orch->tasks_submitted++;
    g_orch_submit_count++;
    g_orch_submit_idx++;
#endif
}

// =============================================================================
// Flow Control
// =============================================================================

void pto2_orchestrator_done(PTO2OrchestratorState* orch) {
    int32_t total_tasks = orch->task_ring.current_index;
    LOG_INFO("=== [Orchestrator] total_tasks=%d ===", total_tasks);
    orch->sm_handle->header->orchestrator_done.store(1, std::memory_order_release);
}

void pto2_orchestrator_wait_all(PTO2OrchestratorState* orch) {
    if (!orch->scheduler) {
        return;  // Can't wait without scheduler reference
    }

    // Spin-wait until scheduler reports all tasks done
    while (!orch->scheduler->is_done()) {
        PTO2_SPIN_PAUSE();
    }
}

bool pto2_orchestrator_has_space(PTO2OrchestratorState* orch) { return pto2_task_ring_has_space(&orch->task_ring); }

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_orchestrator_print_stats(PTO2OrchestratorState* orch) {
    LOG_INFO("=== Orchestrator Statistics ===");
#if PTO2_PROFILING
    LOG_INFO("Tasks submitted:     %lld", (long long)orch->tasks_submitted);
    LOG_INFO("Buffers allocated:   %lld", (long long)orch->buffers_allocated);
    LOG_INFO("Bytes allocated:     %lld", (long long)orch->bytes_allocated);
#endif
    LOG_INFO("Current scope depth: %d", orch->scope_stack_top + 1);
    LOG_INFO("Task ring active:    %d", pto2_task_ring_active_count(&orch->task_ring));
    LOG_INFO("Heap ring used:      %" PRIu64 " / %" PRIu64, orch->heap_ring.top, orch->heap_ring.size);
    LOG_INFO("Dep pool used:       %d / %d", pto2_dep_pool_used(&orch->dep_pool), orch->dep_pool.capacity);
    LOG_INFO("TensorMap valid:     %d", orch->tensor_map.valid_count());
    LOG_INFO("===============================");
}

void pto2_orchestrator_print_scope_stack(PTO2OrchestratorState* orch) {
    LOG_INFO("=== Scope Stack ===");
    LOG_INFO("Depth: %d", orch->scope_stack_top + 1);

    for (int i = 0; i <= orch->scope_stack_top; i++) {
        int32_t begin = orch->scope_begins[i];
        int32_t end = (i < orch->scope_stack_top) ? orch->scope_begins[i + 1] : orch->scope_tasks_size;
        LOG_INFO("  [%d] tasks_owned = %d", i, end - begin);
    }

    LOG_INFO("==================");
}

#if PTO2_ORCH_PROFILING
PTO2OrchProfilingData pto2_orchestrator_get_profiling() {
    PTO2OrchProfilingData d;
    d.sync_cycle = g_orch_sync_cycle;
    d.alloc_cycle = g_orch_alloc_cycle;
    d.params_cycle = g_orch_params_cycle;
    d.lookup_cycle = g_orch_lookup_cycle;
    d.heap_cycle = g_orch_heap_cycle;
    d.insert_cycle = g_orch_insert_cycle;
    d.fanin_cycle = g_orch_fanin_cycle;
    d.finalize_cycle = g_orch_finalize_cycle;
    d.scope_end_cycle = g_orch_scope_end_cycle;
    d.submit_count = g_orch_submit_count;
    d.alloc_wait_cycle = g_orch_alloc_wait_cycle;
    d.heap_wait_cycle = g_orch_heap_wait_cycle;
    d.fanin_wait_cycle = g_orch_fanin_wait_cycle;
    d.finalize_wait_cycle = g_orch_finalize_wait_cycle;
    d.alloc_atomic_count = g_orch_alloc_atomic_count;
    d.params_atomic_count = g_orch_params_atomic_count;
    d.heap_atomic_count = g_orch_heap_atomic_count;
    d.fanin_atomic_count = g_orch_fanin_atomic_count;
    d.finalize_atomic_count = g_orch_finalize_atomic_count;
    d.scope_end_atomic_count = g_orch_scope_end_atomic_count;

    // Reset
    g_orch_sync_cycle = g_orch_alloc_cycle = g_orch_params_cycle = 0;
    g_orch_lookup_cycle = g_orch_heap_cycle = g_orch_insert_cycle = 0;
    g_orch_fanin_cycle = g_orch_finalize_cycle = g_orch_scope_end_cycle = 0;
    g_orch_submit_count = 0;
    g_orch_submit_idx = 0;
    g_orch_alloc_wait_cycle = 0;
    g_orch_heap_wait_cycle = 0;
    g_orch_fanin_wait_cycle = 0;
    g_orch_finalize_wait_cycle = 0;
    g_orch_alloc_atomic_count = 0;
    g_orch_params_atomic_count = 0;
    g_orch_heap_atomic_count = 0;
    g_orch_fanin_atomic_count = 0;
    g_orch_finalize_atomic_count = 0;
    g_orch_scope_end_atomic_count = 0;
    return d;
}
#endif
