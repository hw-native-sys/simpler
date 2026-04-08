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
 * PTO Runtime2 - Orchestrator Implementation
 *
 * Implements orchestrator state management, scope handling, and task submission.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_orchestrator.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/unified_log.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"
#include "tensor.h"

// =============================================================================
// Orchestrator Profiling (compile-time toggle)
// =============================================================================
#if PTO2_ORCH_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/performance_collector_aicpu.h"
// Weak fallback for builds that don't link device_time.cpp (e.g. host).
// The strong symbol from platform/.../device_time.cpp wins in the AICPU build.
//
// IMPORTANT: visibility("hidden") is required to prevent the HOST .so from
// exporting this weak fallback into the global dynamic symbol table via
// RTLD_GLOBAL. Without it, when the AICPU .so is loaded and its PLT entry
// for get_sys_cnt_aicpu is resolved, the dynamic linker finds the HOST .so's
// weak definition first (already in global table) and uses it — returning 0.
// With hidden visibility, the HOST .so does not export this symbol globally,
// so the AICPU .so's PLT resolves to its own strong definition from
// device_time.cpp.
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }
// Weak fallback for builds that don't link performance_collector_aicpu.cpp.
// The strong symbol from the AICPU build wins when profiling is available.
// Also hidden to prevent HOST .so from polluting the global symbol table.
__attribute__((weak, visibility("hidden"))) void
perf_aicpu_record_orch_phase(AicpuPhaseId, uint64_t, uint64_t, uint32_t, uint64_t) {}
// Accumulated cycles per sub-step (only needed for ORCH_PROFILING export)
static uint64_t g_orch_sync_cycle = 0;       // tensormap sync
static uint64_t g_orch_alloc_cycle = 0;      // unified task+heap alloc
static uint64_t g_orch_args_cycle = 0;       // param copy
static uint64_t g_orch_lookup_cycle = 0;     // tensormap lookup + dep building
static uint64_t g_orch_insert_cycle = 0;     // tensormap insert
static uint64_t g_orch_fanin_cycle = 0;      // fanin list + early-return check
static uint64_t g_orch_scope_end_cycle = 0;  // scope_end overhead
static int64_t g_orch_submit_count = 0;
static uint32_t g_orch_submit_idx = 0;
uint64_t g_orch_alloc_wait_cycle = 0;
uint64_t g_orch_fanin_wait_cycle = 0;
uint64_t g_orch_alloc_atomic_count = 0;
uint64_t g_orch_args_atomic_count = 0;
uint64_t g_orch_fanin_atomic_count = 0;
uint64_t g_orch_finalize_atomic_count = 0;
uint64_t g_orch_scope_end_atomic_count = 0;
#define CYCLE_COUNT_START() uint64_t _t0 = get_sys_cnt_aicpu(), _t1
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)                                    \
    do {                                                                              \
        _t1 = get_sys_cnt_aicpu();                                                    \
        acc += (_t1 - _t0);                                                           \
        perf_aicpu_record_orch_phase((phase_id), _t0, _t1, g_orch_submit_idx, (tid)); \
        _t0 = _t1;                                                                    \
    } while (0)
#elif PTO2_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/performance_collector_aicpu.h"
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }
__attribute__((weak, visibility("hidden"))) void
perf_aicpu_record_orch_phase(AicpuPhaseId, uint64_t, uint64_t, uint32_t, uint64_t) {}
// submit_idx needed for swimlane task_id tagging (no cycle accumulation at this level)
static uint32_t g_orch_submit_idx = 0;
#define CYCLE_COUNT_START()                     \
    bool _prof_active = orch->enable_profiling; \
    uint64_t _t0 = _prof_active ? get_sys_cnt_aicpu() : 0, _t1 = 0
#define CYCLE_COUNT_LAP(acc) \
    do {                     \
    } while (0)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)                                        \
    do {                                                                                  \
        if (_prof_active) {                                                               \
            _t1 = get_sys_cnt_aicpu();                                                    \
            perf_aicpu_record_orch_phase((phase_id), _t0, _t1, g_orch_submit_idx, (tid)); \
            _t0 = _t1;                                                                    \
        }                                                                                 \
    } while (0)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)
#endif

static void *pto2_aligned_zalloc(size_t size, size_t alignment) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    memset(ptr, 0, size);
    return ptr;
}

struct PTO2FaninBuilder {
    PTO2TaskSlotState *inline_slots[PTO2_FANIN_INLINE_CAP];
    int32_t count{0};
    int32_t spill_start{0};
    PTO2FaninPool *spill_pool{nullptr};

    template <typename Fn>
    PTO2FaninForEachReturn<Fn> for_each(Fn &&fn) const {
        return pto2_for_each_fanin_storage(inline_slots, count, spill_start, spill_pool, static_cast<Fn &&>(fn));
    }

    bool contains(PTO2TaskSlotState *prod_state) const {
        bool found = false;
        for_each([&](PTO2TaskSlotState *slot_state) {
            if (slot_state == prod_state) {
                found = true;
                return false;
            }
            return true;
        });
        if (found) {
            return true;
        }
        return false;
    }
};

static bool pto2_append_fanin_or_fail(
    PTO2OrchestratorState *orch, PTO2TaskId task_id, int32_t tensor_arg_index, TensorArgType ptype,
    PTO2TaskSlotState *prod_state, PTO2FaninBuilder *fanin_builder, PTO2SchedulerState *sched, PTO2RingFlowControl &fc,
    uint8_t ring_id, const char *reason
) {
    if (fanin_builder->contains(prod_state)) {
        return true;
    }

    if (fanin_builder->count < PTO2_FANIN_INLINE_CAP) {
        fanin_builder->inline_slots[fanin_builder->count++] = prod_state;
        return true;
    }

    if (sched == nullptr || fanin_builder->spill_pool == nullptr) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Fanin Spill Builder Misconfigured!");
        LOG_ERROR("========================================");
        LOG_ERROR("Missing scheduler or fanin spill pool while appending dynamic fanin.");
        LOG_ERROR("  task_id.raw:        %" PRIu64, task_id.raw);
        LOG_ERROR("  tensor_arg_index:   %d", tensor_arg_index);
        LOG_ERROR("  tensor_arg_type:    %d", static_cast<int>(ptype));
        LOG_ERROR("  reason:             %s", reason);
        LOG_ERROR("========================================");
        orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_DEPENDENCY_OVERFLOW, std::memory_order_release);
        orch->fatal = true;
        return false;
    }

    PTO2FaninPool &fanin_pool = *fanin_builder->spill_pool;
    fanin_pool.ensure_space(*sched, fc, ring_id, 1);
    int32_t spill_idx = fanin_pool.top;
    PTO2FaninSpillEntry *entry = fanin_pool.alloc();
    if (entry == nullptr) {
        orch->fatal = true;
        return false;
    }
    if (fanin_builder->count == PTO2_FANIN_INLINE_CAP) {
        fanin_builder->spill_start = spill_idx;
    }
    entry->slot_state = prod_state;
    fanin_builder->count++;
    return true;
}

static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state);

struct PTO2OutputLayout {
    uint64_t offsets[MAX_TENSOR_ARGS] = {};
    uint64_t buffer_sizes[MAX_TENSOR_ARGS] = {};
    int32_t total_output_size = 0;
};

struct PTO2PreparedTask {
    PTO2SchedulerState *sched = nullptr;
    PTO2TaskId task_id = PTO2TaskId::invalid();
    PTO2TaskAllocResult alloc_result = {-1, 0, nullptr, nullptr};
    PTO2TaskDescriptor *task = nullptr;
    PTO2TaskPayload *payload = nullptr;
    PTO2TaskSlotState *slot_state = nullptr;
};

static PTO2OutputLayout pto2_calculate_output_layout(const Arg &args) {
    PTO2OutputLayout layout;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            continue;
        }
        layout.offsets[i] = layout.total_output_size;
        layout.buffer_sizes[i] =
            PTO2_ALIGN_UP(args.tensor(i).create_info->buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
        layout.total_output_size += layout.buffer_sizes[i];
    }
    return layout;
}

static bool
pto2_check_scope_can_accept_task(PTO2OrchestratorState *orch, PTO2TaskAllocator &allocator, uint8_t ring_id) {
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");

    int32_t scope_task_count = orch->scope_tasks_size - orch->scope_begins[orch->scope_stack_top];
    if (scope_task_count < allocator.window_size() - 1) {
        return true;
    }

    int32_t active_count = allocator.active_count();

    LOG_ERROR("========================================");
    LOG_ERROR("FATAL: Scope Deadlock Detected! (ring %d)", ring_id);
    LOG_ERROR("========================================");
    LOG_ERROR("Tasks in current scope (%d) >= task_window_size (%d).", scope_task_count, allocator.window_size());
    LOG_ERROR("  scope_depth:        %d", orch->scope_stack_top + 1);
    LOG_ERROR("  ring_id:            %d", ring_id);
    LOG_ERROR("  scope_task_count:   %d", scope_task_count);
    LOG_ERROR("  active_tasks:       %d / %d", active_count, allocator.window_size());
    LOG_ERROR("Root Cause:");
    LOG_ERROR("  Tasks within a scope hold a fanout_count reference that is only");
    LOG_ERROR("  released at scope_end. When scope task count >= window_size,");
    LOG_ERROR("  no slots can be reclaimed -> deadlock.");
    LOG_ERROR("Solution:");
    LOG_ERROR("  1. Reduce tasks per scope (use batching/unroll)");
    LOG_ERROR("  2. Increase task window (current: %d)", allocator.window_size());
    LOG_ERROR("     Compile-time: PTO2_TASK_WINDOW_SIZE in pto_runtime2_types.h");
    LOG_ERROR("     Runtime env:  PTO2_RING_TASK_WINDOW=<power-of-2>");
    LOG_ERROR("  3. Split work across multiple scopes");
    LOG_ERROR("========================================");
    orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_SCOPE_DEADLOCK, std::memory_order_release);
    orch->fatal = true;
    return false;
}

static void pto2_prefetch_payload(PTO2TaskPayload *payload, int32_t tensor_count, int32_t scalar_count) {
    for (int32_t i = 0; i < tensor_count; i++) {
        __builtin_prefetch(&payload->tensors[i], 1, 3);
        __builtin_prefetch(reinterpret_cast<char *>(&payload->tensors[i]) + 64, 1, 3);
    }
    for (int32_t i = 0; i < scalar_count; i += 8) {
        __builtin_prefetch(&payload->scalars[i], 1, 3);
    }
    __builtin_prefetch(payload, 1, 3);
    __builtin_prefetch(reinterpret_cast<char *>(payload) + 64, 1, 3);
    __builtin_prefetch(reinterpret_cast<char *>(payload) + 128, 1, 3);
}

static bool pto2_prepare_task(
    PTO2OrchestratorState *orch, const Arg &args, int32_t total_output_size, uint8_t active_mask, PTO2PreparedTask *out
) {
    uint8_t ring_id = orch->current_ring_id();
    auto &allocator = orch->rings[ring_id].task_allocator;

    if (!pto2_check_scope_can_accept_task(orch, allocator, ring_id)) {
        return false;
    }

    out->sched = orch->scheduler;
    out->alloc_result = allocator.alloc(total_output_size);
    if (out->alloc_result.failed()) {
        orch->fatal = true;
        return false;
    }

    out->task_id = PTO2TaskId::make(ring_id, static_cast<uint32_t>(out->alloc_result.task_id));
    out->task = &allocator.task_by_slot(out->alloc_result.slot);
    out->payload = &orch->sm_handle->task_payloads[ring_id][out->alloc_result.slot];

    pto2_prefetch_payload(out->payload, args.tensor_count(), args.scalar_count());

    if (out->sched) {
        auto &rs = out->sched->ring_sched_states[ring_id];
        out->slot_state = &rs.get_slot_state_by_slot(out->alloc_result.slot);
        PTO2TaskSlotState &slot_state = *out->slot_state;
        slot_state.fanin_count = 0;
        slot_state.fanout_head = nullptr;
        slot_state.fanout_lock.store(0, std::memory_order_relaxed);
        slot_state.fanout_count = 1;
        slot_state.fanout_refcount.store(0, std::memory_order_release);
        slot_state.fanin_refcount.store(0, std::memory_order_release);
        slot_state.payload = out->payload;
        slot_state.task = out->task;
        slot_state.active_mask = active_mask;
        slot_state.subtask_done_mask.store(0, std::memory_order_relaxed);
        slot_state.ring_id = ring_id;
        scope_tasks_push(orch, &slot_state);
    } else {
        scope_tasks_push(orch, nullptr);
    }

    return true;
}

// =============================================================================
// Orchestrator Initialization
// =============================================================================

bool pto2_orchestrator_init(
    PTO2OrchestratorState *orch, PTO2SharedMemoryHandle *sm_handle, void *gm_heap, uint64_t heap_size,
    int32_t dep_pool_capacity
) {
    *orch = PTO2OrchestratorState{};

    orch->sm_handle = sm_handle;
    orch->gm_heap_base = gm_heap;
    orch->gm_heap_size = heap_size * PTO2_MAX_RING_DEPTH;
    orch->fatal = false;

    // Initialize per-ring resources
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        void *ring_heap_base = reinterpret_cast<char *>(gm_heap) + r * heap_size;
        auto &fc = sm_handle->header->rings[r].fc;

        // Initialize unified task allocator
        orch->rings[r].task_allocator.init(
            sm_handle->task_descriptors[r], sm_handle->header->rings[r].task_window_size, &fc.current_task_index,
            &fc.last_task_alive, ring_heap_base, heap_size, &sm_handle->header->orch_error_code
        );
        size_t fanin_pool_bytes =
            PTO2_ALIGN_UP(static_cast<size_t>(dep_pool_capacity) * sizeof(PTO2FaninSpillEntry), PTO2_ALIGN_SIZE);
        PTO2FaninSpillEntry *fanin_entries =
            reinterpret_cast<PTO2FaninSpillEntry *>(pto2_aligned_zalloc(fanin_pool_bytes, PTO2_ALIGN_SIZE));
        if (!fanin_entries) {
            for (int j = 0; j < r; j++) {
                free(orch->rings[j].fanin_pool.base);
                free(orch->rings[j].dep_pool.base);
            }
            return false;
        }
        orch->rings[r].fanin_pool.init(fanin_entries, dep_pool_capacity, &sm_handle->header->orch_error_code);

        // Allocate and initialize dependency list pool (per-ring)
        PTO2DepListEntry *dep_entries =
            reinterpret_cast<PTO2DepListEntry *>(calloc(dep_pool_capacity, sizeof(PTO2DepListEntry)));
        if (!dep_entries) {
            // Cleanup previously allocated rings
            for (int j = 0; j < r; j++) {
                free(orch->rings[j].fanin_pool.base);
                free(orch->rings[j].dep_pool.base);
            }
            free(orch->rings[r].fanin_pool.base);
            return false;
        }
        orch->rings[r].dep_pool.init(dep_entries, dep_pool_capacity, &sm_handle->header->orch_error_code);
    }

    // Initialize TensorMap with per-ring task window sizes
    int32_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = sm_handle->header->rings[r].task_window_size;
    }
    if (!orch->tensor_map.init_default(task_window_sizes)) {
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            free(orch->rings[r].fanin_pool.base);
            free(orch->rings[r].dep_pool.base);
        }
        return false;
    }
    orch->tensor_map.orch = orch;

    // Initialize scope stack: one flat buffer for task IDs + one array for begin offsets
    uint64_t max_depth = PTO2_MAX_SCOPE_DEPTH;
    int32_t init_cap = PTO2_SCOPE_TASKS_INIT_CAP;
    orch->scope_tasks = reinterpret_cast<PTO2TaskSlotState **>(malloc(init_cap * sizeof(PTO2TaskSlotState *)));
    orch->scope_begins = reinterpret_cast<int32_t *>(malloc(max_depth * sizeof(int32_t)));
    if (!orch->scope_tasks || !orch->scope_begins) {
        free(orch->scope_tasks);
        free(orch->scope_begins);
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            free(orch->rings[r].fanin_pool.base);
            free(orch->rings[r].dep_pool.base);
        }
        orch->tensor_map.destroy();
        return false;
    }
    orch->scope_tasks_size = 0;
    orch->scope_tasks_capacity = init_cap;
    orch->scope_stack_top = -1;
    orch->scope_stack_capacity = max_depth;
    orch->manual_scope_active = false;

    return true;
}

void pto2_orchestrator_destroy(PTO2OrchestratorState *orch) {
    orch->tensor_map.destroy();

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        free(orch->rings[r].fanin_pool.base);
        orch->rings[r].fanin_pool.base = NULL;
        free(orch->rings[r].dep_pool.base);
        orch->rings[r].dep_pool.base = NULL;
    }

    free(orch->scope_tasks);
    orch->scope_tasks = NULL;
    free(orch->scope_begins);
    orch->scope_begins = NULL;
}

void pto2_orchestrator_set_scheduler(PTO2OrchestratorState *orch, PTO2SchedulerState *scheduler) {
    orch->scheduler = scheduler;
}

// =============================================================================
// Scope Management
// =============================================================================

static bool scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state) {
    if (orch->fatal) {
        return false;
    }
    if (orch->scope_tasks_size >= orch->scope_tasks_capacity) {
        int32_t new_cap = orch->scope_tasks_capacity * 2;
        PTO2TaskSlotState **new_buf =
            reinterpret_cast<PTO2TaskSlotState **>(realloc(orch->scope_tasks, new_cap * sizeof(PTO2TaskSlotState *)));
        if (new_buf == nullptr) {
            LOG_ERROR("Failed to grow scope task buffer to %d entries", new_cap);
            orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_OUT_OF_MEMORY, std::memory_order_release);
            orch->fatal = true;
            return false;
        }
        orch->scope_tasks = new_buf;
        orch->scope_tasks_capacity = new_cap;
    }
    orch->scope_tasks[orch->scope_tasks_size++] = task_slot_state;
    return true;
}

static bool in_manual_scope(const PTO2OrchestratorState *orch) {
    return orch->manual_scope_active;
}

static int32_t current_manual_scope_begin(const PTO2OrchestratorState *orch) {
    return orch->scope_begins[orch->scope_stack_top];
}

static int32_t find_current_manual_scope_task_index(const PTO2OrchestratorState *orch, PTO2TaskId task_id) {
    if (!in_manual_scope(orch) || !task_id.is_valid()) {
        return -1;
    }

    int32_t begin = current_manual_scope_begin(orch);
    int32_t count = orch->scope_tasks_size - begin;
    if (count <= 0) {
        return -1;
    }

    PTO2TaskSlotState *first_slot_state = orch->scope_tasks[begin];
    if (first_slot_state == nullptr) {
        return -1;
    }

    PTO2TaskId first_task_id = first_slot_state->task->task_id;
    if (first_task_id.ring() != task_id.ring()) {
        return -1;
    }

    uint32_t first_local = first_task_id.local();
    uint32_t task_local = task_id.local();
    if (task_local < first_local) {
        return -1;
    }

    uint32_t delta = task_local - first_local;
    if (delta >= static_cast<uint32_t>(count)) {
        return -1;
    }

    PTO2TaskSlotState *candidate = orch->scope_tasks[begin + static_cast<int32_t>(delta)];
    if (candidate != nullptr && candidate->task->task_id == task_id) {
        return static_cast<int32_t>(delta);
    }
    return -1;
}

static bool task_owned_by_current_manual_scope(const PTO2OrchestratorState *orch, PTO2TaskId task_id) {
    return find_current_manual_scope_task_index(orch, task_id) >= 0;
}

void pto2_scope_begin(PTO2OrchestratorState *orch, PTO2ScopeMode mode) {
    if (orch->fatal) {
        return;
    }
    assert(orch->scope_stack_top < static_cast<int32_t>(orch->scope_stack_capacity - 1) && "Scope stack overflow");

    if (in_manual_scope(orch)) {
        LOG_ERROR(
            "nested PTO2_SCOPE(PTO2ScopeMode::MANUAL) is not supported in v1; manual scope inside manual scope is not supported"
        );
        orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_INVALID_ARGS, std::memory_order_release);
        orch->fatal = true;
        return;
    }

    ++orch->scope_stack_top;
    orch->scope_begins[orch->scope_stack_top] = orch->scope_tasks_size;
    orch->manual_scope_active = (mode == PTO2ScopeMode::MANUAL);
}

void pto2_scope_end(PTO2OrchestratorState *orch) {
    if (orch->fatal) {
        return;
    }
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

#if PTO2_ORCH_PROFILING
    uint64_t _se0 = get_sys_cnt_aicpu();
#endif

    int32_t top = orch->scope_stack_top;
    int32_t begin = orch->scope_begins[top];
    int32_t count = orch->scope_tasks_size - begin;
    bool manual_scope = orch->manual_scope_active;

    if (!manual_scope) {
        orch->scope_stack_top--;
        orch->manual_scope_active = false;

        if (orch->scheduler && count > 0) {
            orch->scheduler->on_scope_end(&orch->scope_tasks[begin], count);
        }

        orch->scope_tasks_size = begin;

#if PTO2_ORCH_PROFILING
        uint64_t _se1 = get_sys_cnt_aicpu();
        g_orch_scope_end_cycle += (_se1 - _se0);
#endif
        return;
    }

    if (orch->scheduler && count > 0) {
        for (int32_t task_idx = 0; task_idx < count; task_idx++) {
            PTO2TaskSlotState *slot_state = orch->scope_tasks[begin + task_idx];
            PTO2TaskPayload *payload = slot_state->payload;
            if (payload->fanin_actual_count > PTO2_MAX_INPUTS) {
                LOG_ERROR("========================================");
                LOG_ERROR("FATAL: Dependency Overflow Detected!");
                LOG_ERROR("========================================");
                LOG_ERROR("Task requires more than PTO2_MAX_INPUTS unique fanin dependencies.");
                LOG_ERROR("  task_id.raw:        %" PRIu64, slot_state->task->task_id.raw);
                LOG_ERROR("  fanin_count:        %d / %d", payload->fanin_actual_count, PTO2_MAX_INPUTS);
                LOG_ERROR("  reason:             manual dependency bookkeeping");
                LOG_ERROR("This is a runtime dependency-tracking limit.");
                LOG_ERROR("========================================");
                orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_DEPENDENCY_OVERFLOW, std::memory_order_release);
                orch->fatal = true;
                return;
            }
        }

        int32_t dep_pool_mark_prefix = 0;
        for (int32_t task_idx = 0; task_idx < count; task_idx++) {
            PTO2TaskSlotState *slot_state = orch->scope_tasks[begin + task_idx];
            // add_dependency may allocate dep entries for an older consumer after
            // newer tasks were already submitted. Recompute a monotonic dep-pool
            // watermark at publish time so tail reclamation still advances safely.
            if (slot_state->dep_pool_mark < dep_pool_mark_prefix) {
                slot_state->dep_pool_mark = dep_pool_mark_prefix;
            } else {
                dep_pool_mark_prefix = slot_state->dep_pool_mark;
            }
        }
        orch->scheduler->publish_manual_scope_tasks_and_end_scope(&orch->scope_tasks[begin], count);
    }

    // Rewind the task buffer — these entries are no longer needed
    orch->scope_tasks_size = begin;
    orch->scope_stack_top--;
    orch->manual_scope_active = false;

#if PTO2_ORCH_PROFILING
    uint64_t _se1 = get_sys_cnt_aicpu();
    g_orch_scope_end_cycle += (_se1 - _se0);
    // perf_aicpu_record_orch_phase(AicpuPhaseId::ORCH_SCOPE_END, _se0, _se1, g_orch_submit_idx, -1);
#endif
}

// =============================================================================
// Task Submission
// =============================================================================
template <bool kManualSubmit>
static TaskOutputTensors pto2_submit_mixed_task_impl(
    PTO2OrchestratorState *orch, const MixedKernels &mixed_kernels, const Arg &args,
    PTO2TaskId *submitted_task_id = nullptr
) {
    CYCLE_COUNT_START();

    TaskOutputTensors result;

    // Fast path after fatal error — all subsequent submits are no-ops
    if (orch->fatal) {
        return result;
    }

    // Validate Arg construction (errors recorded by add_input/add_output/etc.)
    if (args.has_error) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg Detected!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg ? args.error_msg : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("This is a bug in the orchestration code.");
        LOG_ERROR("========================================");
        orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_INVALID_ARGS, std::memory_order_release);
        orch->fatal = true;
        return result;
    }
    if (!has_aic && has_aiv1 && !has_aiv0) {
        normalized.aiv0_kernel_id = normalized.aiv1_kernel_id;
        normalized.aiv1_kernel_id = INVALID_KERNEL_ID;
        active_mask = pto2_mixed_kernels_to_active_mask(normalized);
    }

    // Submission without an open scope is illegal.
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");

    if constexpr (!kManualSubmit) {
        if (in_manual_scope(orch)) {
            LOG_ERROR("PTO2_SCOPE(PTO2ScopeMode::MANUAL) requires pto2_rt_submit_*_manual task APIs");
            orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_INVALID_ARGS, std::memory_order_release);
            orch->fatal = true;
            return result;
        }
    }

    // Encode require_sync_start into active_mask bit 3 (only meaningful for tasks with block_num > 1)
    if (block_num > 1 && args.launch_spec.require_sync_start()) {
        // Deadlock check: block_num >= total available slots of the required type.
        // For MIX/AIC: limit is total_cluster_count (one AIC per cluster).
        // For AIV:     limit is total_aiv_count.
        PTO2ResourceShape shape = pto2_active_mask_to_shape(active_mask);
        int32_t limit = (shape == PTO2ResourceShape::AIV) ? orch->total_aiv_count : orch->total_cluster_count;
        if (limit > 0 && block_num > limit) {
            LOG_ERROR("FATAL: require_sync_start block_num=%d > limit=%d (deadlock guaranteed)", block_num, limit);
            orch->fatal = true;
            return TaskOutputTensors{};
        }
        active_mask |= PTO2_SUBTASK_FLAG_SYNC_START;
    }
    PTO2OutputLayout layout = pto2_calculate_output_layout(args);
    PTO2PreparedTask prepared;
    if (!pto2_prepare_task(orch, args, layout.total_output_size, active_mask, &prepared)) {
        return result;
    }
    uint8_t ring_id = prepared.task_id.ring();
    PTO2SchedulerState *sched = prepared.sched;
    PTO2RingFlowControl &fc = orch->sm_handle->header->rings[ring_id].fc;
    PTO2TaskId task_id = prepared.task_id;
    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload *payload = prepared.payload;
    int32_t slot = prepared.alloc_result.slot;

    PTO2FaninBuilder fanin_builder;
    fanin_builder.spill_pool = &orch->rings[ring_id].fanin_pool;
    uint64_t manual_local_mask = 0;
    bool needs_tensormap_sync = true;
    if constexpr (kManualSubmit) {
        needs_tensormap_sync = false;
        for (int i = 0; i < args.tensor_count(); i++) {
            TensorArgType ptype = args.tag(i);
            if (ptype == TensorArgType::OUTPUT) {
                continue;
            }

            const Tensor *tensor = args.tensor(i).ptr;
            if (task_owned_by_current_manual_scope(orch, tensor->owner_task_id)) {
                manual_local_mask |= static_cast<uint64_t>(1ULL << i);
                continue;
            }

            bool needs_lookup = (ptype == TensorArgType::INPUT || ptype == TensorArgType::INOUT) && !tensor->manual_dep;
            bool needs_insert =
                (ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING) && !tensor->manual_dep;
            if (needs_lookup || needs_insert) {
                needs_tensormap_sync = true;
            }
        }
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_alloc_cycle, AicpuPhaseId::ORCH_ALLOC, task_id.raw);

#if PTO2_PROFILING
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    // === STEP 2: Sync TensorMap validity and optional cleanup ===
    int32_t sm_last_task_alive = fc.last_task_alive.load(std::memory_order_acquire);

    if (!kManualSubmit || needs_tensormap_sync) {
        orch->tensor_map.sync_tensormap(ring_id, sm_last_task_alive);
    }

    if constexpr (!kManualSubmit) {
        if (sched) {
            orch->rings[ring_id].dep_pool.reclaim(*sched, ring_id, sm_last_task_alive);
        }
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_sync_cycle, AicpuPhaseId::ORCH_SYNC, task_id.raw);

    // === STEP 3: Lookup inputs + materialize runtime-created outputs ===
    for (int i = 0; i < args.tensor_count(); i++) {
        TensorArgType ptype = args.tag(i);
        if (ptype == TensorArgType::OUTPUT) {
            continue;
        }

        const Tensor *tensor = args.tensor(i).ptr;
        if constexpr (kManualSubmit) {
            if ((manual_local_mask & static_cast<uint64_t>(1ULL << i)) != 0) {
                continue;
            }
        }

        PTO2TaskId owner = tensor->owner_task_id;
        if (owner.is_valid() && sched != nullptr) {
            PTO2TaskSlotState *prod_state =
                &sched->ring_sched_states[owner.ring()].get_slot_state_by_task_id(owner.local());
            if (!pto2_append_fanin_or_fail(
                    orch, task_id, i, ptype, prod_state, &fanin_builder, sched, fc, ring_id, "creator retention"
                )) {
                return result;
            }
        }

        if (ptype != TensorArgType::INPUT && ptype != TensorArgType::INOUT) {
            continue;
        }
        if (tensor->manual_dep) {
            continue;
        }

        PTO2LookupResult lookup_result;
        orch->tensor_map.lookup(*tensor, lookup_result);

        for (int r = 0; r < lookup_result.count; r++) {
            PTO2TensorMapEntry &entry = *lookup_result.entries[r].entry;
            auto overlap_status = lookup_result.entries[r].overlap_status;
            auto prod_ring = entry.producer_task_id.ring();
            auto prod_local = entry.producer_task_id.local();
            PTO2TaskSlotState *prod_state =
                &sched->ring_sched_states[prod_ring].get_slot_state_by_task_id(prod_local);
            if (!pto2_append_fanin_or_fail(
                    orch, task_id, i, ptype, prod_state, &fanin_builder, sched, fc, ring_id, "overlap lookup"
                )) {
                return result;
            }
            if (ptype == TensorArgType::INOUT && overlap_status == OverlapStatus::COVERED) {
                orch->tensor_map.remove_entry(entry);
            }
        }
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_lookup_cycle, AicpuPhaseId::ORCH_LOOKUP, task_id.raw);

    // === STEP 4: Register outputs/inouts in TensorMap (must be separate from lookup) ===
    for (int i = 0; i < args.tensor_count(); i++) {
        TensorArgType ptype = args.tag(i);
        if (ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING) {
            if constexpr (kManualSubmit) {
                if ((manual_local_mask & static_cast<uint64_t>(1ULL << i)) != 0) {
                    continue;
                }
            }
            if (!args.tensor(i).ptr->manual_dep) {
                orch->tensor_map.insert(*args.tensor(i).ptr, task_id);
            }
        }
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_insert_cycle, AicpuPhaseId::ORCH_INSERT, task_id.raw);

    // === STEP 5: Batch-write to GM (single cache line burst) ===
    // Deferred from allocation phase to avoid scattered GM writes that get
    // evicted by TensorMap lookup/insert cache pressure.
    __builtin_prefetch(&task, 1, 1);
    task.task_id = task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = normalized.aic_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = normalized.aiv0_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = normalized.aiv1_kernel_id;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    // Prefetch producer slot_states and cur_slot_state (written at init but likely
    // evicted by lookup/insert/heap). param_copy below provides hide time.
    if (sched) {
        auto &rs = sched->ring_sched_states[ring_id];
        __builtin_prefetch(&rs.get_slot_state_by_slot(slot), 1, 0);
        fanin_builder.for_each([](PTO2TaskSlotState *producer_slot_state) {
            __builtin_prefetch(producer_slot_state, 1, 0);
            return true;
        });
    }

    payload->init(args, result, prepared.alloc_result.packed_base, layout.offsets, layout.buffer_sizes);

    // Write owner_task_id into materialized OUTPUT tensors so creator-only dependency
    // tracking remains available even when manual_dep skips OverlapMap publication.
    for (int i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) == TensorArgType::OUTPUT) {
            payload->tensors[i].owner_task_id = task_id;
        }
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_args_cycle, AicpuPhaseId::ORCH_PARAMS, task_id.raw);
#if PTO2_ORCH_PROFILING
    g_orch_args_atomic_count += 2;  // fanout_lock.store + fanout_count.store
#endif

    // === STEP 6: Finalize fanin list ===
    // First build the fanin list
    if (sched) {
        auto &rs = sched->ring_sched_states[ring_id];
        PTO2TaskSlotState &cur_slot_state = rs.get_slot_state_by_slot(slot);
        // Initialize scheduler state BEFORE adding to producer fanout lists,
        // so concurrent on_mixed_task_complete can safely access task_state/fanout_refcount.
        cur_slot_state.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
        cur_slot_state.fanout_refcount.store(0, std::memory_order_relaxed);
        cur_slot_state.completed_subtasks.store(0, std::memory_order_relaxed);
        cur_slot_state.total_required_subtasks =
            static_cast<int16_t>(block_num * __builtin_popcount(pto2_core_mask(active_mask)));
        cur_slot_state.block_num = block_num;
        cur_slot_state.next_block_idx = 0;

        if constexpr (kManualSubmit) {
            int32_t fanin_count = fanin_builder.count;
            int32_t inline_count = std::min(fanin_count, PTO2_FANIN_INLINE_CAP);
            int32_t spill_count = fanin_count - inline_count;
            payload->fanin_actual_count = fanin_count;
            payload->fanin_spill_start = (spill_count > 0) ? fanin_builder.spill_start : 0;
            payload->fanin_spill_pool = (spill_count > 0) ? fanin_builder.spill_pool : nullptr;
            for (int i = 0; i < inline_count; i++) {
                payload->fanin_inline_slot_states[i] = fanin_builder.inline_slots[i];
            }

            auto &dep_pool = orch->rings[ring_id].dep_pool;
            dep_pool.ensure_space(*sched, fc, ring_id, fanin_count);

            int32_t early_finished = 0;
            bool fanout_ok = fanin_builder.for_each([&](PTO2TaskSlotState *producer_slot) {
                PTO2TaskSlotState &producer_slot_state = *producer_slot;
#if PTO2_ORCH_PROFILING
                pto2_fanout_lock(producer_slot_state, g_orch_fanin_atomic_count, g_orch_fanin_wait_cycle);
#else
                pto2_fanout_lock(producer_slot_state);
#endif
                producer_slot_state.fanout_count += 1;
                int32_t prod_state = producer_slot_state.task_state.load(std::memory_order_acquire);
                if (prod_state >= PTO2_TASK_COMPLETED) {
                    early_finished++;
                } else {
                    producer_slot_state.fanout_head =
                        dep_pool.prepend(producer_slot_state.fanout_head, &cur_slot_state);
                    if (producer_slot_state.fanout_head == nullptr) {
                        pto2_fanout_unlock(producer_slot_state);
                        orch->fatal = true;
                        return false;
                    }
                }
                pto2_fanout_unlock(producer_slot_state);
                return true;
            });
            if (!fanout_ok) {
                return result;
            }
            cur_slot_state.fanin_count = fanin_count + 1;
            if (early_finished > 0) {
                cur_slot_state.fanin_refcount.fetch_add(early_finished, std::memory_order_acq_rel);
            }
            cur_slot_state.dep_pool_mark = dep_pool.top;
#if PTO2_ORCH_PROFILING
            g_orch_fanin_atomic_count += fanin_count * 3;
            if (early_finished > 0) {
                g_orch_fanin_atomic_count += 1;  // fanin_refcount.fetch_add
            }
#endif
        } else {
            auto &dep_pool = orch->rings[ring_id].dep_pool;
            int32_t fanin_count = fanin_builder.count;
            int32_t inline_count = std::min(fanin_count, PTO2_FANIN_INLINE_CAP);
            int32_t spill_count = fanin_count - inline_count;
            dep_pool.ensure_space(*sched, fc, ring_id, fanin_count + 1);

            int32_t early_finished = 0;
            cur_slot_state.fanin_count = fanin_count + 1;  // +1 redundance for not being ready too early
            payload->fanin_actual_count = fanin_count;
            payload->fanin_spill_start = (spill_count > 0) ? fanin_builder.spill_start : 0;
            payload->fanin_spill_pool = (spill_count > 0) ? fanin_builder.spill_pool : nullptr;
            for (int i = 0; i < inline_count; i++) {
                payload->fanin_inline_slot_states[i] = fanin_builder.inline_slots[i];
            }
            pto2_for_each_fanin_slot_state(*payload, [&](PTO2TaskSlotState *producer_slot) {
                PTO2TaskSlotState &producer_slot_state = *producer_slot;
#if PTO2_ORCH_PROFILING
                pto2_fanout_lock(producer_slot_state, g_orch_fanin_atomic_count, g_orch_fanin_wait_cycle);
#else
                pto2_fanout_lock(producer_slot_state);
#endif
                // Normal path: prepend consumer to producer's fanout list
                producer_slot_state.fanout_count += 1;
                int32_t prod_state = producer_slot_state.task_state.load(std::memory_order_acquire);
                if (prod_state >= PTO2_TASK_COMPLETED) {
                    // Early return optimization: if producer already completed, we can skip adding dependency and
                    // directly decrement fanin_count
                    early_finished++;
                } else {
                    producer_slot_state.fanout_head =
                        dep_pool.prepend(producer_slot_state.fanout_head, &cur_slot_state);
                }
                pto2_fanout_unlock(producer_slot_state);
                return true;
            });
            int32_t initial_refcount = early_finished + 1;  // +1 for the init release
            int32_t new_rc =
                cur_slot_state.fanin_refcount.fetch_add(initial_refcount, std::memory_order_acq_rel) + initial_refcount;
            if (new_rc >= fanin_count + 1) {
                PTO2ResourceShape shape = pto2_active_mask_to_shape(active_mask);
                sched->ready_queues[static_cast<int32_t>(shape)].push(&cur_slot_state);
            }
            // Record dep pool watermark in local slot state (used by tail reclamation)
            cur_slot_state.dep_pool_mark = orch->rings[ring_id].dep_pool.top;
#if PTO2_ORCH_PROFILING
            // Per producer: fetch_add(fanout_count) + load(task_state) + store(unlock) = 3 atomics
            // Lock atomics (loads + CAS) are counted inside pto2_fanout_lock
            g_orch_fanin_atomic_count += fanin_count * 3;
            if (early_finished > 0) {
                g_orch_fanin_atomic_count += 1;  // fanin_refcount.fetch_add
            }
#endif
        }
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_fanin_cycle, AicpuPhaseId::ORCH_FANIN, task_id.raw);

#if PTO2_PROFILING
    orch->tasks_submitted++;
#if PTO2_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif
    return result;
}

TaskOutputTensors pto2_alloc_tensors(PTO2OrchestratorState *orch, const Arg &args) {
    always_assert(!orch->fatal && "alloc_tensor cannot be called after the runtime becomes fatal");
    always_assert(args.tensor_count() > 0 && "alloc_tensors requires at least one TensorCreateInfo");
    always_assert(args.scalar_count() == 0 && "alloc_tensors only accepts output TensorCreateInfo args");
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        always_assert(
            args.tag(i) == TensorArgType::OUTPUT && "alloc_tensors only accepts output TensorCreateInfo args"
        );
    }

    CYCLE_COUNT_START();

    always_assert(!args.has_error && "alloc_tensors failed to construct output-only Arg");

    PTO2OutputLayout layout = pto2_calculate_output_layout(args);
    PTO2PreparedTask prepared;
    if (!pto2_prepare_task(orch, args, layout.total_output_size, 0, &prepared)) {
        return TaskOutputTensors{};
    }

    uint8_t ring_id = prepared.task_id.ring();
    PTO2RingFlowControl &fc = orch->sm_handle->header->rings[ring_id].fc;
    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload *payload = prepared.payload;

    CYCLE_COUNT_LAP_RECORD(g_orch_alloc_cycle, AicpuPhaseId::ORCH_ALLOC, prepared.task_id.raw);

#if PTO2_PROFILING
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    int32_t sm_last_task_alive = fc.last_task_alive.load(std::memory_order_acquire);
    orch->tensor_map.sync_tensormap(ring_id, sm_last_task_alive);
    if (prepared.sched != nullptr) {
        orch->rings[ring_id].dep_pool.reclaim(*prepared.sched, ring_id, sm_last_task_alive);
    }
    CYCLE_COUNT_LAP_RECORD(g_orch_sync_cycle, AicpuPhaseId::ORCH_SYNC, prepared.task_id.raw);

    task.task_id = prepared.task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = INVALID_KERNEL_ID;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    TaskOutputTensors outputs;
    payload->init(args, outputs, prepared.alloc_result.packed_base, layout.offsets, layout.buffer_sizes);
    payload->fanin_actual_count = 0;
    payload->fanin_spill_start = 0;
    payload->fanin_spill_pool = nullptr;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        payload->tensors[i].owner_task_id = prepared.task_id;
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_args_cycle, AicpuPhaseId::ORCH_PARAMS, prepared.task_id.raw);

    if (prepared.slot_state != nullptr) {
        // Hidden alloc tasks complete inline in the orchestrator before any
        // consumer can exist, so they have no fanout to notify and no worker
        // subtasks to retire. Running the full on_mixed_task_complete path
        // would only pay unnecessary fanout_lock / traversal overhead here.
        // The generic slot initialization done in pto2_prepare_task() is still
        // required so scope_end can release the producer-side reference and
        // drive the slot to CONSUMED, but worker dispatch fields are never
        // observed for hidden alloc tasks.
        prepared.slot_state->task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
    }
    orch->inline_completed_tasks++;

    CYCLE_COUNT_LAP_RECORD(g_orch_fanin_cycle, AicpuPhaseId::ORCH_FANIN, prepared.task_id.raw);

#if PTO2_PROFILING
    orch->tasks_submitted++;
#if PTO2_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif

    return outputs;
}

TaskOutputTensors
pto2_submit_mixed_task(PTO2OrchestratorState *orch, const MixedKernels &mixed_kernels, const Arg &args) {
    if (in_manual_scope(orch)) {
        LOG_ERROR("PTO2_SCOPE(PTO2ScopeMode::MANUAL) requires pto2_rt_submit_*_manual task APIs");
        orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_INVALID_ARGS, std::memory_order_release);
        orch->fatal = true;
        return {};
    }
    return pto2_submit_mixed_task_impl<false>(orch, mixed_kernels, args);
}

PTO2ManualSubmitResult
pto2_submit_mixed_task_manual(PTO2OrchestratorState *orch, const MixedKernels &mixed_kernels, const Arg &args) {
    PTO2ManualSubmitResult result{};
    if (!in_manual_scope(orch)) {
        LOG_ERROR("manual submit APIs require PTO2_SCOPE(PTO2ScopeMode::MANUAL)");
        orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_INVALID_ARGS, std::memory_order_release);
        orch->fatal = true;
        return result;
    }
    PTO2TaskId task_id = PTO2TaskId::invalid();
    TaskOutputTensors outputs = pto2_submit_mixed_task_impl<true>(orch, mixed_kernels, args, &task_id);
    if (orch->fatal || !task_id.is_valid()) {
        return result;
    }
    result.task_id = task_id;
    result.outputs = outputs;
    return result;
}

void pto2_add_dependency(PTO2OrchestratorState *orch, PTO2TaskId producer_id, PTO2TaskId consumer_id) {
    if (orch->fatal) {
        return;
    }

    if (!in_manual_scope(orch)) {
        LOG_ERROR("pto2_rt_add_dependency is only valid inside PTO2_SCOPE(PTO2ScopeMode::MANUAL)");
        orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_INVALID_ARGS, std::memory_order_release);
        orch->fatal = true;
        return;
    }
    if (producer_id == consumer_id) {
        LOG_ERROR("add_dependency does not allow self-dependency");
        orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_INVALID_ARGS, std::memory_order_release);
        orch->fatal = true;
        return;
    }
    int32_t producer_idx = find_current_manual_scope_task_index(orch, producer_id);
    int32_t consumer_idx = find_current_manual_scope_task_index(orch, consumer_id);
    if (producer_idx < 0 || consumer_idx < 0) {
        LOG_ERROR("add_dependency requires producer and consumer to belong to the current manual scope");
        orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_INVALID_ARGS, std::memory_order_release);
        orch->fatal = true;
        return;
    }

    PTO2TaskSlotState *producer_slot_state = orch->scope_tasks[current_manual_scope_begin(orch) + producer_idx];
    PTO2TaskSlotState *consumer_slot_state = orch->scope_tasks[current_manual_scope_begin(orch) + consumer_idx];
    PTO2TaskPayload *consumer_payload = consumer_slot_state->payload;

    bool duplicate = false;
    pto2_for_each_fanin_slot_state(*consumer_payload, [&](PTO2TaskSlotState *fanin_slot_state) {
        if (fanin_slot_state == producer_slot_state) {
            duplicate = true;
            return false;
        }
        return true;
    });
    if (duplicate) {
        return;
    }

    if (consumer_payload->fanin_actual_count >= PTO2_MAX_INPUTS) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Dependency Overflow Detected!");
        LOG_ERROR("========================================");
        LOG_ERROR("Task requires more than PTO2_MAX_INPUTS unique fanin dependencies.");
        LOG_ERROR("  consumer_id.raw:    %" PRIu64, consumer_id.raw);
        LOG_ERROR("  fanin_count:        %d / %d", consumer_payload->fanin_actual_count + 1, PTO2_MAX_INPUTS);
        LOG_ERROR("  reason:             explicit add_dependency");
        LOG_ERROR("========================================");
        orch->sm_handle->header->orch_error_code.store(PTO2_ERROR_DEPENDENCY_OVERFLOW, std::memory_order_release);
        orch->fatal = true;
        return;
    }

    auto &dep_pool = orch->rings[producer_slot_state->ring_id].dep_pool;
    auto &fc = orch->sm_handle->header->rings[producer_slot_state->ring_id].fc;
    dep_pool.ensure_space(*orch->scheduler, fc, producer_slot_state->ring_id, 1);

    PTO2FaninBuilder fanin_builder;
    fanin_builder.count = consumer_payload->fanin_actual_count;
    fanin_builder.spill_start = consumer_payload->fanin_spill_start;
    fanin_builder.spill_pool =
        (consumer_payload->fanin_spill_pool != nullptr) ? consumer_payload->fanin_spill_pool
                                                        : &orch->rings[consumer_slot_state->ring_id].fanin_pool;
    int32_t cached_inline_count = std::min(fanin_builder.count, PTO2_FANIN_INLINE_CAP);
    for (int32_t i = 0; i < cached_inline_count; i++) {
        fanin_builder.inline_slots[i] = consumer_payload->fanin_inline_slot_states[i];
    }

    if (fanin_builder.count < PTO2_FANIN_INLINE_CAP) {
        fanin_builder.inline_slots[fanin_builder.count++] = producer_slot_state;
    } else {
        PTO2FaninPool &fanin_pool = *fanin_builder.spill_pool;
        fanin_pool.ensure_space(*orch->scheduler, fc, consumer_slot_state->ring_id, 1);
        int32_t spill_idx = fanin_pool.top;
        PTO2FaninSpillEntry *entry = fanin_pool.alloc();
        if (entry == nullptr) {
            orch->fatal = true;
            return;
        }
        if (fanin_builder.count == PTO2_FANIN_INLINE_CAP) {
            fanin_builder.spill_start = spill_idx;
        }
        entry->slot_state = producer_slot_state;
        fanin_builder.count++;
    }

    producer_slot_state->fanout_count += 1;
    producer_slot_state->fanout_head = dep_pool.prepend(producer_slot_state->fanout_head, consumer_slot_state);
    if (producer_slot_state->fanout_head == nullptr) {
        orch->fatal = true;
        return;
    }

    int32_t inline_count = std::min(fanin_builder.count, PTO2_FANIN_INLINE_CAP);
    int32_t spill_count = fanin_builder.count - inline_count;
    consumer_payload->fanin_actual_count = fanin_builder.count;
    consumer_payload->fanin_spill_start = (spill_count > 0) ? fanin_builder.spill_start : 0;
    consumer_payload->fanin_spill_pool = (spill_count > 0) ? fanin_builder.spill_pool : nullptr;
    for (int32_t i = 0; i < inline_count; i++) {
        consumer_payload->fanin_inline_slot_states[i] = fanin_builder.inline_slots[i];
    }
    consumer_slot_state->fanin_count += 1;
    consumer_slot_state->dep_pool_mark = dep_pool.top;
}

// =============================================================================
// Flow Control
// =============================================================================

void pto2_orchestrator_done(PTO2OrchestratorState *orch) {
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        int32_t total_tasks = orch->rings[r].task_allocator.active_count();
        if (total_tasks > 0) {
            LOG_INFO("=== [Orchestrator] ring %d: total_tasks=%d ===", r, total_tasks);
        }
        auto &fanin_pool = orch->rings[r].fanin_pool;
        auto &pool = orch->rings[r].dep_pool;
        if (fanin_pool.top > 1) {
            LOG_INFO(
                "=== [FaninPool %d] top=%d tail=%d used=%d high_water=%d capacity=%d ===", r, fanin_pool.top,
                fanin_pool.tail, fanin_pool.top - fanin_pool.tail, fanin_pool.high_water, fanin_pool.capacity
            );
        }
        if (pool.top > 0) {
            LOG_INFO(
                "=== [DepPool %d] top=%d tail=%d used=%d high_water=%d capacity=%d ===", r, pool.top, pool.tail,
                pool.top - pool.tail, pool.high_water, pool.capacity
            );
        }
    }
    orch->sm_handle->header->orchestrator_done.store(1, std::memory_order_release);
#if !PTO2_ORCH_PROFILING && PTO2_PROFILING
    g_orch_submit_idx = 0;
#endif
}

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_orchestrator_print_stats(PTO2OrchestratorState *orch) {
    LOG_INFO("=== Orchestrator Statistics ===");
#if PTO2_PROFILING
    LOG_INFO("Tasks submitted:     %" PRId64, orch->tasks_submitted);
    LOG_INFO("Buffers allocated:   %" PRId64, orch->buffers_allocated);
    LOG_INFO("Bytes allocated:     %" PRId64, orch->bytes_allocated);
#endif
    LOG_INFO("Current scope depth: %d", orch->scope_stack_top + 1);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        int32_t active = orch->rings[r].task_allocator.active_count();
        if (active > 0) {
            LOG_INFO("Ring %d task active:  %d", r, active);
            LOG_INFO(
                "Ring %d heap used:    %" PRIu64 " / %" PRIu64, r, orch->rings[r].task_allocator.heap_top(),
                orch->rings[r].task_allocator.heap_capacity()
            );
            LOG_INFO(
                "Ring %d fanin pool:   %d / %d", r, orch->rings[r].fanin_pool.used(), orch->rings[r].fanin_pool.capacity
            );
            LOG_INFO(
                "Ring %d dep pool:     %d / %d", r, orch->rings[r].dep_pool.used(), orch->rings[r].dep_pool.capacity
            );
        }
    }
    LOG_INFO("TensorMap valid:     %d", orch->tensor_map.valid_count());
    LOG_INFO("===============================");
}

void pto2_orchestrator_print_scope_stack(PTO2OrchestratorState *orch) {
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
    d.args_cycle = g_orch_args_cycle;
    d.lookup_cycle = g_orch_lookup_cycle;
    d.insert_cycle = g_orch_insert_cycle;
    d.fanin_cycle = g_orch_fanin_cycle;
    d.scope_end_cycle = g_orch_scope_end_cycle;
    d.submit_count = g_orch_submit_count;
    d.alloc_wait_cycle = g_orch_alloc_wait_cycle;
    d.fanin_wait_cycle = g_orch_fanin_wait_cycle;
    d.alloc_atomic_count = g_orch_alloc_atomic_count;
    d.args_atomic_count = g_orch_args_atomic_count;
    d.fanin_atomic_count = g_orch_fanin_atomic_count;
    d.finalize_atomic_count = g_orch_finalize_atomic_count;
    d.scope_end_atomic_count = g_orch_scope_end_atomic_count;

    // Reset
    g_orch_sync_cycle = g_orch_alloc_cycle = g_orch_args_cycle = 0;
    g_orch_lookup_cycle = g_orch_insert_cycle = 0;
    g_orch_fanin_cycle = g_orch_scope_end_cycle = 0;
    g_orch_submit_count = 0;
    g_orch_submit_idx = 0;
    g_orch_alloc_wait_cycle = 0;
    g_orch_fanin_wait_cycle = 0;
    g_orch_alloc_atomic_count = 0;
    g_orch_args_atomic_count = 0;
    g_orch_fanin_atomic_count = 0;
    g_orch_finalize_atomic_count = 0;
    g_orch_scope_end_atomic_count = 0;
    return d;
}
#endif
