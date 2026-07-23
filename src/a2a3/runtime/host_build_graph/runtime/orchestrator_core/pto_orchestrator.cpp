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
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "aicpu/dep_gen_collector_aicpu.h"
#include "common/dep_gen.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "pto_dep_compute.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"
#include "tensor.h"

#if SIMPLER_DFX
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/args_dump_aicpu.h"
#endif

// Verify the captured Tensor blob size in DepGenRecord matches the runtime
// Tensor layout. The platform header defines DEP_GEN_TENSOR_SIZE without
// including runtime/tensor.h, so this check lives at the orch callsite.
static_assert(sizeof(Tensor) == DEP_GEN_TENSOR_SIZE, "DepGenRecord::tensors slot size out of sync with sizeof(Tensor)");
// DEP_GEN_MAX_EXPLICIT_DEPS is a diagnostic-side capture cap only; the runtime
// imposes no hard cap on explicit dep count. If a submit exceeds this cap,
// dep_gen_aicpu_record_submit() logs and truncates — runtime correctness is
// unaffected, only the captured replay record is truncated.

// Weak fallbacks: dep_gen_collector_aicpu.cpp provides the strong symbols in
// AICPU builds. Host builds (host_build_graph runtime, future dep_gen replay)
// link these no-op stubs so the runtime translation unit is self-contained.
// Visibility is hidden so the HOST .so doesn't export them into the global
// dynamic symbol table where they'd shadow the AICPU .so's strong symbols
// (same pattern as get_sys_cnt_aicpu / l2_swimlane_aicpu_record_orch_phase below).
extern "C" __attribute__((weak, visibility("hidden"))) bool is_dep_gen_enabled() { return false; }
__attribute__((weak, visibility("hidden"))) void dep_gen_aicpu_record_submit(
    uint64_t, bool, bool, int, const void *const *, const uint8_t *, int, const uint64_t *, int, const int32_t[3]
) {}

// Scope_stats enable gate, queried via the same predicate idiom as
// is_dep_gen_enabled above. The AICPU collector links the strong definition;
// host builds fall back to this weak `false`. Gating here still skips the
// cross-agent occupancy reads that feed the sample when scope_stats is disabled.
extern "C" __attribute__((weak, visibility("hidden"))) bool is_scope_stats_enabled() { return false; }

// Heap-ring wrap report, called from the allocator (pto_ring_buffer.h) on each
// wrap. Strong definition lives in the AICPU collector; host builds fall back to
// this weak no-op so the runtime translation unit stays self-contained.
extern "C" __attribute__((weak, visibility("hidden"))) void scope_stats_note_heap_wrap(int) {}

// AICore register accessor (aicpu/platform_regs.h). The host orchestrator's
// route_ready_once path transitively ODR-uses the early-dispatch doorbell inline
// (pto_scheduler.h ring_one_doorbell), but no core is gated during host
// graph-build, so the doorbell never fires and this weak host fallback only
// satisfies the linker. The AICPU build links the strong definition from
// platform/.../platform_regs.cpp; hidden so the HOST .so does not shadow it.
__attribute__((weak, visibility("hidden"))) volatile uint32_t *get_reg_ptr(uint64_t, RegId) {
    static volatile uint32_t sink = 0;
    return &sink;
}

// =============================================================================
// Orchestrator Profiling (compile-time toggle)
// =============================================================================
#if SIMPLER_ORCH_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/l2_swimlane_collector_aicpu.h"
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
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() {
    // Host fallback: monotonic wall-clock in AICPU cycle units so the host-orch
    // deadlock/timeout backstops fire at their intended wall-clock (see the
    // detailed rationale on the same fallback in pto_runtime2.cpp).
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // Scale sec and nsec separately (divisor is the constant 1e9): avoids a
    // div-by-zero when PLATFORM_PROF_SYS_CNT_FREQ >= 1 GHz and the truncation
    // error a `1e9 / FREQ` divisor would introduce for non-dividing frequencies.
    return static_cast<uint64_t>(ts.tv_sec) * PLATFORM_PROF_SYS_CNT_FREQ +
           static_cast<uint64_t>(ts.tv_nsec) * PLATFORM_PROF_SYS_CNT_FREQ / 1000000000ull;
}
// Weak fallback for builds that don't link l2_swimlane_collector_aicpu.cpp.
// The strong symbol from the AICPU build wins when profiling is available.
// Also hidden to prevent HOST .so from polluting the global symbol table.
__attribute__((weak, visibility("hidden"))) void
l2_swimlane_aicpu_record_orch_phase(uint64_t, uint64_t, uint64_t, uint32_t) {}
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
uint64_t g_orch_scope_end_atomic_count = 0;
// Cycle accumulation is unconditional under SIMPLER_ORCH_PROFILING (that's what
// the flag is for) and feeds the per-sub-step `g_orch_*_cycle` cumulatives
// printed in the cold-path log.
//
// Per-submit ORCH_SUBMIT record is the only swim-lane emit on the orch
// path — one record per submit_task() / alloc_tensors() call spanning
// the entire [start, end] window. Per-sub-step phase records were dropped
// in favour of the cumulatives + per-submit envelope; the dispatcher
// already inserts one record at the end of each submit path via
// CYCLE_COUNT_ORCH_SUBMIT_RECORD.
#define CYCLE_COUNT_START()                                                        \
    bool _prof_active = (orch->l2_swimlane_level >= L2SwimlaneLevel::ORCH_PHASES); \
    uint64_t _t0 = get_sys_cnt_aicpu(), _t1;                                       \
    uint64_t _submit_start_ts = _t0
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)                                                       \
    do {                                                                                          \
        if (_prof_active) {                                                                       \
            l2_swimlane_aicpu_record_orch_phase(_submit_start_ts, _t1, (tid), g_orch_submit_idx); \
        }                                                                                         \
    } while (0)
#elif SIMPLER_DFX
#include "aicpu/device_time.h"
#include "aicpu/l2_swimlane_collector_aicpu.h"
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() {
    // Host fallback: monotonic wall-clock in AICPU cycle units so the host-orch
    // deadlock/timeout backstops fire at their intended wall-clock (see the
    // detailed rationale on the same fallback in pto_runtime2.cpp).
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // Scale sec and nsec separately (divisor is the constant 1e9): avoids a
    // div-by-zero when PLATFORM_PROF_SYS_CNT_FREQ >= 1 GHz and the truncation
    // error a `1e9 / FREQ` divisor would introduce for non-dividing frequencies.
    return static_cast<uint64_t>(ts.tv_sec) * PLATFORM_PROF_SYS_CNT_FREQ +
           static_cast<uint64_t>(ts.tv_nsec) * PLATFORM_PROF_SYS_CNT_FREQ / 1000000000ull;
}
__attribute__((weak, visibility("hidden"))) void
l2_swimlane_aicpu_record_orch_phase(uint64_t, uint64_t, uint64_t, uint32_t) {}
// submit_idx needed for swimlane task_id tagging (no cycle accumulation at this level)
static uint32_t g_orch_submit_idx = 0;
#define CYCLE_COUNT_START()                                                        \
    bool _prof_active = (orch->l2_swimlane_level >= L2SwimlaneLevel::ORCH_PHASES); \
    uint64_t _t0 = _prof_active ? get_sys_cnt_aicpu() : 0, _t1 = 0;                \
    uint64_t _submit_start_ts = _t0
#define CYCLE_COUNT_LAP(acc) \
    do {                     \
    } while (0)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)                                                       \
    do {                                                                                          \
        if (_prof_active) {                                                                       \
            _t1 = get_sys_cnt_aicpu();                                                            \
            l2_swimlane_aicpu_record_orch_phase(_submit_start_ts, _t1, (tid), g_orch_submit_idx); \
        }                                                                                         \
    } while (0)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)
#endif

static int32_t orch_mark_fatal(PTO2OrchestratorState *orch, int32_t error_code) {
    always_assert(orch != nullptr);
    orch->fatal = true;
    if (error_code == PTO2_ERROR_NONE || orch->sm_header == nullptr) {
        return PTO2_ERROR_NONE;
    }

    int32_t expected = PTO2_ERROR_NONE;
    std::atomic<int32_t> &orch_error_code = orch->sm_header->orch_error_code;
    if (orch_error_code.compare_exchange_strong(expected, error_code, std::memory_order_acq_rel)) {
        return error_code;
    }
    return expected;
}

static void
orch_report_fatal_v(PTO2OrchestratorState *orch, int32_t error_code, const char *func, const char *fmt, va_list args) {
    int32_t latched_code = orch_mark_fatal(orch, error_code);

#if SIMPLER_DFX
    // Flush the current scope's peaks BEFORE the FATAL log line, so the
    // diagnostic context (which pool/window filled up) appears right next to
    // the failure reason. on_fatal is latched, so duplicate fatals from
    // different layers don't print multiple stats lines.
    scope_stats_on_fatal();
#endif

    if (fmt == nullptr || fmt[0] == '\0') {
        if (latched_code != PTO2_ERROR_NONE && latched_code != error_code) {
            unified_log_error(func, "FATAL(code=%d, latched=%d)", error_code, latched_code);
        } else {
            unified_log_error(func, "FATAL(code=%d)", error_code);
        }
        return;
    }

    char message[1024];
    vsnprintf(message, sizeof(message), fmt, args);
    if (latched_code != PTO2_ERROR_NONE && latched_code != error_code) {
        unified_log_error(func, "FATAL(code=%d, latched=%d): %s", error_code, latched_code, message);
        return;
    }
    unified_log_error(func, "FATAL(code=%d): %s", error_code, message);
}

void PTO2OrchestratorState::report_fatal(int32_t error_code, const char *func, const char *fmt, ...) {
    auto *orch = this;
    va_list args;
    va_start(args, fmt);
    orch_report_fatal_v(orch, error_code, func, fmt, args);
    va_end(args);
}

static uint32_t next_fanin_seen_epoch(PTO2OrchestratorState *orch) {
    uint32_t next = orch->fanin_seen_current_epoch + 1;
    if (next == 0) {
        memset(
            orch->fanin_seen_epoch, 0, static_cast<size_t>(orch->sm_header->ring.task_window_size) * sizeof(uint32_t)
        );
        next = 1;
    }
    orch->fanin_seen_current_epoch = next;
    return next;
}

// Polling: fanin is a flat array of position-independent producer local ids on
// the payload (no dep-pool spill, no producer pointers). The builder writes them
// directly into payload->fanin_local_ids as producers are appended, deduping by
// slot and hard-capping at PTO2_MAX_FANIN. self_local is this task's own local id
// (the consumer), used to bump each producer's last_consumer_local_id (the
// reclaim gate the host wait_for_consumers polls via completed_watermark).
struct PTO2FaninBuilder {
    PTO2FaninBuilder(PTO2OrchestratorState *orch, PTO2TaskPayload *payload, int32_t self_local, uint32_t seen_epoch) :
        count(0),
        orch(orch),
        seen_epoch(seen_epoch),
        self_local(self_local),
        payload(payload) {}
    int32_t count{0};
    PTO2OrchestratorState *orch{nullptr};
    uint32_t seen_epoch{0};
    int32_t self_local{0};
    PTO2TaskPayload *payload{nullptr};

    bool mark_seen(uint8_t prod_ring, int32_t prod_slot) {
        if (prod_ring >= PTO2_MAX_RING_DEPTH || prod_slot < 0) {
            return false;
        }
        uint32_t *seen = orch->fanin_seen_epoch;
        uint32_t slot = static_cast<uint32_t>(prod_slot);
        if (seen[slot] == seen_epoch) {
            return true;
        }
        seen[slot] = seen_epoch;
        return false;
    }
};

static bool append_fanin_or_fail(
    PTO2OrchestratorState *orch, uint8_t prod_ring, int32_t prod_slot, PTO2TaskSlotState *prod_state,
    PTO2TaskId producer_task_id, PTO2FaninBuilder *fanin_builder
) {
    // Skip a stale/reused producer slot: the cached owner id no longer resolves
    // to this producer (defensive — whole-graph-resident hbg does not reuse slots
    // at build time). A COMPLETED producer IS a real fanin edge under polling (its
    // completion_flags byte is set), so it is not skipped.
    if (prod_state->task == nullptr || prod_state->task->task_id.local() != producer_task_id.local()) {
        return true;
    }
    // Dedup by (ring, slot). Single-ring hbg: prod_ring is always 0.
    if (fanin_builder->mark_seen(prod_ring, prod_slot)) {
        return true;
    }
    if (fanin_builder->count >= PTO2_MAX_FANIN) {
        orch_mark_fatal(orch, PTO2_ERROR_DEP_POOL_OVERFLOW);
        return false;
    }
    fanin_builder->payload->fanin_local_ids[fanin_builder->count++] = static_cast<int32_t>(producer_task_id.local());

    // Reclaim gate: record this task as a consumer of the producer. The producer
    // slot retires once the per-ring completed_watermark reaches this consumer id.
    if (fanin_builder->self_local > prod_state->last_consumer_local_id) {
        prod_state->last_consumer_local_id = fanin_builder->self_local;
    }
    return true;
}

static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state);

struct PTO2PreparedTask {
    PTO2TaskId task_id = PTO2TaskId::invalid();
    PTO2TaskAllocResult alloc_result = {-1, 0, nullptr, nullptr};
    PTO2TaskDescriptor *task = nullptr;
    PTO2TaskPayload *payload = nullptr;
    PTO2TaskSlotState *slot_state = nullptr;
};

static PTO2OutputLayout calculate_output_layout(const L0TaskArgs &args) {
    PTO2OutputLayout layout;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            continue;
        }
        layout.offsets[i] = layout.total_output_size;
        layout.buffer_sizes[i] =
            PTO2_ALIGN_UP(args.tensor(i).create_info().buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
        layout.total_output_size += layout.buffer_sizes[i];
    }
    return layout;
}

static bool check_scope_can_accept_task(PTO2OrchestratorState *orch, PTO2TaskAllocator &allocator, uint8_t ring_id) {
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
    LOG_ERROR("  host_build_graph is whole-graph-resident: the host builds the entire");
    LOG_ERROR("  scope before the device runs, so no slots reclaim during the build.");
    LOG_ERROR("  When scope task count >= window_size the ring overflows.");
    LOG_ERROR("Solution:");
    LOG_ERROR("  1. Reduce tasks per scope (use batching/unroll)");
    LOG_ERROR("  2. Increase task window (current: %d)", allocator.window_size());
    LOG_ERROR("     Compile-time: PTO2_TASK_WINDOW_SIZE in pto_runtime2_types.h");
    LOG_ERROR("     Runtime env:  PTO2_RING_TASK_WINDOW=<power-of-2>");
    LOG_ERROR("  3. Split work across multiple scopes");
    LOG_ERROR("========================================");
    orch_mark_fatal(orch, PTO2_ERROR_SCOPE_DEADLOCK);
    return false;
}

static bool prepare_task(
    PTO2OrchestratorState *orch, const L0TaskArgs &args, int32_t total_output_size, ActiveMask active_mask,
    TaskAttrs task_attrs, PTO2PreparedTask *out
) {
    uint8_t ring_id = 0;
    auto &allocator = orch->ring.task_allocator;

    if (!check_scope_can_accept_task(orch, allocator, ring_id)) {
        return false;
    }

    out->alloc_result = allocator.alloc(total_output_size);
    if (out->alloc_result.failed()) {
        orch_mark_fatal(orch, PTO2_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }

    out->task_id = PTO2TaskId::make(ring_id, static_cast<uint32_t>(out->alloc_result.task_id));
    out->slot_state = &orch->sm_header->ring.get_slot_state_by_slot(out->alloc_result.slot);
    out->task = &orch->sm_header->ring.task_descriptors[out->alloc_result.slot];
    out->payload = &orch->sm_header->ring.task_payloads[out->alloc_result.slot];

    out->payload->prefetch(args.tensor_count(), args.scalar_count());

    // Re-bind payload/task pointers each submit. Value is per-slot constant
    // (same as &task_payloads[slot] / &task_descriptors[slot]), but writing
    // here lets RingSchedState::init() skip the O(window_size) bind loop.
    // Both writes hit the same 64B slot_state cache line we're about to
    // dirty below, so the extra cost is two stores on an already-hot line.
    // Must precede the Orch-side wiring publish at the end of
    // submit_task_common — that publish is the first read of slot_state->task /
    // slot_state->payload by scheduler threads.
    out->slot_state->bind_buffers(out->payload, out->task);

    // prepare_task does NO payload writes: all payload content (tensors/scalars +
    // early-dispatch fields) is initialized in PTO2TaskPayload::init, the
    // single payload-init point, which runs before Orch-side wiring publish.

    // Fields already zeroed by reset_for_reuse() at slot init:
    //   fanout_lock=0, fanout_count=PTO2_FANOUT_SCOPE_BIT, fanout_head=nullptr,
    //   fanin_refcount=0, fanout_refcount=0, completed_subtasks=0, next_block_idx=0
    // Fields immutable after RingSchedState::init():
    //   ring_id
    // task_state is set to PENDING here as the orchestrator populates the slot
    // (host_build_graph does not recycle slots at runtime, so there is no
    // post-CONSUMED reset path).
    out->slot_state->task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    int16_t block_num = args.launch_spec.block_num();
    out->slot_state->total_required_subtasks =
        static_cast<int16_t>(block_num * __builtin_popcount(active_mask.core_mask()));
    out->slot_state->logical_block_num = block_num;
    out->slot_state->active_mask = active_mask;
    out->slot_state->task_attrs = task_attrs;
    // Reclaim gate: seed last_consumer to self, so a producer with no consumers
    // is retirable once completed_watermark >= its own id. Each fanin edge bumps
    // it in append_fanin_or_fail. completion_flags for this slot are already 0
    // (zeroed once at init; whole-graph-resident hbg never reuses a slot).
    out->slot_state->last_consumer_local_id = static_cast<int32_t>(out->task_id.local());
    // payload.fanin_count is set in submit_task_common's STEP 6.
    scope_tasks_push(orch, out->slot_state);

    return true;
}

// =============================================================================
// Scope Management
// =============================================================================

static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state) {
    if (orch->scope_tasks_size >= orch->scope_tasks_capacity) {
        // scope_tasks lives in the per-Worker arena (single backing allocation),
        // so realloc is not legal. Capacity is the total in-flight slot budget
        // (the runtime task window; see reserve_layout) — hitting it means the
        // ring is saturated, so no further push could succeed regardless of
        // buffer growth.
        orch->report_fatal(
            PTO2_ERROR_SCOPE_TASKS_OVERFLOW, __FUNCTION__,
            "scope_tasks buffer saturated at %d entries (all rings full)", orch->scope_tasks_capacity
        );
        return;
    }
    orch->scope_tasks[orch->scope_tasks_size++] = task_slot_state;
}

void PTO2OrchestratorState::begin_scope(PTO2ScopeMode mode) {
    auto *orch = this;
    if (orch->fatal) {
        return;
    }
    assert(orch->scope_stack_top < static_cast<int32_t>(orch->scope_stack_capacity - 1) && "Scope stack overflow");
    if (mode == PTO2ScopeMode::AUTO && orch->in_manual_scope()) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "auto scope nested inside manual scope is not supported");
        return;
    }

    bool already_in_manual_scope = orch->in_manual_scope();
    ++orch->scope_stack_top;
    orch->scope_begins[orch->scope_stack_top] = orch->scope_tasks_size;
    if (mode == PTO2ScopeMode::MANUAL && !already_in_manual_scope) {
        orch->manual_begin_depth = orch->scope_stack_top;
    }
#if SIMPLER_DFX
    // Gate via is_scope_stats_enabled() (weak-false in host builds) BEFORE the
    // collector call: when disabled we pay nothing. Sample the current ring's
    // task/heap start-end and tensormap usage at the scope boundary.
    if (is_scope_stats_enabled()) {
        uint8_t ring_id = 0;
        auto &alloc = orch->ring.task_allocator;
        // Polling: no dep_pool to report (readiness is via completion_flags).
        int32_t dep_pool_tail = 0;
        int32_t dep_pool_top = 0;
        scope_stats_begin(
            ring_id, alloc.task_tail(), alloc.task_head(), alloc.heap_tail(), alloc.heap_top(), dep_pool_tail,
            dep_pool_top, orch->tensor_map.current_used()
        );
    }
#endif
}

void PTO2OrchestratorState::end_scope() {
    auto *orch = this;
    if (orch->fatal) {
        return;
    }
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

    // Snapshot the ring start/end BEFORE the orchestrator drains pending tasks
    // via scheduler->on_scope_end, so the end record reflects the scope's
    // occupancy at close, not the residual after teardown.
#if SIMPLER_DFX
    // Gate via is_scope_stats_enabled() (see begin_scope). One collector call
    // emits the end-boundary record and tears down bookkeeping.
    if (is_scope_stats_enabled()) {
        uint8_t ring_id = 0;
        auto &alloc = orch->ring.task_allocator;
        // Polling: no dep_pool to report (readiness is via completion_flags).
        int32_t dep_pool_tail = 0;
        int32_t dep_pool_top = 0;
        scope_stats_end(
            ring_id, alloc.task_tail(), alloc.task_head(), alloc.heap_tail(), alloc.heap_top(), dep_pool_tail,
            dep_pool_top, orch->tensor_map.current_used()
        );
    }
#endif

#if SIMPLER_ORCH_PROFILING
    uint64_t _se0 = get_sys_cnt_aicpu();
#endif

    bool ending_manual_scope = orch->scope_stack_top == orch->manual_begin_depth;
    int32_t begin = orch->scope_begins[orch->scope_stack_top--];
    int32_t count = orch->scope_tasks_size - begin;
    if (ending_manual_scope) {
        orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
    }

    if (orch->scheduler && count > 0) {
        orch->scheduler->on_scope_end(&orch->scope_tasks[begin], count);
    }

    // Rewind the task buffer — these entries are no longer needed
    orch->scope_tasks_size = begin;

#if SIMPLER_ORCH_PROFILING
    uint64_t _se1 = get_sys_cnt_aicpu();
    g_orch_scope_end_cycle += (_se1 - _se0);
#endif
}

// =============================================================================
// Task Submission
// =============================================================================

// Ensure the tensormap entry pool has room for `needed` inserts before STEP 4
// registers this task's outputs. The pool is watermark-reclaimed like the
// task/heap/fanin pools — retired tasks' entries free once last_task_alive
// advances — so an exhausted pool is back-pressure, not a hard error. Reclaim
// against the single ring's watermark; if still short,
// spin until reclaim actually frees entries, with the same 500 ms wall-clock
// backstop as the task allocator and fanin spill pool. A pool that stays full
// (no entry freed) is a genuine deadlock: latch PTO2_ERROR_TENSORMAP_OVERFLOW
// and bail. Returns false on deadlock or on a fatal already latched by another
// party. Cold path — the fast path returns immediately when the pool has room.
static bool ensure_tensormap_capacity(PTO2OrchestratorState *orch, int32_t needed) {
    PTO2TensorMap &tm = orch->tensor_map;
    if (tm.free_entries() >= needed) {
        return true;
    }

    int32_t alive;
    auto read_alive = [&]() {
        // Relaxed: a self-correcting poll re-read every reclaim tick, so a stale
        // watermark only defers reclaim one tick and never over-frees.
        alive = orch->sm_header->ring.fc.last_task_alive.load(std::memory_order_relaxed);
    };

    read_alive();
    int64_t cur_alive_sum = tm.reclaim_retired_all(alive);  // kept for the deadlock diagnostic
    int32_t prev_free = tm.free_entries();
    if (prev_free >= needed) {
        return true;
    }

    int spin_count = 0;
    uint64_t block_cycle0 = 0;  // wall-clock anchor for the deadlock backstop
    bool block_timing = false;  // false until the first no-reclaim-progress tick
    while (tm.free_entries() < needed) {
        spin_count++;

        // Reclaim (and the all-ring watermark reads it needs) is the costly part of
        // this spin and the only path that frees entries; gate it to a periodic tick.
        // Cold path, but the spin itself is tight.
        if ((spin_count & 31) == 0) {
            read_alive();
            cur_alive_sum = tm.reclaim_retired_all(alive);
            int32_t cur_free = tm.free_entries();
            if (cur_free >= needed) {
                return true;
            }
            // Progress is entries actually freed, NOT watermark movement: a ring can
            // retire zero-output tasks (count_registrable_outputs == 0), advancing
            // last_task_alive without freeing any entry. Gating the backstop on
            // free_entries() keeps a wedged pool from dodging the timeout while some
            // unrelated ring keeps draining.
            if (cur_free > prev_free) {
                spin_count = 0;
                prev_free = cur_free;
                block_timing = false;
            }
        }

        if ((spin_count & 1023) == 0) {
            // A fatal latched elsewhere breaks this otherwise-unbounded spin.
            if (orch->sm_header->orch_error_code.load(std::memory_order_acquire) != PTO2_ERROR_NONE) {
                return false;
            }
            // Absolute-time backstop, matching the task allocator: stable across
            // chips/contention, unlike a fixed spin count. get_sys_cnt_aicpu()
            // is an MMIO read, so sample it only once per 1024 spins.
            uint64_t now = get_sys_cnt_aicpu();
            if (!block_timing) {
                block_cycle0 = now;
                block_timing = true;
            } else if (now - block_cycle0 >= PTO2_ALLOC_DEADLOCK_TIMEOUT_CYCLES) {
                LOG_ERROR("========================================");
                LOG_ERROR("FATAL: TensorMap Entry Pool Deadlock Detected!");
                LOG_ERROR("========================================");
                LOG_ERROR("TensorMap entry pool freed no entries for ~500 ms while a task waits.");
                LOG_ERROR("  - Pool used:   %d / %d", tm.current_used(), tm.pool_capacity());
                LOG_ERROR("  - Needed:      %d entries", needed);
                LOG_ERROR("  - last_task_alive: %" PRId64, cur_alive_sum);
                LOG_ERROR("Diagnosis:");
                LOG_ERROR("  No retiring task is freeing tensormap entries (last_task_alive may");
                LOG_ERROR("  still move on rings with no registered outputs). Check TaskRing");
                LOG_ERROR("  diagnostics for the stalled producer.");
                LOG_ERROR("Solution:");
                LOG_ERROR("  Increase PTO2_TENSORMAP_POOL_SIZE (current: %d).", tm.pool_capacity());
                LOG_ERROR("========================================");
                orch_mark_fatal(orch, PTO2_ERROR_TENSORMAP_OVERFLOW);
                return false;
            }
        }
        SPIN_WAIT_HINT();
    }
    return true;
}

// Shared body for submit_task / submit_dummy_task. Caller has already validated
// args.has_error, decided active_mask (empty for dummy), and resolved the per-slot
// kernel_ids (all INVALID_KERNEL_ID for dummy). Performs tensormap sync, fanin
// computation (explicit_deps + auto), output registration, slot init, and
// Orch-side wiring/ready publication.
static TaskOutputTensors submit_task_common(
    PTO2OrchestratorState *orch, const L0TaskArgs &args, ActiveMask active_mask, TaskAttrs task_attrs,
    int32_t aic_kernel_id, int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    CYCLE_COUNT_START();
    TaskOutputTensors result;
    PTO2OutputLayout layout = calculate_output_layout(args);
    PTO2PreparedTask prepared;
    if (!prepare_task(orch, args, layout.total_output_size, active_mask, task_attrs, &prepared)) {
        return result;
    }
    PTO2SchedulerState *sched = orch->scheduler;
    PTO2RingFlowControl &fc = orch->sm_header->ring.fc;
    PTO2TaskId task_id = prepared.task_id;
    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;
    result.set_task_id(task_id);

    // dep_gen capture point: snapshot the orch submit_task inputs while the
    // tensormap is still in its pre-lookup state for this task. Replay reads
    // these records offline to reconstruct the complete dep graph — the sole
    // source of truth for fanout now that the swimlane hot path no longer
    // records it.
    if (is_dep_gen_enabled()) {
        const void *tensor_ptrs[MAX_TENSOR_ARGS];
        // TensorArgType is `enum class : int32_t` (4 bytes); the on-disk record
        // packs arg_types as uint8_t[16] (5-value enum fits in a byte). Narrow
        // each tag here rather than letting the AICPU writer reinterpret a
        // 4×-wider array as bytes — that path silently lost two of every three
        // tags on little-endian and synthesized phantom self-edges in replay.
        uint8_t arg_types_u8[MAX_TENSOR_ARGS];
        // Clamp to MAX_TENSOR_ARGS even though the Arg builder caps adds at
        // MAX_TENSOR_ARGS: defensive against any future builder bypass /
        // shared-memory bit-flip that could otherwise overrun the two
        // MAX_TENSOR_ARGS-sized stack buffers above.
        const int tc_raw = args.tensor_count();
        const int tc = tc_raw > MAX_TENSOR_ARGS ? MAX_TENSOR_ARGS : tc_raw;
        for (int i = 0; i < tc; i++) {
            // OUTPUT slots carry create_info (not yet a Tensor); skip them —
            // they have no producer to look up and replay's per-tensor loop
            // also skips OUTPUT.
            tensor_ptrs[i] = (args.tag(i) == TensorArgType::OUTPUT) ? nullptr : &args.tensor(i).ref();
            arg_types_u8[i] = static_cast<uint8_t>(args.tag(i));
        }
        const int32_t kernel_ids_capture[3] = {aic_kernel_id, aiv0_kernel_id, aiv1_kernel_id};
        dep_gen_aicpu_record_submit(
            task_id.raw, orch->in_manual_scope(), args.allow_early_resolve(), tc, tensor_ptrs, arg_types_u8,
            static_cast<int>(args.explicit_dep_count()), reinterpret_cast<const uint64_t *>(args.explicit_deps_data()),
            args.launch_spec.block_num(), kernel_ids_capture
        );
    }

    PTO2FaninBuilder fanin_builder(orch, &payload, static_cast<int32_t>(task_id.local()), next_fanin_seen_epoch(orch));

    CYCLE_COUNT_LAP(g_orch_alloc_cycle);

#if SIMPLER_DFX
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    // === STEP 2: Sync TensorMap validity and optional cleanup ===
    // Read current last_task_alive from shared memory for this ring
    int32_t sm_last_task_alive = fc.last_task_alive.load(std::memory_order_acquire);

    orch->tensor_map.sync_tensormap(task_id, sm_last_task_alive);

    CYCLE_COUNT_LAP(g_orch_sync_cycle);

    for (uint32_t i = 0; i < args.explicit_dep_count(); i++) {
        PTO2TaskId dep_task_id = args.explicit_dep(i);
        if (!dep_task_id.is_valid()) {
            orch->report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "Arg.set_dependencies(...) requires valid task ids"
            );
            return result;
        }
        uint8_t dep_ring_id = dep_task_id.ring();
        PTO2SharedMemoryRingHeader &dep_ring = orch->sm_header->ring;
        int32_t dep_local_task_id = static_cast<int32_t>(dep_task_id.local());
        int32_t dep_last_task_alive = dep_ring.fc.last_task_alive.load(std::memory_order_acquire);
        if (dep_local_task_id < dep_last_task_alive) {
            continue;
        }
        int32_t dep_slot = dep_ring.get_slot_by_task_id(dep_local_task_id);
        PTO2TaskSlotState *producer_slot_state = &dep_ring.get_slot_state_by_slot(dep_slot);
        if (!append_fanin_or_fail(orch, dep_ring_id, dep_slot, producer_slot_state, dep_task_id, &fanin_builder)) {
            return result;
        }
    }

    // === STEP 3: Lookup inputs (creator retention + tensormap modifier lookup) ===
    DepInputs dep_inputs{
        args.tensor_count(),       args.tensor_data(), args.tag_data(), static_cast<int32_t>(args.explicit_dep_count()),
        args.explicit_deps_data(),
    };

    auto runtime_emit = [&](PTO2TaskId producer_task_id) -> bool {
        uint8_t prod_ring = producer_task_id.ring();
        PTO2SharedMemoryRingHeader &producer_ring = orch->sm_header->ring;
        int32_t prod_slot = producer_ring.get_slot_by_task_id(static_cast<int32_t>(producer_task_id.local()));
        PTO2TaskSlotState *prod_state = &producer_ring.get_slot_state_by_slot(prod_slot);
        return append_fanin_or_fail(orch, prod_ring, prod_slot, prod_state, producer_task_id, &fanin_builder);
    };

    if (!compute_task_fanin(dep_inputs, orch->tensor_map, orch->in_manual_scope(), runtime_emit)) {
        return result;
    }

    CYCLE_COUNT_LAP(g_orch_lookup_cycle);

    // === STEP 4: Register outputs/inouts in TensorMap (must be separate from lookup) ===
    // Reserve pool capacity for this task's inserts before registering. The pool
    // is reclaimed as last_task_alive advances; an
    // exhausted pool back-pressures here (and detects a wedged watermark) rather
    // than tripping new_entry()'s hard assert mid-registration.
    int32_t tensormap_needed = count_registrable_outputs(dep_inputs, orch->in_manual_scope());
    if (tensormap_needed > 0 && !ensure_tensormap_capacity(orch, tensormap_needed)) {
        return result;
    }
    register_task_outputs(dep_inputs, task_id, orch->tensor_map, orch->in_manual_scope());

    CYCLE_COUNT_LAP(g_orch_insert_cycle);

    // === STEP 5: Batch-write to GM (single cache line burst) ===
    // Deferred from allocation phase to avoid scattered GM writes that get
    // evicted by TensorMap lookup/insert cache pressure.
    __builtin_prefetch(&task, 1, 1);
    task.task_id = task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = aic_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = aiv0_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = aiv1_kernel_id;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    // append_fanin_or_fail wrote each producer's local id straight into
    // payload.fanin_local_ids and bumped its last_consumer_local_id; the count is
    // published in STEP 6 below. payload.init does not touch the fanin region.
    payload.init(args, result, prepared.alloc_result, layout);

    // Dispatch predicate: resolve the (tensor, indices) to an absolute GM address
    // now so the scheduler can read it at the dispatch point with a single load,
    // no Arg/Tensor access. Both branches write predicate.op explicitly because
    // payload slots are ring-reused; op == NONE means "always dispatch".
    {
        const L0TaskPredicate &pred = args.predicate();
        if (pred.op != PredicateOp::NONE && pred.operand.tensor != nullptr && pred.operand.tensor->buffer.addr != 0) {
            uint64_t elem_size = get_element_size(pred.operand.tensor->dtype);
            uint64_t flat_offset = pred.operand.tensor->compute_flat_offset(pred.operand.indices, pred.operand.ndims);
            payload.predicate.addr = pred.operand.tensor->buffer.addr + flat_offset * elem_size;
            payload.predicate.target = pred.target;
            payload.predicate.elem_size = static_cast<uint8_t>(elem_size);
            payload.predicate.op = pred.op;
        } else {
            payload.predicate.addr = 0;
            payload.predicate.op = PredicateOp::NONE;
        }
    }
#if SIMPLER_DFX
    if (is_dump_args_enabled()) {
        if (args.scalar_count() > 0) {
            set_dump_args_task_scalar_dtypes(
                task_id.raw, static_cast<uint32_t>(args.scalar_count()), args.scalar_dtypes()
            );
        }
        // Selective vs full dump is latched at dump_args_init from DumpDataHeader
        // (host-decided before any dispatch), so it is race-free regardless of
        // submission order. Here we only record each marked task's arg mask and
        // metadata flags, which selective collection consults.
        if (args.dump_arg_mask() != 0) {
            set_dump_args_task_mask(task_id.raw, args.dump_arg_mask(), args.dump_arg_index_ambiguous_mask());
        }
    }
#endif

    CYCLE_COUNT_LAP(g_orch_args_cycle);

    // === STEP 6: publish the inline fanin count (device boot classifies) ===
    // Polling + host-orch: append_fanin_or_fail already wrote each producer's
    // local id into payload.fanin_local_ids and bumped its last_consumer_local_id.
    // All that remains is to record how many. There is NO fanout adjacency, NO
    // dep_pool, and NO ready routing here — the device boot scan classifies every
    // task exactly once (fanin_satisfied -> push_ready_routed, else register_wake)
    // before the scheduler dispatch loop starts. Because fanin is now a flat array
    // of position-independent integers, none of this needs host->device pointer
    // relocation.
    payload.fanin_count = fanin_builder.count;
    (void)sched;

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    CYCLE_COUNT_ORCH_SUBMIT_RECORD(task_id.raw);

#if SIMPLER_DFX
    orch->tasks_submitted++;
#if SIMPLER_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif
    return result;
}

TaskOutputTensors PTO2OrchestratorState::submit_task(const MixedKernels &mixed_kernels, const L0TaskArgs &args) {
    auto *orch = this;

    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) {
        return TaskOutputTensors{};
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
        orch_mark_fatal(orch, PTO2_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);
    // === Validate submit inputs ===
    ActiveMask active_mask = mixed_kernels.to_active_mask();
    always_assert(static_cast<bool>(active_mask) && "MixedKernels must have at least one active slot");

    int16_t block_num = args.launch_spec.block_num();
    always_assert(block_num >= 1 && "block_num must be >= 1");

    // Normalize single-AIV tasks: if only aiv1 is set (no aic, no aiv0), move
    // it to the aiv0 slot.  This guarantees the dispatch path can always use
    // PTO2SubtaskSlot::AIV0 for single-AIV shapes without inspecting active_mask.
    // Mixed tasks (AIC+AIV) keep their original AIV identity so the correct
    // hardware channel (AIV0→AIC vs AIV1→AIC) is used at dispatch time.
    MixedKernels normalized = mixed_kernels;
    bool has_aic = active_mask.has_mask(PTO2_SUBTASK_MASK_AIC);
    bool has_aiv0 = active_mask.has_mask(PTO2_SUBTASK_MASK_AIV0);
    bool has_aiv1 = active_mask.has_mask(PTO2_SUBTASK_MASK_AIV1);
    if (!has_aic && has_aiv1 && !has_aiv0) {
        normalized.aiv0_kernel_id = normalized.aiv1_kernel_id;
        normalized.aiv1_kernel_id = INVALID_KERNEL_ID;
        active_mask = normalized.to_active_mask();
    }

    TaskAttrs task_attrs;
    task_attrs.set_early_resolve(args.allow_early_resolve());
    task_attrs.set_timing_slot(args.task_timing_slot());

    // sync_start is only meaningful for tasks with block_num > 1.
    if (block_num > 1 && args.launch_spec.require_sync_start()) {
        // Deadlock check: block_num >= total available slots of the required type.
        // For MIX/AIC: limit is total_cluster_count (one AIC per cluster).
        // For AIV:     limit is total_aiv_count.
        PTO2ResourceShape shape = active_mask.to_shape();
        int32_t limit = (shape == PTO2ResourceShape::AIV) ? orch->total_aiv_count : orch->total_cluster_count;
        if (limit > 0 && block_num > limit) {
            report_fatal(
                PTO2_ERROR_REQUIRE_SYNC_START_INVALID, __FUNCTION__,
                "require_sync_start block_num=%d > limit=%d (deadlock guaranteed)", block_num, limit
            );
            return TaskOutputTensors{};
        }
        task_attrs.set_sync_start();
    }

    if (args.predicate().op != PredicateOp::NONE) {
        task_attrs.set_predicate();
    }

    return submit_task_common(
        orch, args, active_mask, task_attrs, normalized.aic_kernel_id, normalized.aiv0_kernel_id,
        normalized.aiv1_kernel_id
    );
}

// Submit a dependency-only task: full dependency graph participation
// (tensormap lookup/insert, explicit_deps, manual_dep, manual_scope) but no
// AICore dispatch. Empty active_mask routes the slot to the DUMMY ready
// bucket; dispatch loop short-circuits to completion. Accepts the same Arg
// shape as submit_task; scalars are permitted but never consumed.
TaskOutputTensors PTO2OrchestratorState::submit_dummy_task(const L0TaskArgs &args) {
    auto *orch = this;

    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    if (args.has_error) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg in submit_dummy_task!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg ? args.error_msg : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("========================================");
        orch_mark_fatal(orch, PTO2_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);

    // Dummy tasks never dispatch to an AICore, so sync_start / has_predicate do
    // not apply; only the early-dispatch hint and timing tag carry over.
    TaskAttrs task_attrs;
    task_attrs.set_early_resolve(args.allow_early_resolve());
    task_attrs.set_timing_slot(args.task_timing_slot());

    return submit_task_common(
        orch, args, ActiveMask{}, task_attrs, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID
    );
}

TaskOutputTensors PTO2OrchestratorState::alloc_tensors(const L0TaskArgs &args) {
    auto *orch = this;
    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    if (args.tensor_count() <= 0) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors requires at least one TensorCreateInfo");
        return TaskOutputTensors{};
    }
    if (args.scalar_count() != 0) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args");
        return TaskOutputTensors{};
    }
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args"
            );
            return TaskOutputTensors{};
        }
    }

    CYCLE_COUNT_START();

    if (args.has_error) {
        report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }

    PTO2OutputLayout layout = calculate_output_layout(args);
    PTO2PreparedTask prepared;
    // Kernel-less alloc task: no active subtasks, no dispatch-time attributes. The
    // early-dispatch hint is force-set below (see the flag-the-creator note).
    if (!prepare_task(orch, args, layout.total_output_size, ActiveMask{}, TaskAttrs{}, &prepared)) {
        return TaskOutputTensors{};
    }

    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;

    CYCLE_COUNT_LAP(g_orch_alloc_cycle);

#if SIMPLER_DFX
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    task.task_id = prepared.task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = INVALID_KERNEL_ID;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    TaskOutputTensors outputs;
    outputs.set_task_id(prepared.task_id);
    payload.init(args, outputs, prepared.alloc_result, layout);
    payload.fanin_count = 0;  // hidden-alloc tasks have no producer dependencies
    CYCLE_COUNT_LAP(g_orch_args_cycle);

    if (prepared.slot_state != nullptr) {
        // Hidden alloc tasks complete inline in the orchestrator before any
        // consumer can exist, so they have no fanout to notify and no worker
        // subtasks to retire. Running the full on_task_complete path
        // would only pay unnecessary fanout_lock / traversal overhead here.
        // The generic slot initialization done in prepare_task() is still
        // required so scope_end can release the producer-side reference and
        // drive the slot to CONSUMED, but worker dispatch fields are never
        // observed for hidden alloc tasks.
        //
        // Flag the creator so it does NOT suppress its consumers' early-dispatch.
        // Under the direct-only model an unflagged producer disqualifies its
        // consumer, and a pre-completed producer only seeds dispatch_fanin when
        // flagged. A buffer allocation is pure memory whose output is ready at
        // creation — it should always be transparent, never a barrier. Unlike a
        // codegen task there is no Arg-driven hint to honor here, so mark it
        // unconditionally.
        prepared.slot_state->task_attrs.set_early_resolve(true);
        prepared.slot_state->mark_completed();  // host-visible task_state mirror
        // Polling: pre-set the device-visible completion_flags byte in the H2D
        // image. Consumers poll completion_flags (not task_state), so a hidden-alloc
        // producer completed here on the host must publish its flag too — otherwise
        // every consumer register_wakes on a producer that never runs on device and
        // the run hangs. (The device watermark walk transparently steps past this
        // pre-set flag when a later on-device task completes.)
        PTO2SharedMemoryRingHeader &done_ring = orch->sm_header->ring;
        int32_t done_local = static_cast<int32_t>(prepared.task_id.local());
        done_ring.completion_flags[done_local & done_ring.task_window_mask].store(1, std::memory_order_release);
    }
    orch->inline_completed_tasks++;

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    CYCLE_COUNT_ORCH_SUBMIT_RECORD(prepared.task_id.raw);

#if SIMPLER_DFX
    orch->tasks_submitted++;
#if SIMPLER_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif

    return outputs;
}

// =============================================================================
// Flow Control
// =============================================================================

void PTO2OrchestratorState::mark_done() {
    auto *orch = this;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        int32_t total_tasks = orch->ring.task_allocator.active_count();
        if (total_tasks > 0) {
            LOG_DEBUG("=== [Orchestrator] ring %d: total_tasks=%d ===", r, total_tasks);
        }
    }
    orch->sm_header->orchestrator_done.store(1, std::memory_order_release);
    orch->scope_tasks_size = 0;
    orch->scope_stack_top = -1;
    orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
#if !SIMPLER_ORCH_PROFILING && SIMPLER_DFX
    g_orch_submit_idx = 0;
#endif
}

#if SIMPLER_ORCH_PROFILING
PTO2OrchProfilingData orchestrator_get_profiling() {
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
    g_orch_scope_end_atomic_count = 0;
    return d;
}
#endif
