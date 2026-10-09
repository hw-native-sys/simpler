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
 * host_build_graph orchestrator implementation
 *
 * Implements orchestrator state management, scope handling, and task submission.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "host_build_graph/orchestrator.h"

#include <assert.h>
#include <inttypes.h>
#include <limits>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <limits>
#include <new>
#include <utility>

#include "assert_compat.h"
#include "common/host_phase_kind.h"
#include "common/unified_log.h"
#include "host_build_graph/dep_gen_host_graph.h"
#include "host_build_graph/dep_compute.h"
#include "host_build_graph/graph_recording.h"
#include "host_build_graph/graph_submit.h"
#include "host_build_graph/host_phase_trace.h"
#include "host_build_graph/orch_profiling.h"
#include "host_build_graph/orchestrator_internal.h"
#include "host_build_graph/task_id.h"
#include "host_build_graph/runtime_status.h"
#include "host_build_graph/runtime_types.h"
#include "host_build_graph/shared_memory.h"
#include "host_build_graph/tensormap.h"
#include "host_build_graph/types.h"
#include "tensor.h"

// A report that names no code still means "fatal", so it is latched -- and logged --
// under the explicit-fatal code rather than as a zero that reads like "no error".
static constexpr int32_t normalized_fatal_code(int32_t error_code) {
    return error_code == SIMPLER_ERROR_NONE ? SIMPLER_ERROR_EXPLICIT_ORCH_FATAL : error_code;
}

// First-writer-wins, so the latched code names the failure that started the
// cascade. The CAS is load-bearing rather than decorative: a recording worker and
// the bind thread both reach here (see OrchestratorState::fatal_code).
// TaskAllocator::report_capacity_exhausted writes the same field under the same rule.
//
// `latched_out` receives the code that owns the field after this call, whoever put it
// there. The return value is whether *this* call put it there, and it is the only
// answer to that question: a load before the exchange can be overtaken, and the
// latched code cannot stand in for ownership because two reporters may carry the same
// code. A caller that must know whether the run's failure is its own takes the bool.
static bool orch_mark_fatal_owned(OrchestratorState *orch, int32_t error_code, int32_t *latched_out) {
    always_assert(orch != nullptr);
    const int32_t code = normalized_fatal_code(error_code);
    int32_t expected = SIMPLER_ERROR_NONE;
    const bool owned = orch->fatal_code.compare_exchange_strong(expected, code, std::memory_order_acq_rel);
    // A failed exchange loads the winner's code into `expected`.
    if (latched_out != nullptr) *latched_out = owned ? code : expected;
    return owned;
}

namespace simpler::hbg {
int32_t orch_mark_fatal(OrchestratorState *orch, int32_t error_code) {
    int32_t latched = SIMPLER_ERROR_NONE;
    (void)orch_mark_fatal_owned(orch, error_code, &latched);
    return latched;
}
}  // namespace simpler::hbg

// @return whether this report latched the field, i.e. whether this failure is the one
//         the run will be judged by.
static bool
orch_report_fatal_v(OrchestratorState *orch, int32_t error_code, const char *func, const char *fmt, va_list args) {
    const int32_t reported = normalized_fatal_code(error_code);
    // Differs from `reported` only when an earlier fatal already owns the field.
    int32_t latched_code = reported;
    const bool owned = orch_mark_fatal_owned(orch, reported, &latched_code);

    if (fmt == nullptr || fmt[0] == '\0') {
        if (latched_code != reported) {
            unified_log_error(func, "FATAL(code=%d, latched=%d)", reported, latched_code);
        } else {
            unified_log_error(func, "FATAL(code=%d)", reported);
        }
        return owned;
    }

    std::array<char, 1024> message{};
    vsnprintf(message.data(), message.size(), fmt, args);
    if (latched_code != reported) {
        unified_log_error(func, "FATAL(code=%d, latched=%d): %s", reported, latched_code, message.data());
        return owned;
    }
    unified_log_error(func, "FATAL(code=%d): %s", reported, message.data());
    return owned;
}

void OrchestratorState::report_fatal(int32_t error_code, const char *func, const char *fmt, ...) {
    auto *orch = this;
    va_list args;
    va_start(args, fmt);
    (void)orch_report_fatal_v(orch, error_code, func, fmt, args);
    va_end(args);
}

bool OrchestratorState::report_fatal_owned(int32_t error_code, const char *func, const char *fmt, ...) {
    auto *orch = this;
    va_list args;
    va_start(args, fmt);
    const bool owned = orch_report_fatal_v(orch, error_code, func, fmt, args);
    va_end(args);
    return owned;
}

bool OrchestratorState::init(void *sm_base, void *gm_heap, uint64_t heap_size, uint64_t max_tasks) {
    // Reset in place rather than by move-assignment: fatal_code is a std::atomic,
    // which is neither copy- nor move-assignable, and a re-init has to clear every
    // field the previous pass left behind (the pool cursors below rely on it).
    this->~OrchestratorState();
    auto *orch = new (static_cast<void *>(this)) OrchestratorState{};

    always_assert(max_tasks > 0);

    orch->sm_header = reinterpret_cast<SharedMemoryHeader *>(sm_base);

    orch->task_allocator.init(static_cast<int32_t>(max_tasks), gm_heap, heap_size, &orch->fatal_code);

    // The mirror's argument pools. Offset arithmetic on the same base as sm_header,
    // so it holds for whichever SM this orchestrator was pointed at. The cursors
    // reset with the rest of the state above.
    auto *sm_bytes = static_cast<char *>(sm_base);
    const auto pools = sm_layout::segment_offsets(sm_layout::mirror_extents(max_tasks));
    orch->fanin_pool = reinterpret_cast<int32_t *>(sm_bytes + pools.fanin_pool);
    orch->tensor_pool = reinterpret_cast<simpler::hbg::Tensor *>(sm_bytes + pools.tensor_pool);
    orch->scalar_pool = reinterpret_cast<uint64_t *>(sm_bytes + pools.scalar_pool);

    // Polling: no fanin-spill pool — producer ids are inline on the payload.
    const auto slots = static_cast<size_t>(max_tasks);
    orch->fanin_seen_epoch.reset(new (std::nothrow) uint32_t[slots]);
    if (orch->fanin_seen_epoch == nullptr) {
        LOG_ERROR("Orchestrator scratch allocation failed (max_tasks=%" PRIu64 ")", max_tasks);
        return false;
    }
    memset(orch->fanin_seen_epoch.get(), 0, slots * sizeof(uint32_t));

    if (!orch->tensor_map.init_default(static_cast<int32_t>(max_tasks))) {
        return false;
    }

    orch->scope_stack_top = -1;
    orch->manual_begin_depth = CHIP_MAX_SCOPE_DEPTH;

    return true;
}

// Advances the epoch fanin_mark_seen keys against, so a mark left by an earlier task
// never reads as a repeat. The table is cleared only on wraparound.
namespace simpler::hbg {
void next_fanin_seen_epoch(OrchestratorState *orch) {
    uint32_t next = orch->fanin_seen_current_epoch + 1;
    if (next == 0) {
        memset(
            orch->fanin_seen_epoch.get(), 0, static_cast<size_t>(orch->task_allocator.capacity()) * sizeof(uint32_t)
        );
        next = 1;
    }
    orch->fanin_seen_current_epoch = next;
}
}  // namespace simpler::hbg

// True when this producer was already appended under the current epoch. A negative
// local id cannot index the epoch table, so it is reported as not-yet-seen and the
// caller appends it rather than rejecting it; append_fanin_or_fail has already
// established that the producer is a GLOBAL task whose local id is a valid table
// entry.
static bool fanin_mark_seen(OrchestratorState &orch, TaskId producer_task_id) {
    const int32_t prod_local = producer_task_id.local_id();
    if (prod_local < 0) {
        return false;
    }
    uint32_t *seen = orch.fanin_seen_epoch.get();
    uint32_t slot = static_cast<uint32_t>(prod_local);
    if (seen[slot] == orch.fanin_seen_current_epoch) {
        return true;
    }
    seen[slot] = orch.fanin_seen_current_epoch;
    return false;
}

// Polling: fanin is a flat array of position-independent producer local ids in the
// payload's own fanin region (no dep-pool spill, no producer pointers), deduped
// against the current fanin_seen epoch and hard-capped at CHIP_MAX_FANIN.
// fanin_slots and fanin_count are that region and payload.fanin_count itself: the
// region is named by a SelfRelativePtr delta, so the caller resolves it once, and
// the count accumulates in place instead of being copied back at the end.
namespace simpler::hbg {
bool append_fanin_or_fail(
    OrchestratorState &orch, TaskId producer_task_id, int32_t *fanin_slots, int32_t &fanin_count
) {
    // Only a GLOBAL producer has an entry in the task table. A SUB_TASK id's low
    // bits are a packed (Graph task, task index) pair, so using them as a table
    // index names an unrelated task — or, since get_slot_state_by_task_id does not
    // bounds-check, no task at all. A recorded task's id is SUB_TASK and the
    // recorder resolves it against its own body, never here, so a foreign space
    // reaching this point is an id that escaped its Graph — a caller error, not a
    // case to tolerate.
    //
    // The table lookup lives here, after this check, rather than at the three call
    // sites: that keeps the id-space invariant in one place and makes it impossible
    // for a caller to form the out-of-bounds slot reference before reaching it.
    if (!producer_task_id.is_global()) {
        orch.report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__,
            "producer task %#" PRIx64 " is in id space %s, not GLOBAL; host_build_graph resolves every fanin edge "
            "against its one task table",
            TaskId::to_uint64(producer_task_id), producer_task_id.space_name()
        );
        return false;
    }
    // An id past what this run has claimed names no task: get_slot_state_by_task_id
    // does not bounds-check, so accepting it would record a fanin edge to an
    // unclaimed — or out-of-range — slot. Only ids handed back by a previous submit
    // are valid here, and those are below active_count() because hbg mints them in
    // order and never recycles one.
    const int32_t prod_local = producer_task_id.local_id();
    if (prod_local < 0 || prod_local >= orch.task_allocator.active_count()) {
        orch.report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__,
            "producer task %#" PRIx64 " names slot %d, which this run has not submitted (%d claimed)",
            TaskId::to_uint64(producer_task_id), prod_local, orch.task_allocator.active_count()
        );
        return false;
    }
    // Dedup by producer local id, which is also its task-table slot. A local id is
    // its own storage index and hbg never recycles a slot, so the entry that id
    // names is the producer's by construction — there is no stale-slot case to
    // screen for. A COMPLETED producer is a real fanin edge under polling (its
    // task_states byte says so), so it is not screened out either.
    if (fanin_mark_seen(orch, producer_task_id)) {
        return true;
    }
    if (fanin_count >= CHIP_MAX_FANIN) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Fanin Capacity Exhausted!");
        LOG_ERROR("========================================");
        LOG_ERROR("HBG stores every producer dependency in the consumer task's fanin region.");
        LOG_ERROR("  Fanin:     used=%d/%d", fanin_count, CHIP_MAX_FANIN);
        LOG_ERROR("  Requested: at least %d distinct producer dependencies", fanin_count + 1);
        LOG_ERROR("Solution:");
        LOG_ERROR("  Reduce the task fanin to at most CHIP_MAX_FANIN=%d.", CHIP_MAX_FANIN);
        LOG_ERROR("  HBG has no dependency spill pool; runtime_env.ring_dep_pool does not apply.");
        LOG_ERROR("========================================");
        orch_mark_fatal(&orch, SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED);
        return false;
    }
    fanin_slots[fanin_count++] = prod_local;
    return true;
}
}  // namespace simpler::hbg

struct PreparedTask {
    TaskId task_id = TaskId::invalid();
    TaskAllocResult alloc_result = {-1, nullptr, nullptr};
    TaskDescriptor *task = nullptr;
    TaskPayload *payload = nullptr;
    ChipTaskSlotState *slot_state = nullptr;
};

namespace simpler::hbg {
OutputLayout calculate_output_layout(const CoreTaskArgs &args) {
    OutputLayout layout;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            continue;
        }
        layout.offsets[i] = layout.total_output_size;
        layout.buffer_sizes[i] = CHIP_ALIGN_UP(args.tensor(i).create_info().buffer_size_bytes(), PACKED_OUTPUT_ALIGN);
        layout.total_output_size += layout.buffer_sizes[i];
    }
    return layout;
}
}  // namespace simpler::hbg

static bool prepare_task(
    OrchestratorState *orch, const CoreTaskArgs &args, int32_t total_output_size, ActiveMask active_mask,
    TaskAttrs task_attrs, PreparedTask *out
) {
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");
    auto &allocator = orch->task_allocator;

    int16_t block_num = args.launch_spec.block_num();
    int32_t active_subtasks_per_block = __builtin_popcount(active_mask.core_mask());
    int32_t total_required_subtasks = static_cast<int32_t>(block_num) * active_subtasks_per_block;
    if (block_num <= 0 || total_required_subtasks > std::numeric_limits<int16_t>::max()) {
        orch->report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__,
            "block_num=%d with %d active slots requires %d subtasks; expected block_num >= 1 and total <= %d",
            block_num, active_subtasks_per_block, total_required_subtasks, std::numeric_limits<int16_t>::max()
        );
        return false;
    }

    out->alloc_result = allocator.alloc(total_output_size);
    if (out->alloc_result.failed()) {
        orch_mark_fatal(orch, SIMPLER_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }

    out->task_id = TaskId::make_global(out->alloc_result.task_id);
    ChipTaskStorage &storage = orch->sm_header->tasks.storage_at(out->alloc_result.task_id);
    out->slot_state = &storage.slot;
    out->task = &storage.task;
    out->payload = &storage.payload;

    // Bind the three argument regions before prefetch() and init(), both of which
    // dereference them. The scalar cursor advances in whole cache lines because init()
    // rounds its scalar memcpy up to one; a packed advance would let that rounding
    // write into the next task's region. A tensor region is aligned for any count,
    // simpler::hbg::Tensor being two cache lines. The fanin cursor advances at publish, not
    // here — see the comment where it does.
    const uint64_t max_tasks = static_cast<uint64_t>(orch->task_allocator.capacity());
    const int32_t scalar_span = CHIP_ALIGN_UP(args.scalar_count(), ARG_POOL_ALIGN / (int32_t)sizeof(uint64_t));
    debug_assert(static_cast<uint64_t>(orch->tensor_pool_cursor) + args.tensor_count() <= max_tasks * MAX_TENSOR_ARGS);
    debug_assert(static_cast<uint64_t>(orch->scalar_pool_cursor) + scalar_span <= max_tasks * MAX_SCALAR_ARGS);
    debug_assert(static_cast<uint64_t>(orch->fanin_pool_cursor) + CHIP_MAX_FANIN <= max_tasks * CHIP_MAX_FANIN);
    out->payload->bind_regions(
        orch->tensor_pool + orch->tensor_pool_cursor, orch->scalar_pool + orch->scalar_pool_cursor,
        orch->fanin_pool + orch->fanin_pool_cursor
    );
    orch->tensor_pool_cursor += args.tensor_count();
    orch->scalar_pool_cursor += scalar_span;

    // Init-on-write: this slot's dynamic scheduling fields and progress state are
    // initialized here, as the orchestrator claims the slot. whole-graph-resident
    // hbg claims slots [0, total_tasks) exactly once and the device reads no slot
    // past total_tasks, so this claim-time write is the only per-slot SM reset and
    // the unclaimed tail is neither initialized nor read.
    out->slot_state->reset_for_reuse();
    orch->sm_header->tasks.reset_task_state(out->alloc_result.task_id);

    out->payload->prefetch(args.tensor_count(), args.scalar_count());

    // prepare_task does NO payload writes: all payload content (tensors/scalars +
    // early-dispatch fields) is initialized in TaskPayload::init, the
    // single payload-init point, which runs before Orch-side wiring publish.

    // Fields already zeroed by the reset_for_reuse() above:
    //   wake_list_head=nullptr, next_in_wake_list=nullptr,
    //   any_subtask_deferred=false, completed_subtasks=0, next_block_idx=0
    // (host_build_graph does not recycle slots at runtime, so there is no
    // post-CONSUMED reset path).
    out->slot_state->total_required_subtasks = static_cast<int16_t>(total_required_subtasks);
    out->slot_state->logical_block_num = block_num;
    out->slot_state->active_mask = active_mask;
    out->slot_state->task_attrs = task_attrs;
    out->slot_state->task_kind = active_mask.is_dummy() ? TaskKind::DUMMY : TaskKind::KERNEL;
    // payload.fanin_count is left untouched here: submit_task_common zeroes it before
    // its fanin appends, which accumulate into it in place.

    return true;
}

// =============================================================================
// Scope Management
// =============================================================================

void OrchestratorState::begin_scope(ScopeMode mode) {
    auto *orch = this;
    if (orch->is_fatal()) {
        return;
    }
    // A Graph replays as a flat DAG with no scope structure: scope boundaries only
    // shape scheduling on the ordinary path, and the shadow-record path submits no
    // ordinary tasks. So a scope inside a Graph body must not touch the real scope
    // stack. Its manual/auto mode still matters, though — the recorder infers a recorded
    // task's producers with the same compute_task_fanin the ordinary path uses, and
    // that inference is suppressed inside a manual scope — so the depth is tracked on
    // the recording instead.
    if (GraphRecording *recording = active_graph_recording(orch); recording != nullptr) {
        // Reject what the ordinary path rejects. An auto scope inside a manual one is a
        // fatal below; accepting it here would let a Graph record and replay a
        // body that ordinary submission refuses. The push still happens so
        // end_scope stays balanced -- the recording is doomed either way, since
        // graph_commit turns an unsupported recording into SIMPLER_ERROR_INVALID_ARGS.
        if (recording->scope_stack_top >= CHIP_MAX_SCOPE_DEPTH - 1 ||
            (mode == ScopeMode::AUTO && recording->in_manual_scope())) {
            recording->unsupported = true;
        }
        if (recording->scope_stack_top < CHIP_MAX_SCOPE_DEPTH - 1) {
            ++recording->scope_stack_top;
            if (mode == ScopeMode::MANUAL && !recording->in_manual_scope()) {
                recording->manual_begin_depth = recording->scope_stack_top;
            }
        }
        return;
    }
    assert(orch->scope_stack_top < CHIP_MAX_SCOPE_DEPTH - 1 && "Scope stack overflow");
    if (mode == ScopeMode::AUTO && orch->in_manual_scope()) {
        report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "auto scope nested inside manual scope is not supported"
        );
        return;
    }

    bool already_in_manual_scope = orch->in_manual_scope();
    ++orch->scope_stack_top;
    if (mode == ScopeMode::MANUAL && !already_in_manual_scope) {
        orch->manual_begin_depth = orch->scope_stack_top;
    }
}

void OrchestratorState::end_scope() {
    auto *orch = this;
    if (orch->is_fatal()) {
        return;
    }
    // Matches begin_scope: a scope inside a Graph body never touches the real
    // scope stack, only the recording's own manual-scope depth.
    if (GraphRecording *recording = active_graph_recording(orch); recording != nullptr) {
        if (recording->scope_stack_top >= 0) {
            if (recording->manual_begin_depth == recording->scope_stack_top) {
                recording->manual_begin_depth = CHIP_MAX_SCOPE_DEPTH;
            }
            --recording->scope_stack_top;
        }
        return;
    }
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

    if (orch->scope_stack_top == orch->manual_begin_depth) {
        orch->manual_begin_depth = CHIP_MAX_SCOPE_DEPTH;
    }
    --orch->scope_stack_top;
}

// =============================================================================
// Task Submission
// =============================================================================

// Ensure the tensormap entry pool has room for `needed` inserts before STEP 4
// registers this task's outputs. Device completion never reclaims TensorMap
// entries; only synchronous dependency computation can remove a covered
// producer before this check. A pool that is still short here therefore cannot
// become large enough while the host waits: latch
// SIMPLER_ERROR_TENSORMAP_OVERFLOW and bail rather than letting new_entry()'s hard
// assert fire mid-registration. Returns false when the pool is exhausted or a
// fatal is already latched.
namespace simpler::hbg {
bool ensure_tensormap_capacity(OrchestratorState *orch, int32_t needed) {
    ChipTensorMap &tm = orch->tensor_map;
    if (tm.free_entries() >= needed) {
        return true;
    }
    if (orch->is_fatal()) {
        return false;
    }

    LOG_ERROR("========================================");
    LOG_ERROR("FATAL: TensorMap Entry Pool Exhausted!");
    LOG_ERROR("========================================");
    LOG_ERROR("Device completion does not reclaim HBG TensorMap entries.");
    LOG_ERROR("  - Pool used:   %d / %d", tm.current_used(), tm.pool_capacity());
    LOG_ERROR("  - Free:        %d entries", tm.free_entries());
    LOG_ERROR("  - Needed:      %d entries", needed);
    LOG_ERROR("Solution:");
    LOG_ERROR("  Increase CHIP_TENSORMAP_POOL_SIZE (current: %d).", tm.pool_capacity());
    LOG_ERROR("========================================");
    orch_mark_fatal(orch, SIMPLER_ERROR_TENSORMAP_OVERFLOW);
    return false;
}
}  // namespace simpler::hbg

static bool
resolve_dispatch_predicate(OrchestratorState *orch, const CoreTaskPredicate &predicate, DispatchPredicate *resolved) {
    if (resolved == nullptr) return false;
    *resolved = DispatchPredicate{};
    if (predicate.op == PredicateOp::NONE) return true;

    switch (predicate.op) {
    case PredicateOp::EQ:
    case PredicateOp::NE:
    case PredicateOp::GT:
    case PredicateOp::LT:
    case PredicateOp::GE:
    case PredicateOp::LE:
        break;
    case PredicateOp::NONE:
        return true;
    default:
        orch->report_fatal(SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "dispatch predicate has an invalid operator");
        return false;
    }

    const simpler::hbg::Tensor *operand = predicate.operand.tensor;
    if (operand == nullptr || operand->address_space != AddressSpace::DEVICE || operand->buffer.addr == 0 ||
        predicate.operand.ndims == 0 || predicate.operand.ndims > operand->ndims ||
        predicate.operand.ndims > MAX_TENSOR_DIMS) {
        orch->report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "dispatch predicate has an invalid operand tensor"
        );
        return false;
    }

    uint64_t element_offset = operand->start_offset;
    for (uint32_t dim = 0; dim < predicate.operand.ndims; ++dim) {
        if (predicate.operand.indices[dim] >= operand->shapes[dim] ||
            (predicate.operand.indices[dim] != 0 &&
             operand->strides[dim] >
                 (UINT64_MAX - element_offset) / static_cast<uint64_t>(predicate.operand.indices[dim]))) {
            orch->report_fatal(
                SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "dispatch predicate index is outside the operand tensor"
            );
            return false;
        }
        element_offset +=
            static_cast<uint64_t>(predicate.operand.indices[dim]) * static_cast<uint64_t>(operand->strides[dim]);
    }

    const uint64_t element_size = get_element_size(operand->dtype);
    if ((element_size != 1 && element_size != 2 && element_size != 4 && element_size != 8) ||
        operand->buffer.size < element_size || element_offset > (operand->buffer.size - element_size) / element_size) {
        orch->report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "dispatch predicate element is outside the operand buffer"
        );
        return false;
    }
    const uint64_t byte_offset = element_offset * element_size;
    if (operand->buffer.addr > UINT64_MAX - byte_offset ||
        ((operand->buffer.addr + byte_offset) & (element_size - 1)) != 0) {
        orch->report_fatal(SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "dispatch predicate operand address is invalid");
        return false;
    }

    resolved->addr = operand->buffer.addr + byte_offset;
    resolved->target = predicate.target;
    resolved->elem_size = static_cast<uint8_t>(element_size);
    resolved->op = predicate.op;
    return true;
}

// Shared body for submit_task / submit_dummy_task. Caller has already validated
// args.has_error(), decided active_mask (empty for dummy), and resolved the per-slot
// kernel_ids (all INVALID_KERNEL_ID for dummy). Performs tensormap sync, fanin
// computation (explicit_deps + auto), output registration, slot init, and
// Orch-side wiring/ready publication.
static TaskOutputTensors submit_task_common(
    OrchestratorState *orch, const CoreTaskArgs &args, ActiveMask active_mask, TaskAttrs task_attrs,
    int32_t aic_kernel_id, int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    ORCH_STEP_START();
    ORCH_PHASE_START();
    TaskOutputTensors result;
    DispatchPredicate resolved_predicate{};
    if (!resolve_dispatch_predicate(orch, args.predicate(), &resolved_predicate)) return result;
    OutputLayout layout = calculate_output_layout(args);
    PreparedTask prepared;
    if (!prepare_task(orch, args, layout.total_output_size, active_mask, task_attrs, &prepared)) {
        return result;
    }
    TaskId task_id = prepared.task_id;
    TaskDescriptor &task = *prepared.task;
    TaskPayload &payload = *prepared.payload;
    result.set_task_id(task_id);

    // dep_gen capture point: open this task's graph entry before its dependency
    // steps run, so the edges STEP 1 / STEP 3 discover attach to it. The graph
    // is recorded from the dependency path itself, which makes it the runtime's
    // own answer rather than a reconstruction — the sole source of truth for
    // fanout now that the swimlane hot path no longer records it.
    const bool capture_dep_graph = dep_gen_host_graph_enabled();
    if (capture_dep_graph) {
        const std::array<int32_t, SUBTASK_SLOT_COUNT> kernel_ids_capture{
            aic_kernel_id,
            aiv0_kernel_id,
            aiv1_kernel_id,
        };
        dep_gen_host_graph_begin_task(
            task_id, orch->in_manual_scope(), args.allow_early_resolve(), kernel_ids_capture.data(),
            args.launch_spec.block_num(), args.tensor_count(), args.tensor_data(), args.tag_data()
        );
    }

    // The region delta is resolved once here, after prepare_task bound the regions.
    // Zeroing the count gives the appends their starting point, and is this
    // device-read field's only write on the submit path — hbg never zero-fills the
    // task table.
    next_fanin_seen_epoch(orch);
    int32_t *fanin_slots = payload.fanin_data();
    payload.fanin_count = 0;

    ORCH_STEP_LAP(g_orch_alloc_ns);

#if SIMPLER_DFX
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    for (uint32_t i = 0; i < args.explicit_dep_count(); i++) {
        TaskId dep_task_id = args.explicit_dep(i);
        if (!dep_task_id.is_valid()) {
            orch->report_fatal(
                SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "Arg.set_dependencies(...) requires valid task ids"
            );
            return result;
        }
        if (capture_dep_graph) {
            dep_gen_host_graph_add_explicit_edge(dep_task_id);
        }
        if (!append_fanin_or_fail(*orch, dep_task_id, fanin_slots, payload.fanin_count)) {
            return result;
        }
    }

    // === STEP 3: Lookup inputs (creator retention + tensormap modifier lookup) ===
    DepInputs dep_inputs{
        args.tensor_count(),       args.tensor_data(), args.tag_data(), static_cast<int32_t>(args.explicit_dep_count()),
        args.explicit_deps_data(),
    };

    auto runtime_emit = [&](TaskId producer_task_id) -> bool {
        return append_fanin_or_fail(*orch, producer_task_id, fanin_slots, payload.fanin_count);
    };

    // The capture branch instantiates compute_task_fanin with a live Annotate;
    // the plain branch keeps the un-annotated instantiation the hot path had.
    if (capture_dep_graph) {
        const bool ok =
            compute_task_fanin(dep_inputs, orch->tensor_map, orch->in_manual_scope(), runtime_emit, DepGraphAnnotate{});
        // STEP 3 is this task's last capture point, so the entry closes here
        // whether or not the fanin computation succeeded.
        dep_gen_host_graph_end_task();
        if (!ok) {
            return result;
        }
    } else {
        if (!compute_task_fanin(dep_inputs, orch->tensor_map, orch->in_manual_scope(), runtime_emit)) {
            return result;
        }
    }

    ORCH_STEP_LAP(g_orch_lookup_ns);

    // === STEP 4: Register outputs/inouts in TensorMap (must be separate from lookup) ===
    // Reserve pool capacity for this task's inserts before registering, so an
    // exhausted pool reports here rather than tripping new_entry()'s hard assert
    // mid-registration.
    int32_t tensormap_needed = count_registrable_outputs(dep_inputs, orch->in_manual_scope());
    if (tensormap_needed > 0 && !ensure_tensormap_capacity(orch, tensormap_needed)) {
        return result;
    }
    register_task_outputs(dep_inputs, task_id, orch->tensor_map, orch->in_manual_scope());

    ORCH_STEP_LAP(g_orch_insert_ns);

    // === STEP 5: Batch-write to GM (single cache line burst) ===
    // Deferred from allocation phase to avoid scattered GM writes that get
    // evicted by TensorMap lookup/insert cache pressure.
    __builtin_prefetch(&task, 1, 1);
    task.task_id = task_id;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIC)] = aic_kernel_id;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIV0)] = aiv0_kernel_id;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIV1)] = aiv1_kernel_id;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    // append_fanin_or_fail wrote every producer's local id into the payload's fanin
    // region and counted them in payload.fanin_count. payload.init writes
    // tensor_count/scalar_count only and must not touch either, or it would discard
    // that.
    payload.init(args, result, prepared.alloc_result, layout);

    // Predicate validation runs before task allocation. Copy the resolved, bounded
    // operand address into the device payload only after the rest of the payload exists.
    payload.predicate = resolved_predicate;
    ORCH_STEP_LAP(g_orch_args_ns);

    // === STEP 6: close the fanin region (device boot classifies) ===
    // Polling + host-orch: append_fanin_or_fail already wrote each producer's local
    // id into the payload's fanin region and counted them in payload.fanin_count.
    // There is NO fanout adjacency, NO dep_pool, and NO ready routing here — the
    // initial device boot scan classifies
    // each task once. A -1 result from classify_fanin_state routes the task through
    // push_ready_routed; otherwise the returned index selects the producer passed
    // to register_wake. Wake retargeting in register_wake may reclassify a task
    // when the selected producer is already complete.
    // The initial scan happens before the scheduler dispatch loop starts. Fanin is
    // a flat array of position-independent integers, so it crosses to the device
    // unchanged.
    // The region's length is settled, so the cursor closes it at the real count. The
    // equality holds only while nothing between the bind and here bound another fanin
    // region, which is what makes the deferred advance safe.
    debug_assert(orch->fanin_pool_cursor == static_cast<int32_t>(payload.fanin_data() - orch->fanin_pool));
    orch->fanin_pool_cursor += CHIP_ALIGN_UP(payload.fanin_count, ARG_POOL_ALIGN / (int32_t)sizeof(int32_t));

    // Early-dispatch qualification, decided once here where the fanin region is
    // final. The conjunction is order-independent, so it runs on the row as
    // filled; only a CANDIDATE's row is then sorted by producer local id — ids
    // come from the forward-only bump allocator, so ascending id order IS
    // submission order and the sorted row's tail names the latest-submitted
    // producer, the one the publish-list bet hangs on. Non-candidate rows keep
    // their fill order: a flag-free graph schedules exactly as it always has.
    int32_t *const fanin_row = payload.fanin_data();
    SharedMemoryTaskHeader &sm_tasks = orch->sm_header->tasks;
    bool ed_candidate =
        payload.fanin_count > 0 && !task_attrs.has_predicate() && active_mask.to_shape() != ResourceShape::DUMMY;
    for (int32_t i = 0; ed_candidate && i < payload.fanin_count; i++) {
        const ChipTaskSlotState &producer = sm_tasks.get_slot_state_by_task_id(fanin_row[i]);
        if (producer.task_kind == TaskKind::GRAPH) {
            // A Graph shell has no publication event, so its consumers schedule
            // normally.
            ed_candidate = false;
        } else if (!producer.task_attrs.allow_early_resolve()) {
            ed_candidate = false;
        }
    }
    if (ed_candidate) {
        std::sort(fanin_row, fanin_row + payload.fanin_count);
        prepared.slot_state->ed_flags |= ED_FLAG_CANDIDATE;
        // Every producer of a candidate must record its publication state.
        for (int32_t i = 0; i < payload.fanin_count; i++) {
            sm_tasks.get_slot_state_by_task_id(fanin_row[i]).ed_flags |= ED_FLAG_TRACKED;
        }
    }

    ORCH_STEP_LAP(g_orch_fanin_ns);
    ORCH_PHASE_END(HostPhaseKind::OrchSubmitTask, TaskId::to_uint64(task_id));

#if SIMPLER_DFX
    orch->tasks_submitted++;
#if SIMPLER_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif
    return result;
}

TaskOutputTensors OrchestratorState::submit_task(const MixedKernels &mixed_kernels, const CoreTaskArgs &args) {
    if (!require_device_arguments(this, args)) return {};
    auto *orch = this;

    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->is_fatal()) {
        return TaskOutputTensors{};
    }

    // Validate Arg construction (errors recorded by add_input/add_output/etc.)
    if (args.has_error()) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg Detected!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg() ? args.error_msg() : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("This is a bug in the orchestration code.");
        LOG_ERROR("========================================");
        orch_mark_fatal(orch, SIMPLER_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    // === Validate submit inputs ===
    ActiveMask active_mask = mixed_kernels.to_active_mask();
    if (!static_cast<bool>(active_mask)) {
        report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__,
            "MixedKernels names no active slot; set at least one of aic/aiv0/aiv1 kernel_id"
        );
        return TaskOutputTensors{};
    }

    int16_t block_num = args.launch_spec.block_num();

    // Normalize single-AIV tasks: if only aiv1 is set (no aic, no aiv0), move
    // it to the aiv0 slot.  This guarantees the dispatch path can always use
    // SubtaskSlot::AIV0 for single-AIV shapes without inspecting active_mask.
    // Mixed tasks (AIC+AIV) keep their original AIV identity so the correct
    // hardware channel (AIV0→AIC vs AIV1→AIC) is used at dispatch time.
    MixedKernels normalized = mixed_kernels;
    bool has_aic = active_mask.has_mask(SUBTASK_MASK_AIC);
    bool has_aiv0 = active_mask.has_mask(SUBTASK_MASK_AIV0);
    bool has_aiv1 = active_mask.has_mask(SUBTASK_MASK_AIV1);
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
        ResourceShape shape = active_mask.to_shape();
        int32_t limit = (shape == ResourceShape::AIV) ? orch->total_aiv_count : orch->total_cluster_count;
        if (limit > 0 && block_num > limit) {
            report_fatal(
                SIMPLER_ERROR_REQUIRE_SYNC_START_INVALID, __FUNCTION__,
                "require_sync_start block_num=%d > limit=%d (deadlock guaranteed)", block_num, limit
            );
            return TaskOutputTensors{};
        }
        task_attrs.set_sync_start();
    }

    if (args.predicate().op != PredicateOp::NONE) {
        const auto *operand = args.predicate().operand.tensor;
        if (operand != nullptr && operand->address_space != AddressSpace::DEVICE) {
            report_fatal(SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "dispatch predicate requires a DEVICE tensor");
            return {};
        }
        task_attrs.set_predicate();
    }

    if (active_graph_recording(orch) != nullptr) {
        return graph_record_submit_sub_task(
            orch, args, active_mask, task_attrs, normalized.aic_kernel_id, normalized.aiv0_kernel_id,
            normalized.aiv1_kernel_id
        );
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
TaskOutputTensors OrchestratorState::submit_dummy_task(const CoreTaskArgs &args) {
    if (!require_device_arguments(this, args)) return {};
    auto *orch = this;

    if (orch->is_fatal()) {
        return TaskOutputTensors{};
    }

    if (args.has_error()) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg in submit_dummy_task!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg() ? args.error_msg() : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("========================================");
        orch_mark_fatal(orch, SIMPLER_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }

    // Dummy tasks never dispatch to an AICore, so sync_start / has_predicate do
    // not apply; only the early-dispatch hint and timing tag carry over.
    TaskAttrs task_attrs;
    task_attrs.set_early_resolve(args.allow_early_resolve());
    task_attrs.set_timing_slot(args.task_timing_slot());

    if (active_graph_recording(orch) != nullptr) {
        return graph_record_submit_sub_task(
            orch, args, ActiveMask{}, task_attrs, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID
        );
    }

    return submit_task_common(
        orch, args, ActiveMask{}, task_attrs, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID
    );
}

TaskOutputTensors OrchestratorState::alloc_tensors(const CoreTaskArgs &args) {
    if (!require_device_arguments(this, args)) return {};
    auto *orch = this;
    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->is_fatal()) {
        return TaskOutputTensors{};
    }

    if (args.tensor_count() <= 0) {
        report_fatal(SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors requires at least one TensorCreateInfo");
        return TaskOutputTensors{};
    }
    if (args.scalar_count() != 0) {
        report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args"
        );
        return TaskOutputTensors{};
    }
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            report_fatal(
                SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args"
            );
            return TaskOutputTensors{};
        }
    }

    ORCH_STEP_START();
    ORCH_PHASE_START();

    if (args.has_error()) {
        report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg() ? args.error_msg() : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }

    // A Graph body may allocate. The allocation records as a kernel-less sub-task
    // — the same shape submit_dummy_task records — and replay reserves the
    // intermediate heap for every sub-task anyway, so the outputs land at
    // addresses the replayed Definition derives for itself.
    //
    // An allocation is transparent to early-dispatch qualification inside a body
    // exactly as it is at top level: its output is ready at creation, so it must
    // never be the unflagged producer that disqualifies a consumer. The top-level
    // path marks the slot after prepare_task, which this branch returns before, so
    // here the recorded attrs carry the mark instead.
    if (active_graph_recording(orch) != nullptr) {
        TaskAttrs alloc_attrs;
        alloc_attrs.set_early_resolve(true);
        return graph_record_submit_sub_task(
            orch, args, ActiveMask{}, alloc_attrs, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID
        );
    }

    OutputLayout layout = calculate_output_layout(args);
    PreparedTask prepared;
    // Kernel-less alloc task: no active subtasks, no dispatch-time attributes. The
    // early-dispatch hint is force-set below (see the flag-the-creator note).
    if (!prepare_task(orch, args, layout.total_output_size, ActiveMask{}, TaskAttrs{}, &prepared)) {
        return TaskOutputTensors{};
    }

    TaskDescriptor &task = *prepared.task;
    TaskPayload &payload = *prepared.payload;

    ORCH_STEP_LAP(g_orch_alloc_ns);

#if SIMPLER_DFX
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    task.task_id = prepared.task_id;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIC)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIV0)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIV1)] = INVALID_KERNEL_ID;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    TaskOutputTensors outputs;
    outputs.set_task_id(prepared.task_id);
    payload.init(args, outputs, prepared.alloc_result, layout);
    payload.fanin_count = 0;  // hidden-alloc tasks have no producer dependencies
    ORCH_STEP_LAP(g_orch_args_ns);

    if (prepared.slot_state != nullptr) {
        // Hidden alloc tasks complete inline in the orchestrator before any
        // consumer can exist, so they have no fanout to notify and no worker
        // subtasks to retire. Running the full on_task_complete path
        // would only pay unnecessary fanout_lock / traversal overhead here.
        // The generic slot initialization done in prepare_task() is still
        // required — a consumer reads this slot's task_attrs, set below — but
        // worker dispatch fields are never observed for hidden alloc tasks.
        //
        // Flag the creator so it does NOT suppress its consumers' early-dispatch.
        // Under the direct-only model an unflagged producer disqualifies its
        // consumer. A buffer allocation is pure memory whose output is ready at
        // creation — it should always be transparent, never a barrier. Unlike a
        // codegen task there is no Arg-driven hint to honor here, so mark it
        // unconditionally.
        prepared.slot_state->task_attrs.set_early_resolve(true);
        // Polling: pre-set the device-visible task_states byte in the H2D
        // image. That byte is the only completion a consumer polls, so a
        // hidden-alloc producer completed here on the host must publish it —
        // otherwise every consumer register_wakes on a producer that never runs
        // on device and the run hangs.
        SharedMemoryTaskHeader &done_tasks = orch->sm_header->tasks;
        int32_t done_local = prepared.task_id.local_id();
        done_tasks.store_completed(done_local);
    }
    orch->inline_completed_tasks++;

    ORCH_STEP_LAP(g_orch_fanin_ns);
    ORCH_PHASE_END(HostPhaseKind::OrchAllocTensors, TaskId::to_uint64(prepared.task_id));

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

void OrchestratorState::mark_done() {
    auto *orch = this;
    int32_t total_tasks = orch->task_allocator.active_count();
    if (total_tasks > 0) {
        LOG_DEBUG("=== [Orchestrator] total_tasks=%d ===", total_tasks);
    }
    orch->scope_stack_top = -1;
    orch->manual_begin_depth = CHIP_MAX_SCOPE_DEPTH;
    orch_profiling_mark_done();
}
