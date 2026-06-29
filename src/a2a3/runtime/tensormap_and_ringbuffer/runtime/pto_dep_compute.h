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
 * @file pto_dep_compute.h
 * @brief Dependency computation primitives shared by runtime submit_task and dep_gen replay.
 *
 * Two header-only template entry points:
 *
 *   compute_task_fanin     — STEP 3 in submit_task: per-tensor creator retention (Step A)
 *                            + tensormap.lookup for INPUT/INOUT (Step B). Calls back into
 *                            user-supplied `emit` for each producer it identifies.
 *
 *   register_task_outputs  — STEP 4 in submit_task: tensormap.insert for INOUT and
 *                            OUTPUT_EXISTING tensors. No callbacks.
 *
 * Dependency edge kinds (see DepKind in pto_types.h):
 *   Step A (creator / owner_task_id) emits DepKind::RESOURCE  — the consumer reads the
 *       producer's allocated buffer, so the producer must be retained until the consumer
 *       completes.
 *   Step B (tensormap modifier lookup) emits DepKind::EXECUTION — the producer only
 *       modified an already-allocated buffer, so only ordering (not retention) is needed.
 *
 * RESOURCE is strictly stronger than EXECUTION. compute_task_fanin runs in TWO passes
 * (all Step A RESOURCE emits first, then all Step B EXECUTION emits) so that when the
 * same producer is reached via both a creator edge and a modifier edge in one submit,
 * the runtime's mark_seen first-claim dedup always favors RESOURCE (retention wins).
 * The call site splits explicit deps by kind around this call (RESOURCE before,
 * EXECUTION after) for the same reason. Step A does not read or mutate the tensormap,
 * so reordering Step A ahead of Step B leaves the tensormap state and the emitted
 * producer SET identical to the old single-pass; only the edge ORDER changes (which
 * is irrelevant to fanin/fanout accounting).
 *
 * STEP 1 (explicit_deps) is intentionally left at the runtime call site because its
 * `last_task_alive` shortcut + unchecked slot lookup is subtly different from the
 * `slot_state->task->task_id == producer` reuse check in STEP 3. Unifying them would
 * require two emit semantics or a marginal behavior change in transients — not worth
 * the minor structural overlap. Replay handles STEP 1 with a one-line loop of its own.
 *
 * The Emit callback contract:
 *   bool emit(PTO2TaskId producer, DepKind kind);
 *     - return true to continue (whether or not the producer was actually recorded —
 *       producer-not-alive / dedup-hit / etc. all return true silently)
 *     - return false to signal fatal (e.g. fanin spill overflow); caller bails
 *
 * Performance: Emit is a template parameter, not std::function. Both runtime
 * (lambda capturing fanin_builder + sm_header) and replay (lambda capturing edge
 * vector) instantiate at the call site and inline through. Do NOT replace with
 * std::function — it would break the inlining and add ~5 ns/call to the orch hot path.
 */

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_DEP_COMPUTE_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_DEP_COMPUTE_H_

#include <cstdint>

#include "pto_task_id.h"
#include "pto_tensormap.h"
#include "pto_types.h"  // TensorRef
#include "tensor.h"

/**
 * View struct for inputs to compute_task_fanin / register_task_outputs.
 *
 * Both runtime and replay assemble one of these from their own data sources
 * (runtime: from Arg accessors; replay: from SubmitTraceEntry fields). All
 * pointer arrays must remain valid for the duration of the call.
 */
struct DepInputs {
    int32_t tensor_count;
    const TensorRef *tensors;        // length = tensor_count (union; OUTPUT slots' .ptr is unused)
    const TensorArgType *arg_types;  // length = tensor_count
    int32_t explicit_dep_count;
    const PTO2TaskId *explicit_deps;  // length = explicit_dep_count (validity checked by caller)
};

/**
 * Compute fanin for a task being submitted (STEP 3: Step A creator retention +
 * Step B tensormap modifier lookup).
 *
 * Two passes (RESOURCE before EXECUTION — see file header):
 *   Pass 1 (Step A): for every non-OUTPUT tensor with a valid owner_task_id,
 *       emit(owner, RESOURCE). Step A never reads/mutates the tensormap.
 *   Pass 2 (Step B): for INPUT/INOUT (not manual_dep), tensor_map.lookup and
 *       emit(producer, EXECUTION) for each overlapping entry. INOUT+COVERED
 *       triggers tensor_map.remove_entry(entry).
 *
 * @return true on success (or producer-skipped-silently); false if emit signaled
 *         fatal — caller should propagate (after any fatal bookkeeping done by emit).
 */
template <typename Emit>
[[nodiscard]] inline bool
compute_task_fanin(const DepInputs &inputs, PTO2TensorMap &tensor_map, bool in_manual_scope, Emit emit) {
    if (in_manual_scope) {
        return true;
    }

    // -------- Pass 1: Step A — creator retention (RESOURCE) --------
    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        if (inputs.arg_types[i] == TensorArgType::OUTPUT) {
            // Runtime-created OUTPUT tensors have no creator dependency to look up.
            continue;
        }
        const Tensor *tensor = &inputs.tensors[i].ref();
        PTO2TaskId owner = tensor->owner_task_id;
        if (owner.is_valid()) {
            if (!emit(owner, DepKind::RESOURCE)) {
                return false;
            }
        }
    }

    // -------- Pass 2: Step B — tensormap modifier lookup (EXECUTION) --------
    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype != TensorArgType::INPUT && ptype != TensorArgType::INOUT) {
            continue;
        }
        const Tensor *tensor = &inputs.tensors[i].ref();
        if (tensor->manual_dep) {
            continue;
        }

        bool fatal = false;
        tensor_map.lookup(*tensor, [&](PTO2TensorMapEntry &entry, OverlapStatus overlap_status) -> bool {
            if (!emit(entry.producer_task_id, DepKind::EXECUTION)) {
                fatal = true;
                return false;  // stop iteration
            }
            if (ptype == TensorArgType::INOUT && overlap_status == OverlapStatus::COVERED) {
                tensor_map.remove_entry(entry);
            }
            return true;
        });
        if (fatal) {
            return false;
        }
    }
    return true;
}

/**
 * Register a task's outputs in the tensormap (STEP 4 in submit_task).
 *
 * For INOUT and OUTPUT_EXISTING tensors (excluding manual_dep), inserts the
 * tensor into tensor_map keyed by its buffer.addr with `task_id` as producer.
 *
 * No-op when in_manual_scope.
 */
inline void
register_task_outputs(const DepInputs &inputs, PTO2TaskId task_id, PTO2TensorMap &tensor_map, bool in_manual_scope) {
    if (in_manual_scope) {
        return;
    }
    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING) {
            const Tensor *tensor = &inputs.tensors[i].ref();
            if (!tensor->manual_dep) {
                tensor_map.insert(*tensor, task_id);
            }
        }
    }
}

/**
 * Count the tensormap entries register_task_outputs() will insert for this task.
 *
 * Mirrors register_task_outputs()'s selection exactly (INOUT / OUTPUT_EXISTING,
 * excluding manual_dep), so the returned value is the precise number of
 * new_entry() calls that step makes. The orchestrator uses it to reserve pool
 * capacity before inserting. Returns 0 in a manual scope (no registration).
 */
inline int32_t count_registrable_outputs(const DepInputs &inputs, bool in_manual_scope) {
    if (in_manual_scope) {
        return 0;
    }
    int32_t needed = 0;
    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING) {
            if (!inputs.tensors[i].ref().manual_dep) {
                needed++;
            }
        }
    }
    return needed;
}

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_DEP_COMPUTE_H_
