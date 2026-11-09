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
 * @file dep_compute.h
 * @brief Dependency computation primitives shared by runtime submit_task and dep_gen replay.
 *
 * Two header-only template entry points:
 *
 *   compute_task_fanin     — STEP 3 in submit_task: per-tensor creator retention (Step A)
 *                            + tensormap.lookup for conflicting accesses (Step B). Calls back into
 *                            user-supplied `emit` for each producer it identifies.
 *
 *   register_task_accesses  — STEP 4 in submit_task: reader/writer access registration for INPUT,
 *                            INOUT and OUTPUT_EXISTING tensors. No callbacks.
 *
 * STEP 1 (explicit_deps) is intentionally left at the runtime call site because its
 * `last_task_alive` shortcut + unchecked slot lookup is subtly different from the
 * `slot_state->task->task_id == producer` reuse check in STEP 3. Unifying them would
 * require two emit semantics or a marginal behavior change in transients — not worth
 * the minor structural overlap. Replay handles STEP 1 with a one-line loop of its own.
 *
 * The Emit callback contract:
 *   bool emit(TaskId producer, DepFlags kind);
 *     - kind is DEP_WAIT|DEP_RETAIN for a Step-A creator edge (the consumer reads
 *       the producer's allocated buffer, so the producer is retained) and DEP_WAIT
 *       for a Step-B access edge (ordering only; the buffer was allocated
 *       elsewhere). Duplicate producers OR-accumulate their flags.
 *     - return true to continue (whether or not the producer was actually recorded —
 *       producer-not-alive / dedup-hit / etc. all return true silently)
 *     - return false to signal fatal (e.g. fanin spill overflow); caller bails
 *
 * Performance: Emit is a template parameter, not std::function. Both runtime
 * (lambda capturing fanin_builder + sm_header) and replay (lambda capturing edge
 * vector) instantiate at the call site and inline through. Do NOT replace with
 * std::function — it would break the inlining and add ~5 ns/call to the orch hot path.
 */

#pragma once

#include <cstdint>

#include "tensormap_and_ringbuffer/task_id.h"
#include "tensormap.h"
#include "types.h"  // TensorRef
#include "tensor.h"

/**
 * View struct for inputs to compute_task_fanin / register_task_accesses.
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
    const TaskId *explicit_deps;  // length = explicit_dep_count (validity checked by caller)
};

/**
 * Compute fanin for a task being submitted (STEP 3: Step A creator retention +
 * Step B conflicting-access lookup).
 *
 * For each non-OUTPUT tensor:
 *   - If owner_task_id is valid, emit(owner)
 *   - INPUT/INOUT wait for overlapping writers. INOUT/OUTPUT_EXISTING also
 *     wait for overlapping readers. Fully covered accesses are removed only
 *     after their dependency has been emitted successfully.
 *
 * @return true on success (or producer-skipped-silently); false if emit signaled
 *         fatal — caller should propagate (after any fatal bookkeeping done by emit).
 */
template <typename Emit>
[[nodiscard]] inline bool
compute_task_fanin(const DepInputs &inputs, ChipTensorMap &tensor_map, bool in_manual_scope, Emit emit) {
    if (in_manual_scope) {
        return true;
    }

    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype == TensorArgType::OUTPUT) {
            // Runtime-created OUTPUT tensors are not looked up in the TensorMap since
            // they have no dependencies.
            continue;
        }

        const simpler::tmr::Tensor *tensor = &inputs.tensors[i].ref();

        // Step A: creator retention — reading a tensor retains its allocator, so
        // the creator edge carries both ordering and lifetime.
        TaskId owner = tensor->owner_task_id;
        if (owner.is_valid()) {
            if (!emit(owner, DEP_WAIT | DEP_RETAIN)) {
                return false;
            }
        }

        if (tensor->manual_dep) continue;

        const bool writes = ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING;
        auto visit = [&](TensorAccessKind kind) {
            bool ok = true;
            tensor_map.lookup(
                *tensor,
                [&](ChipTensorMapEntry &entry, OverlapStatus overlap_status) {
                    if (!emit(entry.access_task_id, DEP_WAIT)) {
                        ok = false;
                        return false;
                    }
                    if (writes && overlap_status == OverlapStatus::COVERED) {
                        tensor_map.remove_entry(entry);
                    }
                    return true;
                },
                kind
            );
            return ok;
        };
        if ((ptype == TensorArgType::INPUT || ptype == TensorArgType::INOUT) && !visit(TensorAccessKind::WRITER))
            return false;
        if (writes && !visit(TensorAccessKind::READER)) return false;
    }
    return true;
}

/**
 * Register a task's outstanding accesses in the tensormap (STEP 4 in submit_task).
 *
 * INPUT publishes a reader; INOUT and OUTPUT_EXISTING publish writers.
 * manual_dep tensors do not publish accesses. Readers share the task cleanup
 * chain with writers, but have a separate hash index.
 *
 * No-op when in_manual_scope.
 */
inline void
register_task_accesses(const DepInputs &inputs, TaskId task_id, ChipTensorMap &tensor_map, bool in_manual_scope) {
    if (in_manual_scope) {
        return;
    }
    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype == TensorArgType::INPUT || ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING) {
            const simpler::tmr::Tensor *tensor = &inputs.tensors[i].ref();
            if (!tensor->manual_dep) {
                tensor_map.insert(
                    *tensor, task_id,
                    ptype == TensorArgType::INPUT ? TensorAccessKind::READER : TensorAccessKind::WRITER
                );
            }
        }
    }
}

/**
 * Count the tensormap entries register_task_accesses() will insert for this task.
 *
 * Mirrors register_task_accesses()'s selection exactly (INPUT / INOUT / OUTPUT_EXISTING,
 * excluding manual_dep), so the returned value is the precise number of
 * new_entry() calls that step makes. The orchestrator uses it to reserve pool
 * capacity before inserting. Returns 0 in a manual scope (no registration).
 */
inline int32_t count_registrable_accesses(const DepInputs &inputs, bool in_manual_scope) {
    if (in_manual_scope) {
        return 0;
    }
    int32_t needed = 0;
    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype == TensorArgType::INPUT || ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING) {
            if (!inputs.tensors[i].ref().manual_dep) {
                needed++;
            }
        }
    }
    return needed;
}
