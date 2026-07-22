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
 * STEP 1 (explicit_deps) stays at the runtime call site because it also
 * validates task-id ordering. Replay handles it with a matching direct loop.
 *
 * The Emit callback contract:
 *   bool emit(PTO2TaskId producer);
 *     - return true to continue (whether or not the producer was actually recorded —
 *       producer-not-alive / dedup-hit / etc. all return true silently)
 *     - return false to signal fatal (e.g. dep_pool overflow); caller bails
 *
 * Performance: Emit is a template parameter, not std::function. Both runtime
 * (lambda calling append_fanin_or_fail with the consumer slot + seen_epoch) and
 * replay (lambda capturing edge vector) instantiate at the call site and inline
 * through. Do NOT replace with std::function — it would break the inlining and
 * add ~5 ns/call to the orch hot path.
 */

#ifndef SRC_A2A3_RUNTIME_REPLAY_GRAPH_RUNTIME_PTO_DEP_COMPUTE_H_
#define SRC_A2A3_RUNTIME_REPLAY_GRAPH_RUNTIME_PTO_DEP_COMPUTE_H_

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
 * For each non-OUTPUT tensor:
 *   - If owner_task_id is valid, emit(owner)
 *   - For INPUT/INOUT (and not manual_dep), tensor_map.lookup(*tensor) and emit
 *     each matching producer. INOUT+COVERED triggers tensor_map.remove_entry(entry).
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

    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype == TensorArgType::OUTPUT) {
            // Runtime-created OUTPUT tensors are not looked up in the TensorMap since
            // they have no dependencies.
            continue;
        }

        const Tensor *tensor = &inputs.tensors[i].ref();

        // Step A: creator retention — all existing tensors extend their creator lifetime.
        PTO2TaskId owner = tensor->owner_task_id;
        if (owner.is_valid()) {
            if (!emit(owner)) {
                return false;
            }
        }

        // Step B: only INPUT/INOUT need modifier dependency lookup.
        if (ptype != TensorArgType::INPUT && ptype != TensorArgType::INOUT) {
            continue;
        }
        if (tensor->manual_dep) {
            continue;
        }

        bool fatal = false;
        tensor_map.lookup(*tensor, [&](PTO2TensorMapEntry &entry, OverlapStatus overlap_status) -> bool {
            if (!emit(entry.producer_task_id)) {
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

#endif  // SRC_A2A3_RUNTIME_REPLAY_GRAPH_RUNTIME_PTO_DEP_COMPUTE_H_
