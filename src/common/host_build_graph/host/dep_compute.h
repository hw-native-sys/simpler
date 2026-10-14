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
 * @brief Dependency computation primitives for runtime submit_task.
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
 * STEP 1 (explicit_deps) is intentionally left at the runtime call site because its
 * unchecked slot lookup is subtly different from the
 * `slot_state->task->task_id == producer` reuse check in STEP 3. Unifying them would
 * require two emit semantics or a marginal behavior change in transients — not worth
 * the minor structural overlap.
 *
 * The Emit callback contract:
 *   bool emit(TaskId producer);
 *     - return true to continue (whether or not the producer was actually recorded —
 *       producer-not-alive / dedup-hit / etc. all return true silently)
 *     - return false to signal fatal (e.g. fanin spill overflow); caller bails
 *
 * The Annotate callback contract (dep_gen graph capture):
 *   void annotate.creator(int32_t arg_idx, const simpler::hbg::Tensor &consumer, TaskId producer);
 *   void annotate.tensormap(int32_t arg_idx, const simpler::hbg::Tensor &consumer,
 *                           const ChipTensorMapEntry &entry, OverlapStatus overlap);
 * Both fire for exactly the producers `emit` saw, with the tensor-side context an
 * edge needs. `tensormap` runs before the INOUT/COVERED remove_entry(), so `entry`
 * is still live when it is read. NoDepAnnotate is the empty default and compiles
 * away.
 *
 * Performance: Emit and Annotate are template parameters, not std::function. The
 * runtime lambda (capturing the consumer's fanin region + count) instantiates at the
 * call site and inlines through. Do NOT replace with std::function — it would break
 * the inlining and add ~5 ns/call to the orch hot path.
 */

#pragma once

#include <cstdint>

#include "host_build_graph/task_id.h"
#include "host_build_graph/host/tensormap.h"
#include "host_build_graph/types.h"  // TensorRef
#include "host_build_graph/tensor.h"

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
    const TaskId *explicit_deps;  // length = explicit_dep_count (validity checked by caller)
};

/**
 * Empty Annotate: every hook is a no-op, so an un-annotated compute_task_fanin
 * costs exactly what it did before the hooks existed.
 */
struct NoDepAnnotate {
    void creator(int32_t, const simpler::hbg::Tensor &, TaskId) const {}
    void tensormap(int32_t, const simpler::hbg::Tensor &, const ChipTensorMapEntry &, OverlapStatus) const {}
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
template <typename Emit, typename Annotate = NoDepAnnotate>
[[nodiscard]] inline bool compute_task_fanin(
    const DepInputs &inputs, ChipTensorMap &tensor_map, bool in_manual_scope, Emit emit, Annotate annotate = Annotate{}
) {
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

        const simpler::hbg::Tensor *tensor = &inputs.tensors[i].ref();

        // Step A: creator retention — all existing tensors extend their creator lifetime.
        TaskId owner = tensor->owner_task_id;
        if (owner.is_valid()) {
            if (!emit(owner)) {
                return false;
            }
            annotate.creator(i, *tensor, owner);
        }

        // Step B: only INPUT/INOUT need modifier dependency lookup.
        if (ptype != TensorArgType::INPUT && ptype != TensorArgType::INOUT) {
            continue;
        }
        if (tensor->manual_dep) {
            continue;
        }

        bool fatal = false;
        tensor_map.lookup(*tensor, [&](ChipTensorMapEntry &entry, OverlapStatus overlap_status) -> bool {
            if (!emit(entry.producer_task_id)) {
                fatal = true;
                return false;  // stop iteration
            }
            annotate.tensormap(i, *tensor, entry, overlap_status);
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
register_task_outputs(const DepInputs &inputs, TaskId task_id, ChipTensorMap &tensor_map, bool in_manual_scope) {
    if (in_manual_scope) {
        return;
    }
    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING) {
            const simpler::hbg::Tensor *tensor = &inputs.tensors[i].ref();
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
